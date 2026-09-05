"""专业智能体抽象基类。

定义所有 Agent 的统一骨架：名称 / 能力描述（供 Supervisor 决策）、LLM、
专属工具集、LangChain AgentExecutor 的懒构建与执行，以及统一的结果结构。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

from config import settings

logger = logging.getLogger(__name__)


def _text_of(response: Any) -> str:
    """将 LLM 返回归一化为纯文本。"""
    if isinstance(response, str):
        return response
    content = getattr(response, "content", response)
    if isinstance(content, list):
        parts = [
            str(item["text"]) if isinstance(item, dict) and "text" in item else str(item)
            for item in content
        ]
        return "".join(parts)
    return str(content)


class BaseAgent(ABC):
    """所有专业智能体的抽象基类。

    Attributes:
        name: 智能体名称（用于 Supervisor 识别）。
        description: 智能体能力描述（用于 Supervisor 决策）。
        llm: 大语言模型实例。
        tools: 该智能体的工具列表。
        verbose: 是否打印详细执行日志。
        _agent_executor: LangChain AgentExecutor（懒构建）。
    """

    def __init__(
        self,
        name: str,
        description: str,
        llm: BaseLanguageModel,
        tools: Optional[List[BaseTool]] = None,
        verbose: Optional[bool] = None,
    ) -> None:
        """初始化智能体。

        Args:
            name: 智能体名称。
            description: 智能体能力描述。
            llm: 大语言模型实例（需支持 ``invoke``；tool-calling 路径额外
                需要支持 ``bind_tools``，否则自动降级为直答模式）。
            tools: 工具列表；未提供时调用 :meth:`_get_default_tools`。
            verbose: 是否打印详细日志，默认取配置 ``AGENT_VERBOSE``。
        """
        self.name: str = name
        self.description: str = description
        self.llm: BaseLanguageModel = llm
        self.verbose: bool = (
            settings.AGENT_VERBOSE if verbose is None else bool(verbose)
        )
        # 注意：tools 需在 llm 等子类依赖属性就位后再构建（子类的
        # _get_default_tools 可能引用 self.llm / self.rag_chain 等）
        self.tools: List[BaseTool] = tools if tools is not None else self._get_default_tools()
        self._agent_executor: Optional[Any] = None
        self._direct_mode: bool = False
        logger.info(
            "智能体 %s 初始化完成：%d 个工具（%s）",
            self.name,
            len(self.tools),
            ", ".join(tool.name for tool in self.tools) or "无",
        )

    # ------------------------------------------------------------------ #
    # 子类必须实现
    # ------------------------------------------------------------------ #
    @abstractmethod
    def _get_default_tools(self) -> List[BaseTool]:
        """返回该智能体的默认工具集（子类必须实现）。"""

    @abstractmethod
    def _get_system_prompt(self) -> str:
        """返回该智能体的系统提示词（子类必须实现）。"""

    # ------------------------------------------------------------------ #
    # AgentExecutor 构建
    # ------------------------------------------------------------------ #
    def _build_agent(self) -> Any:
        """构建 LangChain AgentExecutor（tool-calling），并缓存到实例。

        采用 ``create_tool_calling_agent``；若当前 LLM 不支持工具调用
        （无 ``bind_tools``）或依赖不可用，记录 WARNING 并降级为直答模式
        （:meth:`_direct_answer`），保证基本问答可用。

        Returns:
            AgentExecutor 实例；直答模式下返回 None。

        Raises:
            ValueError: 智能体没有任何工具。
        """
        if self._agent_executor is not None:
            return self._agent_executor
        if not self.tools:
            raise ValueError(f"智能体 {self.name} 缺少工具，无法构建执行器")
        try:
            from langchain.agents import AgentExecutor, create_tool_calling_agent
            from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", self._get_system_prompt()),
                    ("human", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                ]
            )
            agent = create_tool_calling_agent(self.llm, self.tools, prompt)
            self._agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=self.verbose,
                max_iterations=settings.AGENT_MAX_ITERATIONS,
                handle_parsing_errors=True,
                return_intermediate_steps=True,
            )
        except (ImportError, AttributeError, TypeError, ValueError, NotImplementedError) as exc:
            logger.warning(
                "智能体 %s 无法构建 tool-calling 执行器（%s），已降级为直答模式",
                self.name,
                exc,
            )
            self._direct_mode = True
            self._agent_executor = None
        return self._agent_executor

    def _direct_answer(self, query: str) -> str:
        """直答降级：将系统提示与任务拼接后单次调用 LLM。"""
        prompt_text = f"{self._get_system_prompt()}\n\n用户任务：{query}"
        return _text_of(self.llm.invoke(prompt_text)).strip()

    # ------------------------------------------------------------------ #
    # 对外接口
    # ------------------------------------------------------------------ #
    def run(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        """执行一次智能体任务。

        Args:
            query: 用户输入或由 Supervisor 下发的子任务描述。
            **kwargs: 预留参数（当前仅记录日志，供后续扩展传入会话上下文等）。

        Returns:
            统一结果字典：``output``（最终文本）、``intermediate_steps``
            （工具调用中间步）、``tools_used``（实际使用过的工具名列表）。
            直答模式下 ``intermediate_steps`` 与 ``tools_used`` 为空。

        Raises:
            RuntimeError: 执行失败（含模型不可用等）。
        """
        if kwargs:
            logger.debug("智能体 %s 收到附加参数: %s", self.name, sorted(kwargs))
        self._build_agent()

        if self._direct_mode or self._agent_executor is None:
            output = self._direct_answer(query)
            logger.info("智能体 %s（直答模式）完成", self.name)
            return {"output": output, "intermediate_steps": [], "tools_used": []}

        try:
            result = self._agent_executor.invoke({"input": query})
        except Exception as exc:  # noqa: BLE001 - 统一包装为清晰异常
            logger.error("智能体 %s 执行失败: %s", self.name, exc)
            raise RuntimeError(f"智能体 {self.name} 执行失败: {exc}") from exc

        output = str(result.get("output", "")).strip()
        intermediate = list(result.get("intermediate_steps", []) or [])
        tools_used = sorted({step[0].tool for step in intermediate}) if intermediate else []
        logger.info(
            "智能体 %s 完成：输出 %d 字符，使用工具 %s",
            self.name,
            len(output),
            tools_used or "无",
        )
        return {
            "output": output,
            "intermediate_steps": intermediate,
            "tools_used": tools_used,
        }

    def get_tools_description(self) -> str:
        """返回工具描述（供 Supervisor 决策 / 展示）。"""
        return "\n".join(
            f"- {tool.name}: {tool.description}" for tool in self.tools
        )
