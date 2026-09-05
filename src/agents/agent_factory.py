"""智能体注册与工厂。

根据类型名创建对应的专业智能体实例；同时提供 ``get_all_agents``
一次创建全部智能体（供 Supervisor 初始化使用），以及 Supervisor 相关的
快捷创建方法（``create_supervisor`` / ``create_all_with_supervisor``）。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from agents.base_agent import BaseAgent
from agents.code_agent import CodeAgent
from agents.rag_agent import RAGAgent
from agents.summarizer_agent import SummarizerAgent
from agents.supervisor import SupervisorAgent
from agents.web_agent import WebSearchAgent

logger = logging.getLogger(__name__)

#: 类型名 -> 智能体类（Supervisor 派发依据）
_AGENT_TYPES: Dict[str, Any] = {
    "rag": RAGAgent,
    "code": CodeAgent,
    "web": WebSearchAgent,
    "summarizer": SummarizerAgent,
}

#: Supervisor 构造函数可接受的额外参数（其余按注册智能体用途过滤）
_SUPERVISOR_ARGS: frozenset = frozenset(
    {
        "tools",
        "verbose",
        "enable_parallel",
        "max_planning_attempts",
        "subtask_timeout",
    }
)


class AgentFactory:
    """按名称创建专业智能体的工厂类。"""

    _agent_map: Dict[str, Any] = _AGENT_TYPES

    @classmethod
    def create(cls, agent_type: str, **kwargs: Any) -> BaseAgent:
        """创建指定类型的智能体。

        Args:
            agent_type: 智能体类型名（rag / code / web / summarizer）。
            **kwargs: 透传给对应智能体构造函数的参数。

        Returns:
            对应类型的智能体实例。

        Raises:
            ValueError: 未知的智能体类型。
        """
        key = str(agent_type).strip().lower()
        if key not in cls._agent_map:
            raise ValueError(
                f"未知智能体类型: {agent_type!r}，可选：{' / '.join(cls._agent_map)}"
            )
        return cls._agent_map[key](**kwargs)

    @classmethod
    def get_all_agents(
        cls,
        llm: Any = None,
        rag_chain: Any = None,
        **kwargs: Any,
    ) -> Dict[str, BaseAgent]:
        """一次创建所有智能体实例（供 Supervisor 默认配置）。

        ``rag`` / ``summarizer`` 依赖 RAG 链，未提供 ``rag_chain`` 时跳过
        并记录 WARNING；其余智能体始终创建。

        Args:
            llm: 供各智能体共享的大语言模型实例。
            rag_chain: 高级 RAG 链（可选，缺少则跳过 RAG / Summarizer）。
            **kwargs: 其余可选参数（workspace_dir / search_api 等）。

        Returns:
            ``{类型名: 智能体实例}`` 字典。
        """
        agents: Dict[str, BaseAgent] = {}
        for name, agent_class in cls._agent_map.items():
            if name in ("rag", "summarizer"):
                if rag_chain is None:
                    logger.warning("未提供 rag_chain，跳过 %s 智能体", name)
                    continue
                instance = agent_class(llm=llm, rag_chain=rag_chain, **kwargs)
            else:
                instance = agent_class(llm=llm, **kwargs)
            agents[name] = instance
        return agents

    @classmethod
    def create_supervisor(
        cls,
        llm: Any,
        agents: Dict[str, BaseAgent],
        **kwargs: Any,
    ) -> SupervisorAgent:
        """创建监督者智能体，并注册所有专业子智能体。

        仅透传 Supervisor 支持的参数（enable_parallel / max_planning_attempts /
        subtask_timeout / tools / verbose），其余面向子智能体的配置
        （rag_chain / workspace_dir / search_api 等）不会透传给 Supervisor。

        Args:
            llm: 大语言模型实例。
            agents: ``{"rag": ..., "code": ..., ...}`` 专业智能体字典。
            **kwargs: Supervisor 可选参数。

        Returns:
            SupervisorAgent 实例。
        """
        supervisor_kwargs: Dict[str, Any] = {
            key: value for key, value in kwargs.items() if key in _SUPERVISOR_ARGS
        }
        ignored = sorted(set(kwargs) - _SUPERVISOR_ARGS)
        if ignored:
            logger.debug("忽略面向子智能体的参数（Supervisor 不使用）: %s", ignored)
        return SupervisorAgent(llm=llm, agents=agents, **supervisor_kwargs)

    @classmethod
    def create_all_with_supervisor(
        cls,
        llm: Any,
        rag_chain: Any = None,
        **kwargs: Any,
    ) -> Dict[str, BaseAgent]:
        """创建全部专业智能体并装配 Supervisor（便于快速部署）。

        Args:
            llm: 大语言模型实例（专业智能体与 Supervisor 共享）。
            rag_chain: 高级 RAG 链；缺失时自动跳过 RAG / Summarizer。
            **kwargs: 其余参数（workspace_dir / search_api / enable_parallel 等）。

        Returns:
            ``{"supervisor": supervisor, "rag": ..., "code": ..., ...}`` 字典。
        """
        specialists = cls.get_all_agents(llm=llm, rag_chain=rag_chain, **kwargs)
        supervisor = cls.create_supervisor(llm=llm, agents=specialists, **kwargs)
        return {"supervisor": supervisor, **specialists}
