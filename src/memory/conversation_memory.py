"""多会话对话记忆管理模块。

提供 :class:`ConversationMemoryManager`，为 RAG 多轮对话提供上下文记忆能力。
支持三种记忆策略：

- ``buffer``: 保留全部消息，不修剪（适合短对话）；
- ``buffer_window``: 滑动窗口。对外仅暴露最近 ``max_window_size`` 条消息，
  且当内部消息数超过 ``2 * max_window_size`` 时物理裁剪最早消息（节省内存）；
- ``summary``: 借助 LLM 将最早溢出的消息增量压缩为摘要（SystemMessage），
  适合超长对话场景下的 Token 控制。

同时提供：

1. 与 LangChain :class:`BaseChatMemory` 兼容的鸭子接口（``save_context`` /
   ``load_memory_variables``），便于接入 LangChain Chain / LangGraph 节点；
2. 基于 ``session_id`` 的多会话隔离——同一会话 id 全局复用同一实例（单例）；
3. :meth:`build_context_with_history`，为 Day 5 注入 RAG 提示词预留的历史拼接。

设计上不依赖 ``langchain.memory``，仅使用 ``langchain_core.messages`` 消息原语，
保持轻量。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain_core import messages as lc_messages
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from config import settings

logger = logging.getLogger(__name__)

#: 支持的记忆类型
_SUPPORTED_TYPES: tuple[str, ...] = ("buffer", "buffer_window", "summary")

#: 消息 type -> 对话文本中的角色标签
_TYPE_LABELS: Dict[str, str] = {
    "human": "用户",
    "ai": "AI",
    "system": "系统",
    "tool": "工具",
    "function": "工具",
    "generic": "消息",
    "ai_chunk": "AI",
    "human_chunk": "用户",
    "system_chunk": "系统",
}

#: summary 模式下用于引导 LLM 压缩的系统提示
_SUMMARY_SYSTEM_PROMPT: str = (
    "你是一个对话记忆压缩助手。请把给定的历史对话压缩成一段简洁的中文摘要，"
    "尽可能保留：用户的身份信息与偏好、已讨论过的关键事实与结论、"
    "尚未解决或仍待回答的问题。不要编造对话中不存在的内容。"
)


def _content_to_text(content: Any) -> str:
    """将消息 content（字符串 / 多模态块列表 / 其他）展平为纯文本。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", block)))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def _role_label(message: BaseMessage) -> str:
    """根据消息类型返回可读的角色标签（如 用户 / AI / 系统）。"""
    label = _TYPE_LABELS.get(message.type)
    if label is not None:
        return label
    # 兜底：ChatMessage 等使用 role；其余回退为原始 type
    return str(getattr(message, "role", None) or message.type)


def _render_line(message: BaseMessage) -> str:
    """将单条消息渲染为 ``角色: 内容`` 的一行文本。"""
    return f"{_role_label(message)}: {_content_to_text(message.content)}"


def _serialize_message(message: BaseMessage) -> Dict[str, Any]:
    """将单条消息序列化为可 JSON 化的字典（含类型与可选标量字段）。"""
    data: Dict[str, Any] = {
        "class": type(message).__name__,
        "content": message.content,
    }
    for field in ("name", "tool_call_id"):
        value = getattr(message, field, None)
        if value is not None:
            data[field] = value
    return data


def _deserialize_message(data: Dict[str, Any]) -> Optional[BaseMessage]:
    """从字典重建消息；无法重建时返回 None 并告警（调用方跳过）。"""
    class_name = data.get("class", "")
    message_class = getattr(lc_messages, class_name, None)
    if (
        not isinstance(message_class, type)
        or not issubclass(message_class, BaseMessage)
        or message_class is BaseMessage
    ):
        logger.warning("无法识别的消息类型 %r，已跳过该条记录", class_name)
        return None
    try:
        kwargs: Dict[str, Any] = {"content": data.get("content")}
        for field in ("name", "tool_call_id"):
            if field in data and data[field] is not None:
                kwargs[field] = data[field]
        return message_class(**kwargs)
    except Exception as exc:  # noqa: BLE001 - 反序列化容错
        logger.warning("重建消息 %r 失败（%s），已跳过该条记录", class_name, exc)
        return None


class ConversationMemoryManager:
    """对话记忆管理器（多会话隔离，同一 session_id 全局复用）。

    消息存储在实例属性 ``messages`` 中；对外暴露的可见历史由
    :meth:`get_messages` 提供：

    - ``buffer`` / ``buffer_window``: 直接返回内部消息列表的拷贝；
    - ``summary``: 返回「摘要 SystemMessage + 近期未压缩消息」。

    类型未显式传入的参数（传 ``None``）将从全局配置 ``config.settings`` 读取，
    以支持通过 .env 统一调整，避免硬编码。

    Attributes:
        session_id: 会话唯一标识；为 None 表示独立（不入全局会话缓存）。
        messages: 内部存储的消息列表（summary 类型仅保留未压缩的近期消息）。
    """

    #: session_id -> 已创建的会话实例（类级缓存，实现会话级单例）
    _store: Dict[str, "ConversationMemoryManager"] = {}

    def __new__(
        cls,
        max_window_size: Optional[int] = None,
        memory_type: Optional[str] = None,
        llm: Optional[BaseLanguageModel] = None,
        summary_max_tokens: Optional[int] = None,
        return_messages: Optional[bool] = None,
        session_id: Optional[str] = None,
    ) -> "ConversationMemoryManager":
        """若目标会话已存在则直接复用实例（忽略本次其余参数）。"""
        if session_id is not None and session_id in cls._store:
            instance = cls._store[session_id]
            logger.warning(
                "会话 %r 已存在 ConversationMemoryManager 实例，直接复用（忽略本次其余参数）",
                session_id,
            )
            return instance
        return super().__new__(cls)

    def __init__(
        self,
        max_window_size: Optional[int] = None,
        memory_type: Optional[str] = None,
        llm: Optional[BaseLanguageModel] = None,
        summary_max_tokens: Optional[int] = None,
        return_messages: Optional[bool] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """初始化对话记忆管理器。

        Args:
            max_window_size: 保留 / 暴露给外部的最近消息条数
                （``buffer_window`` 生效；``summary`` 的压缩阈值），
                默认取配置 ``MEMORY_WINDOW_SIZE``。
            memory_type: 记忆类型（小写不敏感）：
                ``buffer`` 全量保留；``buffer_window`` 滑动窗口；
                ``summary`` LLM 摘要压缩。默认取配置 ``MEMORY_TYPE``。
            llm: 用于 ``summary`` 类型的 LLM；缺省时 ``summary`` 自动降级为
                ``buffer_window`` 并记录 WARNING。
            summary_max_tokens: 生成摘要的最大 Token 约束，
                默认取配置 ``MEMORY_SUMMARY_MAX_TOKENS``。
            return_messages: 记忆变量以消息对象返回（True）还是文本（False），
                默认取配置 ``MEMORY_RETURN_MESSAGES``。
            session_id: 会话唯一标识；提供后进入全局会话缓存，
                相同 session_id 复用同一实例。

        Raises:
            ValueError: ``memory_type`` 不支持、``max_window_size`` 或
                ``summary_max_tokens`` 小于 1。
        """
        if getattr(self, "_initialized", False):
            return

        self.max_window_size: int = (
            settings.MEMORY_WINDOW_SIZE
            if max_window_size is None
            else int(max_window_size)
        )
        if self.max_window_size < 1:
            raise ValueError("max_window_size 必须大于等于 1")

        raw_type = settings.MEMORY_TYPE if memory_type is None else memory_type
        self.memory_type: str = raw_type.strip().lower()
        if self.memory_type not in _SUPPORTED_TYPES:
            raise ValueError(
                f"不支持的记忆类型 {self.memory_type!r}，"
                f"可选：{' / '.join(_SUPPORTED_TYPES)}"
            )

        self.llm: Optional[BaseLanguageModel] = llm
        self.summary_max_tokens: int = (
            settings.MEMORY_SUMMARY_MAX_TOKENS
            if summary_max_tokens is None
            else int(summary_max_tokens)
        )
        if self.summary_max_tokens < 1:
            raise ValueError("summary_max_tokens 必须大于等于 1")

        self.return_messages: bool = (
            settings.MEMORY_RETURN_MESSAGES
            if return_messages is None
            else bool(return_messages)
        )
        self.session_id: Optional[str] = session_id

        #: 内部消息列表（summary 类型只保留未压缩的近期消息）
        self.messages: List[BaseMessage] = []
        #: 滚动摘要文本（仅 summary 类型使用）
        self._summary_text: Optional[str] = None

        if self.memory_type == "summary" and self.llm is None:
            logger.warning(
                "memory_type='summary' 需要传入 llm 实例；未提供，已降级为 'buffer_window'"
            )
            self.memory_type = "buffer_window"

        self._initialized = True
        if session_id is not None:
            type(self)._store[session_id] = self

    # ------------------------------------------------------------------ #
    # 会话级单例入口
    # ------------------------------------------------------------------ #
    @classmethod
    def get_session(
        cls, session_id: str, **kwargs: Any
    ) -> "ConversationMemoryManager":
        """返回指定会话的记忆管理器（不存在则创建）。

        Args:
            session_id: 会话唯一标识。
            **kwargs: 透传给 :class:`ConversationMemoryManager` 的构造参数
                （仅首次创建时生效）。

        Returns:
            对应会话的记忆管理器实例。
        """
        if session_id in cls._store:
            logger.warning("会话 %r 已存在，复用现有实例（忽略其余参数）", session_id)
            return cls._store[session_id]
        return cls(session_id=session_id, **kwargs)

    # ------------------------------------------------------------------ #
    # 消息添加
    # ------------------------------------------------------------------ #
    def add_user_message(self, message: str) -> None:
        """添加一条用户消息（封装为 HumanMessage 后存储）。"""
        self.add_message(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        """添加一条 AI 回复（封装为 AIMessage 后存储）。"""
        self.add_message(AIMessage(content=message))

    def add_message(self, message: BaseMessage) -> None:
        """通用添加消息，支持任意 ``BaseMessage`` 子类型。

        Args:
            message: 任意消息对象（HumanMessage / AIMessage / SystemMessage 等）。

        Raises:
            TypeError: 传入非 ``BaseMessage`` 子类对象。
        """
        if not isinstance(message, BaseMessage):
            raise TypeError(
                f"仅支持 BaseMessage 子类，收到 {type(message).__name__}"
            )
        self.messages.append(message)
        logger.debug(
            "会话 %s 新增消息：type=%s",
            self.session_id or "(standalone)",
            message.type,
        )
        # 按记忆策略自动修剪 / 压缩（buffer 为空操作）
        self.prune_if_needed()

    # ------------------------------------------------------------------ #
    # 消息读取
    # ------------------------------------------------------------------ #
    def _visible_messages(self) -> List[BaseMessage]:
        """组装对外可见的历史消息列表（summary 类型头部拼接摘要）。"""
        if self.memory_type == "summary" and self._summary_text:
            return [SystemMessage(content=self._summary_text)] + list(self.messages)
        return list(self.messages)

    def get_messages(self) -> List[BaseMessage]:
        """获取当前会话的所有可见消息（返回拷贝，避免外部误改）。

        Returns:
            消息列表：buffer / buffer_window 为内部全量；summary 为
            摘要消息 + 近期未压缩消息。
        """
        return self._visible_messages()

    def get_messages_windowed(self) -> List[BaseMessage]:
        """获取窗口内消息：仅返回最近 ``max_window_size`` 条可见消息。

        Returns:
            最近 ``max_window_size`` 条消息（不足则全部返回）。
        """
        return self._visible_messages()[-self.max_window_size :]

    def get_chat_history(self) -> str:
        """将可见历史拼接为可读文本（用于 prompt 注入）。

        每行格式为 ``角色: 内容``，行间以换行分隔，例如::

            用户: 你好
            AI: 你好！有什么可以帮你？

        Returns:
            拼接后的历史字符串。
        """
        return "\n".join(_render_line(msg) for msg in self._visible_messages())

    def clear(self) -> None:
        """清空当前会话的所有消息（含滚动摘要）。"""
        removed = len(self.messages)
        self.messages = []
        self._summary_text = None
        logger.info("会话 %s 已清空记忆（移除 %d 条消息）", self.session_id or "(standalone)", removed)

    # ------------------------------------------------------------------ #
    # 修剪 / 压缩策略
    # ------------------------------------------------------------------ #
    def prune_if_needed(self) -> None:
        """按记忆策略修剪 / 压缩历史消息。

        - ``buffer``: 不修剪；
        - ``buffer_window``: 消息数超过 ``2 * max_window_size`` 时，
          物理删除最早消息，仅保留最近 ``max_window_size`` 条；
        - ``summary``: 未压缩消息数超过 ``max_window_size`` 时，将最早溢出的
          消息增量压缩为摘要（旧消息替换为一条摘要 SystemMessage）。
          LLM 调用失败时降级为 ``buffer_window`` 并记录 WARNING。
        """
        if self.memory_type == "summary":
            self._prune_summary()
        elif self.memory_type == "buffer_window":
            self._prune_buffer_window()

    def _prune_buffer_window(self) -> None:
        """buffer_window 物理裁剪：超过阈值仅保留最近窗口消息。"""
        if len(self.messages) <= self.max_window_size * 2:
            return
        removed = len(self.messages) - self.max_window_size
        self.messages = self.messages[-self.max_window_size :]
        logger.info(
            "buffer_window 修剪：删除最早 %d 条消息，保留最近 %d 条",
            removed,
            self.max_window_size,
        )

    def _prune_summary(self) -> None:
        """summary 增量压缩：溢出部分并入滚动摘要。"""
        if self.llm is None:
            logger.warning("summary 类型缺少 llm，降级为 buffer_window")
            self.memory_type = "buffer_window"
            self._prune_buffer_window()
            return
        if len(self.messages) <= self.max_window_size:
            return
        overflow = self.messages[: -self.max_window_size]
        keep = self.messages[-self.max_window_size :]
        try:
            summary_text = self._build_summary(overflow)
        except Exception as exc:  # noqa: BLE001 - LLM 服务不可用等
            logger.warning(
                "摘要生成失败（%s），已降级为 buffer_window 并保留原消息", exc
            )
            self.memory_type = "buffer_window"
            self._prune_buffer_window()
            return
        self._summary_text = summary_text
        self.messages = keep
        logger.info(
            "summary 压缩：将 %d 条旧消息并入摘要（保留最近 %d 条）",
            len(overflow),
            self.max_window_size,
        )

    def _build_summary(self, overflow: List[BaseMessage]) -> str:
        """调用 LLM 将溢出消息（含此前滚动摘要）压缩为一段摘要文本。"""
        lines: List[str] = []
        if self._summary_text:
            lines.extend(["【此前摘要】", self._summary_text, ""])
        lines.append("【新增待压缩对话】")
        lines.extend(_render_line(msg) for msg in overflow)
        prompt_text = "\n".join(lines)
        prompt = HumanMessage(
            content=(
                f"{prompt_text}\n\n"
                f"请用中文生成摘要，控制在约 {self.summary_max_tokens} 个 token 以内。"
            )
        )
        response: Any = self.llm.invoke(  # type: ignore[union-attr]
            [SystemMessage(content=_SUMMARY_SYSTEM_PROMPT), prompt]
        )
        if isinstance(response, str):
            text = response
        else:
            content = getattr(response, "content", None)
            text = _content_to_text(content) if content is not None else str(response)
        if not text.strip():
            raise ValueError("LLM 返回了空摘要")
        return text.strip()

    # ------------------------------------------------------------------ #
    # LangChain BaseChatMemory 兼容接口（鸭子类型）
    # ------------------------------------------------------------------ #
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """保存一轮对话（与 LangChain ``BaseChatMemory.save_context`` 兼容）。

        从 ``inputs`` 提取用户输入（优先键 ``input`` / ``question``），
        从 ``outputs`` 提取 AI 输出（优先键 ``output`` / ``answer``）；
        若取值已是 ``BaseMessage`` 则原样存储，否则包装为对应消息。

        Args:
            inputs: 链输入字典。
            outputs: 链输出字典。
        """
        input_value = self._extract_value(inputs, ("input", "question"))
        output_value = self._extract_value(outputs, ("output", "answer"))
        if input_value is not None:
            message = input_value if isinstance(input_value, BaseMessage) else HumanMessage(content=_content_to_text(input_value))
            self.add_message(message)
        if output_value is not None:
            message = output_value if isinstance(output_value, BaseMessage) else AIMessage(content=_content_to_text(output_value))
            self.add_message(message)

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """加载记忆变量，供链 / 提示词注入（与 LangChain 兼容）。

        Args:
            inputs: 链输入（当前实现仅用于接口对齐，不参与取值）。

        Returns:
            形如 ``{"history": <消息列表或文本>}`` 的字典；返回形态由
            ``return_messages`` 决定。``buffer_window`` 类型自动收窄为
            最近窗口，summary 类型包含摘要消息。
        """
        del inputs  # 预留：后续可据此做相关性筛选
        messages = (
            self._visible_messages()
            if self.memory_type != "buffer_window"
            else self.get_messages_windowed()
        )
        if self.return_messages:
            history: Any = messages
        else:
            history = "\n".join(_render_line(msg) for msg in messages)
        return {"history": history}

    @staticmethod
    def _extract_value(
        data: Dict[str, Any], preferred_keys: tuple[str, ...]
    ) -> Any:
        """按优先级取字典值；均缺失时退回到第一个非空值。"""
        for key in preferred_keys:
            if key in data:
                return data[key]
        for value in data.values():
            if value is not None:
                return value
        return None

    # ------------------------------------------------------------------ #
    # 与 RAGChain 集成的辅助方法（Day 5 预留）
    # ------------------------------------------------------------------ #
    def build_context_with_history(self, query: str, max_tokens: int = 2000) -> str:
        """构建包含历史对话的上下文，用于注入 RAG prompt。

        策略：``buffer_window`` 类型先收窄到最近 ``max_window_size`` 轮，
        其余类型使用可见历史；随后自最新消息向前累计，只保留能在
        ``max_tokens`` 预算内完整放入的整行历史（预算按字符计，
        对中文场景近似友好）；若最新一条本身超长，则单独截断该条。

        Args:
            query: 当前用户问题（预留：后续可据此做历史片段相关性筛选）。
            max_tokens: 历史上下文的最大字符预算（中文场景按字符近似）。

        Returns:
            格式化后的历史字符串（``用户: ...\\nAI: ...``），不含当前问题。
        """
        del query
        budget = int(max_tokens)
        if budget <= 0:
            return ""

        if self.memory_type == "buffer_window":
            selected = self._visible_messages()[-self.max_window_size :]
        else:
            selected = self._visible_messages()

        # 自最新消息向前累计，保留能完整放下的行；保证顺序仍为时间正序
        picked: List[str] = []
        used = 0
        for message in reversed(selected):
            line = _render_line(message)
            if used + len(line) <= budget:
                picked.append(line)
                used += len(line)
            else:
                if not picked:
                    # 连最新一条都无法完整放入：按预算截断该条，避免上下文为空
                    picked.append(line[:budget])
                break
        picked.reverse()
        return "\n".join(picked)

    # ------------------------------------------------------------------ #
    # 序列化 / 反序列化
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        """将记忆导出为可 JSON 化的字典（便于持久化 / 跨会话迁移）。

        Returns:
            包含配置、内部消息与滚动摘要的字典。
        """
        return {
            "session_id": self.session_id,
            "memory_type": self.memory_type,
            "max_window_size": self.max_window_size,
            "summary_max_tokens": self.summary_max_tokens,
            "return_messages": self.return_messages,
            "summary_text": self._summary_text,
            "messages": [_serialize_message(msg) for msg in self.messages],
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """从字典恢复记忆状态（用于从数据库 / 文件恢复会话）。

        覆盖当前实例的配置与消息；无法反序列化的单条消息会被跳过并告警。

        Args:
            data: :meth:`to_dict` 产出的字典。
        """
        memory_type = str(data.get("memory_type", self.memory_type)).lower()
        if memory_type not in _SUPPORTED_TYPES:
            raise ValueError(f"字典中的记忆类型 {memory_type!r} 不受支持")
        self.memory_type = memory_type
        self.max_window_size = max(1, int(data.get("max_window_size", self.max_window_size)))
        self.summary_max_tokens = max(1, int(data.get("summary_max_tokens", self.summary_max_tokens)))
        self.return_messages = bool(data.get("return_messages", self.return_messages))
        self.session_id = data.get("session_id") or self.session_id

        restored: List[BaseMessage] = []
        for item in data.get("messages", []):
            message = _deserialize_message(item) if isinstance(item, dict) else None
            if message is not None:
                restored.append(message)
        self.messages = restored
        summary_text = data.get("summary_text")
        self._summary_text = str(summary_text) if summary_text else None
        logger.info(
            "会话 %s 从字典恢复记忆：%d 条消息",
            self.session_id or "(standalone)",
            len(restored),
        )
