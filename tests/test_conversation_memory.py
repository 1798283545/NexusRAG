"""ConversationMemoryManager 对话记忆单元测试。

覆盖：消息添加与读取、buffer_window 窗口收窄与物理修剪、summary 摘要压缩
（含 LLM 缺失 / 失败降级）、多会话隔离（session_id 单例）、序列化 / 反序列化、
LangChain BaseChatMemory 兼容接口、get_chat_history / build_context_with_history。

summary 场景使用确定性哑 LLM（记录调用次数与最后一次 prompt，可注入失败），
无需外部 API / 模型下载。
"""

import logging
import uuid

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from memory import ConversationMemoryManager


class _SummaryLLM:
    """哑摘要 LLM：invoke 返回固定摘要，记录调用次数与最后一次 prompt。"""

    def __init__(self, summary: str = "【摘要】用户问好，AI 进行了回应。", fail: bool = False) -> None:
        self.summary = summary
        self.fail = fail
        self.call_count = 0
        self.last_messages: list | None = None

    def invoke(self, messages: list) -> str:
        self.call_count += 1
        self.last_messages = messages
        if self.fail:
            raise RuntimeError("mock summary LLM outage")
        return self.summary


def _make(**kwargs) -> ConversationMemoryManager:
    """构造显式 buffer 类型的记忆管理器（避免受 .env 默认值影响）。"""
    defaults = {"memory_type": "buffer"}
    defaults.update(kwargs)
    return ConversationMemoryManager(**defaults)


def _contents(messages: list[BaseMessage]) -> list[str]:
    """提取消息列表的文本内容。"""
    return [message.content for message in messages]


def _add_turns(memory: ConversationMemoryManager, turns: list[tuple[str, str]]) -> None:
    """按 (用户, AI) 轮次批量写入对话。"""
    for user, ai in turns:
        memory.add_user_message(user)
        memory.add_ai_message(ai)


# ---------------------------------------------------------------------- #
# 初始化与参数校验
# ---------------------------------------------------------------------- #
def test_initial_state_and_message_count():
    """初始化空记忆；添加用户 / AI 消息后数量与类型正确。"""
    memory = _make()
    assert memory.get_messages() == []

    memory.add_user_message("你好")
    memory.add_ai_message("你好！有什么可以帮你？")
    messages = memory.get_messages()

    assert len(messages) == 2
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)
    assert _contents(messages) == ["你好", "你好！有什么可以帮你？"]
    # get_messages 返回拷贝，外部修改不影响内部状态
    messages.append(HumanMessage(content="篡改"))
    assert len(memory.get_messages()) == 2


def test_unsupported_memory_type_raises():
    """不支持的记忆类型应抛出 ValueError。"""
    with pytest.raises(ValueError, match="不支持的记忆类型"):
        ConversationMemoryManager(memory_type="rolling")


def test_add_message_accepts_arbitrary_types():
    """add_message 支持 SystemMessage 等任意 BaseMessage 子类型。"""
    memory = _make()
    memory.add_message(SystemMessage(content="请用中文回答"))
    memory.add_message(HumanMessage(content="介绍一下 RAG"))

    messages = memory.get_messages()
    assert isinstance(messages[0], SystemMessage)
    assert len(messages) == 2
    assert memory.get_chat_history() == "系统: 请用中文回答\n用户: 介绍一下 RAG"


# ---------------------------------------------------------------------- #
# buffer_window 滑动窗口
# ---------------------------------------------------------------------- #
def test_buffer_window_returns_latest_only():
    """max_window_size=2：get_messages_windowed 只返回最近 2 条。"""
    memory = _make(memory_type="buffer_window", max_window_size=2)
    for index in range(4):
        memory.add_user_message(f"m{index}")

    # 4 == 2 * 2，未触发物理修剪，内部仍保留全量
    assert len(memory.get_messages()) == 4
    assert _contents(memory.get_messages_windowed()) == ["m2", "m3"]


def test_buffer_window_physically_prunes_when_doubled():
    """超过 2 倍窗口后物理删除最早消息，仅保留最近窗口。"""
    memory = _make(memory_type="buffer_window", max_window_size=2)
    for index in range(5):
        memory.add_user_message(f"m{index}")

    # 5 > 2 * 2，触发物理修剪
    assert _contents(memory.get_messages()) == ["m3", "m4"]
    assert _contents(memory.get_messages_windowed()) == ["m3", "m4"]


# ---------------------------------------------------------------------- #
# summary 摘要压缩
# ---------------------------------------------------------------------- #
def test_summary_compresses_overflowing_messages():
    """消息超过窗口时调用 LLM 增量压缩，旧消息替换为摘要 SystemMessage。"""
    llm = _SummaryLLM()
    memory = _make(memory_type="summary", max_window_size=3, llm=llm)
    memory.add_user_message("早上好")
    memory.add_ai_message("早上好！请问有什么需要帮助？")
    memory.add_user_message("我叫小明")
    assert llm.call_count == 0  # 3 条 == 窗口，未触发压缩

    memory.add_ai_message("你好，小明！")  # 4 条 > 窗口 3，触发压缩
    assert llm.call_count == 1
    assert memory.memory_type == "summary"

    # 内部仅保留最近窗口内的原始消息，且头部拼接摘要
    assert len(memory.messages) == 3
    assert all(content != "早上好" for content in _contents(memory.messages))

    visible = memory.get_messages()
    assert isinstance(visible[0], SystemMessage)
    assert visible[0].content == llm.summary
    assert len(visible) == 4

    # 摘要 prompt 中应包含被压缩的旧消息原文
    assert "早上好" in str(llm.last_messages[-1].content)

    # 继续增长：再次溢出时增量并入摘要（调用计数 +1）
    memory.add_user_message("今天有什么新文档吗")
    assert llm.call_count == 2
    assert len(memory.messages) == 3
    assert len(memory.get_messages()) == 4


def test_summary_without_llm_downgrades_to_buffer_window(caplog):
    """summary 未提供 llm 时应降级为 buffer_window 并记录 WARNING。"""
    with caplog.at_level(logging.WARNING, logger="memory.conversation_memory"):
        memory = ConversationMemoryManager(memory_type="summary")
    assert memory.memory_type == "buffer_window"
    assert "降级" in caplog.text


def test_summary_llm_failure_downgrades(caplog):
    """summary 生成失败时应降级为 buffer_window 并保留原消息。"""
    llm = _SummaryLLM(fail=True)
    memory = _make(memory_type="summary", max_window_size=2, llm=llm)
    with caplog.at_level(logging.WARNING, logger="memory.conversation_memory"):
        memory.add_user_message("q1")
        memory.add_ai_message("a1")
        memory.add_user_message("q2")  # 第 5 条前触发压缩失败
    assert memory.memory_type == "buffer_window"
    assert len(memory.messages) == 3  # 原消息未被破坏
    assert "降级" in caplog.text


# ---------------------------------------------------------------------- #
# 会话隔离
# ---------------------------------------------------------------------- #
def test_session_isolation_and_singleton_reuse():
    """不同 session_id 记忆隔离；同 id 复用同一实例。"""
    session_a = f"sess-a-{uuid.uuid4().hex[:8]}"
    session_b = f"sess-b-{uuid.uuid4().hex[:8]}"

    memory_a = ConversationMemoryManager.get_session(session_a, memory_type="buffer")
    memory_b = ConversationMemoryManager.get_session(session_b, memory_type="buffer")

    memory_a.add_user_message("A 的独有问题")
    memory_a.add_ai_message("A 的回答")
    memory_b.add_user_message("B 的问题")

    assert len(memory_a.get_messages()) == 2
    assert len(memory_b.get_messages()) == 1
    assert _contents(memory_b.get_messages()) == ["B 的问题"]

    # 单例复用：再次获取同一会话返回同一实例，且忽略新的配置参数
    memory_a2 = ConversationMemoryManager.get_session(
        session_a, memory_type="buffer_window", max_window_size=1
    )
    assert memory_a2 is memory_a
    assert memory_a2.memory_type == "buffer"


# ---------------------------------------------------------------------- #
# 序列化 / 反序列化
# ---------------------------------------------------------------------- #
def test_to_dict_and_from_dict_roundtrip():
    """导出字典后重建管理器，消息顺序、类型与历史文本一致。"""
    memory = _make()
    memory.add_message(SystemMessage(content="请用中文回答"))
    _add_turns(memory, [("什么是 RAG？", "检索增强生成。"), ("它有什么优点？", "减少幻觉、可溯源。")])

    data = memory.to_dict()
    assert data["memory_type"] == "buffer"
    assert len(data["messages"]) == 5
    assert data["summary_text"] is None

    restored = _make()
    restored.from_dict(data)

    original = [(type(m).__name__, m.content) for m in memory.get_messages()]
    recovered = [(type(m).__name__, m.content) for m in restored.get_messages()]
    assert recovered == original
    assert restored.get_chat_history() == memory.get_chat_history()


def test_from_dict_restores_summary_and_window():
    """summary 会话的滚动摘要与窗口内消息可被 from_dict 恢复。"""
    llm = _SummaryLLM()
    memory = _make(memory_type="summary", max_window_size=2, llm=llm)
    _add_turns(memory, [("q1", "a1"), ("q2", "a2")])  # 第 4 条触发压缩
    assert memory._summary_text is not None

    restored = ConversationMemoryManager(
        memory_type="summary", max_window_size=2, llm=_SummaryLLM()
    )
    restored.from_dict(memory.to_dict())

    assert restored.memory_type == "summary"
    assert restored._summary_text == memory._summary_text
    visible = restored.get_messages()
    assert isinstance(visible[0], SystemMessage)
    assert len(restored.messages) == memory.max_window_size


# ---------------------------------------------------------------------- #
# LangChain BaseChatMemory 兼容接口
# ---------------------------------------------------------------------- #
def test_save_context_and_load_memory_variables_return_messages():
    """save_context / load_memory_variables 返回消息对象形态。"""
    memory = _make(return_messages=True)
    memory.save_context({"input": "介绍一下自己"}, {"output": "我是 NexusRAG。"})

    variables = memory.load_memory_variables({})
    assert list(variables.keys()) == ["history"]
    history = variables["history"]
    assert isinstance(history, list)
    assert isinstance(history[0], HumanMessage)
    assert isinstance(history[1], AIMessage)
    assert _contents(history) == ["介绍一下自己", "我是 NexusRAG。"]


def test_save_context_text_mode_and_question_answer_keys():
    """文本模式返回拼接字符串；支持 question / answer 键。"""
    memory = _make(return_messages=False)
    memory.save_context({"question": "1+1 等于几？"}, {"answer": "等于 2。"})

    history = memory.load_memory_variables({})["history"]
    assert isinstance(history, str)
    assert "用户: 1+1 等于几？" in history
    assert "AI: 等于 2。" in history


# ---------------------------------------------------------------------- #
# 文本历史与 RAG 上下文构建
# ---------------------------------------------------------------------- #
def test_get_chat_history_format():
    """get_chat_history 输出 ``角色: 内容`` 可读文本。"""
    memory = _make()
    _add_turns(memory, [("你好", "你好！有什么可以帮你？")])
    assert memory.get_chat_history() == "用户: 你好\nAI: 你好！有什么可以帮你？"


def test_build_context_keeps_recent_window_within_budget():
    """build_context_with_history 保留最近窗口并受字符预算约束。"""
    memory = _make(memory_type="buffer_window", max_window_size=2)
    _add_turns(
        memory,
        [
            ("第一问", "回答一"),
            ("第二问", "回答二"),
        ],
    )
    context = memory.build_context_with_history(query="第二问", max_tokens=200)
    assert context == "用户: 第二问\nAI: 回答二"
    assert "第一问" not in context
    assert len(context) <= 200


def test_build_context_truncates_oversized_single_message():
    """最新单条消息超预算时应被截断而非返回空字符串。"""
    memory = _make()
    _add_turns(memory, [("这是一个非常非常长的用户问题", "短回答")])
    context = memory.build_context_with_history(query="", max_tokens=6)
    assert isinstance(context, str)
    assert 0 < len(context) <= 6
