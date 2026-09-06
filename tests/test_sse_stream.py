"""SSE 流式对话测试（第四阶段 Day 4）。

- 离线用例：通过 ``fastapi.testclient`` 校验事件帧顺序 / 错误帧 / 记忆落盘 /
  思考开关 / 缓冲批量，全部使用 Fake RAG 链，无需联网或密钥；
- 链级用例：直接消费 ``RAGChain.stream_query`` 的事件契约（依赖不可导入时跳过）；
- 联调用例：真实 ``httpx`` 客户端脚本，设置 ``NEXUSRAG_LIVE_SSE=1`` 并启动
  服务（localhost:8000）后才会执行。
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_rag_chain
from api.main import app
from config import settings
import api.routes.chat as chat_module

client = TestClient(app)


# --------------------------------------------------------------------------- #
# 离线替身
# --------------------------------------------------------------------------- #
class _EventChain:
    """按给定事件列表回放的 Fake RAG 链。"""

    def __init__(self, events: Optional[List[Dict[str, Any]]] = None) -> None:
        self.events = events or []
        self.calls: List[Dict[str, Any]] = []

    def stream_query(self, query: str, k: Optional[int] = None, **kwargs: Any):
        self.calls.append({"query": query, "k": k, **kwargs})
        for event in self.events:
            yield event

    def query(self, *_: Any, **__: Any) -> Dict[str, Any]:
        return {"answer": "", "source_documents": []}


class _Memory:
    """记录型会话记忆替身（history 为已有消息元组列表）。"""

    def __init__(self, history: Optional[List[tuple]] = None) -> None:
        self.msgs: List[tuple] = list(history or [])
        self.user_added: List[str] = []
        self.ai_added: List[str] = []

    def get_messages_windowed(self) -> list:
        return list(self.msgs)

    def add_user_message(self, text: str) -> None:
        self.msgs.append(("u", text))
        self.user_added.append(text)

    def add_ai_message(self, text: str) -> None:
        self.msgs.append(("a", text))
        self.ai_added.append(text)


def _happy_chain() -> _EventChain:
    return _EventChain(
        [
            {"type": "sources", "sources": [{"content": "doc", "metadata": {"source": "a.txt"}}]},
            {"type": "thinking", "content": "正在生成回答…", "step": 2},
            {"type": "token", "content": "你"},
            {"type": "token", "content": "好"},
            {"type": "token", "content": "世界"},
            {"type": "done", "total_tokens": 3, "execution_time": 0.42},
        ]
    )


@pytest.fixture(autouse=True)
def _clean_overrides():
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


def _parse_frames(text: str) -> List[Dict[str, Any]]:
    """把 SSE 响应文本解析为事件字典列表。"""
    return [
        json.loads(line[6:])
        for line in text.splitlines()
        if line.startswith("data: ")
    ]


def _stream_text(payload: Dict[str, Any]) -> str:
    chain = _happy_chain()
    app.dependency_overrides[get_rag_chain] = lambda: chain
    with client.stream("POST", "/api/chat/stream", json=payload) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert resp.headers.get("cache-control") == "no-cache"
        return "".join(resp.iter_text())


# --------------------------------------------------------------------------- #
# 事件契约
# --------------------------------------------------------------------------- #
def test_sse_stream_emits_full_event_sequence(monkeypatch):
    """事件顺序：session → sources/thinking → token… → done。"""
    monkeypatch.setattr(settings, "SSE_ENABLE_THINKING", True)
    frames = _parse_frames(_stream_text({"query": "你好", "use_memory": False}))

    types = [frame["type"] for frame in frames]
    assert types[0] == "session"
    assert frames[0]["session_id"]

    assert types.index("sources") < types.index("done")
    assert "thinking" in types
    assert types.count("token") == 3

    done = frames[-1]
    assert done["type"] == "done"
    assert done["total_tokens"] == 3
    assert done["execution_time"] >= 0.0

    sources = next(f for f in frames if f["type"] == "sources")
    assert sources["sources"][0]["content"] == "doc"


def test_sse_stream_forwards_error_event(monkeypatch):
    """链路 error 事件应透传为终止帧，且不再出现 done。"""
    monkeypatch.setattr(settings, "SSE_ENABLE_THINKING", True)
    chain = _EventChain(
        [
            {"type": "token", "content": "部分"},
            {"type": "error", "error": "llm_stream_error", "detail": "boom（已流式输出 1 个 token）"},
        ]
    )
    app.dependency_overrides[get_rag_chain] = lambda: chain
    with client.stream(
        "POST", "/api/chat/stream", json={"query": "q", "use_memory": False}
    ) as resp:
        frames = _parse_frames("".join(resp.iter_text()))

    types = [frame["type"] for frame in frames]
    assert "done" not in types
    assert types[-1] == "error"
    assert frames[-1]["error"] == "llm_stream_error"


def test_sse_stream_saves_memory_after_done(monkeypatch):
    """done 前完整回答写入记忆（user 在前、ai 在后）。"""
    monkeypatch.setattr(settings, "SSE_ENABLE_THINKING", True)
    memory = _Memory()
    monkeypatch.setattr(chat_module, "_memory_for", lambda sid, use: memory if use else None)

    frames = _parse_frames(
        _stream_text({"query": "你好", "use_memory": True, "session_id": "sse-mem-1"})
    )
    assert frames[0]["session_id"] == "sse-mem-1"
    assert frames[-1]["type"] == "done"

    assert memory.user_added == ["你好"]
    assert memory.ai_added == ["你好世界"]


def test_sse_stream_reports_history_then_writes_memory(monkeypatch):
    """存在历史时先发 thinking 提示，再完成本轮读写。"""
    monkeypatch.setattr(settings, "SSE_ENABLE_THINKING", True)
    memory = _Memory(history=[("u", "上一轮问题"), ("a", "上一轮回答"), ("u", "第三句")])
    monkeypatch.setattr(chat_module, "_memory_for", lambda sid, use: memory if use else None)

    frames = _parse_frames(_stream_text({"query": "继续", "use_memory": True}))
    history_thinking = [
        frame for frame in frames if frame["type"] == "thinking" and "对话历史" in frame["content"]
    ]
    assert history_thinking, "应发送“已加载对话历史”思考帧"
    assert "共 3 条历史消息" in history_thinking[0]["content"]

    assert memory.user_added == ["继续"]
    assert memory.ai_added == ["你好世界"]


def test_sse_stream_respects_thinking_disabled(monkeypatch):
    """SSE_ENABLE_THINKING=False 时抑制 thinking 帧，token 不受影响。"""
    monkeypatch.setattr(settings, "SSE_ENABLE_THINKING", False)
    frames = _parse_frames(_stream_text({"query": "你好", "use_memory": False}))

    types = [frame["type"] for frame in frames]
    assert "thinking" not in types
    assert types.count("token") == 3
    assert types[-1] == "done"


def test_sse_stream_buffers_tokens(monkeypatch):
    """SSE_BUFFER_SIZE>1 时将多个 token 合并为单个 token 帧。"""
    monkeypatch.setattr(settings, "SSE_ENABLE_THINKING", False)
    monkeypatch.setattr(settings, "SSE_BUFFER_SIZE", 3)
    monkeypatch.setattr(settings, "SSE_CHUNK_DELAY", 0.0)
    frames = _parse_frames(_stream_text({"query": "你好", "use_memory": False}))

    token_frames = [frame for frame in frames if frame["type"] == "token"]
    assert len(token_frames) == 1
    assert token_frames[0]["content"] == "你好世界"


# --------------------------------------------------------------------------- #
# 链级事件契约（RAGChain 可导入时执行）
# --------------------------------------------------------------------------- #
def test_rag_chain_stream_query_event_contract():
    """stream_query 依次产出 thinking/sources/token/done 事件。"""
    try:
        from chains import RAGChain
    except ImportError as exc:  # langchain 版本漂移导致模块暂不可导入
        pytest.skip(f"RAGChain 无法导入（{exc}），跳过链级契约测试")
    from langchain_core.documents import Document

    chain = RAGChain.__new__(RAGChain)
    doc = Document(page_content="NexusRAG 说明", metadata={"file_name": "guide.txt", "source": "guide.txt"})
    chain.k = 4
    chain._prepare = lambda query, k=None, filter=None, threshold=None: (
        "[1] NexusRAG 说明",
        [{"file_name": "guide.txt", "page_number": None, "source": "guide.txt", "score": 0.9, "content": "NexusRAG 说明"}],
        [(doc, 0.9)],
    )
    chain._qa_prompt = type("Prompt", (), {"format_messages": lambda self, **kw: []})()
    chain._get_llm = lambda: type("LLM", (), {"stream": lambda self, msgs: iter(["你", "好"])})()

    events = list(chain.stream_query("NexusRAG 是什么？", k=2))
    types = [event["type"] for event in events]

    assert types[0] == "thinking"
    assert "sources" in types
    sources = next(event for event in events if event["type"] == "sources")
    assert sources["sources"][0]["metadata"]["file_name"] == "guide.txt"
    assert "".join(e["content"] for e in events if e["type"] == "token") == "你好"
    done = events[-1]
    assert done["type"] == "done"
    assert done["total_tokens"] == 2
    assert done["execution_time"] >= 0.0


def test_rag_chain_stream_query_llm_failure_emits_error():
    """LLM 中途失败：已产出 token 保留，随后 error 事件终止。"""
    try:
        from chains import RAGChain
    except ImportError as exc:  # langchain 版本漂移导致模块暂不可导入
        pytest.skip(f"RAGChain 无法导入（{exc}），跳过链级契约测试")

    chain = RAGChain.__new__(RAGChain)
    chain.k = 4
    chain._prepare = lambda query, k=None, filter=None, threshold=None: (
        "（未检索到相关文档内容）",
        [],
        [],
    )
    chain._qa_prompt = type("Prompt", (), {"format_messages": lambda self, **kw: []})()

    class _BrokenLLM:
        def stream(self, messages):
            yield "部分输出"
            raise RuntimeError("mock mid-stream failure")

    chain._get_llm = lambda: _BrokenLLM()

    events = list(chain.stream_query("问题", k=2))
    tokens = [event["content"] for event in events if event["type"] == "token"]
    assert tokens == ["部分输出"]
    last = events[-1]
    assert last["type"] == "error"
    assert last["error"] == "llm_stream_error"
    assert "mock mid-stream failure" in last["detail"]


# --------------------------------------------------------------------------- #
# 联调用例（需真实服务：NEXUSRAG_LIVE_SSE=1）
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    os.getenv("NEXUSRAG_LIVE_SSE") != "1",
    reason="设置 NEXUSRAG_LIVE_SSE=1 并启动服务（localhost:8000）后执行联调",
)
def test_live_sse_stream_with_httpx():
    """真实 httpx 客户端逐帧读取 SSE 流（打字机 + 来源 + done）。"""
    import httpx

    with httpx.Client(timeout=60.0) as http:
        with http.stream(
            "POST",
            "http://localhost:8000/api/chat/stream",
            json={"query": "请介绍一下你自己", "use_memory": False},
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")

            sources_received = False
            token_chars: List[str] = []
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = json.loads(line[6:])
                event_type = data.get("type")
                if event_type == "sources":
                    sources_received = True
                elif event_type == "token":
                    token_chars.append(str(data.get("content") or ""))
                elif event_type == "done":
                    break

            assert sources_received is True
            assert "".join(token_chars).strip() != ""
