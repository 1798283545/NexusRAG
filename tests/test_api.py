"""API 集成测试：健康检查 / 文档管理 / 对话（含流式）/ 智能体 / 工作流。

通过 ``fastapi.testclient`` 走完整 HTTP 层验证路由、请求校验、序列化与
统一错误响应；所有重量级依赖（RAG 链、向量存储、智能体、工作流、LLM）
一律用 Fake / monkeypatch 替换，避免真实调用 OpenAI 与 ChromaDB。
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_rag_chain
from api.main import app
from config import settings
import api.routes.agents as agents_module
import api.routes.chat as chat_module
import api.routes.health as health_module
import api.routes.workflows as workflows_module

client = TestClient(app)


# --------------------------------------------------------------------------- #
# 共享替身
# --------------------------------------------------------------------------- #
class FakeVSM:
    """向量存储替身：内存内实现 API 路由所需的最小方法集。"""

    def __init__(
        self,
        docs: Optional[List[Any]] = None,
        delete_result: int = 0,
        stats: Optional[Dict[str, Any]] = None,
        clear_ok: bool = True,
    ) -> None:
        self.docs = docs or []
        self.delete_result = delete_result
        self.stats = stats or {"document_count": len(self.docs)}
        self.clear_ok = clear_ok
        self.filter_calls: List[Dict[str, Any]] = []
        self.clear_called = False

    def delete_by_filter(self, filter_: Dict[str, Any]) -> int:
        self.filter_calls.append(filter_)
        return self.delete_result

    def get_all_documents(self) -> List[Any]:
        return self.docs

    def get_collection_stats(self) -> Dict[str, Any]:
        return dict(self.stats)

    def clear_collection(self) -> bool:
        self.clear_called = True
        return self.clear_ok


class FakeRAGChain:
    """RAG 链替身：query / stream_query / process_document 与向量存储透传。"""

    def __init__(self, vsm: Optional[FakeVSM] = None, chunk_count: int = 2) -> None:
        self.vsm = vsm or FakeVSM()
        self.chunk_count = chunk_count
        self.processed: List[str] = []
        self.queries: List[Dict[str, Any]] = []

    @property
    def vector_store_manager(self) -> FakeVSM:
        return self.vsm

    def query(self, query: str, k: Optional[int] = None, **_: Any) -> Dict[str, Any]:
        self.queries.append({"query": query, "k": k})
        return {
            "answer": "基于知识库的测试回答。",
            "source_documents": [{"source": "guide.txt", "page": 1}],
        }

    def stream_query(self, query: str, k: Optional[int] = None, **_: Any):
        """按 SSE 事件契约产出：sources → thinking → token×2 → done。"""
        del query, k
        yield {
            "type": "sources",
            "sources": [{"content": "guide 内容", "metadata": {"source": "guide.txt"}}],
        }
        yield {"type": "thinking", "content": "正在生成回答…", "step": 1}
        yield {"type": "token", "content": "流式"}
        yield {"type": "token", "content": "token"}
        yield {"type": "done", "total_tokens": 2, "execution_time": 0.01}

    def process_document(self, file_path: str) -> int:
        self.processed.append(file_path)
        return self.chunk_count


class FakeAgent:
    """单个智能体替身。"""

    def __init__(self, output: str = "agent-ok", tools_used: Optional[List[str]] = None) -> None:
        self.output = output
        self.tools_used = tools_used or ["tool-a"]
        self.last_kwargs: Dict[str, Any] = {}

    def run(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        self.last_kwargs = {"query": query, **kwargs}
        return {"output": self.output, "tools_used": self.tools_used}


class FakeWorkflow:
    """LangGraph 工作流替身：状态以字典在内存中保存。"""

    def __init__(self, state: Optional[Dict[str, Any]] = None, raise_on_run: bool = False) -> None:
        self.enable_hitl = False
        self.state = state or {
            "plan": [],
            "results": [],
            "final_answer": "工作流最终回答。",
            "pending_action": None,
            "error": None,
        }
        self.raise_on_run = raise_on_run
        self.last_query: Optional[str] = None
        self.last_feedback: Optional[str] = None

    def run(self, query: str, thread_id: str) -> None:
        if self.raise_on_run:
            raise RuntimeError("workflow boom")
        self.last_query = query

    def resume(self, thread_id: str, feedback: str) -> None:
        self.last_feedback = feedback

    def get_state(self, session_id: str) -> Dict[str, Any]:
        return dict(self.state)


@pytest.fixture(autouse=True)
def _clean_overrides():
    """每个用例结束后清理依赖覆盖，避免串扰。"""
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


def _override_rag(chain: Any) -> None:
    """把 RAG 链依赖替换为替身。"""
    app.dependency_overrides[get_rag_chain] = lambda: chain


# --------------------------------------------------------------------------- #
# 路由注册与文档
# --------------------------------------------------------------------------- #
def test_all_routers_registered():
    """五个业务模块的路由均应挂载到主应用（/api 命名空间）。"""
    paths = set(app.openapi()["paths"])
    expected = {
        "/api/health",
        "/api/documents/upload",
        "/api/documents/upload-multiple",
        "/api/documents/list",
        "/api/documents/",
        "/api/chat/",
        "/api/chat/stream",
        "/api/agents/run",
        "/api/agents/status",
        "/api/workflows/run",
        "/api/workflows/resume",
        "/api/workflows/status/{session_id}",
    }
    assert expected.issubset(paths)


def test_docs_page_available():
    """Swagger 文档地址应可访问。"""
    assert client.get("/api/docs").status_code == 200


# --------------------------------------------------------------------------- #
# 健康检查
# --------------------------------------------------------------------------- #
def test_health_check(monkeypatch):
    """组件就绪时返回 healthy。"""
    monkeypatch.setattr(health_module, "get_vector_store", lambda: object())
    monkeypatch.setattr(health_module, "get_llm", lambda: object())
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "sk-fake")

    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["version"] == "0.1.0"
    assert body["components"]["vector_store"] is True
    assert body["components"]["llm"] is True


def test_health_check_degraded_without_api_key(monkeypatch):
    """缺少 LLM 密钥时整体降级为 degraded。"""
    monkeypatch.setattr(health_module, "get_vector_store", lambda: object())
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "")

    resp = client.get("/api/health/")
    assert resp.status_code == 200
    assert resp.json()["status"] == "degraded"
    assert resp.json()["components"]["llm"] is False


# --------------------------------------------------------------------------- #
# 文档管理
# --------------------------------------------------------------------------- #
def _fake_document_meta(**meta) -> SimpleNamespace:
    base = {"file_name": "guide.txt", "file_type": "txt", "source": "/tmp/guide.txt"}
    base.update(meta)
    return SimpleNamespace(metadata=base)


def test_upload_document():
    """上传单文档返回块数并清理旧同名向量。"""
    vsm = FakeVSM()
    chain = FakeRAGChain(vsm=vsm, chunk_count=3)
    _override_rag(chain)

    resp = client.post(
        "/api/documents/upload",
        files={"file": ("guide.txt", b"hello nexusrag", "text/plain")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["file_name"] == "guide.txt"
    assert body["chunk_count"] == 3
    assert chain.processed and chain.processed[0].endswith(".txt")
    # 同名清理：上传前先按 file_name 删除一次
    assert vsm.filter_calls == [{"file_name": "guide.txt"}]


def test_upload_multiple_documents():
    """批量上传按文件个数逐个索引。"""
    chain = FakeRAGChain(chunk_count=1)
    _override_rag(chain)

    resp = client.post(
        "/api/documents/upload-multiple",
        files=[
            ("files", ("a.txt", b"aaa", "text/plain")),
            ("files", ("b.md", b"bbb", "text/markdown")),
        ],
    )
    assert resp.status_code == 200
    assert [item["file_name"] for item in resp.json()] == ["a.txt", "b.md"]


def test_upload_missing_file_returns_422():
    """缺少 file 字段时由框架统一返回 422 校验错误。"""
    _override_rag(FakeRAGChain())
    resp = client.post("/api/documents/upload", files={})
    assert resp.status_code == 422
    assert resp.json()["error"] == "validation_error"


def test_list_documents_groups_by_name():
    """文档列表按文件名聚合块数与来源。"""
    chain = FakeRAGChain(
        vsm=FakeVSM(
            docs=[
                _fake_document_meta(),
                _fake_document_meta(),
                _fake_document_meta(file_name="a.md", source="/tmp/a.md", file_type="md"),
            ]
        )
    )
    _override_rag(chain)

    resp = client.get("/api/documents/list")
    assert resp.status_code == 200
    items = resp.json()["documents"]
    by_name = {item["name"]: item for item in items}
    assert by_name["guide.txt"]["chunk_count"] == 2
    assert by_name["a.md"]["chunk_count"] == 1


def test_delete_document():
    """删除存在的文档返回删除块数。"""
    vsm = FakeVSM(delete_result=5)
    _override_rag(FakeRAGChain(vsm=vsm))

    resp = client.delete("/api/documents/guide.txt")
    assert resp.status_code == 200
    assert resp.json()["deleted_count"] == 5


def test_delete_missing_document_returns_404():
    """删除不存在的文档应返回统一 404 JSON。"""
    vsm = FakeVSM(delete_result=0)
    _override_rag(FakeRAGChain(vsm=vsm))

    resp = client.delete("/api/documents/nonexist.txt")
    assert resp.status_code == 404
    body = resp.json()
    assert body["error"] == "http_error"
    assert "未找到文档" in body["detail"]


def test_delete_all_documents():
    """清空集合返回删除前的文档块总量。"""
    vsm = FakeVSM(stats={"document_count": 8})
    _override_rag(FakeRAGChain(vsm=vsm))

    resp = client.delete("/api/documents/")
    assert resp.status_code == 200
    assert resp.json()["deleted_count"] == 8
    assert vsm.clear_called is True


# --------------------------------------------------------------------------- #
# 对话（普通 + 流式 + 记忆）
# --------------------------------------------------------------------------- #
def test_chat_non_stream():
    """普通问答应返回回答、来源与会话 ID。"""
    chain = FakeRAGChain()
    _override_rag(chain)

    resp = client.post(
        "/api/chat/",
        json={"query": "NexusRAG 是什么？", "session_id": "s-1", "use_memory": False},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == "s-1"
    assert "测试回答" in body["answer"]
    assert body["sources"] == [{"source": "guide.txt", "page": 1}]
    assert chain.queries[0]["query"] == "NexusRAG 是什么？"


def test_chat_writes_memory(monkeypatch):
    """use_memory=True 时按顺序写入用户与 AI 消息。"""
    chain = FakeRAGChain()
    _override_rag(chain)
    memory_log: List[str] = []

    class _FakeMemory:
        def add_user_message(self, msg: str) -> None:
            memory_log.append(f"u:{msg}")

        def add_ai_message(self, msg: str) -> None:
            memory_log.append(f"a:{msg}")

    monkeypatch.setattr(chat_module, "_memory_for", lambda sid, use: _FakeMemory() if use else None)

    resp = client.post(
        "/api/chat/", json={"query": "还记得我吗？", "use_memory": True}
    )
    assert resp.status_code == 200
    assert memory_log and memory_log[0] == "u:还记得我吗？"
    assert memory_log[-1].startswith("a:")


def test_chat_invalid_query_returns_422():
    """空 query 触发 Pydantic 校验并返回统一 422 JSON。"""
    _override_rag(FakeRAGChain())
    resp = client.post("/api/chat/", json={"query": ""})
    assert resp.status_code == 422
    assert resp.json()["error"] == "validation_error"


def test_chat_stream_sse(monkeypatch):
    """SSE 流式响应依序产出 session → sources/thinking → token → done 帧。"""
    monkeypatch.setattr(settings, "SSE_ENABLE_THINKING", True)
    chain = FakeRAGChain()
    _override_rag(chain)

    with client.stream(
        "POST", "/api/chat/stream", json={"query": "流式问题", "use_memory": False}
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        text = "".join(resp.iter_text())

    frames = [
        json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")
    ]
    types = [frame["type"] for frame in frames]
    assert types[0] == "session"
    assert "sources" in types
    assert "thinking" in types
    assert types.count("token") == 2
    assert types[-1] == "done"
    assert frames[-1]["total_tokens"] == 2


# --------------------------------------------------------------------------- #
# 智能体
# --------------------------------------------------------------------------- #
def test_agent_run_ok(monkeypatch):
    """运行已注册智能体并返回 output / tools_used。"""
    agent = FakeAgent(tools_used=["retriever", "summarizer"])
    monkeypatch.setattr(
        agents_module,
        "get_agents",
        lambda: {"supervisor": FakeAgent(), "rag": agent},
    )

    resp = client.post(
        "/api/agents/run", json={"agent_type": "rag", "query": "总结一下", "parameters": {}}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["agent_type"] == "rag"
    assert body["output"] == "agent-ok"
    assert body["tools_used"] == ["retriever", "summarizer"]
    assert agent.last_kwargs["query"] == "总结一下"


def test_agent_run_unregistered_returns_404(monkeypatch):
    """未注册的智能体类型应返回 404。"""
    monkeypatch.setattr(agents_module, "get_agents", lambda: {"supervisor": FakeAgent()})
    resp = client.post("/api/agents/run", json={"agent_type": "rag", "query": "x"})
    assert resp.status_code == 404


def test_agent_run_env_unavailable_returns_503(monkeypatch):
    """底层环境不可用时（get_agents 抛错）应返回 503。"""
    def _boom():
        raise RuntimeError("chroma down")

    monkeypatch.setattr(agents_module, "get_agents", _boom)
    resp = client.post("/api/agents/run", json={"agent_type": "rag", "query": "x"})
    assert resp.status_code == 503


def test_agent_run_invalid_type_returns_422(monkeypatch):
    """非法 agent_type 触发校验错误。"""
    monkeypatch.setattr(agents_module, "get_agents", lambda: {})
    resp = client.post("/api/agents/run", json={"agent_type": "alien", "query": "x"})
    assert resp.status_code == 422


def test_agent_status(monkeypatch):
    """状态接口按注册情况返回 ready / error。"""
    agents = {
        "supervisor": FakeAgent(),
        "rag": FakeAgent(),
        "code": FakeAgent(),
        "web": FakeAgent(),
        "summarizer": FakeAgent(),
    }
    monkeypatch.setattr(agents_module, "get_agents", lambda: agents)
    body = client.get("/api/agents/status").json()
    assert all(state == "ready" for state in body["agents"].values())


# --------------------------------------------------------------------------- #
# 工作流
# --------------------------------------------------------------------------- #
def test_workflow_run_done(monkeypatch):
    """工作流运行至完成应返回 done 与最终回答。"""
    monkeypatch.setattr(workflows_module, "get_workflow", lambda: FakeWorkflow())

    resp = client.post("/api/workflows/run", json={"query": "复杂任务", "session_id": "wf-1"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == "wf-1"
    assert body["status"] == "done"
    assert "最终回答" in body["final_answer"]


def test_workflow_run_awaiting_review(monkeypatch):
    """命中人工审核点时返回 awaiting_review 与 pending_action。"""
    workflow = FakeWorkflow(
        state={
            "plan": [],
            "results": [],
            "final_answer": "",
            "pending_action": {"tool": "human_review", "question": "批准该步骤？"},
            "error": None,
        }
    )
    monkeypatch.setattr(workflows_module, "get_workflow", lambda: workflow)

    resp = client.post("/api/workflows/run", json={"query": "需审核", "session_id": "wf-2"})
    body = resp.json()
    assert body["status"] == "awaiting_review"
    assert body["pending_action"]["tool"] == "human_review"


def test_workflow_run_hitl_conflict(monkeypatch):
    """客户端传入与服务端全局 HITL 配置冲突时应 400。"""
    workflow = FakeWorkflow()  # enable_hitl=False
    monkeypatch.setattr(workflows_module, "get_workflow", lambda: workflow)

    resp = client.post("/api/workflows/run", json={"query": "q", "enable_hitl": True})
    assert resp.status_code == 400


def test_workflow_run_failure_returns_500(monkeypatch):
    """工作流执行抛错应映射为 500。"""
    monkeypatch.setattr(
        workflows_module, "get_workflow", lambda: FakeWorkflow(raise_on_run=True)
    )
    resp = client.post("/api/workflows/run", json={"query": "boom"})
    assert resp.status_code == 500


def test_workflow_env_unavailable_returns_503(monkeypatch):
    """工作流环境不可用（构建抛错）应返回 503。"""
    def _boom():
        raise RuntimeError("no supervisor")

    monkeypatch.setattr(workflows_module, "get_workflow", _boom)
    resp = client.post("/api/workflows/run", json={"query": "q"})
    assert resp.status_code == 503


def test_workflow_status_and_resume(monkeypatch):
    """run 后可通过 status 查询、通过 resume 继续。"""
    workflow = FakeWorkflow()
    monkeypatch.setattr(workflows_module, "get_workflow", lambda: workflow)

    run_resp = client.post("/api/workflows/run", json={"query": "任务", "session_id": "wf-9"})
    assert run_resp.status_code == 200

    status_resp = client.get("/api/workflows/status/wf-9")
    assert status_resp.status_code == 200
    assert status_resp.json()["status"] == "done"

    resume_resp = client.post(
        "/api/workflows/resume", json={"session_id": "wf-9", "feedback": "approve"}
    )
    assert resume_resp.status_code == 200
    assert workflow.last_feedback == "approve"


def test_workflow_unknown_session_returns_404(monkeypatch):
    """未启动过的会话不应出现在 status / resume。"""
    monkeypatch.setattr(workflows_module, "get_workflow", lambda: FakeWorkflow())
    assert client.get("/api/workflows/status/no-such-session").status_code == 404
    resp = client.post(
        "/api/workflows/resume", json={"session_id": "no-such-session", "feedback": "approve"}
    )
    assert resp.status_code == 404
