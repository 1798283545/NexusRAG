"""NexusRAG 后端 REST / SSE 客户端封装（Streamlit 前端专用）。

职责：

- 封装健康检查 / 文档管理 / 对话（含 SSE 流式）/ 智能体 / 工作流全部接口；
- SSE 流式按帧解析为事件字典（事件契约见 ``api/schemas.py`` 中
  ``StreamSessionEvent`` / ``StreamSourcesEvent`` / ``StreamThinkingEvent`` /
  ``StreamTokenEvent`` / ``StreamErrorEvent`` / ``StreamDoneEvent``）；
- 每次调用使用短生命周期 ``httpx.Client``，避免跨会话线程共享连接；
- 后端异常统一转成携带可读信息的 :class:`BackendError`。

用法::

    from ui.utils.api_client import APIClient

    client = APIClient()  # 默认取 NEXUSRAG_API_URL / src.config.settings
    for event in client.chat_stream("你好", session_id="abc"):
        ...
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple
from urllib.parse import quote

import httpx

#: 普通请求超时：连接 10s，读 / 写 60s
_REQUEST_TIMEOUT = httpx.Timeout(60.0, connect=10.0)
#: SSE 流式超时：连接 10s，读间隔最长 300s（打字机慢速输出不被打断）
_STREAM_TIMEOUT = httpx.Timeout(300.0, connect=10.0)


def default_base_url() -> str:
    """解析后端默认地址：环境变量 > src/config.py(settings) > localhost。"""
    url = os.getenv("NEXUSRAG_API_URL", "").strip().rstrip("/")
    if url:
        return url
    try:
        from config import settings  # 复用后端配置（可编辑安装后可用）

        host = str(getattr(settings, "API_HOST", "0.0.0.0"))
        port = int(getattr(settings, "API_PORT", 8000))
        if host in ("0.0.0.0", "::", ""):
            host = "127.0.0.1"
        return f"http://{host}:{port}"
    except Exception:  # noqa: BLE001 - 配置缺失时退回默认地址
        return "http://127.0.0.1:8000"


class BackendError(RuntimeError):
    """后端请求失败（携带 HTTP 状态码与可读错误信息）。"""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        payload: Any = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload


class APIClient:
    """NexusRAG REST / SSE 客户端。"""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: Optional[httpx.Timeout] = None,
    ) -> None:
        self.base_url = (base_url or default_base_url()).rstrip("/")
        self._timeout = timeout or _REQUEST_TIMEOUT
        self._stream_timeout = _STREAM_TIMEOUT

    # ------------------------------------------------------------------ #
    # 底层请求工具
    # ------------------------------------------------------------------ #
    @staticmethod
    def _raise_for_response(response: httpx.Response) -> None:
        """把非 2xx 响应转成 :class:`BackendError`（尽力解析后端错误体）。"""
        if response.is_success:
            return
        detail: Optional[str] = None
        body: Any = None
        try:
            body = response.json()
        except Exception:  # noqa: BLE001 - 非 JSON 错误体原样透传
            body = response.text
        if isinstance(body, dict):
            detail = body.get("detail") or body.get("error") or body.get("message")
            if isinstance(detail, str) and detail:
                pass
            else:
                detail = None
        message = detail if detail else f"后端返回 HTTP {response.status_code}"
        raise BackendError(message, status_code=response.status_code, payload=body)

    def _request_json(self, method: str, path: str, **kwargs: Any) -> Any:
        """发起一次 JSON 请求并返回解析结果。"""
        with httpx.Client(timeout=self._timeout) as http:
            response = http.request(method, self.base_url + path, **kwargs)
        self._raise_for_response(response)
        if not response.content:
            return {}
        return response.json()

    # ------------------------------------------------------------------ #
    # 健康检查
    # ------------------------------------------------------------------ #
    def health_check(self) -> Dict[str, Any]:
        """系统健康状态（status / version / components）。"""
        return self._request_json("GET", "/api/health/")

    # ------------------------------------------------------------------ #
    # 文档管理
    # ------------------------------------------------------------------ #
    def list_documents(self) -> List[Dict[str, Any]]:
        """返回已索引文档列表。"""
        data = self._request_json("GET", "/api/documents/list")
        return list((data or {}).get("documents") or [])

    def upload_document(
        self,
        filename: str,
        content: bytes,
        content_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """上传并索引单个文档（文件以字节形式提交）。"""
        files = {
            "file": (filename, content, content_type or "application/octet-stream")
        }
        return self._request_json("POST", "/api/documents/upload", files=files)

    def upload_documents(
        self,
        files: Sequence[Tuple[str, bytes, Optional[str]]],
    ) -> List[Dict[str, Any]]:
        """批量上传并索引（按顺序逐个处理）。"""
        payload = [
            ("file", (name, content, ctype or "application/octet-stream"))
            for name, content, ctype in files
        ]
        return self._request_json("POST", "/api/documents/upload-multiple", files=payload)

    def delete_document(self, document_name: str) -> Dict[str, Any]:
        """按文件名删除文档的全部向量块。"""
        safe = quote(document_name, safe="")
        return self._request_json("DELETE", f"/api/documents/{safe}")

    def clear_documents(self) -> Dict[str, Any]:
        """清空知识库（危险操作）。"""
        return self._request_json("DELETE", "/api/documents/")

    # ------------------------------------------------------------------ #
    # 对话（非流式）
    # ------------------------------------------------------------------ #
    def chat(
        self,
        query: str,
        session_id: Optional[str] = None,
        use_memory: bool = True,
        k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """执行一次非流式 RAG 问答。"""
        payload: Dict[str, Any] = {
            "query": query,
            "use_memory": bool(use_memory),
            "stream": False,
        }
        if session_id:
            payload["session_id"] = session_id
        if k is not None:
            payload["k"] = k
        return self._request_json("POST", "/api/chat/", json=payload)

    # ------------------------------------------------------------------ #
    # 对话（SSE 流式）
    # ------------------------------------------------------------------ #
    def chat_stream(
        self,
        query: str,
        session_id: Optional[str] = None,
        use_memory: bool = True,
        k: Optional[int] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """SSE 流式问答，逐帧产出事件字典（session/sources/thinking/token/done/error）。"""
        payload: Dict[str, Any] = {
            "query": query,
            "use_memory": bool(use_memory),
            "stream": True,
        }
        if session_id:
            payload["session_id"] = session_id
        if k is not None:
            payload["k"] = k

        with httpx.Client(timeout=self._stream_timeout) as http:
            with http.stream(
                "POST", self.base_url + "/api/chat/stream", json=payload
            ) as response:
                self._raise_for_response(response)
                for line in response.iter_lines():
                    line = (line or "").strip()
                    if not line.startswith("data:"):
                        continue
                    data_text = line[len("data:") :].strip()
                    if not data_text:
                        continue
                    try:
                        yield json.loads(data_text)
                    except json.JSONDecodeError:  # noqa: PERF203 - 坏帧跳过
                        continue

    # ------------------------------------------------------------------ #
    # 智能体
    # ------------------------------------------------------------------ #
    def get_agent_status(self) -> Dict[str, Any]:
        """已注册智能体及其状态（ready / error）。"""
        return self._request_json("GET", "/api/agents/status")

    def run_agent(
        self,
        agent_type: str,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """执行指定智能体（同步返回输出与工具列表）。"""
        payload = {
            "agent_type": agent_type,
            "query": query,
            "parameters": parameters or {},
        }
        return self._request_json("POST", "/api/agents/run", json=payload)

    # ------------------------------------------------------------------ #
    # 工作流
    # ------------------------------------------------------------------ #
    def run_workflow(
        self,
        query: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """启动多智能体工作流（HITL 由服务端全局配置决定）。"""
        payload: Dict[str, Any] = {"query": query}
        if session_id:
            payload["session_id"] = session_id
        return self._request_json("POST", "/api/workflows/run", json=payload)

    def resume_workflow(self, session_id: str, feedback: str) -> Dict[str, Any]:
        """恢复被人工审核暂停的工作流（approve / reject / 修改说明）。"""
        payload = {"session_id": session_id, "feedback": feedback}
        return self._request_json("POST", "/api/workflows/resume", json=payload)

    def workflow_status(self, session_id: str) -> Dict[str, Any]:
        """查询工作流线程当前状态。"""
        safe = quote(session_id, safe="")
        return self._request_json("GET", f"/api/workflows/status/{safe}")
