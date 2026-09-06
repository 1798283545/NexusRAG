"""NexusRAG API 请求 / 响应模型（Pydantic）。

第四阶段 Day 1-3：为文档管理、对话（含流式）、智能体控制、工作流执行与
健康检查定义统一、自描述的接口契约，同时作为 OpenAPI（Swagger）文档依据。
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# 通用模型
# --------------------------------------------------------------------------- #
class StatusResponse(BaseModel):
    """通用操作状态响应。"""

    status: str = "success"
    message: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)


class ErrorResponse(BaseModel):
    """统一错误响应体（配合 exceptions 模块使用）。"""

    error: str = Field(..., description="错误类型（机器可读的短名）")
    detail: Optional[str] = Field(None, description="人类可读的错误细节")


# --------------------------------------------------------------------------- #
# 文档管理模型
# --------------------------------------------------------------------------- #
class DocumentUploadResponse(BaseModel):
    """单文件上传 / 索引结果。"""

    file_name: str = Field(..., description="原始文件名")
    file_type: str = Field("", description="文件扩展名（不含点）")
    chunk_count: int = Field(0, description="入库的文本块数量")
    document_id: str = Field("", description="文档唯一标识（此处为源文件路径）")
    status: str = Field("success", description="处理状态")


class DocumentItem(BaseModel):
    """文档列表中的单项信息。"""

    id: str = Field(..., description="文档唯一标识（源文件绝对路径）")
    name: str = Field(..., description="文档名（文件名）")
    file_type: str = Field("", description="文件类型（扩展名）")
    chunk_count: int = Field(0, description="该文档对应的文本块数量")
    created_at: Optional[str] = Field(None, description="文档创建/修改时间（ISO）")
    source: str = Field("", description="来源路径")


class DocumentListResponse(BaseModel):
    """文档列表响应。"""

    documents: List[DocumentItem] = Field(default_factory=list)


class DocumentDeleteResponse(BaseModel):
    """删除文档响应。"""

    deleted_count: int = Field(0, description="实际删除的文档块数量")
    status: str = "success"


# --------------------------------------------------------------------------- #
# 对话模型
# --------------------------------------------------------------------------- #
class ChatRequest(BaseModel):
    """对话请求体。"""

    query: str = Field(..., min_length=1, max_length=8000, description="用户问题")
    session_id: Optional[str] = Field(None, description="会话 ID，用于多轮对话")
    k: Optional[int] = Field(None, ge=1, le=50, description="检索返回的文档块数")
    use_memory: bool = Field(True, description="是否使用并记录会话历史记忆")
    stream: bool = Field(False, description="是否流式输出（SSE）")


class ChatResponse(BaseModel):
    """非流式对话响应。"""

    answer: str = Field("", description="模型回答（含 Sources 溯源）")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="引用来源")
    session_id: str = Field("", description="实际使用的会话 ID")
    execution_time: float = Field(0.0, description="耗时（秒）")


# --------------------------------------------------------------------------- #
# SSE 流式事件模型（与 /api/chat/stream 的事件帧一一对应）
# --------------------------------------------------------------------------- #
class StreamSessionEvent(BaseModel):
    """流开始事件：告知前端会话 ID。"""

    type: Literal["session"] = "session"
    session_id: str = Field("", description="会话 ID")


class StreamSourcesEvent(BaseModel):
    """来源事件：在生成前发送已检索到的引用文档。"""

    type: Literal["sources"] = "sources"
    sources: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="引用来源列表：`[{\"content\": str, \"metadata\": {...}}]`",
    )


class StreamThinkingEvent(BaseModel):
    """思考事件：检索 / 重排 / 加载历史等阶段性提示（前端可折叠展示）。"""

    type: Literal["thinking"] = "thinking"
    content: str = Field("", description="思考过程说明")
    step: Optional[int] = Field(None, description="思考步骤序号（从 1 递增）")


class StreamTokenEvent(BaseModel):
    """Token 事件：逐 token 输出（打字机效果）。"""

    type: Literal["token"] = "token"
    content: str = Field("", description="本次发送的文本片段（可能含 1~N 个 token）")


class StreamErrorEvent(BaseModel):
    """错误事件：流式中断 / 超时 / 服务异常时的终止帧。"""

    type: Literal["error"] = "error"
    error: str = Field("", description="机器可读的错误类型（如 llm_stream_error）")
    detail: Optional[str] = Field(None, description="人类可读的错误细节")


class StreamDoneEvent(BaseModel):
    """完成事件：流式生成正常结束。"""

    type: Literal["done"] = "done"
    total_tokens: int = Field(0, description="本次生成的 token 总数")
    execution_time: float = Field(0.0, description="实际执行耗时（秒，不含客户端打字延迟）")


#: 所有 SSE 事件类型的联合（用于 OpenAPI 文档描述）
StreamEvent = Union[
    StreamSessionEvent,
    StreamSourcesEvent,
    StreamThinkingEvent,
    StreamTokenEvent,
    StreamErrorEvent,
    StreamDoneEvent,
]


# --------------------------------------------------------------------------- #
# 智能体模型
# --------------------------------------------------------------------------- #
class AgentType(str, Enum):
    """受支持的智能体类型。"""

    SUPERVISOR = "supervisor"
    RAG = "rag"
    CODE = "code"
    WEB = "web"
    SUMMARIZER = "summarizer"


class AgentRunRequest(BaseModel):
    """智能体执行请求体。"""

    agent_type: AgentType = Field(..., description="要执行的智能体")
    query: str = Field(..., min_length=1, description="任务描述")
    parameters: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="透传给智能体的额外参数"
    )


class AgentRunResponse(BaseModel):
    """智能体执行响应。"""

    agent_type: str = Field(..., description="执行的智能体类型")
    output: str = Field("", description="执行结果文本")
    tools_used: List[str] = Field(default_factory=list, description="实际使用的工具名")
    execution_time: float = Field(0.0, description="耗时（秒）")


class AgentStatusResponse(BaseModel):
    """智能体状态响应。"""

    agents: Dict[str, str] = Field(default_factory=dict, description="智能体名 → 状态")


# --------------------------------------------------------------------------- #
# 工作流模型
# --------------------------------------------------------------------------- #
class WorkflowRunRequest(BaseModel):
    """工作流执行请求体。"""

    query: str = Field(..., min_length=1, description="复杂任务描述")
    session_id: Optional[str] = Field(None, description="会话线程 ID（缺省自动生成）")
    enable_hitl: Optional[bool] = Field(
        None,
        description="可选；HITL 在服务启动时由 WORKFLOW_ENABLE_HITL 全局决定，"
        "传入不一致的值将被拒绝（避免同一线程审核语义混乱）。",
    )


class WorkflowRunResponse(BaseModel):
    """工作流执行 / 恢复响应。

    ``status`` 为 ``done``（已整合出最终回答）或 ``awaiting_review``
    （在 ``human_review`` 节点暂停，等待调用 :meth:`WorkflowResumeRequest`）。
    """

    session_id: str = Field("", description="会话线程 ID")
    status: str = Field("done", description="done / awaiting_review")
    final_answer: str = Field("", description="最终回答（暂停时为空）")
    plan: List[Dict[str, Any]] = Field(default_factory=list, description="任务计划")
    execution_trace: List[Dict[str, Any]] = Field(default_factory=list, description="执行痕迹")
    pending_action: Optional[Dict[str, Any]] = Field(None, description="待人工审核的操作")
    error: Optional[str] = Field(None, description="执行错误信息")
    execution_time: float = Field(0.0, description="耗时（秒）")


class WorkflowResumeRequest(BaseModel):
    """恢复被人工审核暂停的工作流。"""

    session_id: str = Field(..., description="会话线程 ID")
    feedback: str = Field(
        ...,
        description="人工反馈：approve / reject，或携带修改说明的自由文本",
    )


# --------------------------------------------------------------------------- #
# 健康检查
# --------------------------------------------------------------------------- #
class HealthResponse(BaseModel):
    """健康检查响应。"""

    status: str = Field("healthy", description="整体状态")
    version: str = Field("0.1.0", description="服务版本")
    components: Dict[str, bool] = Field(
        default_factory=dict, description="各组件可用性（vector_store / llm / bm25_index）"
    )
