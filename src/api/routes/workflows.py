"""工作流执行接口。

将 LangGraph 多智能体工作流（Supervisor + 专业 Agent + HITL）暴露为 REST：

- ``POST /run``：启动一次工作流执行。若命中人工审核点，返回
  ``status=awaiting_review`` 与 ``pending_action``（用于前端弹窗）；
- ``POST /resume``：提交人工反馈（approve / reject / 修改说明）后继续执行；
- ``GET /status/{session_id}``：查询某线程当前执行状态。

注意：会话线程基于进程内 MemorySaver 检查点，服务重启后历史线程失效。
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException

from api.dependencies import get_workflow
from api.schemas import (
    WorkflowResumeRequest,
    WorkflowRunRequest,
    WorkflowRunResponse,
)

router = APIRouter()
logger = logging.getLogger(__name__)

#: 已启动过的会话线程集合（用于校验 session_id 是否有效）
_sessions: Dict[str, float] = {}


def _workflow() -> Any:
    """懒构建工作流；底层环境不可用时转成 503。"""
    try:
        return get_workflow()
    except Exception as exc:  # noqa: BLE001
        logger.exception("工作流环境初始化失败")
        raise HTTPException(status_code=503, detail=f"工作流环境不可用：{exc}") from exc


def _snapshot(workflow: Any, session_id: str, elapsed: float) -> WorkflowRunResponse:
    """从 LangGraph 检查点快照组装响应（支持 done / awaiting_review / error）。"""
    state: Dict[str, Any] = workflow.get_state(session_id)
    pending: Optional[Dict[str, Any]] = state.get("pending_action")
    error: Optional[str] = state.get("error")
    final_answer: str = state.get("final_answer") or ""

    if pending:
        status = "awaiting_review"
    elif error and not final_answer:
        status = "error"
    else:
        status = "done"

    return WorkflowRunResponse(
        session_id=session_id,
        status=status,
        final_answer=final_answer,
        plan=list(state.get("plan") or []),
        execution_trace=list(state.get("results") or []),
        pending_action=pending,
        error=error,
        execution_time=round(elapsed, 3),
    )


@router.post("/run", response_model=WorkflowRunResponse, summary="启动工作流")
def run_workflow(request: WorkflowRunRequest) -> WorkflowRunResponse:
    """启动一次多智能体协作工作流，运行至完成或首个人工审核点。"""
    workflow = _workflow()

    if request.enable_hitl is not None and request.enable_hitl != workflow.enable_hitl:
        raise HTTPException(
            status_code=400,
            detail=(
                f"enable_hitl 在服务启动时全局配置（当前={workflow.enable_hitl}），"
                "请通过 WORKFLOW_ENABLE_HITL 环境变量调整后重启服务。"
            ),
        )

    session_id = request.session_id or uuid.uuid4().hex
    started = time.perf_counter()
    try:
        workflow.run(query=request.query, thread_id=session_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("工作流执行异常（session=%s）", session_id)
        raise HTTPException(status_code=500, detail=f"工作流执行失败：{exc}") from exc
    elapsed = time.perf_counter() - started

    _sessions[session_id] = time.time()
    return _snapshot(workflow, session_id, elapsed)


@router.post("/resume", response_model=WorkflowRunResponse, summary="恢复工作流")
def resume_workflow(request: WorkflowResumeRequest) -> WorkflowRunResponse:
    """提交人工反馈并继续执行被暂停的工作流（可多次直至完成）。"""
    if request.session_id not in _sessions:
        raise HTTPException(status_code=404, detail="会话不存在或已失效")
    workflow = _workflow()

    started = time.perf_counter()
    try:
        workflow.resume(thread_id=request.session_id, feedback=request.feedback)
    except Exception as exc:  # noqa: BLE001
        logger.exception("工作流恢复异常（session=%s）", request.session_id)
        raise HTTPException(status_code=500, detail=f"工作流恢复失败：{exc}") from exc
    elapsed = time.perf_counter() - started

    return _snapshot(workflow, request.session_id, elapsed)


@router.get("/status/{session_id}", response_model=WorkflowRunResponse, summary="查询工作流状态")
def workflow_status(session_id: str) -> WorkflowRunResponse:
    """查询指定线程的当前状态（可用于 HITL 轮询与结果获取）。"""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="会话不存在或已失效")
    workflow = _workflow()
    try:
        return _snapshot(workflow, session_id, 0.0)
    except Exception as exc:  # noqa: BLE001
        logger.exception("读取工作流状态失败（session=%s）", session_id)
        raise HTTPException(status_code=500, detail=f"读取状态失败：{exc}") from exc
