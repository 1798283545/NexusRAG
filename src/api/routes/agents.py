"""智能体控制接口。

暴露 Supervisor 与各专业智能体的直接执行能力，以及运行时状态查询。
实际执行由全局缓存（``dependencies.get_agents``）中的 Agent 实例完成；
底层依赖（LLM / ChromaDB）不可用时返回可读的 503 错误而非裸 500。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from api.dependencies import get_agents
from api.schemas import AgentRunRequest, AgentRunResponse, AgentStatusResponse

router = APIRouter()
logger = logging.getLogger(__name__)

_AGENT_KEYS = ("supervisor", "rag", "code", "web", "summarizer")


def _output_of(result: Any) -> str:
    """把 Agent.run 的返回归一化为文本输出。"""
    if isinstance(result, dict):
        output = result.get("output")
        if output:
            return str(output).strip()
        # Supervisor 等复合结果：优先回显子任务汇总
        sub_results = result.get("results")
        if isinstance(sub_results, list):
            total = len(sub_results)
            success = sum(1 for r in sub_results if r.get("status") == "success")
            return f"完成 {success}/{total} 个子任务，详见执行明细。"
        return str(result)
    return str(result)


@router.post("/run", response_model=AgentRunResponse, summary="执行指定智能体")
def run_agent(request: AgentRunRequest) -> AgentRunResponse:
    """调用指定智能体执行单个任务（同步返回最终输出）。"""
    try:
        agents: Dict[str, Any] = get_agents()
    except Exception as exc:  # noqa: BLE001
        logger.exception("智能体环境初始化失败")
        raise HTTPException(status_code=503, detail=f"智能体环境不可用：{exc}") from exc

    agent = agents.get(request.agent_type.value)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Agent {request.agent_type.value!r} 未注册")

    started = time.perf_counter()
    try:
        result = agent.run(query=request.query, **request.parameters)
    except Exception as exc:  # noqa: BLE001 - Agent 内部失败归并为 500
        logger.exception("Agent %s 执行失败", request.agent_type.value)
        raise HTTPException(status_code=500, detail=f"Agent 执行失败：{exc}") from exc
    elapsed = time.perf_counter() - started

    tools_used = (result.get("tools_used") or []) if isinstance(result, dict) else []
    return AgentRunResponse(
        agent_type=request.agent_type.value,
        output=_output_of(result),
        tools_used=[str(tool) for tool in tools_used],
        execution_time=round(elapsed, 3),
    )


@router.get("/status", response_model=AgentStatusResponse, summary="智能体状态")
def agent_status() -> AgentStatusResponse:
    """返回已注册智能体及其状态（ready / error）。"""
    statuses: Dict[str, str] = {key: "error" for key in _AGENT_KEYS}
    try:
        agents = get_agents()
        for key in _AGENT_KEYS:
            statuses[key] = "ready" if key in agents else "error"
    except Exception as exc:  # noqa: BLE001
        logger.warning("智能体环境不可用: %s", exc)
    return AgentStatusResponse(agents=statuses)
