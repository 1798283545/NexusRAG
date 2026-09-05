"""LangGraph 多智能体协作工作流。

以有状态状态图（``StateGraph``）串联 Supervisor 与各专业智能体，实现：

- **有状态编排**：任务计划 / 子任务结果 / 当前智能体在状态中流转；
- **条件边路由**：Supervisor 决策后跳转对应 Agent 节点，完成后循环回到
  Supervisor，直到计划全部执行完或达到最大迭代次数；
- **Human-in-the-Loop**：高风险子任务（如执行代码 / 删除文件）在工作流中
  暂停，经 ``interrupt`` 等待人工审核（批准 / 修改 / 拒绝），再继续或终止。

节点：``supervisor → (rag_agent | code_agent | web_agent | summarizer_agent |
human_review | synthesizer) → ... → synthesizer → END``。

依赖 LangGraph（``pip install langgraph>=0.2.0``），Human-in-the-Loop 还需
checkpointer（本模块默认使用内存版 ``MemorySaver``）。
"""

from __future__ import annotations

import logging
import re
import time
from functools import partial
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage

from agents.base_agent import BaseAgent
from agents.supervisor import SupervisorAgent, SubTask
from config import settings

logger = logging.getLogger(__name__)

#: Supervisor 派发后的可执行 Agent 节点名
AGENT_NODES: Dict[str, str] = {
    "rag": "rag_agent",
    "code": "code_agent",
    "web": "web_agent",
    "summarizer": "summarizer_agent",
}

#: 一个子任务执行结束后的「终态」状态集合（不会再重跑）
TERMINAL_STATUSES = {"success", "failed", "unavailable", "blocked", "timeout"}

#: 视为高风险操作的关键词（命中即触发人工审核）
_RISK_PATTERNS = (
    re.compile(r"删除|移除|清空|覆盖", re.IGNORECASE),
    re.compile(r"\b(delete|drop|remove|overwrite|rm\s*-rf)\b", re.IGNORECASE),
    re.compile(r"执行代码|运行脚本|调用.*sql|写入数据库", re.IGNORECASE),
)

# --------------------------------------------------------------------------- #
# 版本兼容导入（langgraph 各小版本对导出路径有差异）
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - 由运行环境决定
    from langgraph.graph.message import add_messages
except ImportError:  # 旧版本 (<0.2.4)
    from langgraph.graph import add_messages  # type: ignore[no-redef]

try:  # pragma: no cover
    from langgraph.graph import END as _LANGGRAPH_END
except ImportError:  # pragma: no cover
    _LANGGRAPH_END = "__end__"  # type: ignore[assignment]

try:  # pragma: no cover
    from langgraph.checkpoint.memory import MemorySaver
except ImportError:  # pragma: no cover
    MemorySaver = None  # type: ignore[assignment,misc]


def _get_interrupt() -> Callable[..., Any]:
    """按版本获取 ``interrupt`` 函数（Human-in-the-Loop 专用）。"""
    try:
        from langgraph.types import interrupt

        return interrupt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "当前 langgraph 版本不支持 interrupt()，请升级 langgraph>=0.2.20 以启用 "
            "Human-in-the-Loop（或设置 enable_hitl=False）"
        ) from exc


def _get_command() -> Optional[Callable[..., Any]]:
    """按版本获取 ``Command``（用于恢复被中断的工作流）。"""
    try:
        from langgraph.types import Command

        return Command
    except ImportError:  # pragma: no cover
        return None


def _get_state_graph() -> Any:
    """按版本导入 StateGraph / START / END。"""
    from langgraph.graph import START, StateGraph

    return StateGraph, START, _LANGGRAPH_END


# --------------------------------------------------------------------------- #
# 状态定义
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - TypedDict 在 3.11+ 由 typing 提供
    from typing import Annotated, TypedDict
except ImportError:  # pragma: no cover
    from typing_extensions import Annotated, TypedDict  # type: ignore[no-redef]


class MultiAgentState(TypedDict, total=False):
    """多智能体工作流状态。

    Attributes:
        messages: 对话消息（``add_messages`` 规约累加）。
        query: 当前用户输入。
        plan: 任务计划（子任务字典列表，含 task_id / agent_name /
            description / dependencies / requires_approval）。
        results: 各子任务执行结果（task_id / agent_name / status / output /
            error / elapsed_ms）。
        current_agent: 当前正在执行的智能体。
        next_agent: 下一个要进入的节点（supervisor 决策结果）。
        iteration: 当前迭代次数（每次经过 supervisor 递增）。
        max_iterations: 最大迭代次数（防止死循环）。
        human_feedback: 人工反馈（approved / rejected / modified）。
        pending_action: 待人工审核的操作（含子任务信息）。
        approved_task_ids: 已通过人工审核、允许执行的任务编号。
        active_task: Supervisor 当前正在派发的子任务。
        final_answer: 整合后的最终回答。
        error: 错误信息。
    """

    messages: Annotated[List[BaseMessage], add_messages]
    query: str
    plan: List[Dict[str, Any]]
    results: List[Dict[str, Any]]
    current_agent: Optional[str]
    next_agent: Optional[str]
    iteration: int
    max_iterations: int
    human_feedback: Optional[str]
    pending_action: Optional[Dict[str, Any]]
    approved_task_ids: List[int]
    active_task: Optional[Dict[str, Any]]
    final_answer: Optional[str]
    error: Optional[str]


def build_initial_state(query: str, max_iterations: int = 10) -> Dict[str, Any]:
    """构造一次工作流运行的初始状态。"""
    return {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "plan": [],
        "results": [],
        "current_agent": None,
        "next_agent": None,
        "iteration": 0,
        "max_iterations": int(max_iterations),
        "human_feedback": None,
        "pending_action": None,
        "approved_task_ids": [],
        "active_task": None,
        "final_answer": None,
        "error": None,
    }


# --------------------------------------------------------------------------- #
# 共享辅助
# --------------------------------------------------------------------------- #
def _task_dict(task: Any) -> Dict[str, Any]:
    """将 SubTask / dict 统一转为普通 dict（保证状态可序列化）。"""
    if isinstance(task, SubTask):
        data = task.model_dump()
        data["requires_approval"] = bool(getattr(task, "requires_approval", False))
        return data
    return dict(task)


def _obtain_plan(supervisor: Any, query: str) -> List[Dict[str, Any]]:
    """从 Supervisor 获取结构化任务计划（兼容多种接口形态）。

    支持的调用顺序：``plan(query)`` → ``plan_task(query)`` →
    ``_plan_task(query)``。调用方负责为 ``requires_approval`` 等字段兜底。
    """
    planner: Optional[Callable[[str], Any]] = None
    for attr in ("plan", "plan_task", "_plan_task"):
        if hasattr(supervisor, attr):
            planner = getattr(supervisor, attr)
            break
    if planner is None:
        raise TypeError(
            "Supervisor 需提供 plan(query) / plan_task(query) 接口以产出结构化计划"
        )
    raw = planner(query) or []
    tasks: List[Dict[str, Any]] = []
    for item in raw:
        data = _task_dict(item)
        if data.get("agent_name") in AGENT_NODES:
            data.setdefault("dependencies", [])
            data.setdefault("requires_approval", False)
            tasks.append(data)
    return tasks


def _is_risky(task: Dict[str, Any]) -> bool:
    """判断子任务是否属于高风险操作（命中即触发人工审核）。"""
    agent_name = str(task.get("agent_name", "")).lower()
    if agent_name == "code":
        return True  # 执行代码默认为高风险
    description = str(task.get("description", ""))
    return any(pattern.search(description) for pattern in _RISK_PATTERNS)


def _done_task_ids(results: List[Dict[str, Any]]) -> set:
    """返回已进入终态（不再执行）的子任务编号集合。"""
    return {
        int(record["task_id"])
        for record in results
        if record.get("status") in TERMINAL_STATUSES
    }


def _first_ready_task(
    plan: List[Dict[str, Any]], done_ids: set
) -> Optional[Dict[str, Any]]:
    """按 task_id 升序选出第一个「依赖已满足且未执行」的子任务。"""
    done_ids = done_ids or set()
    for task in sorted(plan, key=lambda t: int(t.get("task_id", 0))):
        if task.get("task_id") in done_ids:
            continue
        deps = task.get("dependencies") or []
        if all(int(dep) in done_ids for dep in deps):
            return task
    return None


def _node_label(agent_name: str) -> str:
    """注册名 → 工作流节点名（未知则回到 synthesizer）。"""
    return AGENT_NODES.get(str(agent_name), "synthesizer")


def _record(
    task: Dict[str, Any], status: str, output: str = "", error: str = "", elapsed_ms: float = 0.0
) -> Dict[str, Any]:
    """构造单条子任务执行记录。"""
    return {
        "task_id": task.get("task_id"),
        "agent_name": task.get("agent_name"),
        "description": task.get("description", ""),
        "status": status,
        "output": output or "",
        "error": error or "",
        "elapsed_ms": round(elapsed_ms, 1),
    }


# --------------------------------------------------------------------------- #
# 工作流节点
# --------------------------------------------------------------------------- #
def supervisor_node(
    state: Dict[str, Any],
    supervisor: Any,
    enable_hitl: bool = True,
    available_agents: Optional[set] = None,
) -> Dict[str, Any]:
    """Supervisor 节点：规划 + 路由决策。

    行为：
    1. 首次进入（``plan`` 为空）时调用 Supervisor 生成任务计划并写入状态；
    2. 每次进入将 ``iteration`` +1；
    3. 依依赖关系选出下一个可执行子任务：
       - 目标是「未注册的智能体」→ 记录 unavailable 并自动跳过（软降级，
         避免把状态路由到图中不存在的节点）；
       - 无可执行任务 / 达到迭代上限 → ``next_agent="finish"``；
       - 任务需人工审核（HITL 开启且命中高风险）→ 写入 ``pending_action``；
       - 否则 → ``next_agent`` 指向对应 Agent 节点。

    Args:
        enable_hitl: 是否启用人工审核（False 时 ``requires_approval``
            任务自动放行，避免路由到不存在的审核节点）。
        available_agents: 已注册到图中的智能体键集合（``rag/code/...``），
            用于跳过计划中不可执行的子任务；缺省视为全部可用。
    """
    query = state.get("query", "")
    plan = state.get("plan") or []
    iteration = int(state.get("iteration", 0)) + 1
    max_iterations = int(state.get("max_iterations", settings.WORKFLOW_MAX_ITERATIONS))

    update: Dict[str, Any] = {"iteration": iteration, "current_agent": None}

    if not plan:
        try:
            plan = _obtain_plan(supervisor, query)
        except Exception as exc:  # noqa: BLE001 - 规划失败不应中断图
            logger.error("Supervisor 规划失败: %s", exc)
            plan = []
        if plan:
            logger.info("Supervisor 规划出 %d 个子任务", len(plan))
        update["plan"] = plan

    if available_agents is None:
        available_agents = set(AGENT_NODES.keys())

    results = list(state.get("results", []))
    approved_ids = {int(i) for i in (state.get("approved_task_ids") or [])}

    # 逐个挑选「依赖已满足」的子任务；遇到未注册智能体的任务即记录
    # unavailable 并跳过，直到选中可执行任务或计划耗尽。
    task: Optional[Dict[str, Any]] = None
    while True:
        candidate = _first_ready_task(plan, _done_task_ids(results))
        if candidate is None:
            break
        if str(candidate.get("agent_name")) not in available_agents:
            logger.warning(
                "智能体 %r 未注册，子任务 %s 标记为不可执行并跳过",
                candidate.get("agent_name"),
                candidate.get("task_id"),
            )
            results.append(
                _record(
                    candidate,
                    "unavailable",
                    error=f"智能体 {candidate.get('agent_name')!r} 未注册，任务被跳过",
                )
            )
            continue
        task = candidate
        break

    if len(results) > len(state.get("results", [])):
        update["results"] = results

    if task is None or iteration >= max_iterations:
        if iteration >= max_iterations and task is not None:
            update["error"] = f"达到最大迭代次数（{max_iterations}），停止派发新任务"
        update["next_agent"] = "finish"
        update["active_task"] = None
        update["pending_action"] = None
        return update

    needs_approval = bool(task.get("requires_approval")) or _is_risky(task)
    already_approved = int(task["task_id"]) in approved_ids

    node = _node_label(task.get("agent_name", ""))
    update["active_task"] = task
    update["next_agent"] = node

    if needs_approval and not already_approved:
        if not enable_hitl:
            # HITL 关闭：无法挂起审核节点，自动放行并记录警告
            logger.warning("HITL 已关闭，子任务 %s 自动放行", task.get("task_id"))
        else:
            update["pending_action"] = {
                "task_id": task.get("task_id"),
                "agent_name": task.get("agent_name"),
                "description": task.get("description", ""),
                "requires_approval": True,
            }
            logger.info("子任务 %s 需人工审核，暂停等待", task.get("task_id"))
    else:
        update["pending_action"] = None
        update["current_agent"] = task.get("agent_name")
    return update


def _make_agent_node(agent_key: str, agent: Any) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """构造单个专业 Agent 的执行节点。"""

    def agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
        plan = state.get("plan") or []
        results = list(state.get("results", []))
        done_ids = _done_task_ids(results)
        task = _first_ready_task(plan, done_ids)
        if task is None or str(task.get("agent_name")) != agent_key:
            # Supervisor 与节点选取逻辑不一致时的防御：直接回到 supervisor
            return {"current_agent": None}

        start = time.perf_counter()
        try:
            output = agent.run(task.get("description", ""))
            text = output.get("output", "") if isinstance(output, dict) else str(output)
            record = _record(task, "success", output=str(text or "").strip())
        except Exception as exc:  # noqa: BLE001 - 子任务失败不阻塞工作流
            logger.error("Agent %s 执行子任务 %s 失败: %s", agent_key, task.get("task_id"), exc)
            record = _record(task, "failed", error=str(exc))
        record["elapsed_ms"] = round((time.perf_counter() - start) * 1000, 1)

        return {"results": results + [record], "current_agent": None}

    return agent_node


def human_review_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """人工审核节点：使用 ``interrupt`` 暂停并等待人工反馈。

    反馈取值（区分大小写不敏感）：
    - ``approve`` / ``批准``：放行当前子任务；
    - ``reject`` / ``拒绝``：终止整个工作流（记录 error）；
    - 其他内容（如 ``modify`` + 修改说明）：视为「修改后放行」，
      修改内容会并入子任务描述。
    """
    pending_action = state.get("pending_action") or {}
    interrupt = _get_interrupt()

    payload = interrupt(
        {
            "action": pending_action,
            "message": "请审核以下操作是否允许执行：",
            "options": ["approve", "modify", "reject"],
            "agent_name": pending_action.get("agent_name"),
            "description": pending_action.get("description"),
            "task_id": pending_action.get("task_id"),
        }
    )

    # 兼容「字符串」或「dict{feedback,...}」两种人工输入形态
    raw_feedback = payload.get("feedback") if isinstance(payload, dict) else payload
    note = payload.get("note", "") if isinstance(payload, dict) else ""
    feedback = str(raw_feedback or "").strip().lower()
    task_id = int(pending_action.get("task_id", 0) or 0)

    approved_ids = [int(i) for i in (state.get("approved_task_ids") or [])]
    update: Dict[str, Any] = {
        "pending_action": None,
        "human_feedback": feedback,
    }

    if feedback in ("approve", "approved", "yes", "ok", "批准", "允许", "通过"):
        update["approved_task_ids"] = approved_ids + ([task_id] if task_id else [])
        logger.info("人工审核通过子任务 %s", task_id)
    elif feedback in ("reject", "rejected", "no", "拒绝", "终止"):
        update["error"] = "操作被用户拒绝"
        update["next_agent"] = "finish"
        logger.warning("人工审核拒绝子任务 %s，工作流终止", task_id)
    else:
        # modify：把修改意见写回计划后放行
        plan = list(state.get("plan", []))
        if task_id:
            for item in plan:
                if int(item.get("task_id", 0)) == task_id:
                    extra = f"（人工修改：{note or raw_feedback}）"
                    item["description"] = str(item.get("description", "")) + extra
            update["approved_task_ids"] = approved_ids + [task_id]
        update["plan"] = plan
        update["human_feedback"] = "modified"
        logger.info("人工修改子任务 %s 后放行", task_id)
    return update


def synthesizer_node(
    state: Dict[str, Any], supervisor: Any
) -> Dict[str, Any]:
    """整合节点：收集所有子任务结果生成最终回答。"""
    results = list(state.get("results", []))
    query = state.get("query", "")
    error = state.get("error")

    try:
        synthesize = getattr(supervisor, "_synthesize_results", None)
        if callable(synthesize):
            final_answer = synthesize(results, query)
        else:
            final_answer = _fallback_synthesize(results, query, error)
    except Exception as exc:  # noqa: BLE001
        logger.error("结果整合失败，使用兜底拼接: %s", exc)
        final_answer = _fallback_synthesize(results, query, error)

    update: Dict[str, Any] = {"final_answer": final_answer or "", "current_agent": None}
    if error:
        update["error"] = error
    return update


def _fallback_synthesize(
    results: List[Dict[str, Any]], query: str, error: Optional[str] = None
) -> str:
    """整合兜底：把各子任务输出拼接为可读文本。"""
    if not results:
        reason = error or "没有可用的子任务结果"
        return f"（无法完成整合：{reason}）"
    parts = [f"用户问题：{query}"]
    for record in results:
        status = record.get("status")
        body = record.get("output") or record.get("error") or "（无输出）"
        parts.append(f"[子任务 {record.get('task_id')} · {record.get('agent_name')} · {status}] {body}")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# 条件路由
# --------------------------------------------------------------------------- #
def route_after_supervisor(state: Dict[str, Any]) -> str:
    """Supervisor 节点后的条件路由。

    优先级：待审核操作 > finish/迭代上限（整合）> Supervisor 指定节点。
    未知节点名一律回落到 synthesizer，避免图执行异常。
    """
    # 1) 需要人工审核
    if state.get("pending_action"):
        return "human_review"
    next_agent = state.get("next_agent")
    # 2) 全部完成 / 达到上限
    if next_agent in ("finish", "__end__") or int(state.get("iteration", 0)) >= int(
        state.get("max_iterations", settings.WORKFLOW_MAX_ITERATIONS)
    ):
        return "synthesizer"
    # 3) 派发到具体 Agent 节点
    known = set(AGENT_NODES.values())
    if next_agent in known:
        return str(next_agent)
    if next_agent and next_agent in AGENT_NODES:
        return AGENT_NODES[next_agent]
    logger.warning("未知的 next_agent=%r，回落到 synthesizer", next_agent)
    return "synthesizer"


def route_after_human_review(state: Dict[str, Any]) -> str:
    """人工审核节点后的条件路由：拒绝 → 结束；否则回到 supervisor 继续。"""
    if state.get("human_feedback") in ("rejected", "reject", "no", "拒绝"):
        return "end"
    return "continue"


# --------------------------------------------------------------------------- #
# 图构建
# --------------------------------------------------------------------------- #
def build_multi_agent_workflow(
    supervisor: SupervisorAgent,
    agents: Dict[str, BaseAgent],
    enable_hitl: bool = True,
    max_iterations: int = 10,
    checkpointer: Optional[Any] = None,
) -> Any:
    """构建并编译多智能体工作流状态图。

    Args:
        supervisor: SupervisorAgent（或其协议实现），用于规划与结果整合。
        agents: 专业智能体注册表，key 需为 ``rag / code / web / summarizer``。
        enable_hitl: 是否加入人工审核节点（Human-in-the-Loop）。
        max_iterations: 最大循环迭代次数（超过后强制进入整合阶段）。
        checkpointer: LangGraph 检查点；启用 HITL 时必需，默认使用
            ``MemorySaver``（进程内）。如需跨进程恢复请替换为
            SqliteSaver/PostgresSaver 并配置 ``WORKFLOW_CHECKPOINT_DIR``。

    Returns:
        编译后的 LangGraph ``CompiledStateGraph``。
    """
    StateGraph, START, END = _get_state_graph()

    # 为每个 Agent 注册节点（绑定具体智能体实例）；只注册实际存在的智能体
    registered_keys = [key for key in AGENT_NODES if key in agents]
    agent_nodes = {
        AGENT_NODES[key]: _make_agent_node(key, agents[key]) for key in registered_keys
    }

    workflow = StateGraph(MultiAgentState)
    workflow.add_node(
        "supervisor",
        partial(
            supervisor_node,
            supervisor=supervisor,
            enable_hitl=enable_hitl,
            available_agents=set(registered_keys),
        ),
    )
    for label, node in agent_nodes.items():
        workflow.add_node(label, node)
    workflow.add_node("synthesizer", partial(synthesizer_node, supervisor=supervisor))

    # 人工审核节点（可选）
    if enable_hitl:
        workflow.add_node("human_review", human_review_node)

    workflow.set_entry_point("supervisor")

    # supervisor → 条件路由（只映射实际注册的 Agent 节点，避免图编译失败）
    route_map: Dict[str, Any] = {label: label for label in agent_nodes}
    route_map.update(
        {
            "synthesizer": "synthesizer",
            "__end__": END,
        }
    )
    if enable_hitl:
        route_map["human_review"] = "human_review"
    workflow.add_conditional_edges("supervisor", route_after_supervisor, route_map)

    # Agent 执行完成后回到 supervisor（循环，直到计划完成）
    for label in agent_nodes:
        workflow.add_edge(label, "supervisor")

    if enable_hitl:
        workflow.add_conditional_edges(
            "human_review",
            route_after_human_review,
            {"continue": "supervisor", "end": END},
        )

    workflow.add_edge("synthesizer", END)

    # 编译（HITL 必须携带 checkpointer）
    if enable_hitl and checkpointer is None:
        if MemorySaver is None:
            raise RuntimeError("未安装 langgraph-checkpoint，无法启用 Human-in-the-Loop")
        checkpointer = MemorySaver()

    app = workflow.compile(checkpointer=checkpointer)
    logger.info(
        "多智能体工作流编译完成（nodes=%s, hitl=%s, max_iterations=%d）",
        list(workflow.nodes) or "built",
        enable_hitl,
        max_iterations,
    )
    return app


# --------------------------------------------------------------------------- #
# 工作流执行器
# --------------------------------------------------------------------------- #
class MultiAgentWorkflow:
    """多智能体工作流执行器（封装编译后的 StateGraph）。

    提供同步执行、逐节点流式输出、人工审核恢复与前端事件转换。
    """

    def __init__(
        self,
        supervisor: SupervisorAgent,
        agents: Dict[str, BaseAgent],
        enable_hitl: bool = True,
        max_iterations: int = 10,
        checkpointer: Optional[Any] = None,
    ) -> None:
        self.supervisor = supervisor
        self.agents = {key: agents[key] for key in agents if key in AGENT_NODES}
        self.enable_hitl = enable_hitl
        self.max_iterations = int(
            settings.WORKFLOW_MAX_ITERATIONS
            if max_iterations is None
            else max_iterations
        )
        self._app = None
        self._build_graph(checkpointer=checkpointer)

    def _build_graph(self, checkpointer: Optional[Any] = None) -> None:
        self._app = build_multi_agent_workflow(
            supervisor=self.supervisor,
            agents=self.agents,
            enable_hitl=self.enable_hitl,
            max_iterations=self.max_iterations,
            checkpointer=checkpointer,
        )

    @staticmethod
    def _config(thread_id: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        base = dict(config or {})
        base.setdefault("configurable", {})["thread_id"] = thread_id
        return base

    def run(
        self,
        query: str,
        thread_id: str = "default",
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """同步执行整个工作流。

        Returns:
            ``{"final_answer", "trace", "plan", "status", "execution_time"}``
        """
        started = time.perf_counter()
        cfg = self._config(thread_id, config)
        final_state = self._app.invoke(build_initial_state(query, self.max_iterations), cfg)
        return self._summarize(final_state, started)

    def stream(
        self,
        query: str,
        thread_id: str = "default",
        config: Optional[Dict[str, Any]] = None,
    ):
        """流式执行：逐步产出节点级更新事件。

        事件为 ``{节点名: 状态更新}`` 字典；若被 interrupt 暂停，会产生
        含 ``__interrupt__`` 的事件，调用方可用 :meth:`resume` 恢复。
        """
        cfg = self._config(thread_id, config)
        for event in self._app.stream(
            build_initial_state(query, self.max_iterations),
            cfg,
            stream_mode="updates",
        ):
            yield event

    def resume(
        self,
        thread_id: str,
        feedback: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """恢复被人工审核中断的工作流。

        Args:
            thread_id: 会话线程 ID。
            feedback: ``approve`` / ``reject`` / ``modify``（或任意修改说明）。

        Returns:
            工作流最终（或下一中断点）状态摘要。
        """
        cfg = self._config(thread_id, config)
        Command = _get_command()
        started = time.perf_counter()

        if Command is not None:
            # 现代 langgraph：直接以 Command(resume=...) 恢复
            final_state = self._app.invoke(Command(resume=feedback), cfg)
        else:  # pragma: no cover - 兼容旧版本
            snapshot = self._app.get_state(cfg)
            values = dict(snapshot.values)
            values["human_feedback"] = feedback
            self._app.update_state(cfg, values, as_node="human_review")
            final_state = self._app.invoke(None, cfg)
        return self._summarize(final_state, started)

    def get_state(self, thread_id: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """读取某线程的当前状态快照（含中断后待审核现场）。"""
        snapshot = self._app.get_state(self._config(thread_id, config))
        return dict(snapshot.values)

    @staticmethod
    def _summarize(state: Dict[str, Any], started: Optional[float] = None) -> Dict[str, Any]:
        """把 LangGraph 状态归一化为对外结果结构。"""
        return {
            "final_answer": state.get("final_answer") or "",
            "trace": list(state.get("results", [])),
            "plan": list(state.get("plan", [])),
            "status": "ok" if not state.get("error") else "error",
            "error": state.get("error"),
            "execution_time": round(time.perf_counter() - started, 3) if started else 0.0,
        }

    def get_final_answer(self, thread_id: str, config: Optional[Dict[str, Any]] = None) -> str:
        """便捷方法：读取某线程最终的最终回答。"""
        state = self.get_state(thread_id, config)
        return state.get("final_answer") or ""

    # ------------------------------------------------------------------ #
    # 前端集成
    # ------------------------------------------------------------------ #
    def to_frontend_event(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """将工作流状态转换为前端可消费的事件结构。"""
        return {
            "type": "node_update",
            "current_node": state.get("current_agent"),
            "next_agent": state.get("next_agent"),
            "iteration": state.get("iteration", 0),
            "plan": state.get("plan", []),
            "results": state.get("results", []),
            "pending_action": state.get("pending_action"),
            "final_answer": state.get("final_answer"),
            "error": state.get("error"),
        }
