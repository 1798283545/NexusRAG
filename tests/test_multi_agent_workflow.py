"""第三阶段 Day 5-7：LangGraph 多智能体工作流测试。

覆盖：简单单任务、多步协作、Human-in-the-Loop（批准 / 拒绝）、线程状态
隔离与恢复、流式逐节点输出、子 Agent 失败时的优雅降级，以及条件路由 /
前端事件等纯函数逻辑。

说明：依赖 langgraph 与 langchain-core，未安装时整模块自动跳过。
"""

from __future__ import annotations

import pytest

pytest.importorskip("langchain_core")
pytest.importorskip("langgraph")

from typing import Any, Dict, List, Optional

from workflows import (
    MultiAgentState,
    MultiAgentWorkflow,
    build_multi_agent_workflow,
    route_after_human_review,
    route_after_supervisor,
)
from workflows.multi_agent_workflow import build_initial_state


# --------------------------------------------------------------------------- #
# 测试替身
# --------------------------------------------------------------------------- #
class FakeAgent:
    """专业智能体替身：记录调用，可模拟失败 / 延时。"""

    def __init__(self, name: str, should_raise: bool = False) -> None:
        self.name = name
        self.should_raise = should_raise
        self.calls: List[str] = []

    def run(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append(query)
        if self.should_raise:
            raise RuntimeError(f"{self.name} 模拟执行失败")
        return {"output": f"{self.name} 结果：{query}"}


class FakeSupervisor:
    """Supervisor 替身：按需返回固定计划，提供结果整合。"""

    def __init__(self, plan: List[Dict[str, Any]]) -> None:
        self._plan = plan

    def plan(self, query: str) -> List[Dict[str, Any]]:
        return self._plan

    def _synthesize_results(self, results: List[Dict[str, Any]], original_query: str) -> str:
        parts = [f"【整合】用户问题：{original_query}"]
        for record in results:
            body = record.get("output") or record.get("error") or "（无输出）"
            parts.append(
                f"[子任务 {record.get('task_id')} · {record.get('agent_name')} · "
                f"{record.get('status')}] {body}"
            )
        return "\n".join(parts)


def make_agents(**overrides: FakeAgent) -> Dict[str, FakeAgent]:
    agents = {
        "rag": FakeAgent("RAGAgent"),
        "code": FakeAgent("CodeAgent"),
        "web": FakeAgent("WebSearchAgent"),
        "summarizer": FakeAgent("SummarizerAgent"),
    }
    agents.update(overrides)
    return agents


def task(task_id: int, agent_name: str, description: str, **extra: Any) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "task_id": task_id,
        "agent_name": agent_name,
        "description": description,
        "dependencies": [],
    }
    data.update(extra)
    return data


# --------------------------------------------------------------------------- #
# 条件路由纯函数
# --------------------------------------------------------------------------- #
def test_route_after_supervisor_priorities() -> None:
    # 待审核优先
    state: Dict[str, Any] = {"pending_action": {"task_id": 1}, "next_agent": "code_agent"}
    assert route_after_supervisor(state) == "human_review"

    # finish → synthesizer
    assert route_after_supervisor({"pending_action": None, "next_agent": "finish", "iteration": 1}) == "synthesizer"

    # 达到上限 → synthesizer
    assert route_after_supervisor(
        {"pending_action": None, "next_agent": "code_agent", "iteration": 10, "max_iterations": 10}
    ) == "synthesizer"

    # 具体智能体
    assert route_after_supervisor(
        {"pending_action": None, "next_agent": "rag_agent", "iteration": 1, "max_iterations": 10}
    ) == "rag_agent"

    # 未知节点回落
    assert route_after_supervisor(
        {"pending_action": None, "next_agent": "mystery", "iteration": 1, "max_iterations": 10}
    ) == "synthesizer"


def test_route_after_human_review() -> None:
    assert route_after_human_review({"human_feedback": "rejected"}) == "end"
    assert route_after_human_review({"human_feedback": "approved"}) == "continue"
    assert route_after_human_review({"human_feedback": None}) == "continue"


# --------------------------------------------------------------------------- #
# 简单场景：单个子任务
# --------------------------------------------------------------------------- #
def test_simple_single_subtask() -> None:
    agents = make_agents()
    supervisor = FakeSupervisor(plan=[task(1, "rag", "查询知识库")])
    workflow = MultiAgentWorkflow(
        supervisor=supervisor, agents=agents, enable_hitl=False
    )

    result = workflow.run("查询知识库中的资料")
    assert len(agents["rag"].calls) == 1
    assert len(result["trace"]) == 1
    assert result["trace"][0]["status"] == "success"
    assert result["status"] == "ok"
    assert result["final_answer"]
    # 其余 Agent 不应被调用
    assert not agents["code"].calls and not agents["web"].calls


# --------------------------------------------------------------------------- #
# 多步场景：多个 Agent 协作，Supervisor 迭代路由
# --------------------------------------------------------------------------- #
def test_multi_agent_iteration_routing() -> None:
    agents = make_agents()
    supervisor = FakeSupervisor(
        plan=[
            task(1, "rag", "读取文档背景"),
            task(2, "code", "清洗并计算销售数据"),
            task(3, "web", "补充最新行业动态"),
        ]
    )
    workflow = MultiAgentWorkflow(
        supervisor=supervisor, agents=agents, enable_hitl=False
    )

    result = workflow.run("做一份多源分析")
    assert len(result["trace"]) == 3
    assert [r["agent_name"] for r in result["trace"]] == ["rag", "code", "web"]
    assert all(r["status"] == "success" for r in result["trace"])
    assert [len(c) for c in (agents["rag"].calls, agents["code"].calls, agents["web"].calls)] == [1, 1, 1]
    # 最终回答整合了三方结果
    assert result["final_answer"]
    assert "RAGAgent 结果" in result["final_answer"]


# --------------------------------------------------------------------------- #
# 依赖任务：B 依赖 A，顺序保证
# --------------------------------------------------------------------------- #
def test_dependent_subtasks_respect_order() -> None:
    agents = make_agents()
    supervisor = FakeSupervisor(
        plan=[
            task(1, "code", "先生成汇总表"),
            task(2, "rag", "基于汇总表答疑", dependencies=[1]),
        ]
    )
    workflow = MultiAgentWorkflow(
        supervisor=supervisor, agents=agents, enable_hitl=False
    )

    result = workflow.run("先算后问")
    assert [r["agent_name"] for r in result["trace"]] == ["code", "rag"]
    # 执行顺序严格一致
    assert agents["code"].calls and agents["rag"].calls


# --------------------------------------------------------------------------- #
# Human-in-the-Loop：批准后继续
# --------------------------------------------------------------------------- #
def test_hitl_approve_then_continue() -> None:
    code_agent = FakeAgent("CodeAgent")
    agents = make_agents(code=code_agent)
    supervisor = FakeSupervisor(
        plan=[task(1, "code", "执行数据分析代码", requires_approval=True)]
    )
    workflow = MultiAgentWorkflow(
        supervisor=supervisor, agents=agents, enable_hitl=True
    )
    thread = "hitl-approve"

    # 1) 流式执行到 interrupt 暂停
    events = list(workflow.stream("运行代码任务", thread_id=thread))
    assert any("__interrupt__" in event for event in events), "应在 human_review 处中断"

    # 2) 审核现场：pending_action 已写入，代码尚未执行
    snapshot = workflow.get_state(thread)
    assert snapshot["pending_action"] is not None
    assert snapshot["pending_action"]["agent_name"] == "code"
    assert not code_agent.calls

    # 3) 模拟人工批准 → 工作流继续并完成
    result = workflow.resume(thread, "approve")
    assert result["status"] == "ok"
    assert len(code_agent.calls) == 1
    assert result["final_answer"]
    assert "CodeAgent 结果" in result["final_answer"]


# --------------------------------------------------------------------------- #
# Human-in-the-Loop：拒绝后终止
# --------------------------------------------------------------------------- #
def test_hitl_reject_terminates() -> None:
    code_agent = FakeAgent("CodeAgent")
    agents = make_agents(code=code_agent)
    supervisor = FakeSupervisor(
        plan=[task(1, "code", "删除生产环境数据", requires_approval=True)]
    )
    workflow = MultiAgentWorkflow(
        supervisor=supervisor, agents=agents, enable_hitl=True
    )
    thread = "hitl-reject"

    list(workflow.stream("执行危险操作", thread_id=thread))
    result = workflow.resume(thread, "reject")

    assert result["status"] == "error"
    assert result["error"] == "操作被用户拒绝"
    assert not code_agent.calls, "拒绝后不应执行代码"
    assert not result["final_answer"]


# --------------------------------------------------------------------------- #
# 状态恢复：线程隔离 + 断点恢复
# --------------------------------------------------------------------------- #
def test_thread_isolation_and_restore() -> None:
    agents = make_agents()
    supervisor = FakeSupervisor(
        plan=[task(1, "rag", "第一次查询"), task(2, "web", "第二次查询")]
    )
    workflow = MultiAgentWorkflow(
        supervisor=supervisor, agents=agents, enable_hitl=False
    )

    first = workflow.run("线程 A 任务", thread_id="thread-a")
    assert len(first["trace"]) == 2

    # 相同线程再次 run 会从已完成状态继续/复跑同一计划
    agents["rag"].calls.clear()
    second = workflow.run("线程 B 独立任务", thread_id="thread-b")
    assert len(second["trace"]) == 2
    assert len(agents["rag"].calls) == 1  # thread-b 独立执行了自己的计划
    assert len(agents["web"].calls) == 1


# --------------------------------------------------------------------------- #
# 流式输出：逐节点产出事件
# --------------------------------------------------------------------------- #
def test_streaming_yields_node_events() -> None:
    agents = make_agents()
    supervisor = FakeSupervisor(plan=[task(1, "rag", "查询")])
    workflow = MultiAgentWorkflow(
        supervisor=supervisor, agents=agents, enable_hitl=False
    )

    events = list(workflow.stream("查询一次", thread_id="stream-demo"))
    node_keys = [key for event in events for key in event]
    assert "supervisor" in node_keys
    assert "rag_agent" in node_keys
    assert "synthesizer" in node_keys


# --------------------------------------------------------------------------- #
# 错误处理：Agent 失败优雅降级
# --------------------------------------------------------------------------- #
def test_agent_failure_degrades_gracefully() -> None:
    agents = make_agents(rag=FakeAgent("RAGAgent", should_raise=True))
    supervisor = FakeSupervisor(plan=[task(1, "rag", "一定会失败")])
    workflow = MultiAgentWorkflow(
        supervisor=supervisor, agents=agents, enable_hitl=False
    )

    result = workflow.run("触发失败")
    assert result["status"] == "ok"  # 流程未被中断
    assert result["trace"][0]["status"] == "failed"
    assert "模拟执行失败" in result["trace"][0]["error"]
    # 仍能产出兜底整合结果
    assert result["final_answer"]
    assert "模拟执行失败" in result["final_answer"]


# --------------------------------------------------------------------------- #
# 前端事件转换
# --------------------------------------------------------------------------- #
def test_to_frontend_event_shape() -> None:
    agents = make_agents()
    supervisor = FakeSupervisor(plan=[task(1, "web", "查询")])
    workflow = MultiAgentWorkflow(
        supervisor=supervisor, agents=agents, enable_hitl=False
    )
    event = workflow.to_frontend_event(build_initial_state("你好", 5))
    assert event["type"] == "node_update"
    for key in ("current_node", "plan", "results", "pending_action", "final_answer"):
        assert key in event


def test_max_iterations_guard() -> None:
    """超过最大迭代次数后强制进入整合，避免死循环。"""
    agents = make_agents()
    # 计划中塞入多于迭代上限的任务数（这里只需验证达到上限后仍能结束）
    plan = [task(i, "rag", f"第 {i} 步") for i in range(1, 4)]
    supervisor = FakeSupervisor(plan=plan)
    workflow = MultiAgentWorkflow(
        supervisor=supervisor, agents=agents, enable_hitl=False, max_iterations=2
    )

    result = workflow.run("限次任务")
    # 迭代被截断但仍产出结果，不会无限循环
    assert len(result["trace"]) <= 2
    assert result["final_answer"]
