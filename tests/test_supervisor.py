"""第三阶段 Day 3-4：SupervisorAgent 单元测试。

覆盖：初始化与工具注册、单任务分发、混合任务拆解、依赖任务串行、
并行执行、智能体缺失降级、子任务失败容错、规划重试与全败降级、
输出结构完整性以及 LangGraph 状态导出。

全部使用确定性 stub（不联网、不加载真实模型）。
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional

import pytest

from agents import AgentFactory, SupervisorAgent
from agents.agent_factory import _SUPERVISOR_ARGS
from agents.supervisor import SubTask, TaskPlan


# --------------------------------------------------------------------------- #
# 测试替身
# --------------------------------------------------------------------------- #
class FakeSpecialist:
    """专业智能体替身：可记录调用、模拟耗时 / 抛错。"""

    def __init__(
        self,
        name: str,
        key: str,
        description: str = "测试用智能体",
        recorder: Optional[list] = None,
        delay: float = 0.0,
        should_raise: bool = False,
        tracker: Optional["_ConcurrencyTracker"] = None,
    ) -> None:
        self.name = name
        self.key = key
        self.description = description
        self.recorder = recorder
        self.delay = delay
        self.should_raise = should_raise
        self.tracker = tracker

    def run(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        if self.tracker is not None:
            self.tracker.enter(self.key)
        if self.recorder is not None:
            self.recorder.append(self.key)
        if self.delay:
            time.sleep(self.delay)
        try:
            if self.should_raise:
                raise RuntimeError(f"{self.name} 模拟执行失败")
            return {"output": f"{self.name} 已完成：{query}"}
        finally:
            if self.tracker is not None:
                self.tracker.exit(self.key)


class _ConcurrencyTracker:
    """统计并发峰值与调用顺序。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.order: List[str] = []

    def enter(self, name: str) -> None:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.order.append(f"{name}:start")

    def exit(self, name: str) -> None:
        with self._lock:
            self.active -= 1
            self.order.append(f"{name}:end")


class FakePlanLLM:
    """监督者用 LLM 替身：按提示词类别分流。

    - 规划提示（含 ``JSON 结构输出``）：从 ``plan_responses`` 队列取响应；
    - 整合提示（含 ``子任务结果：``）：透传子任务片段；
    - 其余（直接回答 / 降级）：返回 ``direct_text``。
    """

    def __init__(
        self,
        plan_responses: Optional[List[Any]] = None,
        direct_text: str = "监督者兜底回答",
    ) -> None:
        self.plan_responses = list(plan_responses or [])
        self.direct_text = direct_text
        self.calls: List[str] = []

    def with_structured_output(self, schema: Any) -> "FakePlanLLM":
        # 模拟支持 structured output 的 ChatModel
        return self

    def invoke(self, message: Any) -> Any:
        text = message if isinstance(message, str) else str(message)
        self.calls.append(text)
        if "JSON 结构输出" in text:
            if self.plan_responses:
                return self.plan_responses.pop(0)
            return TaskPlan(plan=[])  # 空计划触发解析重试
        if "子任务结果：" in text:
            start = text.find("子任务结果：") + len("子任务结果：")
            end = text.find("\n\n要求：", start)
            segment = text[start:end if end != -1 else len(text)]
            return f"（整合结果）{segment.strip()[:500]}"
        return self.direct_text

    def plan_call_count(self) -> int:
        return sum(1 for call in self.calls if "JSON 结构输出" in call)


def make_subtask(
    task_id: int, agent_name: str, description: str, dependencies: Optional[List[int]] = None
) -> SubTask:
    return SubTask(
        task_id=task_id,
        agent_name=agent_name,
        description=description,
        dependencies=list(dependencies or []),
    )


def make_specialists(recorder: Optional[list] = None) -> Dict[str, FakeSpecialist]:
    """构造四个常用专业智能体替身。"""
    specs: Dict[str, FakeSpecialist] = {}
    for key, name in [
        ("rag", "RAGAgent"),
        ("code", "CodeAgent"),
        ("web", "WebSearchAgent"),
        ("summarizer", "SummarizerAgent"),
    ]:
        specs[key] = FakeSpecialist(
            name=name, key=key, recorder=recorder, description=f"{name}（测试替身）"
        )
    return specs


def make_supervisor(
    agents: Optional[Dict[str, FakeSpecialist]] = None,
    llm: Optional[FakePlanLLM] = None,
    **kwargs: Any,
) -> SupervisorAgent:
    return SupervisorAgent(
        llm=llm or FakePlanLLM(),
        agents=agents or make_specialists(),
        verbose=False,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# 1. 初始化：子智能体被正确封装为工具
# --------------------------------------------------------------------------- #
def test_supervisor_registers_subagents_as_tools() -> None:
    supervisor = make_supervisor()
    assert len(supervisor.tools) == 4
    assert {tool.name for tool in supervisor.tools} == {
        "RAGAgent",
        "CodeAgent",
        "WebSearchAgent",
        "SummarizerAgent",
    }
    for tool in supervisor.tools:
        assert "调用专业智能体" in tool.description
    assert supervisor.get_agent_status() == {
        "rag": "ready",
        "code": "ready",
        "web": "ready",
        "summarizer": "ready",
    }
    description_text = supervisor.get_available_agents_description()
    assert "summarizer" in description_text and "rag" in description_text


# --------------------------------------------------------------------------- #
# 2. 单任务场景：路由到 SummarizerAgent
# --------------------------------------------------------------------------- #
def test_single_task_routed_to_summarizer() -> None:
    llm = FakePlanLLM(
        plan_responses=[
            TaskPlan(
                plan=[make_subtask(1, "summarizer", "总结文档A", [])]
            )
        ]
    )
    supervisor = make_supervisor(llm=llm, enable_parallel=False)

    result = supervisor.run("总结文档A")
    assert result["plan"][0]["agent_name"] == "summarizer"
    assert len(result["results"]) == 1
    record = result["results"][0]
    assert record["status"] == "success"
    assert record["agent_name"] == "summarizer"
    assert "总结文档A" in record["output"]
    assert "SummarizerAgent" in result["execution_trace"]


# --------------------------------------------------------------------------- #
# 3. 混合任务：拆成 Summarizer + WebSearch 两个子任务
# --------------------------------------------------------------------------- #
def test_mixed_task_dispatched_to_two_agents(recorder=None) -> None:
    order: List[str] = []
    llm = FakePlanLLM(
        plan_responses=[
            TaskPlan(
                plan=[
                    make_subtask(1, "summarizer", "总结文档A", []),
                    make_subtask(2, "web", "搜索最新行业动态", []),
                ]
            )
        ]
    )
    agents = make_specialists(recorder=order)
    supervisor = make_supervisor(agents=agents, llm=llm, enable_parallel=False)

    result = supervisor.run("总结文档A并搜索最新行业动态")
    plan = result["plan"]
    assert len(plan) == 2
    assert {item["agent_name"] for item in plan} == {"summarizer", "web"}
    statuses = {item["agent_name"]: item["status"] for item in result["results"]}
    assert statuses == {"summarizer": "success", "web": "success"}
    assert sorted(order) == ["summarizer", "web"]


# --------------------------------------------------------------------------- #
# 4. 依赖任务：B 依赖 A，保证顺序与正确等待
# --------------------------------------------------------------------------- #
def test_dependent_tasks_are_serialized() -> None:
    order: List[str] = []
    llm = FakePlanLLM(
        plan_responses=[
            TaskPlan(
                plan=[
                    make_subtask(1, "code", "先计算数据", []),
                    make_subtask(2, "rag", "基于结果查询", [1]),
                ]
            )
        ]
    )
    agents = make_specialists(recorder=order)
    supervisor = make_supervisor(agents=agents, llm=llm, enable_parallel=True)

    result = supervisor.run("先算再查")
    assert [item["status"] for item in result["results"]] == ["success", "success"]
    # 即便启用并行，依赖任务也必须严格按序执行
    assert order == ["code", "rag"]


# --------------------------------------------------------------------------- #
# 5. 并行执行：独立子任务真正并发
# --------------------------------------------------------------------------- #
def test_independent_tasks_run_in_parallel() -> None:
    tracker = _ConcurrencyTracker()
    agents = make_specialists(recorder=None)
    agents["code"].tracker = tracker
    agents["code"].delay = 0.3
    agents["web"].tracker = tracker
    agents["web"].delay = 0.3

    llm = FakePlanLLM(
        plan_responses=[
            TaskPlan(
                plan=[
                    make_subtask(1, "code", "跑一个计算", []),
                    make_subtask(2, "web", "查一条资讯", []),
                ]
            )
        ]
    )
    supervisor = make_supervisor(agents=agents, llm=llm, enable_parallel=True)

    started = time.perf_counter()
    result = supervisor.run("并行任务")
    elapsed = time.perf_counter() - started

    # 两个 0.3s 任务：串行约 0.6s，并行显著更短
    assert tracker.max_active >= 2, "两个独立子任务应并发执行"
    assert elapsed < 0.55, f"并行执行耗时异常：{elapsed:.2f}s"
    assert all(item["status"] == "success" for item in result["results"])


# --------------------------------------------------------------------------- #
# 6. 智能体不可用：跳过并记录，不阻塞其余任务
# --------------------------------------------------------------------------- #
def test_unavailable_agent_is_skipped() -> None:
    agents = {"code": FakeSpecialist(name="CodeAgent", key="code")}
    llm = FakePlanLLM(
        plan_responses=[
            TaskPlan(
                plan=[
                    make_subtask(1, "ghost", "不存在的智能体", []),
                    make_subtask(2, "code", "真实子任务", []),
                ]
            )
        ]
    )
    supervisor = make_supervisor(agents=agents, llm=llm, enable_parallel=False)

    result = supervisor.run("执行任务")
    by_id = {item["task_id"]: item for item in result["results"]}
    assert by_id[1]["status"] == "unavailable"
    assert "未注册" in by_id[1]["error"]
    assert by_id[2]["status"] == "success"
    # 单个失败不影响整体回答
    assert result["output"]


# --------------------------------------------------------------------------- #
# 7. 子任务抛错：记录失败但不阻塞整体
# --------------------------------------------------------------------------- #
def test_failed_subtask_does_not_block_others() -> None:
    agents = make_specialists()
    agents["code"].should_raise = True
    llm = FakePlanLLM(
        plan_responses=[
            TaskPlan(
                plan=[
                    make_subtask(1, "code", "会失败的任务", []),
                    make_subtask(2, "summarizer", "能成功的任务", []),
                ]
            )
        ]
    )
    supervisor = make_supervisor(agents=agents, llm=llm, enable_parallel=False)

    result = supervisor.run("混合失败任务")
    statuses = {item["task_id"]: item["status"] for item in result["results"]}
    assert statuses == {1: "failed", 2: "success"}
    assert "模拟执行失败" in result["results"][0]["error"]
    assert result["output"]


# --------------------------------------------------------------------------- #
# 8. 规划解析重试：前两次失败、第三次成功
# --------------------------------------------------------------------------- #
def test_planning_retries_then_succeeds() -> None:
    llm = FakePlanLLM(
        plan_responses=[
            "这不是 JSON {",  # 第 1 次：解析失败
            {"unexpected": "schema"},  # 第 2 次：结构错误
            TaskPlan(plan=[make_subtask(1, "code", "最终任务", [])]),  # 第 3 次：成功
        ]
    )
    supervisor = make_supervisor(llm=llm, max_planning_attempts=3)

    result = supervisor.run("多次规划")
    assert llm.plan_call_count() == 3, "应当重试 3 次（2 次失败 + 1 次成功）"
    assert len(result["plan"]) == 1
    assert result["results"][0]["status"] == "success"


# --------------------------------------------------------------------------- #
# 9. 规划全败：降级为监督者直接回答
# --------------------------------------------------------------------------- #
def test_planning_all_failed_degrades_to_direct_answer() -> None:
    llm = FakePlanLLM(plan_responses=["坏数据", "坏数据", "坏数据"])
    supervisor = make_supervisor(llm=llm, max_planning_attempts=3)

    result = supervisor.run("无法规划的任务")
    assert llm.plan_call_count() == 3
    assert result["plan"] == []
    assert result["results"] == []
    assert result["output"] == "监督者兜底回答"
    step_names = [item["step"] for item in result["steps"]]
    assert "degrade" in step_names


# --------------------------------------------------------------------------- #
# 10. 输出结构完整性 + LangGraph 状态
# --------------------------------------------------------------------------- #
def test_output_structure_and_langgraph_state() -> None:
    llm = FakePlanLLM(
        plan_responses=[
            TaskPlan(plan=[make_subtask(1, "code", "跑一个任务", [])])
        ]
    )
    supervisor = make_supervisor(llm=llm, enable_parallel=False)

    result = supervisor.run("结构化输出验证")
    for key in ("output", "plan", "results", "steps", "execution_trace", "elapsed_ms"):
        assert key in result, f"结果缺少字段 {key}"

    state = supervisor.to_langgraph_state()
    assert set(state) == {"messages", "plan", "results", "current_step", "next_agent"}
    assert state["messages"][0]["role"] == "user"
    assert state["current_step"] == 1


# --------------------------------------------------------------------------- #
# 11. 工厂：Supervisor 快速装配
# --------------------------------------------------------------------------- #
def test_factory_create_supervisor_with_fake_agents() -> None:
    agents = make_specialists()
    supervisor = AgentFactory.create_supervisor(
        llm=FakePlanLLM(), agents=agents, enable_parallel=False
    )
    assert isinstance(supervisor, SupervisorAgent)
    assert len(supervisor.tools) == 4
    assert supervisor.enable_parallel is False


def test_factory_create_all_with_supervisor_skips_rag_without_chain() -> None:
    bundle = AgentFactory.create_all_with_supervisor(
        llm=FakePlanLLM(), rag_chain=None, workspace_dir="./workspace"
    )
    assert set(bundle) == {"supervisor", "code", "web"}
    assert isinstance(bundle["supervisor"], SupervisorAgent)
    assert set(bundle["supervisor"].agents) == {"code", "web"}


def test_supervisor_filter_supported_kwargs() -> None:
    assert "rag_chain" not in _SUPERVISOR_ARGS
    assert {"enable_parallel", "max_planning_attempts", "subtask_timeout"} <= _SUPERVISOR_ARGS
