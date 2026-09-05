"""监督者智能体（SupervisorAgent）。

作为多智能体系统的「大脑」，负责任务理解 → 拆解规划 → 并行/串行调度 →
结果整合。继承 :class:`BaseAgent`，本身是一个可被 LangGraph 编排的单元：

执行链路：``_plan_task -> _execute_plan -> _synthesize_results``。

- 规划：通过 ``with_structured_output``（缺失时退回 JSON 文本解析）要求
  LLM 输出结构化任务计划（:class:`TaskPlan`），失败按 ``max_planning_attempts``
  重试，最终失败降级为直接回答；
- 调度：按依赖关系分批执行，独立子任务可用 ``ThreadPoolExecutor`` 并行，
  每个子任务带超时；
- 容错：单个子任务失败 / 智能体缺失不阻塞整体，最终结果统一整合并附
  ``execution_trace`` 便于前端展示与 LangGraph 状态构建。
"""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from agents.base_agent import BaseAgent, _text_of
from config import settings

logger = logging.getLogger(__name__)

#: 事件类型（用于 steps / execution_trace）
_PLAN = "plan"
_EXECUTE = "execute"
_SYNTHESIZE = "synthesize"
_DEGRADE = "degrade"
_BLOCKED = "blocked"


class SubTask(BaseModel):
    """单条子任务。

    Attributes:
        task_id: 子任务唯一标识（正整数）。
        agent_name: 负责执行的智能体名称（须为 ``agents.keys()`` 中的值）。
        description: 该子任务要完成的自然语言描述。
        dependencies: 依赖的子任务 ``task_id`` 列表（先完成依赖再执行本任务）。
    """

    task_id: int = Field(description="子任务唯一编号")
    agent_name: str = Field(description="负责执行的智能体名称")
    description: str = Field(description="子任务描述，需自包含、可独立执行")
    dependencies: List[int] = Field(default_factory=list, description="依赖的子任务编号列表")


class TaskPlan(BaseModel):
    """LLM 输出的结构化任务计划。"""

    plan: List[SubTask] = Field(description="子任务列表")


class _AgentQueryArgs(BaseModel):
    query: str = Field(description="交给子智能体执行的子任务描述")


class _AgentCallTool(BaseTool):
    """将一个专业智能体包装为可供 Supervisor 调用的工具。

    工具名称 = 智能体名称（如 ``RAGAgent``），``_run`` 转发给子智能体的
    ``run(query)`` 并提取文本输出。
    """

    name: str = ""
    description: str = ""
    args_schema: type[BaseModel] = _AgentQueryArgs
    agent: Any = None

    def _run(self, query: str, **kwargs: Any) -> str:
        try:
            result = self.agent.run(query)
        except Exception as exc:  # noqa: BLE001 - 以文本回传便于监督者决策
            logger.exception("子智能体 %s 执行异常", self.agent.name)
            return f"（子智能体 {getattr(self.agent, 'name', '?')} 执行失败：{exc}）"
        if isinstance(result, dict):
            text = result.get("output")
            return str(text).strip() if text else str(result)
        return str(result).strip()

    async def _arun(self, query: str, **kwargs: Any) -> str:  # pragma: no cover
        return self._run(query, **kwargs)

    def __init__(self, agent: Any, name: str, description: str, **kwargs: Any) -> None:
        kwargs["agent"] = agent
        kwargs["name"] = name
        kwargs["description"] = description
        super().__init__(**kwargs)


class SupervisorAgent(BaseAgent):
    """监督者智能体：理解、拆解、分发与整合。

    Args:
        llm: 大语言模型（用于规划与最终整合）。
        agents: ``{"rag": rag_agent, "code": code_agent, ...}`` 专业智能体注册表。
        enable_parallel: 是否对无依赖子任务并行执行；默认取配置
            ``SUPERVISOR_ENABLE_PARALLEL``。
        max_planning_attempts: 规划解析失败后的最大重试次数；默认取配置
            ``SUPERVISOR_MAX_PLANNING_ATTEMPTS``。
        subtask_timeout: 单个子任务超时秒数；默认取配置
            ``SUPERVISOR_SUBTASK_TIMEOUT``。
        **kwargs: 透传给 BaseAgent 的其余参数（tools / verbose 等）。
    """

    def __init__(
        self,
        llm: Any,
        agents: Dict[str, BaseAgent],
        enable_parallel: Optional[bool] = None,
        max_planning_attempts: Optional[int] = None,
        subtask_timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        self.agents: Dict[str, BaseAgent] = dict(agents or {})
        self.enable_parallel: bool = (
            bool(settings.SUPERVISOR_ENABLE_PARALLEL)
            if enable_parallel is None
            else bool(enable_parallel)
        )
        self.max_planning_attempts: int = int(
            settings.SUPERVISOR_MAX_PLANNING_ATTEMPTS
            if max_planning_attempts is None
            else max_planning_attempts
        )
        self.subtask_timeout: float = float(
            settings.SUPERVISOR_SUBTASK_TIMEOUT
            if subtask_timeout is None
            else subtask_timeout
        )
        # SUPERVISOR_VERBOSE 仅在调用方未显式指定 verbose 时生效
        if kwargs.get("verbose") is None:
            kwargs["verbose"] = settings.SUPERVISOR_VERBOSE

        # 记录最近一次执行状态，供 to_langgraph_state() 使用
        self._last_state: Dict[str, Any] = {
            "query": None,
            "plan": [],
            "results": [],
            "steps": [],
        }

        super().__init__(
            name="SupervisorAgent",
            description=(
                "监督者智能体：能理解复杂指令，将任务拆解为子任务，"
                "并分发给专业的子智能体执行。"
            ),
            llm=llm,
            **kwargs,
        )

    # ------------------------------------------------------------------ #
    # BaseAgent 抽象实现
    # ------------------------------------------------------------------ #
    def _get_default_tools(self) -> List[BaseTool]:
        """将每个专业智能体封装为一个可调用工具。"""
        return self._create_agent_tools()

    def _create_agent_tools(self) -> List[BaseTool]:
        """把注册的子智能体逐一封装成 :class:`_AgentCallTool`。

        工具名称 = 智能体名称（如 ``RAGAgent``），描述 = 智能体描述；
        便于上层 AgentExecutor / 模型调用对应智能体。
        """
        tools: List[BaseTool] = []
        for name, agent in self.agents.items():
            canonical = getattr(agent, "name", None) or name
            desc = getattr(agent, "description", "") or f"执行「{name}」相关子任务"
            tools.append(
                _AgentCallTool(
                    agent=agent,
                    name=str(canonical),
                    description=(
                        f"调用专业智能体「{canonical}」（注册名 {name}）执行一个子任务。"
                        f"输入为该子任务的完整描述。\n能力说明：{desc}"
                    ),
                )
            )
        return tools

    def _get_system_prompt(self) -> str:
        """返回监督者系统提示词（已渲染各智能体能力描述）。"""
        template = (
            "你是监督者智能体，负责协调多个专业智能体完成复杂任务。\n\n"
            "可用智能体：\n{agent_descriptions}\n\n"
            "你的工作流程：\n"
            "1. 理解用户输入的任务本质\n"
            "2. 将复杂任务拆解为可并行或串行的子任务\n"
            "3. 为每个子任务选择最合适的智能体\n"
            "4. 调用智能体执行子任务\n"
            "5. 整合所有结果，形成完整回答返回给用户\n\n"
            "决策原则：\n"
            "- 信息查询 → RAGAgent\n"
            "- 数据处理/计算/图表 → CodeAgent\n"
            "- 实时信息/新闻 → WebSearchAgent\n"
            "- 长文档摘要/对比 → SummarizerAgent\n"
            "- 混合任务 → 拆解后分别调用\n\n"
            "输出要求：\n"
            "- 对每个调用步骤给出简要说明\n"
            "- 最终回答需整合所有子任务结果\n"
            "- 若某智能体执行失败，尝试替代方案或说明原因"
        )
        return template.replace(
            "{agent_descriptions}", self.get_available_agents_description()
        )

    # ------------------------------------------------------------------ #
    # 对外只读接口
    # ------------------------------------------------------------------ #
    def get_agent_status(self) -> Dict[str, str]:
        """返回各子智能体状态（ready / unavailable）。"""
        status: Dict[str, str] = {}
        for name, agent in self.agents.items():
            if agent is not None and hasattr(agent, "run"):
                status[name] = "ready"
            else:
                status[name] = "unavailable"
                logger.warning("智能体 %s 未初始化或不可用", name)
        return status

    def get_available_agents_description(self) -> str:
        """生成「注册名 - 能力描述」文本，用于注入系统提示 / 规划提示。"""
        if not self.agents:
            return "（当前没有可用的专业智能体）"
        return "\n".join(
            f"- {name}（{getattr(agent, 'name', name)}）: "
            f"{getattr(agent, 'description', '') or '未提供描述'}"
            for name, agent in self.agents.items()
        )

    # ------------------------------------------------------------------ #
    # 主入口（重写 BaseAgent.run）
    # ------------------------------------------------------------------ #
    def run(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        """执行完整监督流程：规划 → 调度 → 整合。

        Args:
            query: 用户输入或上级下发的复杂任务。
            **kwargs: 预留参数（记录日志）。

        Returns:
            统一结果字典，包含：
            ``output``（最终回答）、``plan``（任务计划）、``results``
            （各子任务执行明细）、``steps``（结构化执行步骤）与
            ``execution_trace``（人类可读执行链路）。
        """
        started = time.perf_counter()
        steps: List[Dict[str, Any]] = []
        logger.info("监督者开始处理任务（并行=%s）：%s", self.enable_parallel, query)

        # ① 规划
        self._record(steps, _PLAN, detail=f"规划任务：{query}")
        plan: List[SubTask] = self._plan_task(query)
        if not plan:
            self._record(
                steps,
                _DEGRADE,
                detail="多次规划失败，降级为监督者直接回答",
            )
            output = self._fallback_direct(query)
            results: List[Dict[str, Any]] = []
        else:
            # ② 执行
            self._record(
                steps, _EXECUTE, detail=f"开始执行 {len(plan)} 个子任务"
            )
            results = self._execute_plan(plan, steps=steps)
            # ③ 整合
            self._record(steps, _SYNTHESIZE, detail="整合各子任务结果")
            output = self._synthesize_results(results, query)

        plan_payload = [task.model_dump() for task in plan]
        trace = self._build_trace(steps)
        total_ms = (time.perf_counter() - started) * 1000

        # 供日志与 LangGraph 状态使用
        self._last_state = {
            "query": query,
            "plan": plan_payload,
            "results": results,
            "steps": steps,
            "output": output,
        }

        result = {
            "output": output,
            "plan": plan_payload,
            "results": results,
            "steps": steps,
            "execution_trace": trace,
            "elapsed_ms": round(total_ms, 1),
        }
        logger.info(
            "监督者完成（%.1fms）链路: %s", total_ms, trace.replace("\n", " | ")
        )
        return result

    # ------------------------------------------------------------------ #
    # ① 任务规划
    # ------------------------------------------------------------------ #
    def _plan_task(self, query: str) -> List[SubTask]:
        """调用 LLM 生成结构化任务计划；解析失败按次数重试。

        Returns:
            子任务列表；彻底失败时返回空列表（由调用方降级）。
        """
        prompt = self._build_plan_prompt(query)
        for attempt in range(1, self.max_planning_attempts + 1):
            try:
                raw = self._request_plan(prompt)
                subtasks = self._parse_plan(raw)
                if subtasks:
                    logger.info(
                        "规划成功（第 %d 次尝试）：%d 个子任务",
                        attempt,
                        len(subtasks),
                    )
                    return subtasks
                raise ValueError("LLM 返回空任务计划")
            except Exception as exc:  # noqa: BLE001 - 统一重试
                logger.warning(
                    "任务规划解析失败（第 %d/%d 次）：%s",
                    attempt,
                    self.max_planning_attempts,
                    exc,
                )
        logger.error("任务规划连续失败 %d 次，将降级为直接回答", self.max_planning_attempts)
        return []

    def _build_plan_prompt(self, query: str) -> str:
        """构造规划提示词（要求输出结构化 JSON 计划）。"""
        return (
            "请将以下任务拆解为子任务，并为每个子任务指定最合适的智能体。\n\n"
            "可用智能体：\n"
            f"{self.get_available_agents_description()}\n\n"
            f"用户任务：{query}\n\n"
            "约束：\n"
            "1. agent_name 必须是上述可用智能体的注册名；\n"
            "2. 每个子任务需自包含、可独立执行；\n"
            "3. 若子任务 B 依赖子任务 A 的结果，则 B.dependencies 需包含 A 的 task_id；\n"
            "4. 简单任务可只拆出 1 个子任务。\n\n"
            '请按如下 JSON 结构输出（不要输出其他文字）：\n'
            '{"plan": [{"task_id": 1, "agent_name": "...", '
            '"description": "...", "dependencies": []}]}'
        )

    def _request_plan(self, prompt: str) -> Any:
        """请求 LLM 输出计划。

        优先使用 ``with_structured_output`` 得到结构化对象；若当前 LLM
        不支持（无该方法或抛 NotImplementedError），退回普通 invoke +
        JSON 文本解析。
        """
        structured = getattr(self.llm, "with_structured_output", None)
        if structured is not None:
            try:
                return structured(TaskPlan).invoke(prompt)
            except (AttributeError, NotImplementedError, TypeError):
                logger.debug("with_structured_output 不可用，退回 JSON 文本解析")
        return self.llm.invoke(prompt)

    def _parse_plan(self, raw: Any) -> List[SubTask]:
        """把 LLM 返回（TaskPlan / dict / list / JSON 文本）归一化为子任务列表。"""
        obj: Any = raw
        if isinstance(raw, str):
            obj = self._extract_json(raw)
        if isinstance(obj, TaskPlan):
            items = list(obj.plan)
        elif isinstance(obj, dict):
            if isinstance(obj.get("plan"), list):
                items = obj["plan"]
            elif "task_id" in obj or "agent_name" in obj:
                items = [obj]  # 兼容单子任务 dict
            else:
                raise ValueError("计划 dict 中缺少 plan 字段")
        elif isinstance(obj, list):
            items = obj
        else:
            raise ValueError(f"无法识别的计划结构: {type(raw).__name__}")

        subtasks: List[SubTask] = []
        for index, item in enumerate(items, start=1):
            if isinstance(item, SubTask):
                subtask = item
            else:
                subtask = SubTask.model_validate(item)
            # 补齐 / 纠正 task_id，保证唯一
            if subtask.task_id in {t.task_id for t in subtasks}:
                subtask = subtask.model_copy(update={"task_id": index})
            subtasks.append(subtask)
        return subtasks

    @staticmethod
    def _extract_json(text: str) -> Any:
        """从 LLM 文本中稳健提取 JSON（容忍代码块 / 前后缀文字）。"""
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.MULTILINE).strip()
        candidates = [cleaned]
        start_bracket = cleaned.find("[")
        if start_bracket != -1:
            candidates.append(cleaned[start_bracket:])
        start_brace = cleaned.find("{")
        if start_brace != -1:
            candidates.append(cleaned[start_brace:])
        last_bracket = cleaned.rfind("]")
        if last_bracket != -1:
            candidates.append(cleaned[cleaned.find("[") : last_bracket + 1])
        last_brace = cleaned.rfind("}")
        if last_brace != -1 and cleaned.find("{") != -1:
            candidates.append(cleaned[cleaned.find("{") : last_brace + 1])

        last_error: Optional[Exception] = None
        for candidate in candidates:
            try:
                return json.loads(candidate)
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
        raise ValueError(f"无法从文本解析 JSON 计划: {last_error}") from last_error

    # ------------------------------------------------------------------ #
    # ② 计划执行
    # ------------------------------------------------------------------ #
    def _execute_plan(
        self, plan: List[SubTask], steps: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """按依赖关系分批执行任务；支持并行与单任务超时。

        Returns:
            子任务执行结果列表（含 status=success/failed/unavailable/timeout/blocked）。
        """
        results: List[Dict[str, Any]] = []
        done_ids: set[int] = set()
        result_by_id: Dict[int, Dict[str, Any]] = {}
        pending: Dict[int, SubTask] = {task.task_id: task for task in plan}

        while pending:
            ready_ids = sorted(
                task_id
                for task_id, task in pending.items()
                if all(dep in done_ids for dep in task.dependencies)
            )
            if not ready_ids:
                # 存在死锁/无效依赖：标记阻塞并终止，避免死循环
                for task_id, task in pending.items():
                    logger.error("子任务 %d 依赖无法满足（死锁），标记为阻塞", task_id)
                    blocked = self._make_result(task, "blocked", error="依赖无法满足，任务被阻塞")
                    results.append(blocked)
                    result_by_id[task_id] = blocked
                    if steps is not None:
                        self._record(
                            steps,
                            _BLOCKED,
                            agent=getattr(self.agents.get(task.agent_name), "name", task.agent_name),
                            task_id=task_id,
                            detail="依赖无法满足",
                        )
                break

            ready_tasks = [pending[tid] for tid in ready_ids]
            if self.enable_parallel and len(ready_tasks) > 1:
                batch_results = self._run_batch_parallel(ready_tasks, steps=steps)
            else:
                batch_results = [self._run_subtask(task, steps=steps) for task in ready_tasks]

            for record in batch_results:
                task_id = record["task_id"]
                results.append(record)
                done_ids.add(task_id)
                result_by_id[task_id] = record
                pending.pop(task_id, None)

        return results

    def _run_batch_parallel(
        self, tasks: List[SubTask], steps: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """使用线程池并行执行一批无依赖子任务，并为每个任务设置超时。"""
        records: List[Dict[str, Any]] = []
        if not tasks:
            return records
        executor = ThreadPoolExecutor(
            max_workers=min(len(tasks), 8), thread_name_prefix="supervisor-subtask"
        )
        try:
            future_map = {
                executor.submit(self._run_subtask, task, steps=steps): task
                for task in tasks
            }
            for future, task in future_map.items():
                try:
                    records.append(future.result(timeout=self.subtask_timeout))
                except Exception as exc:  # noqa: BLE001 - 超时/未知异常统一标记失败
                    timeout = isinstance(exc, TimeoutError)
                    detail = f"执行超时（>{self.subtask_timeout}s）" if timeout else f"{exc}"
                    logger.error("子任务 %d 并行执行失败: %s", task.task_id, detail)
                    records.append(self._make_result(task, "timeout" if timeout else "failed", error=detail))
        finally:
            # 不等待仍在运行的任务（超时任务标记失败即可），避免挂死主流程
            executor.shutdown(wait=False, cancel_futures=True)
        return records

    def _run_subtask(
        self, task: SubTask, steps: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """执行单个子任务：查找智能体 -> 调用 run() -> 记录耗时。"""
        start = time.perf_counter()
        agent = self.agents.get(task.agent_name)
        agent_label = getattr(agent, "name", None) or task.agent_name
        if steps is not None:
            self._record(
                steps,
                _EXECUTE,
                agent=agent_label,
                task_id=task.task_id,
                detail=task.description,
            )

        if agent is None or not hasattr(agent, "run"):
            logger.warning(
                "智能体 %s 不可用，子任务 %d 将被跳过", task.agent_name, task.task_id
            )
            record = self._make_result(
                task,
                "unavailable",
                error=f"智能体 {task.agent_name!r} 未注册，任务被跳过",
            )
        else:
            try:
                raw_output = agent.run(task.description)
                output = self._extract_output(raw_output)
                record = self._make_result(task, "success", output=output)
            except Exception as exc:  # noqa: BLE001 - 单个失败不阻塞整体
                logger.error("子任务 %d（%s）执行失败: %s", task.task_id, task.agent_name, exc)
                record = self._make_result(task, "failed", error=str(exc))

        record["elapsed_ms"] = round((time.perf_counter() - start) * 1000, 1)
        return record

    @staticmethod
    def _make_result(
        task: SubTask, status: str, output: str = "", error: str = ""
    ) -> Dict[str, Any]:
        """构造单个子任务的统一结果记录。"""
        return {
            "task_id": task.task_id,
            "agent_name": task.agent_name,
            "description": task.description,
            "dependencies": list(task.dependencies),
            "status": status,
            "output": output or "",
            "error": error or "",
            "elapsed_ms": 0.0,
        }

    @staticmethod
    def _extract_output(raw: Any) -> str:
        """把子智能体 run() 返回值归一化为文本。"""
        if isinstance(raw, dict):
            text = raw.get("output", "")
            return str(text).strip() if text else str(raw)
        return str(raw).strip()

    # ------------------------------------------------------------------ #
    # ③ 结果整合
    # ------------------------------------------------------------------ #
    def _synthesize_results(self, results: List[Dict[str, Any]], original_query: str) -> str:
        """使用 LLM 将各子任务结果整合为连贯的完整回答；失败则拼接兜底。"""
        succeeded = [r for r in results if r.get("status") == "success"]
        if not succeeded:
            return "（所有子任务均未成功执行，无法整合有效结果。）"

        blocks: List[str] = []
        for record in results:
            if record.get("status") == "success":
                body = record.get("output", "").strip() or "（无输出）"
            else:
                reason = record.get("error") or record.get("status", "unknown")
                body = f"（子任务失败：{reason}）"
            blocks.append(
                f"[子任务 {record['task_id']} · {record.get('agent_name')} · "
                f"{record.get('status')}]\n{body}"
            )
        context = "\n\n".join(blocks)

        prompt = (
            "你是监督者智能体，请将以下各专业智能体的执行结果整合为一份"
            "连贯、完整、面向用户的最终回答。\n\n"
            f"用户问题：{original_query}\n\n"
            f"子任务结果：\n{context}\n\n"
            "要求：回答结构清晰；标注关键结论来自哪个子任务；"
            "若有子任务失败，简要说明原因。"
        )
        try:
            response = self.llm.invoke(prompt)
            answer = _text_of(response).strip()
            if answer:
                return answer
        except Exception as exc:  # noqa: BLE001 - 整合失败使用拼接兜底
            logger.error("结果整合失败，使用拼接结果: %s", exc)
        return context

    def _fallback_direct(self, query: str) -> str:
        """规划彻底失败后的兜底：监督者直接回答。"""
        logger.warning("监督者对任务「%s」启用直接回答降级", query)
        try:
            return self._direct_answer(query)
        except Exception as exc:  # noqa: BLE001
            logger.error("监督者直接回答失败: %s", exc)
            return "（抱歉，任务规划与执行均失败，请尝试简化问题后重试。）"

    # ------------------------------------------------------------------ #
    # 步骤记录 / 可观测性
    # ------------------------------------------------------------------ #
    @staticmethod
    def _record(
        steps: List[Dict[str, Any]],
        kind: str,
        agent: Optional[str] = None,
        task_id: Optional[int] = None,
        detail: str = "",
    ) -> None:
        """追加一条执行步骤记录（含时间戳）。"""
        steps.append(
            {
                "step": kind,
                "agent": agent,
                "task_id": task_id,
                "detail": detail,
                "time": datetime.now().isoformat(timespec="seconds"),
            }
        )

    @staticmethod
    def _build_trace(steps: List[Dict[str, Any]]) -> str:
        """将结构化步骤渲染为人类可读的执行链路文本。"""
        labels: Dict[str, str] = {
            _PLAN: "[规划]",
            _EXECUTE: "[执行]",
            _SYNTHESIZE: "[整合]",
            _DEGRADE: "[降级直答]",
            _BLOCKED: "[阻塞]",
        }
        tokens: List[str] = []
        for step in steps:
            label = labels.get(step.get("step"), step.get("step", "?"))
            if step.get("step") == _EXECUTE:
                agent = step.get("agent") or "?"
                tokens.append(f"{label}：{agent}（任务 {step.get('task_id')}）")
            else:
                tokens.append(label)
        return " → ".join(tokens) if tokens else "（空执行链路）"

    # ------------------------------------------------------------------ #
    # LangGraph 集成
    # ------------------------------------------------------------------ #
    def to_langgraph_state(self) -> Dict[str, Any]:
        """将当前（最近一次）执行状态转换为 LangGraph State 形态。

        Returns:
            包含 ``messages / plan / results / current_step / next_agent``
            的字典，便于后续与 LangGraph 工作流对接。
        """
        state = self._last_state
        query = state.get("query")
        output = state.get("output")
        messages: List[Dict[str, str]] = []
        if query:
            messages.append({"role": "user", "content": query})
        if output:
            messages.append({"role": "assistant", "content": output})
        return {
            "messages": messages,
            "plan": state.get("plan", []),
            "results": state.get("results", []),
            "current_step": len(state.get("results", [])),
            "next_agent": None,
        }
