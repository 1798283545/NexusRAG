#!/usr/bin/env python
"""演示 LangGraph 多智能体协作工作流。

以有状态状态图编排 Supervisor 与专业智能体，展示：
- Supervisor 规划 → 条件路由 → 子任务执行 → 回归 Supervisor 的循环编排；
- 高风险子任务在 ``human_review`` 节点暂停（Human-in-the-Loop）；
- 模拟人工批准后继续执行，最终整合生成报告。

用法（在项目根目录执行）::

    python scripts/demo_workflow.py
    python scripts/demo_workflow.py "分析知识库中的销售数据，并搜索最新行业趋势，最后生成报告" --no-auto-approve

说明：需要先在 .env 中配置 OPENAI_API_KEY；未连接 ChromaDB / 未配置知识库时，
RAGAgent 与 SummarizerAgent 会被自动跳过。
"""
from __future__ import annotations

import argparse
import logging
import sys
import uuid
from pathlib import Path

# 将项目 src 目录加入模块搜索路径（无需先执行 pip install）
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def build_llm() -> object:
    """构建对话 LLM（基于 .env 的 MODEL_NAME）。"""
    from langchain_openai import ChatOpenAI

    from config import settings

    return ChatOpenAI(
        model_name=settings.MODEL_NAME,
        temperature=settings.TEMPERATURE,
        max_tokens=settings.MAX_TOKENS,
    )


def build_rag_chain(llm: object) -> object | None:
    """尝试连接 ChromaDB 构建 RAG 链；失败返回 None（跳过 RAG 系智能体）。"""
    try:
        from chains import RAGChain
        from retrievers import VectorStoreManager

        store = VectorStoreManager()
        return RAGChain(vector_store_manager=store, llm=llm)
    except Exception as exc:  # noqa: BLE001 - 知识库不可用时优雅降级
        logging.getLogger(__name__).warning(
            "无法连接知识库（%s），将跳过 RAGAgent / SummarizerAgent", exc
        )
        return None


def print_event(event: dict) -> None:
    """按节点名打印流式事件（用于直观展示执行链路）。"""
    for node, payload in event.items():
        if node == "__interrupt__":
            pending = payload[0].get("action", {}) if payload else {}
            print(f"    ✋ 暂停审核：{pending.get('agent_name')} → {pending.get('description')}")
        elif node == "supervisor":
            plan = (payload or {}).get("plan")
            if plan:
                print(f"    🤔 Supervisor 规划出 {len(plan)} 个子任务")
        elif node == "rag_agent":
            print("    📚 RAG Agent 执行中...")
        elif node == "code_agent":
            print("    💻 Code Agent 执行中...")
        elif node == "web_agent":
            print("    🌐 Web Agent 执行中...")
        elif node == "summarizer_agent":
            print("    🗂️  Summarizer Agent 执行中...")
        elif node == "human_review":
            print("    👤 人工审核中...")
        elif node == "synthesizer":
            print("    📝 正在整合最终回答...")


def main() -> int:
    from config import settings
    from workflows import MultiAgentWorkflow

    print("🚀 NexusRAG LangGraph 多智能体工作流演示开始...")

    if not settings.OPENAI_API_KEY:
        print("❌ 未在 .env 中配置 OPENAI_API_KEY，无法调用 LLM 完成规划与生成。")
        print("   请执行: cp .env.example .env，然后填入你的 API Key 后重试。")
        return 1

    parser = argparse.ArgumentParser(description="LangGraph 多智能体工作流演示")
    default_query = (
        "分析知识库中关于 NexusRAG 销售数据的资料，"
        "并搜索 2026 年 RAG 技术的最新进展，最后生成一份综合报告"
    )
    parser.add_argument("query", nargs="?", default=default_query, help="复杂任务描述")
    parser.add_argument(
        "--no-auto-approve",
        action="store_true",
        help="关闭自动批准，改为手动输入人工反馈",
    )
    args = parser.parse_args()

    llm = build_llm()
    rag_chain = build_rag_chain(llm)

    # 1. 初始化所有智能体（含 Supervisor）
    from agents import AgentFactory

    bundle = AgentFactory.create_all_with_supervisor(
        llm=llm,
        rag_chain=rag_chain,
        workspace_dir=settings.AGENT_WORKSPACE_DIR,
    )
    supervisor = bundle["supervisor"]
    specialists = {key: agent for key, agent in bundle.items() if key != "supervisor"}
    print(f"已注册专业智能体：{', '.join(sorted(specialists))}")
    print("=" * 70)

    # 2. 构建工作流
    workflow = MultiAgentWorkflow(
        supervisor=supervisor,
        agents=specialists,
        enable_hitl=settings.WORKFLOW_ENABLE_HITL,
        max_iterations=settings.WORKFLOW_MAX_ITERATIONS,
    )
    thread = f"demo-{uuid.uuid4().hex[:8]}"
    print(f"\n用户请求：{args.query}\n" + "-" * 70)
    print(f"[会话线程] {thread}")

    # 3. 流式执行，逐步展示节点运行过程（遇到审核时自动暂停）
    print("[执行链路]")
    paused = False
    for event in workflow.stream(args.query, thread_id=thread):
        print_event(event)
        if "__interrupt__" in event:
            paused = True

    # 4. Human-in-the-Loop：处理可能出现的全部审核点
    if paused:
        print("-" * 70)
    while paused:
        state = workflow.get_state(thread)
        pending = state.get("pending_action")
        if not pending:
            break
        task_desc = pending.get("description", "")
        if args.no_auto_approve:
            feedback = input(
                f"请审核（approve / reject / 修改说明）「{task_desc}」: "
            ).strip()
        else:
            feedback = "approve"
            print(f"✋ 自动批准高风险操作：「{task_desc}」")
        workflow.resume(thread, feedback)
        # resume 可能停在下一处审核点，继续循环直至无待审核操作

    # 5. 读取最终结果
    state = workflow.get_state(thread)
    result = state
    print("\n[子任务明细]")
    for record in result.get("results", []):
        mark = "✅" if record.get("status") == "success" else "⚠️"
        print(
            f"  {mark} 任务 {record['task_id']} → {record['agent_name']} "
            f"[{record.get('status')}]（{record.get('elapsed_ms', 0):.0f} ms）"
        )

    print("\n[最终回答]")
    print(result.get("final_answer", "（无输出）"))
    print("=" * 70)
    print(f"✨ 演示结束（总耗时见上方子任务明细）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
