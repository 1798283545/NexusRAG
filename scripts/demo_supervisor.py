#!/usr/bin/env python
"""演示 Supervisor 的多智能体协作能力。

用法（在项目根目录执行）::

    python scripts/demo_supervisor.py
    python scripts/demo_supervisor.py "总结一下知识库中的竞品分析，并搜索最新的 AI 行业动态"

说明：需要先在 .env 中配置 OPENAI_API_KEY；未连接 ChromaDB / 未配置知识库时，
RAGAgent 与 SummarizerAgent 会被自动跳过，其余智能体仍可工作。
"""
from __future__ import annotations

import argparse
import logging
import sys
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


def main() -> int:
    from config import settings

    print("🚀 NexusRAG Supervisor 多智能体演示开始...")

    if not settings.OPENAI_API_KEY:
        print("❌ 未在 .env 中配置 OPENAI_API_KEY，无法调用 LLM 完成规划与生成。")
        print("   请执行: cp .env.example .env，然后填入你的 API Key 后重试。")
        return 1

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
    print(f"\n已注册专业智能体：{', '.join(sorted(bundle) - {'supervisor'})}")
    print("=" * 70)

    # 2. 输入一个复杂任务
    default_task = "总结知识库中关于「NexusRAG」的介绍，并搜索 2026 年 RAG 技术的最新进展。"
    parser = argparse.ArgumentParser(description="Supervisor 演示")
    parser.add_argument("task", nargs="?", default=default_task, help="需要执行的复杂任务")
    args = parser.parse_args()
    print(f"\n用户任务：{args.task}\n" + "-" * 70)

    # 3. 运行并打印执行链路
    result = supervisor.run(args.task)

    print("\n[执行链路]")
    print(result.get("execution_trace", "（无）"))
    print(f"（总耗时 {result.get('elapsed_ms', 0):.0f} ms）")

    print("\n[子任务明细]")
    for record in result.get("results", []):
        status = record.get("status")
        mark = "✅" if status == "success" else ("⚠️" if status in ("failed", "unavailable") else "🕐")
        print(
            f"  {mark} 任务 {record['task_id']} → {record['agent_name']} "
            f"[{status}]（{record.get('elapsed_ms', 0):.0f} ms）"
        )

    # 4. 输出最终回答
    print("\n[最终回答]")
    print(result.get("output", "（无输出）"))
    print("=" * 70)
    print("✨ 演示结束。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
