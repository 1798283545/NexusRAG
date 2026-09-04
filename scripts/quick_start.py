#!/usr/bin/env python
"""快速体验 NexusRAG 的完整流程。

用法（在项目根目录执行）::

    python scripts/quick_start.py

说明：需要先在 .env 中配置 OPENAI_API_KEY（默认 Embedding 与对话模型均走 OpenAI）。
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# 将项目 src 目录加入模块搜索路径（无需先执行 pip install）
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import settings
from chains import RAGChain
from retrievers import VectorStoreManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main() -> int:
    print("🚀 NexusRAG 快速体验开始...")

    if not settings.OPENAI_API_KEY:
        print("❌ 未在 .env 中配置 OPENAI_API_KEY，无法完成向量化与生成。")
        print("   请执行: cp .env.example .env，然后填入你的 API Key 后重试。")
        return 1

    # 1. 创建测试文档
    demo_file = _ROOT / "demo_document.txt"
    demo_file.write_text(
        "NexusRAG 是一个基于 LangChain 构建的多智能体文档协作平台。\n"
        "它支持 PDF、Word、Markdown 等多种格式的文档处理。\n"
        "核心能力包括：混合检索、多智能体协作、人机协同审核。",
        encoding="utf-8",
    )

    # 2. 初始化（本地持久化向量库）
    vsm = VectorStoreManager(
        collection_name="demo",
        persist_directory=str(_ROOT / "demo_chroma"),
    )
    rag = RAGChain(vsm, chunk_size=200, chunk_overlap=50)

    try:
        # 3. 处理文档（重复体验前先清空集合，保证幂等）
        print("📄 正在处理文档...")
        vsm.clear_collection()
        count = rag.process_document(str(demo_file))
        print(f"✅ 已索引 {count} 个文档块")

        # 4. 提问
        question = "NexusRAG 支持哪些文档格式？"
        print(f"\n💬 示例问题：{question}")
        result = rag.query(question)
        print(f"\n🤖 回答：{result['answer']}")

        # 5. 清理向量
        rag.delete_document(str(demo_file))
        print("\n✅ 体验完成！")
        return 0
    finally:
        # 无论成功或失败都移除临时演示文件
        demo_file.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
