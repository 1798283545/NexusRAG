#!/usr/bin/env python
"""健康检查脚本 - 验证所有核心模块是否正常工作。

用法（在项目根目录执行）::

    python scripts/check_health.py
"""
from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

# 将项目 src 目录加入模块搜索路径（无需先执行 pip install）
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

logging.basicConfig(level=logging.INFO)


def check_module(module_name: str, test_func) -> bool:
    """执行单个模块的检查函数并输出结果。"""
    try:
        test_func()
        print(f"✅ {module_name} 正常")
        return True
    except Exception as exc:  # noqa: BLE001 - 健康检查需汇总全部异常
        print(f"❌ {module_name} 异常: {exc}")
        return False


def main() -> int:
    print("🔍 NexusRAG 健康检查开始...\n")
    all_ok = True

    # 1. 检查 config
    def test_config() -> None:
        from config import settings

        assert settings.OPENAI_API_KEY is not None

    all_ok &= check_module("config", test_config)

    # 2. 检查 loaders（get_loader 要求文件真实存在，故使用临时文件）
    def test_loaders() -> None:
        from loaders import LoaderFactory

        with tempfile.TemporaryDirectory() as tmp_dir:
            sample = Path(tmp_dir) / "sample.txt"
            sample.write_text("健康检查内容", encoding="utf-8")
            docs = LoaderFactory.get_loader(str(sample)).load()
            assert len(docs) == 1

    all_ok &= check_module("loaders", test_loaders)

    # 3. 检查 splitters
    def test_splitters() -> None:
        from langchain_core.documents import Document

        from loaders import split_documents

        chunks = split_documents(
            [Document(page_content="测试文本", metadata={"file_type": "txt"})],
            strategy="recursive",
        )
        assert len(chunks) >= 1

    all_ok &= check_module("splitters", test_splitters)

    # 4. 检查 vector_store
    def test_vector_store() -> None:
        from retrievers import VectorStoreManager

        vsm = VectorStoreManager(
            collection_name="health_check",
            persist_directory=str(Path(tempfile.mkdtemp())),
        )
        vsm.get_collection_stats()
        vsm.clear_collection()

    all_ok &= check_module("vector_store", test_vector_store)

    # 5. 检查 rag_chain（构造链本身不发起网络调用）
    def test_rag_chain() -> None:
        from chains import RAGChain
        from retrievers import VectorStoreManager

        vsm = VectorStoreManager(
            collection_name="health_check",
            persist_directory=str(Path(tempfile.mkdtemp())),
        )
        rag = RAGChain(vsm)
        assert rag.k >= 1

    all_ok &= check_module("rag_chain", test_rag_chain)

    print("\n" + "=" * 40)
    if all_ok:
        print("🎉 所有模块健康检查通过！")
        return 0
    print("⚠️ 部分模块异常，请检查日志。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
