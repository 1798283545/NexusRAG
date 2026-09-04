"""NexusRAG 链式编排模块。

对外暴露最简 RAG 链（RAGChain）及其 Prompt 模板常量。
"""

from .rag_chain import QA_PROMPT_TEMPLATE, RAGChain, SUMMARY_PROMPT_TEMPLATE

__all__ = [
    "RAGChain",
    "QA_PROMPT_TEMPLATE",
    "SUMMARY_PROMPT_TEMPLATE",
]
