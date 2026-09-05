"""工具模块内部共享的小工具（不对外导出）。

提供：LLM 输出归一化、LLM 文本生成、从 RAG 链检索文档等公共能力，
供各专业工具复用，避免重复实现。
"""

from __future__ import annotations

import logging
from typing import Any, List, Tuple

from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


def as_text(response: Any) -> str:
    """将 LLM 返回（字符串 / AIMessage / 多模态块列表）归一化为纯文本。"""
    if isinstance(response, str):
        return response
    content = getattr(response, "content", response)
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
        return "".join(parts)
    return str(content)


def generate_text(llm: BaseLanguageModel, system: str, user: str) -> str:
    """以「系统 + 用户」消息调用 LLM，返回纯文本。"""
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return as_text(response).strip()


def retrieve_docs(
    rag_chain: Any, query: str, k: int = 4
) -> List[Tuple[Document, float]]:
    """从 RAG 链底层的向量库检索带分数文档。

    Args:
        rag_chain: RAGChain（或其向量库封装）实例。
        query: 检索查询。
        k: 返回条数。

    Returns:
        ``(Document, similarity)`` 列表，相似度降序。
    """
    store = rag_chain.vector_store_manager
    return store.similarity_search_with_score(query, k=k)


def format_sources(
    scored: List[Tuple[Document, float]], limit: int = 20
) -> str:
    """将检索结果格式化为供 LLM 阅读的编号文本块。"""
    blocks: List[str] = []
    for index, (doc, score) in enumerate(scored[:limit], start=1):
        source = doc.metadata.get("source", "") or doc.metadata.get("file_name", "")
        blocks.append(f"[{index}] 来源:{source}（分数:{score:.3f}）\n{doc.page_content}")
    return "\n\n".join(blocks)
