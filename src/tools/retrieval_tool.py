"""知识库检索工具。

从 RAG 链底层向量库检索与查询最相关的文档片段（不做生成），
供 RAGAgent 使用。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from tools._utils import format_sources, retrieve_docs


class _RetrievalArgs(BaseModel):
    query: str = Field(description="检索查询语句")
    k: int = Field(default=4, description="返回的文档片段数量")


class RetrievalTool(BaseTool):
    """从知识库检索相关文档片段并返回（含来源与相似度）。"""

    name: str = "retriever"
    description: str = (
        "从企业知识库中检索与查询最相关的文档片段，返回带来源编号的文本。"
        "当只需要定位原文而不需要生成回答时使用。"
    )
    args_schema: type[BaseModel] = _RetrievalArgs
    rag_chain: Any = None

    def _run(self, query: str, k: int = 4, **kwargs) -> str:
        scored = retrieve_docs(self.rag_chain, query, k=max(1, int(k)))
        if not scored:
            return "（未检索到相关文档内容）"
        return format_sources(scored, limit=max(1, int(k)))

    async def _arun(self, query: str, k: int = 4, **kwargs) -> str:  # pragma: no cover
        return self._run(query, k=k, **kwargs)

    def __init__(self, rag_chain: Any, **kwargs) -> None:
        kwargs["rag_chain"] = rag_chain
        super().__init__(**kwargs)
