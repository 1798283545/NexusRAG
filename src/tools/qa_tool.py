"""基于知识库的文档问答工具。

调用 RAG 链完成「检索 + 生成」，返回带来源溯源的回答文本，供 RAGAgent 使用。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool


class _QAArgs(BaseModel):
    query: str = Field(description="需要基于知识库回答的问题")
    k: int = Field(default=4, description="检索参考的文档片段数量")


class DocumentQATool(BaseTool):
    """基于知识库检索到的文档回答具体问题（附带 Sources 溯源）。"""

    name: str = "document_qa"
    description: str = (
        "基于企业知识库中的文档回答用户问题。回答会附带引用来源"
        "（文件名 / 页码）。当用户需要根据内部资料获取结论时使用。"
    )
    args_schema: type[BaseModel] = _QAArgs
    rag_chain: Any = None

    def _run(self, query: str, k: int = 4, **kwargs) -> str:
        try:
            result = self.rag_chain.query(query, k=max(1, int(k)))
        except Exception as exc:  # noqa: BLE001 - 以文本形式回传错误便于 Agent 决策
            return f"（文档问答失败：{exc}）"
        answer = result.get("answer", "")
        sources = result.get("source_documents", [])
        summary = (
            f"引用 {len(sources)} 个来源；置信度 {float(result.get('confidence', 0.0)):.3f}。"
            if sources
            else "未引用任何文档来源。"
        )
        return f"{answer}\n{summary}"

    async def _arun(self, query: str, k: int = 4, **kwargs) -> str:  # pragma: no cover
        return self._run(query, k=k, **kwargs)

    def __init__(self, rag_chain: Any, **kwargs) -> None:
        kwargs["rag_chain"] = rag_chain
        super().__init__(**kwargs)
