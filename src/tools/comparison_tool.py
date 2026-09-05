"""文档对比工具。

分别检索两份主题相关的文档内容，用 LLM 对比异同，供 SummarizerAgent 使用。
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

from tools._utils import generate_text, retrieve_docs


class _CompareArgs(BaseModel):
    query_a: str = Field(description="文档 A 的检索主题（用于定位第一份文档）")
    query_b: str = Field(description="文档 B 的检索主题（用于定位第二份文档）")


class DocumentComparisonTool(BaseTool):
    """检索两份主题文档并对比其异同。"""

    name: str = "document_comparer"
    description: str = (
        "对比两份文档 / 两种方案的内容异同：先按 query_a、query_b 从知识库"
        "检索对应内容，再输出差异与相同点列表。"
    )
    args_schema: type[BaseModel] = _CompareArgs
    rag_chain: Any = None
    llm: Optional[BaseLanguageModel] = None

    def _fetch(self, query: str) -> str:
        scored = retrieve_docs(self.rag_chain, query, k=3)
        return "\n\n".join(doc.page_content for doc, _ in scored)[:6000] or "（无内容）"

    def _run(self, query_a: str, query_b: str, **kwargs) -> str:
        if self.llm is None:
            return "（缺少 LLM 实例，无法进行对比）"
        content_a = self._fetch(query_a)
        content_b = self._fetch(query_b)
        user = (
            f"请对比以下两份文档的异同，输出「相同点」与「差异点」两个列表：\n\n"
            f"【文档 A】\n{content_a}\n\n【文档 B】\n{content_b}"
        )
        return generate_text(
            self.llm, "你是文档对比专家，输出需层次分明、要点清晰。", user
        )

    async def _arun(self, query_a: str, query_b: str, **kwargs) -> str:  # pragma: no cover
        return self._run(query_a, query_b, **kwargs)

    def __init__(
        self,
        rag_chain: Any,
        llm: Optional[BaseLanguageModel] = None,
        **kwargs,
    ) -> None:
        kwargs["rag_chain"] = rag_chain
        kwargs["llm"] = llm
        super().__init__(**kwargs)
