"""文档摘要工具。

基于知识库检索相关文档片段，使用 LLM 生成受控长度的结构化摘要，
供 SummarizerAgent 使用。
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

from tools._utils import generate_text, retrieve_docs


class _SummaryArgs(BaseModel):
    query: str = Field(description="指定要总结的主题 / 文档方向的关键词或问题")
    max_length: int = Field(default=500, description="摘要最大字符数")


class DocumentSummaryTool(BaseTool):
    """检索知识库内容并用 LLM 生成长度受限的文档摘要。"""

    name: str = "document_summarizer"
    description: str = (
        "检索知识库中与主题相关的文档片段，并生成简洁中文摘要。"
        "适用：长文档总结、报告提炼。"
    )
    args_schema: type[BaseModel] = _SummaryArgs
    rag_chain: Any = None
    llm: Optional[BaseLanguageModel] = None

    def _run(self, query: str, max_length: int = 500, **kwargs) -> str:
        if self.llm is None:
            return "（缺少 LLM 实例，无法生成摘要）"
        scored = retrieve_docs(self.rag_chain, query, k=8)
        if not scored:
            return "（未检索到相关文档内容）"
        context = "\n\n".join(doc.page_content for doc, _ in scored)
        if len(context) > 12000:
            context = context[:12000] + "……（内容过长已截断）"
        user = (
            f"请基于以下知识库内容生成中文摘要，控制在 {int(max_length)} 字符以内，"
            "突出核心观点、结论与关键数据：\n\n" + context
        )
        summary = generate_text(
            self.llm, "你是文档摘要专家，输出简洁、结构化的摘要。", user
        )
        sources = ", ".join(
            {str(doc.metadata.get("file_name") or doc.metadata.get("source", "未知来源")) for doc, _ in scored}
        )
        return f"【摘要】{summary}\n【相关来源】{sources}"

    async def _arun(self, query: str, max_length: int = 500, **kwargs) -> str:  # pragma: no cover
        return self._run(query, max_length=max_length, **kwargs)

    def __init__(
        self,
        rag_chain: Any,
        max_length: int = 500,
        llm: Optional[BaseLanguageModel] = None,
        **kwargs,
    ) -> None:
        kwargs["rag_chain"] = rag_chain
        kwargs["llm"] = llm
        super().__init__(**kwargs)
