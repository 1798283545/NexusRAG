"""关键点提取工具。

检索知识库内容，用 LLM 提炼「要点 / 核心观点」列表，供 SummarizerAgent 使用。
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

from tools._utils import generate_text, retrieve_docs


class _KeyPointArgs(BaseModel):
    query: str = Field(description="指定要提炼要点的主题 / 文档方向")
    max_points: int = Field(default=5, description="要点数量上限")


class KeyPointExtractionTool(BaseTool):
    """从知识库相关文档中提取关键要点列表。"""

    name: str = "keypoint_extractor"
    description: str = (
        "检索知识库相关文档，提炼出 N 条关键要点 / 核心观点。"
        "适用：长文档要点抽取、会议纪要提炼。"
    )
    args_schema: type[BaseModel] = _KeyPointArgs
    rag_chain: Any = None
    llm: Optional[BaseLanguageModel] = None

    def _run(self, query: str, max_points: int = 5, **kwargs) -> str:
        if self.llm is None:
            return "（缺少 LLM 实例，无法提取要点）"
        scored = retrieve_docs(self.rag_chain, query, k=8)
        if not scored:
            return "（未检索到相关文档内容）"
        context = "\n\n".join(doc.page_content for doc, _ in scored)[:12000]
        user = (
            f"请从以下知识库内容中提炼不超过 {int(max_points)} 条关键要点，"
            "每条用「•」开头并保留关键信息：\n\n" + context
        )
        return generate_text(
            self.llm, "你是信息提炼专家，要点需准确、精炼、不重复。", user
        )

    async def _arun(self, query: str, max_points: int = 5, **kwargs) -> str:  # pragma: no cover
        return self._run(query, max_points=max_points, **kwargs)

    def __init__(
        self,
        rag_chain: Any,
        llm: Optional[BaseLanguageModel] = None,
        **kwargs,
    ) -> None:
        kwargs["rag_chain"] = rag_chain
        kwargs["llm"] = llm
        super().__init__(**kwargs)
