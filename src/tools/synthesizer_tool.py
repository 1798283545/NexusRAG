"""信息整合工具。

使用 LLM 将多个信息源片段整合为结构完整、带来源标注的回答，
供 WebSearchAgent 作为收尾步骤使用。
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

from tools._utils import generate_text


class _SynthesizeArgs(BaseModel):
    query: str = Field(description="需要整合回答的问题")
    snippets: List[str] = Field(description="多个信息源片段列表（每段注明来源）")


class InformationSynthesizerTool(BaseTool):
    """基于多个信息源片段整合出一篇完整、带来源的回答。"""

    name: str = "information_synthesizer"
    description: str = (
        "整合多条网络信息片段，形成结构完整、逻辑清晰、注明来源的最终回答。"
        "适合在收集完所有搜索资料后作为收尾调用。"
    )
    args_schema: type[BaseModel] = _SynthesizeArgs
    llm: Optional[BaseLanguageModel] = None

    def _run(self, query: str, snippets: List[str], **kwargs) -> str:
        if self.llm is None:
            return "（缺少 LLM 实例，无法进行信息整合）"
        numbered = "\n\n".join(
            f"[来源 {index}] {item}" for index, item in enumerate(snippets, start=1)
        )
        user = (
            f"问题：{query}\n\n以下为搜索到的多条信息片段：\n{numbered}\n\n"
            "请整合这些信息形成完整回答；回答需标注各结论对应的来源编号。"
        )
        return generate_text(
            self.llm,
            "你是信息整合专家，回答需覆盖要点、消除矛盾、明确标注来源。",
            user,
        )

    async def _arun(self, query: str, snippets: List[str], **kwargs) -> str:  # pragma: no cover
        return self._run(query, snippets, **kwargs)

    def __init__(self, llm: BaseLanguageModel, **kwargs) -> None:
        kwargs["llm"] = llm
        super().__init__(**kwargs)
