"""文档摘要专业智能体：负责文档摘要与要点提炼。"""

from __future__ import annotations

from typing import Any, List, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

from agents.base_agent import BaseAgent
from tools.comparison_tool import DocumentComparisonTool
from tools.keypoint_tool import KeyPointExtractionTool
from tools.summary_tool import DocumentSummaryTool


class SummarizerAgent(BaseAgent):
    """长文档摘要 / 要点提取 / 多文档对比智能体。"""

    def __init__(
        self,
        llm: BaseLanguageModel,
        rag_chain: Any,
        max_length: int = 500,
        **kwargs: Any,
    ) -> None:
        """初始化 SummarizerAgent。

        Args:
            llm: 大语言模型实例（同时用于摘要工具生成）。
            rag_chain: 高级 RAG 链（RAGChain），提供检索相关文档能力。
            max_length: 摘要最大字符数。
            **kwargs: 透传给 BaseAgent 的其余参数。
        """
        self.rag_chain: Any = rag_chain
        self.max_length: int = max(50, int(max_length))
        super().__init__(
            name="SummarizerAgent",
            description=(
                "负责对文档进行结构化摘要和要点提取。"
                "适用场景：长文档总结、报告提取、关键信息提炼。"
            ),
            llm=llm,
            **kwargs,
        )

    def _get_default_tools(self) -> List[BaseTool]:
        """返回默认工具：摘要 / 要点提取 / 文档对比。"""
        return [
            DocumentSummaryTool(rag_chain=self.rag_chain, max_length=self.max_length, llm=self.llm),
            KeyPointExtractionTool(rag_chain=self.rag_chain, llm=self.llm),
            DocumentComparisonTool(rag_chain=self.rag_chain, llm=self.llm),
        ]

    def _get_system_prompt(self) -> str:
        return """你是文档摘要专业智能体，专门负责提炼文档核心内容。

你的工作流程：
1. 使用 DocumentSummaryTool 生成文档摘要
2. 使用 KeyPointExtractionTool 提取关键要点
3. 使用 DocumentComparisonTool 对比多份文档
4. 输出格式要求：结构清晰、层次分明、突出重点"""
