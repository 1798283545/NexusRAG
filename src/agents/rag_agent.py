"""RAG 专业智能体：负责知识库检索与问答。"""

from __future__ import annotations

from typing import Any, List

from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

from agents.base_agent import BaseAgent
from tools.qa_tool import DocumentQATool
from tools.retrieval_tool import RetrievalTool


class RAGAgent(BaseAgent):
    """基于知识库的检索与问答智能体。

    提供两个工具：:class:`RetrievalTool`（只检索原文）与
    :class:`DocumentQATool`（检索 + 生成回答）。
    """

    def __init__(
        self,
        llm: BaseLanguageModel,
        rag_chain: Any,
        **kwargs: Any,
    ) -> None:
        """初始化 RAGAgent。

        Args:
            llm: 大语言模型实例。
            rag_chain: 高级 RAG 链（RAGChain），提供检索与问答能力。
            **kwargs: 透传给 BaseAgent 的其余参数（tools / verbose 等）。
        """
        self.rag_chain: Any = rag_chain
        super().__init__(
            name="RAGAgent",
            description=(
                "负责从知识库中检索文档并回答基于文档的问题。"
                "适用场景：文档问答、信息查询、知识检索。"
            ),
            llm=llm,
            **kwargs,
        )

    def _get_default_tools(self) -> List[BaseTool]:
        """返回默认工具：检索文档 + 基于文档问答。"""
        return [
            RetrievalTool(rag_chain=self.rag_chain),
            DocumentQATool(rag_chain=self.rag_chain),
        ]

    def _get_system_prompt(self) -> str:
        return """你是 RAG 专业智能体，专门负责从知识库中检索信息并回答用户问题。

你的工作流程：
1. 当用户需要基于文档回答问题时，使用 DocumentQATool
2. 当用户只需要找到相关文档内容时，使用 RetrievalTool
3. 始终在回答中引用文档来源（文件名、页码等）"""
