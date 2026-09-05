"""联网搜索专业智能体：负责联网搜索与信息整合。"""

from __future__ import annotations

from typing import Any, List

from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

from agents.base_agent import BaseAgent
from tools.synthesizer_tool import InformationSynthesizerTool
from tools.web_extractor_tool import WebContentExtractorTool
from tools.web_search_tool import WebSearchTool


class WebSearchAgent(BaseAgent):
    """实时联网搜索与多源信息整合智能体。"""

    def __init__(
        self,
        llm: BaseLanguageModel,
        search_api: str = "duckduckgo",
        max_results: int = 5,
        **kwargs: Any,
    ) -> None:
        """初始化 WebSearchAgent。

        Args:
            llm: 大语言模型实例。
            search_api: 搜索引擎（当前内置 ``duckduckgo``）。
            max_results: 每次搜索返回的结果条数。
            **kwargs: 透传给 BaseAgent 的其余参数。
        """
        self.search_api: str = search_api
        self.max_results: int = max(1, int(max_results))
        super().__init__(
            name="WebSearchAgent",
            description=(
                "负责联网搜索最新信息并整合汇总。"
                "适用场景：实时信息查询、新闻获取、外部信息验证。"
            ),
            llm=llm,
            **kwargs,
        )

    def _get_default_tools(self) -> List[BaseTool]:
        """返回默认工具：网络搜索 / 网页内容提取 / 信息整合。"""
        return [
            WebSearchTool(search_api=self.search_api, max_results=self.max_results),
            WebContentExtractorTool(),
            InformationSynthesizerTool(llm=self.llm),
        ]

    def _get_system_prompt(self) -> str:
        return """你是联网搜索专业智能体，专门负责获取和整合网络信息。

你的工作流程：
1. 当用户需要实时信息时，使用 WebSearchTool
2. 从搜索结果中提取关键信息，用 WebContentExtractorTool 获取详细内容
3. 最后用 InformationSynthesizerTool 整合成完整回答
4. 务必注明信息来源（URL 或网站名称）"""
