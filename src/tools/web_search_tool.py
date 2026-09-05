"""联网搜索工具。

使用 duckduckgo-search（可选）进行实时搜索，返回「标题 + 摘要 + URL」列表。
支持 ``search_api`` 配置（当前内置 duckduckgo；google/bing 预留，可自行扩展）。
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool


class _SearchArgs(BaseModel):
    query: str = Field(description="搜索关键词或问题")


class WebSearchTool(BaseTool):
    """进行联网搜索，返回带标题 / 摘要 / URL 的结果列表。"""

    name: str = "web_search"
    description: str = (
        "联网搜索最新信息，返回若干条结果（标题 + 摘要 + URL）。"
        "适用：实时信息查询、新闻获取、外部信息验证。"
    )
    args_schema: type[BaseModel] = _SearchArgs
    search_api: str = "duckduckgo"
    max_results: int = 5

    def _run(self, query: str, **kwargs) -> str:
        if self.search_api.lower() not in ("duckduckgo", "ddgs"):
            return (
                f"（暂不支持搜索引擎 {self.search_api!r}，当前仅内置 duckduckgo）"
            )
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return "（缺少依赖 duckduckgo-search，请执行 pip install duckduckgo-search）"
        try:
            with DDGS() as client:
                results: List[dict] = list(client.text(query, max_results=self.max_results))
        except Exception as exc:  # noqa: BLE001 - 网络异常以文本回传
            return f"（联网搜索失败：{exc}）"
        if not results:
            return "（未搜索到结果）"
        lines: List[str] = []
        for index, item in enumerate(results[: self.max_results], start=1):
            title = item.get("title", "")
            body = item.get("body", "")
            href = item.get("href", item.get("url", ""))
            lines.append(f"{index}. {title}\n   摘要：{body}\n   URL：{href}")
        return "\n".join(lines)

    async def _arun(self, query: str, **kwargs) -> str:  # pragma: no cover
        return self._run(query, **kwargs)

    def __init__(self, search_api: str = "duckduckgo", max_results: int = 5, **kwargs) -> None:
        kwargs["search_api"] = search_api
        kwargs["max_results"] = int(max_results)
        super().__init__(**kwargs)
