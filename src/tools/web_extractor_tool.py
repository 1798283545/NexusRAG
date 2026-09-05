"""网页正文提取工具。

使用 requests + BeautifulSoup 抓取并提取网页正文文本（截断到固定长度），
供 WebSearchAgent 深入阅读链接内容。
"""

from __future__ import annotations

import logging
import re

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


class _ExtractArgs(BaseModel):
    url: str = Field(description="需要提取正文的网页 URL")
    max_chars: int = Field(default=3000, description="返回正文的最大字符数")


class WebContentExtractorTool(BaseTool):
    """抓取网页并提取正文纯文本（截断返回）。"""

    name: str = "web_content_extractor"
    description: str = (
        "输入一个网页 URL，返回该网页的正文纯文本内容。"
        "适用：深入阅读搜索结果链接、抓取文章全文。"
    )
    args_schema: type[BaseModel] = _ExtractArgs
    timeout: int = 10

    def _run(self, url: str, max_chars: int = 3000, **kwargs) -> str:
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            return "（缺少依赖 requests / beautifulsoup4，请执行 pip install requests beautifulsoup4 lxml）"
        try:
            response = requests.get(url, timeout=self.timeout, headers={"User-Agent": "Mozilla/5.0 NexusRAG"})
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = re.sub(r"\s+", " ", soup.get_text(separator=" ")).strip()
        except Exception as exc:  # noqa: BLE001 - 网络 / 解析异常以文本回传
            logger.warning("网页提取失败: %s（%s）", url, exc)
            return f"（网页提取失败：{exc}）"
        if not text:
            return "（未能从该页面提取到正文）"
        return text[: int(max_chars)]

    async def _arun(self, url: str, max_chars: int = 3000, **kwargs) -> str:  # pragma: no cover
        return self._run(url, max_chars=max_chars, **kwargs)
