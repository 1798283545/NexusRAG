"""NexusRAG 文档加载与分块模块。

对外暴露统一的文档加载工厂（LoaderFactory）与文本分块入口（split_documents）。
"""

from .loader_factory import (
    BaseLoader,
    LoaderError,
    LoaderFactory,
    UnsupportedFormatError,
)
from .splitters import (
    MarkdownSplitter,
    RecursiveSplitter,
    SemanticSplitter,
    split_documents,
)

__all__ = [
    "BaseLoader",
    "LoaderError",
    "LoaderFactory",
    "UnsupportedFormatError",
    "RecursiveSplitter",
    "SemanticSplitter",
    "MarkdownSplitter",
    "split_documents",
]
