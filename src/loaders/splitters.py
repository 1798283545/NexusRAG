"""智能文本分块策略模块。

将加载器输出的长文档切割为适合向量检索的短文本块。
支持三种策略（recursive / semantic / markdown），统一入口见 :func:`split_documents`。

Requires: pip install langchain-experimental sentence-transformers
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
from typing import Any, Dict, List, Optional, Tuple

from langchain.text_splitter import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

#: 递归分割的默认分隔符（按优先级，兼顾中英文标点）
_RECURSIVE_SEPARATORS: List[str] = ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]


def _enrich_document(
    original: Document,
    pieces: List[Tuple[str, Dict[str, Any]]],
    strategy: str,
) -> List[Document]:
    """将原始文档切分得到的文本块封装为带增强元数据的 Document 列表。

    Args:
        original: 被切分的原始文档。
        pieces: ``(文本内容, 额外元数据)`` 二元组列表。
        strategy: 实际使用的分块策略名。

    Returns:
        分割后的 Document 列表，每个子块继承原始元数据并新增
        chunk_index / chunk_total / chunk_strategy / chunk_size /
        page_number / parent_source 字段。
    """
    total = len(pieces)
    chunks: List[Document] = []
    for index, (text, extra_metadata) in enumerate(pieces):
        content = text.strip()
        if not content:
            continue
        metadata: Dict[str, Any] = {**original.metadata, **(extra_metadata or {})}
        metadata.update(
            {
                "chunk_index": index,
                "chunk_total": total,
                "chunk_strategy": strategy,
                "chunk_size": len(content),
                "page_number": original.metadata.get("page_number"),
                "parent_source": original.metadata.get("source"),
            }
        )
        chunks.append(Document(page_content=content, metadata=metadata))
    return chunks


def _module_available(module_name: str) -> bool:
    """判断模块是否可导入。

    Args:
        module_name: 模块名（如 ``sentence_transformers``）。

    Returns:
        可导入返回 True，否则返回 False。
    """
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _semantic_dependencies_ready() -> bool:
    """semantic 策略所需的可选依赖是否齐全。"""
    return _module_available("sentence_transformers") and _module_available(
        "langchain_experimental"
    )


def _create_semantic_chunker(
    *,
    embedding_model: str,
    breakpoint_threshold_type: str,
    breakpoint_threshold: Optional[float],
    buffer_size: int,
    number_of_chunks: Optional[int],
) -> Any:
    """构建 SemanticChunker 实例（延迟导入可选依赖）。

    Args:
        embedding_model: HuggingFace 嵌入模型名。
        breakpoint_threshold_type: 断点阈值类型。
        breakpoint_threshold: 余弦相似度断点阈值（可为 None，按类型动态计算）。
        buffer_size: 缓冲区大小。
        number_of_chunks: 期望的分块数（可为 None）。

    Returns:
        SemanticChunker 实例。

    Raises:
        ImportError: langchain-experimental / sentence-transformers 未安装。
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_experimental.text_splitter import SemanticChunker

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    params: Dict[str, Any] = {
        "embeddings": embeddings,
        "buffer_size": buffer_size,
        "breakpoint_threshold_type": breakpoint_threshold_type,
        "number_of_chunks": number_of_chunks,
    }
    # 兼容 langchain_experimental 新旧版本参数命名
    if breakpoint_threshold is not None:
        threshold_key = (
            "breakpoint_threshold_amount"
            if "breakpoint_threshold_amount"
            in inspect.signature(SemanticChunker.__init__).parameters
            else "breakpoint_threshold"
        )
        params[threshold_key] = breakpoint_threshold
    logger.info(
        "创建 SemanticChunker（模型=%s，断点阈值类型=%s）",
        embedding_model,
        breakpoint_threshold_type,
    )
    return SemanticChunker(**params)


class RecursiveSplitter:
    """递归字符分割策略（通用场景，保底方案）。

    使用 :class:`langchain.text_splitter.RecursiveCharacterTextSplitter`，
    按分隔符优先级递归切分，直至块大小满足要求；分隔符序列同时覆盖中英文标点。
    """

    strategy: str = "recursive"

    def split(
        self,
        documents: List[Document],
        chunk_size: int,
        chunk_overlap: int,
        **kwargs: Any,
    ) -> List[Document]:
        """执行递归字符分割。

        Args:
            documents: 待切分的原始文档列表。
            chunk_size: 目标块大小（字符数）。
            chunk_overlap: 相邻块之间的重叠字符数。
            **kwargs: 透传给 RecursiveCharacterTextSplitter 的额外参数，
                仅支持 ``separators`` / ``length_function`` / ``keep_separator`` /
                ``strip_whitespace`` / ``add_start_index`` / ``is_separator_regex``。

        Returns:
            分割并增强元数据后的 Document 列表。
        """
        allowed_kwargs = {
            "separators",
            "length_function",
            "keep_separator",
            "strip_whitespace",
            "add_start_index",
            "is_separator_regex",
        }
        unknown = set(kwargs) - allowed_kwargs
        if unknown:
            logger.warning("recursive 策略忽略不支持的参数: %s", sorted(unknown))
        splitter_kwargs = {k: v for k, v in kwargs.items() if k in allowed_kwargs}
        separators = list(splitter_kwargs.pop("separators", _RECURSIVE_SEPARATORS))
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            **splitter_kwargs,
        )
        results: List[Document] = []
        for doc in documents:
            pieces = [(t, {}) for t in splitter.split_text(doc.page_content) if t.strip()]
            results.extend(_enrich_document(doc, pieces, self.strategy))
        logger.info(
            "recursive 分块完成: %d 个文档 → %d 个文本块（chunk_size=%d, overlap=%d）",
            len(documents),
            len(results),
            chunk_size,
            chunk_overlap,
        )
        return results


class SemanticSplitter:
    """语义嵌入分割策略（按主题段落切分）。

    使用 :class:`langchain_experimental.text_splitter.SemanticChunker`，配合
    HuggingFaceEmbeddings 计算句子向量，通过余弦相似度断点动态决定切分位置。
    """

    strategy: str = "semantic"

    def split(
        self,
        documents: List[Document],
        chunk_size: int,
        chunk_overlap: int,
        **kwargs: Any,
    ) -> List[Document]:
        """执行语义嵌入分割。

        Args:
            documents: 待切分的原始文档列表。
            chunk_size: 目标块大小（字符数，仅作为参考，不强制限制）。
            chunk_overlap: 相邻块之间的重叠字符数（语义切分中不使用）。
            **kwargs: 支持 ``embedding_model`` / ``breakpoint_threshold_type`` /
                ``breakpoint_threshold`` / ``buffer_size`` / ``number_of_chunks``。

        Returns:
            分割并增强元数据后的 Document 列表。
        """
        embedding_model = kwargs.get("embedding_model", "all-MiniLM-L6-v2")
        chunker = _create_semantic_chunker(
            embedding_model=embedding_model,
            breakpoint_threshold_type=kwargs.get("breakpoint_threshold_type", "percentile"),
            breakpoint_threshold=kwargs.get("breakpoint_threshold"),
            buffer_size=int(kwargs.get("buffer_size", 1)),
            number_of_chunks=kwargs.get("number_of_chunks"),
        )
        results: List[Document] = []
        for doc in documents:
            pieces = [(t, {}) for t in chunker.split_text(doc.page_content) if t.strip()]
            results.extend(_enrich_document(doc, pieces, self.strategy))
        logger.info(
            "semantic 分块完成: %d 个文档 → %d 个文本块（模型=%s）",
            len(documents),
            len(results),
            embedding_model,
        )
        return results


class MarkdownSplitter:
    """Markdown 结构分割策略（保留标题层级）。

    使用 :class:`langchain.text_splitter.MarkdownHeaderTextSplitter` 按
    H1~H6 标题切分，标题层级写入子块元数据；若某个标题下的内容超过
    ``chunk_size``，再用递归分割器做二次切分。非 Markdown 文档自动降级为
    recursive 策略。
    """

    strategy: str = "markdown"

    @staticmethod
    def _is_markdown(doc: Document) -> bool:
        """判断文档是否为 Markdown 类型。"""
        file_type = str(doc.metadata.get("file_type") or "").lower()
        source = str(doc.metadata.get("source") or "").lower()
        return file_type == "md" or source.endswith(".md")

    def split(
        self,
        documents: List[Document],
        chunk_size: int,
        chunk_overlap: int,
        **kwargs: Any,
    ) -> List[Document]:
        """执行 Markdown 结构分割。

        Args:
            documents: 待切分的原始文档列表。
            chunk_size: 单个标题下内容超过该值时的二次切分目标块大小。
            chunk_overlap: 二次切分的重叠字符数。
            **kwargs: 支持 ``strip_headers``（是否从正文移除标题行）。

        Returns:
            分割并增强元数据后的 Document 列表。
        """
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#" * i, f"H{i}") for i in range(1, 7)],
            strip_headers=kwargs.get("strip_headers", True),
        )
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=list(_RECURSIVE_SEPARATORS),
        )
        results: List[Document] = []
        for doc in documents:
            if not self._is_markdown(doc):
                logger.warning(
                    "文档 %s 不是 Markdown 文件（file_type=%s），markdown 策略降级为 recursive",
                    doc.metadata.get("source", "<unknown>"),
                    doc.metadata.get("file_type", "<unknown>"),
                )
                pieces = [
                    (t, {}) for t in recursive_splitter.split_text(doc.page_content) if t.strip()
                ]
                results.extend(_enrich_document(doc, pieces, "recursive"))
                continue

            md_pieces: List[Tuple[str, Dict[str, Any]]] = []
            for section in header_splitter.split_text(doc.page_content):
                if not section.page_content.strip():
                    continue
                header_metadata = dict(section.metadata)
                if len(section.page_content) <= chunk_size:
                    md_pieces.append((section.page_content, header_metadata))
                else:
                    for sub in recursive_splitter.split_text(section.page_content):
                        if sub.strip():
                            md_pieces.append((sub, header_metadata))
            results.extend(_enrich_document(doc, md_pieces, self.strategy))
        logger.info(
            "markdown 分块完成: %d 个文档 → %d 个文本块",
            len(documents),
            len(results),
        )
        return results


#: 策略名到分割器实例的映射
_STRATEGIES: Dict[str, Any] = {
    "recursive": RecursiveSplitter(),
    "semantic": SemanticSplitter(),
    "markdown": MarkdownSplitter(),
}


def split_documents(
    documents: List[Document],
    strategy: str = "recursive",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    **kwargs: Any,
) -> List[Document]:
    """统一的文档分割入口。

    Args:
        documents: 待切分的 Document 列表（通常来自文档加载器）。
        strategy: 分块策略，取值 ``'recursive' | 'semantic' | 'markdown'``。
        chunk_size: 目标块大小（字符数），默认 500。
        chunk_overlap: 相邻块之间的重叠字符数，默认 50。
        **kwargs: 透传给具体策略的额外参数。

    Returns:
        分割后的 Document 列表。每个子块继承原始元数据，并新增:
        ``chunk_index``（块序号，从 0 开始）、``chunk_total``（分块总数）、
        ``chunk_strategy``（策略名）、``chunk_size``（实际块字符数）、
        ``page_number``（继承自原文档页码，无则 None）、
        ``parent_source``（原文档 source 路径）。

    Raises:
        ValueError: 策略名未知，或所选策略执行失败。

    Notes:
        - 选择 semantic 但未安装 sentence-transformers 时，自动降级为 recursive；
        - 对非 Markdown 文档强制使用 markdown 时，自动降级为 recursive；
        - 上述降级行为会打印 WARNING 日志。
    """
    if not documents:
        logger.warning("split_documents 收到空文档列表，直接返回空结果")
        return []

    name = (strategy or "recursive").strip().lower()
    if name not in _STRATEGIES:
        raise ValueError(f"未知的分块策略: {strategy!r}，可选: recursive | semantic | markdown")

    non_empty = [d for d in documents if d.page_content and d.page_content.strip()]
    if len(non_empty) != len(documents):
        logger.warning("跳过 %d 个空白文档", len(documents) - len(non_empty))
    if not non_empty:
        return []

    # semantic 策略：可选依赖缺失时自动降级
    if name == "semantic" and not _semantic_dependencies_ready():
        logger.warning("未安装 sentence-transformers / langchain-experimental，semantic 策略降级为 recursive")
        name = "recursive"

    try:
        splitter = _STRATEGIES[name]
        results = splitter.split(non_empty, chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001 - 统一转为 ValueError
        logger.error("分块失败（strategy=%s）: %s", name, exc)
        raise ValueError(f"文档分块失败（strategy={name}）: {exc}") from exc

    logger.info(
        "分块完成（strategy=%s）: %d 个输入文档 → %d 个文本块",
        name,
        len(non_empty),
        len(results),
    )
    return results
