"""混合检索器模块。

将向量检索（语义）与 BM25 关键词检索结合，提供两种融合策略：

- ``weighted``（加权和）：按固定权重线性组合两路分数。适合已明确两路召回
  重要性的场景（如冷启动调参时手工设定配比）；
- ``rrf``（Reciprocal Rank Fusion）：基于排序位置融合，无需调权重，
  适合不确定两路可靠性、希望稳健提升整体召回率的场景。

Requires: pip install rank-bm25
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

from config import settings
from retrievers.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

#: 支持的融合策略
_SUPPORTED_STRATEGIES = ("weighted", "rrf")

#: BM25 token 化正则：英文/数字/下划线词 + 单个 CJK 汉字（无需分词器依赖）
_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]")


def _tokenize(text: str) -> List[str]:
    """将文本切分为 BM25 可用的 token 列表。

    英文按连续字母/数字成词并统一小写；中文逐字成 token，避免依赖 jieba
    等可选分词库。
    """
    return [token.lower() for token in _TOKEN_PATTERN.findall(text or "")]


class HybridRetriever:
    """混合检索器（向量 + BM25，双路召回后融合）。

    Args:
        vector_store_manager: 向量存储管理器（必须）。
        bm25_corpus: BM25 语料库（文档文本列表）。不传时初始化自动从
            ``vector_store_manager.get_all_documents()`` 拉取全部文档。
        bm25_metadatas: 与 ``bm25_corpus`` 一一对应的元数据列表；从向量库
            拉取时自动生成，无需手动传入。
        fusion_strategy: 融合策略 ``'weighted' | 'rrf'``，默认取配置。
        weights: 加权融合权重 ``(向量权重, BM25 权重)``，默认取配置各 0.5。
        rrf_k: RRF 算法常数（越大越平滑），默认取配置 60。
        top_k: 最终返回文档数，默认取配置 10。
        enable_reranking: 是否启用重排（Day 10 实现，当前为占位参数）。

    Raises:
        ValueError: 融合策略不支持，或 top_k 非法。
        ImportError: 缺少依赖 rank-bm25。

    Notes:
        - BM25 原始分数无界，检索后通过 :meth:`_normalize_scores` 归一化到 [0, 1]；
        - 向量库新增/删除文档后，请调用 :meth:`refresh_bm25_index` 重建索引；
        - 向向量库增量写入时，建议分批并控制批次大小，BM25 重建前先等待写入完成。
    """

    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        bm25_corpus: Optional[List[str]] = None,
        bm25_metadatas: Optional[List[Dict[str, Any]]] = None,
        fusion_strategy: Optional[str] = None,
        weights: Optional[Tuple[float, float]] = None,
        rrf_k: Optional[int] = None,
        top_k: Optional[int] = None,
        enable_reranking: bool = False,
    ) -> None:
        if not isinstance(vector_store_manager, VectorStoreManager):
            raise TypeError(
                f"vector_store_manager 必须是 VectorStoreManager 实例，"
                f"实际为 {type(vector_store_manager).__name__}"
            )
        self.vector_store_manager = vector_store_manager

        # 语料来源标记：None → 自动从向量库拉取（refresh 时重新拉取）
        self._corpus_from_store = bm25_corpus is None
        self._bm25_corpus: List[str] = list(bm25_corpus) if bm25_corpus is not None else []
        self._bm25_metadatas: List[Dict[str, Any]] = (
            list(bm25_metadatas) if bm25_metadatas is not None else []
        )

        strategy = (fusion_strategy or settings.HYBRID_FUSION_STRATEGY).strip().lower()
        if strategy not in _SUPPORTED_STRATEGIES:
            raise ValueError(
                f"不支持的融合策略: {strategy!r}，可选: {' | '.join(_SUPPORTED_STRATEGIES)}"
            )
        self.fusion_strategy = strategy

        resolved_weights = weights or (
            settings.HYBRID_WEIGHT_VECTOR,
            settings.HYBRID_WEIGHT_BM25,
        )
        if len(resolved_weights) != 2:
            raise ValueError("weights 必须为 (向量权重, BM25 权重) 二元组")
        self.weights: Tuple[float, float] = (float(resolved_weights[0]), float(resolved_weights[1]))

        self.rrf_k: int = int(rrf_k or settings.HYBRID_RRF_K)
        self.top_k: int = int(top_k or settings.HYBRID_TOP_K)
        if self.top_k < 1:
            raise ValueError(f"top_k 必须大于 0，实际为 {self.top_k}")
        if self.rrf_k <= 0:
            raise ValueError(f"rrf_k 必须大于 0，实际为 {self.rrf_k}")

        self.enable_reranking = bool(enable_reranking)
        if self.enable_reranking:
            logger.warning("enable_reranking=True 为预留参数，Cross-Encoder 重排将在 Day 10 实现")

        #: BM25 索引对象（懒构建，构造时会尝试导入 rank_bm25）
        self._bm25: Optional[Any] = None
        start = time.perf_counter()
        self._build_bm25_index()
        logger.info(
            "HybridRetriever 初始化完成（strategy=%s, weights=%s, rrf_k=%d, top_k=%d，"
            "BM25 构建耗时 %.4fs）",
            self.fusion_strategy,
            self.weights,
            self.rrf_k,
            self.top_k,
            time.perf_counter() - start,
        )

    # ------------------------------------------------------------------ #
    # BM25 索引构建
    # ------------------------------------------------------------------ #
    def _load_corpus(self) -> None:
        """加载 BM25 语料：语料来源为向量库时重新拉取全部文档。"""
        if self._corpus_from_store:
            logger.info(
                "bm25_corpus 未提供，正在从向量库（collection=%s）拉取全部文档构建索引，"
                "文档量较大时可能耗时……",
                self.vector_store_manager.collection_name,
            )
            documents = self.vector_store_manager.get_all_documents()
            self._bm25_corpus = [doc.page_content for doc in documents]
            self._bm25_metadatas = [dict(doc.metadata) for doc in documents]
        elif not self._bm25_metadatas:
            # 外部语料未带元数据时，补齐空字典并保持对齐
            self._bm25_metadatas = [{} for _ in self._bm25_corpus]

        if len(self._bm25_metadatas) != len(self._bm25_corpus):
            raise ValueError(
                f"bm25_metadatas 长度（{len(self._bm25_metadatas)}）"
                f"必须与 bm25_corpus 长度（{len(self._bm25_corpus)}）一致"
            )
        logger.info("BM25 语料就绪: 共 %d 条文档", len(self._bm25_corpus))

    def _build_bm25_index(self) -> None:
        """构建 BM25 索引（语料为空时自动从向量库拉取）。

        Raises:
            ImportError: 未安装 rank-bm25（提示安装命令）。
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:  # pragma: no cover - 依赖缺失提示
            raise ImportError(
                "使用 HybridRetriever 需要安装 rank-bm25，请执行: pip install rank-bm25"
            ) from exc

        self._load_corpus()
        tokenized_corpus = [_tokenize(text) for text in self._bm25_corpus]
        self._bm25 = BM25Okapi(tokenized_corpus)
        logger.info("BM25 索引构建完成: %d 条语料", len(tokenized_corpus))

    def refresh_bm25_index(self) -> None:
        """刷新 BM25 索引。

        当向量库新增 / 删除文档后调用：若语料源自向量库，会重新拉取全部
        文档以保证与向量库同步。
        """
        self._build_bm25_index()

    # ------------------------------------------------------------------ #
    # 单路检索
    # ------------------------------------------------------------------ #
    def _vector_retrieve(self, query: str, k: int, filter: Optional[Dict[str, Any]] = None) -> List[Tuple[Document, float]]:
        """执行向量检索，返回文档与其归一化相似度（夹取到 [0, 1]）。

        Args:
            query: 查询文本。
            k: 返回条数。
            filter: 可选的元数据过滤条件。

        Returns:
            ``(Document, score)`` 列表，分数为 0~1 的余弦相似度。
        """
        try:
            scored = self.vector_store_manager.similarity_search_with_score(query, k=k, filter=filter)
        except Exception as exc:  # noqa: BLE001
            logger.error("向量检索失败: %s", exc)
            raise RuntimeError(f"向量检索失败: {exc}") from exc
        results = [(doc, max(0.0, min(1.0, score))) for doc, score in scored]
        logger.info("向量检索完成: 查询=%r，返回 %d 条", query[:60], len(results))
        return results

    def _bm25_retrieve(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """执行 BM25 检索，返回文档与归一化到 [0, 1] 的分数。

        Args:
            query: 查询文本。
            k: 返回条数。

        Returns:
            ``(Document, score)`` 列表；无命中时返回空列表。
        """
        if self._bm25 is None:
            logger.warning("BM25 索引为空，跳过 BM25 检索")
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        # 仅保留正分命中项，按原始分降序，取候选池供归一化与融合
        ranked = sorted(
            ((index, scores[index]) for index in range(len(scores)) if scores[index] > 0),
            key=lambda item: item[1],
            reverse=True,
        )
        if not ranked:
            logger.info("BM25 检索完成: 无命中（查询=%r）", query[:60])
            return []

        pool = ranked[:k]
        normalized = self._normalize_scores([score for _, score in pool])
        results = [
            (
                Document(
                    page_content=self._bm25_corpus[index],
                    metadata={**self._bm25_metadatas[index]},
                ),
                normalized[position],
            )
            for position, (index, _) in enumerate(pool)
        ]
        logger.info("BM25 检索完成: 查询=%r，返回 %d 条", query[:60], len(results))
        return results

    @staticmethod
    def _normalize_scores(scores: List[float]) -> List[float]:
        """对 BM25 原始分数做 min-max 归一化到 [0, 1]。

        Args:
            scores: 待归一化的原始分数列表（非空）。

        Returns:
            归一化后的分数列表；若所有分数相等则统一返回 1.0。
        """
        if not scores:
            return []
        minimum, maximum = min(scores), max(scores)
        span = maximum - minimum
        if span <= 1e-12:
            return [1.0 for _ in scores]
        return [(score - minimum) / span for score in scores]

    @staticmethod
    def _doc_key(doc: Document) -> Tuple[str, str, str]:
        """生成文档去重键（source + chunk_index + 内容）。"""
        metadata = doc.metadata
        source = str(metadata.get("source") or metadata.get("file_name") or "")
        chunk = str(metadata.get("chunk_index", ""))
        return (source, chunk, doc.page_content)

    # ------------------------------------------------------------------ #
    # 融合策略
    # ------------------------------------------------------------------ #
    def _weighted_scored(self, vec_results: List[Tuple[Document, float]], bm25_results: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """加权线性融合两路结果，返回带综合分数的去重列表。"""
        vector_weight, bm25_weight = self.weights
        merged: Dict[Tuple[str, str, str], List[Any]] = {}
        for doc, score in vec_results:
            key = self._doc_key(doc)
            merged.setdefault(key, [doc, 0.0, 0.0])[1] = score
        for doc, score in bm25_results:
            key = self._doc_key(doc)
            merged.setdefault(key, [doc, 0.0, 0.0])[2] = score

        scored = [
            (entry[0], vector_weight * entry[1] + bm25_weight * entry[2])
            for entry in merged.values()
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored

    def _rrf_scored(self, vec_results: List[Tuple[Document, float]], bm25_results: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """基于排序位置的 RRF 融合，返回带综合分数的去重列表。"""
        acc: Dict[Tuple[str, str, str], List[Any]] = {}
        for ranked in (vec_results, bm25_results):
            for rank, (doc, _score) in enumerate(ranked, start=1):
                key = self._doc_key(doc)
                if key not in acc:
                    acc[key] = [doc, 0.0]
                acc[key][1] += 1.0 / (rank + self.rrf_k)

        scored = [(entry[0], entry[1]) for entry in acc.values()]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored

    def _fuse(self, vec_results: List[Tuple[Document, float]], bm25_results: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """按当前策略融合两路非空结果。"""
        if self.fusion_strategy == "weighted":
            return self._weighted_scored(vec_results, bm25_results)
        return self._rrf_scored(vec_results, bm25_results)

    def _weighted_fusion(self, vec_results: List[Tuple[Document, float]], bm25_results: List[Tuple[Document, float]]) -> List[Document]:
        """加权融合两路结果并返回 Top-K 文档（去重）。"""
        return [doc for doc, _ in self._weighted_scored(vec_results, bm25_results)[: self.top_k]]

    def _rrf_fusion(self, vec_results: List[Tuple[Document, float]], bm25_results: List[Tuple[Document, float]]) -> List[Document]:
        """RRF 融合两路结果并返回 Top-K 文档（去重）。"""
        return [doc for doc, _ in self._rrf_scored(vec_results, bm25_results)[: self.top_k]]

    # ------------------------------------------------------------------ #
    # 对外检索接口
    # ------------------------------------------------------------------ #
    def _search(self, query: str, top_k: int, filter: Optional[Dict[str, Any]] = None) -> List[Tuple[Document, float]]:
        """执行双路检索并按策略融合。

        向量 / BM25 任一为空时直接返回另一路结果；两者皆空返回空列表。
        """
        vec_time = time.perf_counter()
        vec_results = self._vector_retrieve(query, k=top_k, filter=filter)
        logger.info("向量检索耗时: %.4fs", time.perf_counter() - vec_time)

        bm_time = time.perf_counter()
        bm25_results = self._bm25_retrieve(query, k=top_k)
        logger.info("BM25 检索耗时: %.4fs", time.perf_counter() - bm_time)

        if not vec_results and not bm25_results:
            logger.warning("向量检索与 BM25 检索均无结果（查询=%r）", query[:60])
            return []

        if not vec_results:
            logger.info("向量检索无结果，直接返回 BM25 结果（%d 条）", len(bm25_results))
            return bm25_results[:top_k]

        if not bm25_results:
            logger.info("BM25 检索无结果，直接返回向量结果（%d 条）", len(vec_results))
            return vec_results[:top_k]

        fuse_time = time.perf_counter()
        fused = self._fuse(vec_results, bm25_results)[:top_k]
        logger.info(
            "融合完成（strategy=%s）: 向量 %d 条 + BM25 %d 条 → %d 条，耗时 %.4fs",
            self.fusion_strategy,
            len(vec_results),
            len(bm25_results),
            len(fused),
            time.perf_counter() - fuse_time,
        )
        return fused

    def get_relevant_documents(self, query: str, top_k: Optional[int] = None, **kwargs: Any) -> List[Document]:
        """对外检索接口：返回融合排序后的文档列表（不含分数）。

        Args:
            query: 查询文本。
            top_k: 返回条数（默认使用初始化时配置的 top_k）。
            **kwargs: 支持 ``filter``（向量检索的元数据过滤条件）等扩展参数。

        Returns:
            按综合分数降序排列的 Document 列表。
        """
        target_k = int(top_k or self.top_k)
        scored = self._search(query, target_k, filter=kwargs.get("filter"))
        return [doc for doc, _ in scored[:target_k]]

    def get_relevant_documents_with_scores(self, query: str, top_k: Optional[int] = None, **kwargs: Any) -> List[Tuple[Document, float]]:
        """带综合分数的检索接口（便于调试与后续重排）。

        Args:
            query: 查询文本。
            top_k: 返回条数（默认使用初始化时配置的 top_k）。
            **kwargs: 支持 ``filter`` 等扩展参数。

        Returns:
            ``(Document, score)`` 列表，分数为融合后的综合得分。
        """
        target_k = int(top_k or self.top_k)
        return self._search(query, target_k, filter=kwargs.get("filter"))[:target_k]
