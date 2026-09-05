"""检索结果重排（Reranker）模块。

# Requires: pip install sentence-transformers

使用 Cross-Encoder 对「查询 + 候选文档」的配对进行精细打分，从而修正仅靠
双塔向量相似度 / 关键词命中产生的排序误差（尤其适合语义相近但表述差异大的
文档）。模型加载为惰性单例：

- 首次真正执行打分时才会加载模型，并以 ``lru_cache`` 按参数缓存引擎，
  避免同一进程内重复下载 / 载入同一模型；
- 若 sentence-transformers 的 CrossEncoder 加载失败（如网络异常），
  WARNING 降级为 transformers 文本分类管线再次尝试；
- 两者均不可用 / 均失败时抛出 ``RuntimeError``，携带依赖与排障建议。

性能提示：重排耗时与候选文档数近似成正比（Cross-Encoder 需逐对前向计算），
建议上游先将候选截断到 30~50 条再送入 Reranker。
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, List, Optional, Tuple

from langchain_core.documents import Document

from config import settings

logger = logging.getLogger(__name__)

#: 支持的设备；auto 表示「GPU 可用则 GPU，否则 CPU」
_SUPPORTED_DEVICES: tuple[str, ...] = ("cpu", "cuda", "auto")

#: 缺失 sentence-transformers 时的安装提示
_MISSING_DEPS_HINT: str = (
    "加载 Reranker 失败：缺少 sentence-transformers，"
    "请执行 `pip install sentence-transformers` 后重试。"
)

#: 模型加载 / 推理失败时的通用排障建议
_LOAD_FAIL_HINT: str = (
    "请依次排查：1) 网络连通性与 HuggingFace 镜像；"
    "2) model_name 是否真实存在（或本地路径是否正确）；"
    "3) 依赖是否完整（pip install sentence-transformers）。"
)


def _text_of(doc: Document) -> str:
    """抽取文档正文为纯文本（兼容非字符串 page_content）。"""
    content = doc.page_content
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return str(content)


class _PipelineEngine:
    """transformers 文本分类管线的轻量适配层（CrossEncoder 的降级实现）。

    将 ``(query, doc)`` 配对以 ``query [SEP] doc`` 形式喂入分类管线，
    返回标签分数列表（分数越高代表越相关，仅作相对排序参考）。
    """

    def __init__(self, classifier: Any, max_length: int) -> None:
        self._classifier = classifier
        self._max_length = max_length

    def predict(self, pairs: List[Tuple[str, str]], batch_size: Optional[int] = None) -> List[float]:
        """对配对列表打分。

        Args:
            pairs: (query, doc) 配对列表。
            batch_size: 管线批大小（由调用方决定，缺省用默认值）。

        Returns:
            分数列表。
        """
        texts = [f"{query} [SEP] {doc}" for query, doc in pairs]
        kwargs: dict = {"batch_size": batch_size} if batch_size else {}
        try:
            results = self._classifier(texts, truncation=True, max_length=self._max_length, **kwargs)
        except TypeError:
            # 个别管线不支持在调用期传入 max_length / truncation
            results = self._classifier(texts, **kwargs)
        return [float(item["score"]) for item in results]


@lru_cache(maxsize=8)
def _create_cross_encoder(model_name: str, device: str, max_length: int, use_fp16: bool) -> Any:
    """按参数缓存并构建 CrossEncoder 引擎（同参数全局仅加载一次）。"""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError(_MISSING_DEPS_HINT) from exc

    engine = CrossEncoder(model_name, max_length=max_length, device=device)
    if use_fp16 and device == "cuda":
        try:
            engine.model.half()
        except Exception as exc:  # noqa: BLE001 - fp16 为可选加速，失败不阻断
            logger.warning("启用 FP16 失败（%s），已回退 FP32", exc)
    logger.info(
        "CrossEncoder 引擎加载完成：%s（device=%s, max_length=%d）",
        model_name,
        device,
        max_length,
    )
    return engine


class Reranker:
    """Cross-Encoder 检索重排器。

    模型加载采用**惰性单例**：构造对象本身不加载模型；首次执行
    :meth:`rerank` / :meth:`rerank_with_scores` 时才加载（见 :attr:`engine`），
    底层引擎按 ``(model_name, device, max_length, use_fp16)`` 缓存复用。
    类型未显式传入的参数（传 ``None``）将从全局配置 ``config.settings`` 读取。

    Attributes:
        model_name: HuggingFace Cross-Encoder 模型名（或本地目录）。
        device: 推理设备（``cpu`` / ``cuda`` / ``auto``）。
        batch_size: 批量推理大小。
        max_length: 输入序列最大长度（超长截断）。
        use_fp16: 是否启用半精度（仅对 GPU 生效，CPU 上自动忽略并 WARNING）。
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        use_fp16: Optional[bool] = None,
    ) -> None:
        """初始化 Reranker。

        Args:
            model_name: HuggingFace 上的 Cross-Encoder 模型名称（或本地路径），
                默认取配置 ``RERANKER_MODEL``。
            device: 推理设备 ``cpu`` / ``cuda`` / ``auto``；
                默认取配置 ``RERANKER_DEVICE``。
            batch_size: 批量推理大小（优化速度），默认取配置
                ``RERANKER_BATCH_SIZE``。
            max_length: 输入序列最大长度（截断），默认取配置
                ``RERANKER_MAX_LENGTH``。
            use_fp16: 是否使用半精度加速（仅 GPU），默认取配置
                ``RERANKER_USE_FP16``。

        Raises:
            ValueError: 设备不受支持、batch_size / max_length 小于 1。
        """
        self.model_name: str = (
            settings.RERANKER_MODEL if model_name is None else str(model_name)
        )
        raw_device = settings.RERANKER_DEVICE if device is None else str(device)
        self.device: str = raw_device.strip().lower()
        if self.device not in _SUPPORTED_DEVICES:
            raise ValueError(
                f"不支持的推理设备 {self.device!r}，可选：{' / '.join(_SUPPORTED_DEVICES)}"
            )
        self.batch_size: int = (
            settings.RERANKER_BATCH_SIZE if batch_size is None else int(batch_size)
        )
        if self.batch_size < 1:
            raise ValueError("batch_size 必须大于等于 1")
        self.max_length: int = (
            settings.RERANKER_MAX_LENGTH if max_length is None else int(max_length)
        )
        if self.max_length < 1:
            raise ValueError("max_length 必须大于等于 1")
        self.use_fp16: bool = (
            settings.RERANKER_USE_FP16 if use_fp16 is None else bool(use_fp16)
        )

        #: 惰性加载的底层引擎；首次打分前为 None
        self._model: Optional[Any] = None

    # ------------------------------------------------------------------ #
    # 模型加载
    # ------------------------------------------------------------------ #
    @property
    def engine(self) -> Any:
        """底层引擎（惰性加载，进程内只加载一次）。"""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self) -> Any:
        """加载 Cross-Encoder 引擎；失败时降级 transformers 管线。

        Returns:
            具备 ``predict(pairs, batch_size=None)`` 接口的引擎对象。

        Raises:
            RuntimeError: sentence-transformers 缺失、或降级加载也失败时，
                携带依赖安装与网络排障提示。
        """
        device = self._resolve_device()
        try:
            engine = _create_cross_encoder(
                self.model_name, device, self.max_length, self.use_fp16
            )
        except RuntimeError:
            # 缺少依赖等明确错误：直接上抛，不再尝试降级
            raise
        except Exception as exc:  # noqa: BLE001 - 网络 / 模型不存在等
            logger.warning(
                "CrossEncoder 加载失败（%s），降级尝试 transformers 文本分类管线",
                exc,
            )
            engine = self._load_transformers_fallback(device)
        logger.info(
            "Reranker 模型加载成功：%s（device=%s, batch_size=%d）",
            self.model_name,
            device,
            self.batch_size,
        )
        return engine

    def _resolve_device(self) -> str:
        """解析实际推理设备（``auto`` -> GPU 可用则 cuda，否则 cpu）。"""
        if self.device != "auto":
            return self.device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:  # noqa: BLE001 - torch 缺失等，视为仅 CPU
            logger.warning("检测 CUDA 失败（可能缺少 torch），回退到 CPU")
        return "cpu"

    def _load_transformers_fallback(self, device: str) -> Any:
        """使用 transformers 文本分类管线降级加载。

        Args:
            device: 解析后的推理设备（当前降级实现交由管线自动选择）。

        Returns:
            :class:`_PipelineEngine` 实例。

        Raises:
            RuntimeError: transformers 缺失或管线构建失败。
        """
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "Reranker 降级加载失败：缺少 transformers。"
                "请执行 `pip install sentence-transformers`（含 transformers 依赖）。"
            ) from exc
        logger.warning(
            "已降级使用 transformers 文本分类管线加载 %s（device=%s）",
            self.model_name,
            device,
        )
        try:
            classifier = pipeline("text-classification", model=self.model_name)
        except Exception as exc:  # noqa: BLE001 - 网络 / 模型名错误等
            raise RuntimeError(
                f"Reranker 模型加载失败（{self.model_name!r}），"
                f"CrossEncoder 与 transformers 降级均不可用：{exc}。{_LOAD_FAIL_HINT}"
            ) from exc
        return _PipelineEngine(classifier, self.max_length)

    # ------------------------------------------------------------------ #
    # 打分与排序核心
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_top_k(top_k: int) -> None:
        """校验 top_k 参数。"""
        if top_k < 1:
            raise ValueError("top_k 必须大于等于 1")

    def _predict_scores(self, query: str, documents: List[Document]) -> List[float]:
        """调用引擎对 (query, doc) 配对批量打分。

        Args:
            query: 查询文本。
            documents: 候选文档列表。

        Returns:
            与 ``documents`` 顺序一致的分数列表。
        """
        pairs = [(query, _text_of(doc)) for doc in documents]
        logger.debug(
            "开始打分 %d 个候选（query=%r, batch_size=%d）",
            len(pairs),
            query[:50],
            self.batch_size,
        )
        try:
            scores = self.engine.predict(pairs, batch_size=self.batch_size)
        except TypeError:
            # 兼容不接受 batch_size 关键字的旧引擎
            scores = self.engine.predict(pairs)
        return [float(score) for score in scores]

    def _score_and_sort(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int],
    ) -> List[Tuple[Document, float]]:
        """打分并按分数降序返回 (文档, 分数) 列表（稳定排序，可截断）。"""
        scores = self._predict_scores(query, documents)
        ranked = sorted(
            zip(documents, scores), key=lambda item: item[1], reverse=True
        )
        if top_k is not None:
            ranked = ranked[:top_k]
        return ranked

    # ------------------------------------------------------------------ #
    # 对外接口
    # ------------------------------------------------------------------ #
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
    ) -> List[Document]:
        """按相关性重排候选文档。

        小规模快捷路径：候选为空 / 仅 1 条 / 条数不足 ``top_k`` 时直接返回
        原列表，不触发模型加载（无需重排）。

        Args:
            query: 查询文本。
            documents: 初步检索返回的候选文档。
            top_k: 只保留分数最高的前 k 条；None 表示全部返回。

        Returns:
            按相关性分数降序排列的文档列表（并截断到 top_k）。

        Raises:
            ValueError: top_k 小于 1。
            RuntimeError: 模型加载失败（含排障建议）。
        """
        result = list(documents)
        if not result or len(result) < 2:
            return result
        if top_k is not None:
            self._validate_top_k(top_k)
            if len(result) < top_k:
                return result
        return [doc for doc, _ in self._score_and_sort(query, result, top_k)]

    def rerank_with_scores(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
    ) -> List[Tuple[Document, float]]:
        """带分数重排，便于调试与阈值过滤。

        注意：模型输出分数范围取决于所选引擎——ms-marco 类 Cross-Encoder
        通常直接返回 logits（可正可负，单调越高越相关）；若通过
        ``default_activation_function=softmax`` 或 bge-reranker 的 softmax
        变体则接近 0~1。本方法返回**原始引擎分数**，仅保证同一查询内部
        单调可比。

        Args:
            query: 查询文本。
            documents: 候选文档列表。
            top_k: 只保留分数最高的前 k 条；None 表示全部返回。

        Returns:
            [(文档, 相关性分数)] 按分数降序排列。

        Raises:
            ValueError: top_k 小于 1。
            RuntimeError: 模型加载失败（含排障建议）。
        """
        result = list(documents)
        if not result:
            return []
        if top_k is not None:
            self._validate_top_k(top_k)
        return self._score_and_sort(query, result, top_k)

    def filter_by_threshold(
        self,
        query: str,
        documents: List[Document],
        threshold: float = 0.5,
    ) -> List[Document]:
        """阈值过滤：仅保留分数严格高于 ``threshold`` 的文档。

        适合过滤低质量检索结果。过滤结果沿用重排后的降序。

        Args:
            query: 查询文本。
            documents: 候选文档列表。
            threshold: 分数下限（含等于者被丢弃）。

        Returns:
            分数高于阈值的文档列表（降序）。

        Raises:
            RuntimeError: 模型加载失败（含排障建议）。
        """
        if not documents:
            return []
        threshold = float(threshold)
        return [
            doc
            for doc, score in self._score_and_sort(query, list(documents), None)
            if score > threshold
        ]

    def rerank_batch(
        self,
        queries: List[str],
        doc_lists: List[List[Document]],
        top_k: Optional[int] = None,
    ) -> List[List[Document]]:
        """批量重排多个查询（适用于 API 场景，提升吞吐量）。

        Args:
            queries: 查询文本列表。
            doc_lists: 与 ``queries`` 一一对应的候选文档列表。
            top_k: 每个查询保留的前 k 条；None 表示全部返回。

        Returns:
            与 ``queries`` 等长的重排结果列表。

        Raises:
            ValueError: 两个列表长度不一致，或 top_k 小于 1。
            RuntimeError: 模型加载失败（含排障建议）。
        """
        if len(queries) != len(doc_lists):
            raise ValueError(
                f"queries（{len(queries)}）与 doc_lists（{len(doc_lists)}）长度不一致"
            )
        if top_k is not None:
            self._validate_top_k(top_k)
        return [
            self.rerank(query, docs, top_k=top_k)
            for query, docs in zip(queries, doc_lists)
        ]
