"""Reranker 重排单元测试。

测试全程离线：Reranker 的模型加载为惰性且可注入，子类重写
``_load_model`` 注入确定性伪 Cross-Encoder（按「查询词元在文档中的覆盖率」
打分），因此无需下载 / 加载真实模型。

覆盖：初始化参数、惰性加载与加载一次、重排排序质量、top_k 截断、
小规模快速路径（不触发加载）、带分数返回、阈值过滤、批量重排、
参数校验，以及模型加载失败时的 RuntimeError 与降级管线路径。
"""

import re

import pytest
from langchain_core.documents import Document

from retrievers import reranker as reranker_module
from retrievers.reranker import Reranker

#: 5 个主题各异的候选文档（索引 0 最相关，索引 4 完全无关）
CORPUS: list[str] = [
    "LangChain 的 RAG 检索增强生成流程，可结合向量检索与生成回答。",
    "检索增强生成 RAG 结合检索与生成，可降低幻觉并支持来源溯源。",
    "LangChain 是一个用于构建 LLM 应用的框架，支持链式调用与工具集成。",
    "ChromaDB 是向量数据库，用于存储嵌入与执行向量相似度检索。",
    "Streamlit 是数据应用 UI 框架，与本次查询主题无关。",
]

#: 与 CORPUS[0] 词元完全覆盖的查询（确保其得分最高）
QUERY = "LangChain RAG 检索增强生成 向量"

_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]")


def _tokens(text: str) -> set[str]:
    """与混合检索器一致的轻量 token 化（英文词 + 单个汉字）。"""
    return set(_TOKEN_PATTERN.findall(text.lower()))


def _docs() -> list[Document]:
    """按 CORPUS 构造测试文档。"""
    return [Document(page_content=text, metadata={"source": f"doc-{index}.txt"}) for index, text in enumerate(CORPUS)]


def _engine_score(query: str, doc: str) -> float:
    """伪 Cross-Encoder 分数：查询词元在文档中的覆盖率（0~1）。"""
    query_tokens = _tokens(query)
    if not query_tokens:
        return 0.0
    return len(query_tokens & _tokens(doc)) / len(query_tokens)


class _FakeCrossEncoder:
    """确定性伪 Cross-Encoder：具备与真模型一致的 predict 接口。"""

    def predict(self, pairs: list[tuple[str, str]], batch_size: int | None = None) -> list[float]:
        return [_engine_score(query, doc) for query, doc in pairs]


class _FakeReranker(Reranker):
    """注入伪引擎的 Reranker：统计模型加载次数，避免任何网络依赖。"""

    def __init__(self, *args, engine=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._fake_engine = engine if engine is not None else _FakeCrossEncoder()
        self.load_count = 0

    def _load_model(self):
        self.load_count += 1
        return self._fake_engine


# ---------------------------------------------------------------------- #
# 初始化与惰性加载
# ---------------------------------------------------------------------- #
def test_init_resolves_explicit_and_settings_defaults():
    """显式参数覆盖默认值；未传参数回退到全局配置（构造不加载模型）。"""
    explicit = Reranker(
        model_name="local/foo", device="cpu", batch_size=8, max_length=256, use_fp16=False
    )
    assert explicit.model_name == "local/foo"
    assert explicit.device == "cpu"
    assert explicit.batch_size == 8
    assert explicit.max_length == 256
    assert explicit.use_fp16 is False
    assert explicit._model is None  # 惰性：构造阶段不加载

    defaults = Reranker()
    from config import settings

    assert defaults.model_name == settings.RERANKER_MODEL
    assert defaults.device == settings.RERANKER_DEVICE
    assert defaults.batch_size == settings.RERANKER_BATCH_SIZE
    assert defaults.max_length == settings.RERANKER_MAX_LENGTH


def test_engine_loaded_lazily_and_cached():
    """引擎在首次打分时加载，且同一实例只加载一次。"""
    reranker = _FakeReranker()
    docs = _docs()
    assert reranker.load_count == 0

    first = reranker.rerank(QUERY, docs)
    assert reranker.load_count == 1
    assert first[0].page_content == CORPUS[0]

    # 二次调用复用已加载引擎，不再触发加载
    reranker.rerank(QUERY, docs)
    assert reranker.load_count == 1


def test_invalid_device_raises():
    """不支持的推理设备应抛出 ValueError。"""
    with pytest.raises(ValueError, match="不支持的推理设备"):
        Reranker(device="npu")


# ---------------------------------------------------------------------- #
# 重排排序
# ---------------------------------------------------------------------- #
def test_rerank_orders_most_relevant_first():
    """相关文档应排到最前，无关文档沉底。"""
    reranker = _FakeReranker()
    result = reranker.rerank(QUERY, _docs())

    assert len(result) == len(CORPUS)
    assert result[0].page_content == CORPUS[0]
    assert result[-1].page_content == CORPUS[4]


def test_rerank_respects_top_k():
    """指定 top_k 后只保留分数最高的前 k 条。"""
    reranker = _FakeReranker()
    result = reranker.rerank(QUERY, _docs(), top_k=2)

    assert len(result) == 2
    assert [doc.page_content for doc in result] == [CORPUS[0], CORPUS[1]]


def test_rerank_skips_engine_when_fewer_than_top_k():
    """候选少于 top_k 时直接返回原列表，不触发模型加载。"""
    reranker = _FakeReranker()
    docs = _docs()[:3]

    result = reranker.rerank(QUERY, docs, top_k=5)
    assert [doc.page_content for doc in result] == CORPUS[:3]
    assert reranker.load_count == 0


def test_rerank_empty_or_single_does_not_load_engine():
    """空列表 / 单文档无需重排，直接返回且不触发模型加载。"""
    reranker = _FakeReranker()

    assert reranker.rerank(QUERY, []) == []
    assert reranker.rerank(QUERY, _docs()[:1])[0].page_content == CORPUS[0]
    assert reranker.load_count == 0


def test_invalid_top_k_raises():
    """top_k 小于 1 应抛出 ValueError，且不触发模型加载。"""
    reranker = _FakeReranker()
    with pytest.raises(ValueError, match="top_k 必须大于等于 1"):
        reranker.rerank_with_scores(QUERY, _docs(), top_k=0)
    assert reranker.load_count == 0


# ---------------------------------------------------------------------- #
# 带分数重排
# ---------------------------------------------------------------------- #
def test_rerank_with_scores_returns_descending_pairs():
    """rerank_with_scores 返回 (文档, 分数) 元组，分数非增且位于 0~1。"""
    reranker = _FakeReranker()
    pairs = reranker.rerank_with_scores(QUERY, _docs())

    assert len(pairs) == len(CORPUS)
    for doc, score in pairs:
        assert isinstance(doc, Document)
        assert 0.0 <= score <= 1.0
    scores = [score for _, score in pairs]
    assert all(left >= right for left, right in zip(scores, scores[1:]))
    # 最相关文档得分应显著高于完全无关文档
    assert scores[0] > scores[-1]


# ---------------------------------------------------------------------- #
# 阈值过滤
# ---------------------------------------------------------------------- #
def test_filter_by_threshold_drops_low_score_docs():
    """低于等于阈值的文档被过滤，高于阈值的保留（沿用降序）。"""
    reranker = _FakeReranker()
    # QUERY 与 CORPUS[0] 覆盖率 1.0，CORPUS[1] 约为 0.7，其余更低
    filtered = reranker.filter_by_threshold(QUERY, _docs(), threshold=0.8)

    assert [doc.page_content for doc in filtered] == [CORPUS[0]]


# ---------------------------------------------------------------------- #
# 批量重排
# ---------------------------------------------------------------------- #
def test_rerank_batch_multiple_queries():
    """批量重排多个查询，返回与查询一一对应的结果。"""
    reranker = _FakeReranker()
    second_query = "ChromaDB 向量数据库 存储"
    queries = [QUERY, second_query]
    doc_lists = [_docs()[:3], _docs()[3:]]

    results = reranker.rerank_batch(queries, doc_lists)

    assert len(results) == 2
    assert len(results[0]) == 3
    assert results[0][0].page_content == CORPUS[0]
    assert results[1][0].page_content == CORPUS[3]


def test_rerank_batch_mismatched_lengths_raises():
    """queries 与 doc_lists 长度不一致应抛出 ValueError。"""
    reranker = _FakeReranker()
    with pytest.raises(ValueError, match="长度不一致"):
        reranker.rerank_batch([QUERY], [_docs(), _docs()])


# ---------------------------------------------------------------------- #
# 加载失败与降级
# ---------------------------------------------------------------------- #
def test_model_load_failure_raises_runtime_error_with_hint(monkeypatch):
    """模型加载失败应抛出携带安装 / 排障建议的 RuntimeError。"""

    def boom(*args, **kwargs):
        raise RuntimeError("缺少 sentence-transformers，请执行 pip install sentence-transformers")

    monkeypatch.setattr(reranker_module, "_create_cross_encoder", boom)
    reranker = Reranker(model_name="unreachable/model", device="cpu")

    with pytest.raises(RuntimeError, match="pip install"):
        reranker.rerank(QUERY, _docs()[:2])


def test_fallback_engine_used_when_cross_encoder_load_fails(monkeypatch):
    """CrossEncoder 抛一般异常时降级使用 transformers 管线（此处注入伪引擎）。"""
    fake = _FakeCrossEncoder()

    class _FallbackReranker(_FakeReranker):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, engine=fake, **kwargs)
            self.fallback_count = 0

        def _load_transformers_fallback(self, device):
            self.fallback_count += 1
            return self._fake_engine

    def boom(*args, **kwargs):
        raise ValueError("mock network down")

    monkeypatch.setattr(reranker_module, "_create_cross_encoder", boom)
    reranker = _FallbackReranker(model_name="unreachable/model")

    result = reranker.rerank(QUERY, _docs())
    assert reranker.fallback_count == 1
    assert result[0].page_content == CORPUS[0]
