"""HybridRetriever 混合检索单元测试。

覆盖：加权融合 / RRF 融合（含排序差异）、向量为空回退 BM25、语料自动从
向量库构建、refresh 后新增文档可检索、双路皆空优雅返回。

使用确定性词元 n-gram 伪 Embeddings，使「含查询词的文档」在向量路也有较高
相似度；BM25 路为真实关键词检索（rank-bm25 需已安装）。
"""

import hashlib
import uuid

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

pytest.importorskip("rank_bm25")

from retrievers import HybridRetriever, VectorStoreManager  # noqa: E402

#: 6 个不同主题的测试语料（前 5 个用于 ≥5 个不同主题要求）
CORPUS: list[str] = [
    "LangChain is a framework for building LLM applications. 它支持链式调用与工具集成。",
    "ChromaDB is a vector database that stores embeddings. 用于向量相似度检索。",
    "BM25 Okapi ranks documents by keyword frequency. 属于稀疏关键词检索算法。",
    "FastAPI helps build high-performance web APIs. 用于构建 HTTP 服务。",
    "Streamlit is a framework for building data apps UI. 用于构建交互式前端界面。",
    "Pydantic validates data models. 用于配置与数据校验。",
]


def _tokens(text: str) -> list[str]:
    """与混合检索器一致的轻量 token 化（英文词 + 单个汉字）。"""
    import re

    return re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]", text.lower())


class _NgramEmbeddings(Embeddings):
    """确定性伪 Embeddings：按 token / 2-gram 哈希到 64 维，词元相近者余弦更高。"""

    _DIM = 64

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @classmethod
    def _embed(cls, text: str) -> list[float]:
        vector = [0.0] * cls._DIM
        grams = set()
        for token in _tokens(text):
            grams.add(token)
            for index in range(len(token) - 1):
                grams.add(token[index : index + 2])
        for gram in grams:
            digest = hashlib.md5(gram.encode("utf-8")).hexdigest()
            vector[int(digest, 16) % cls._DIM] += 1.0
        return vector


def _doc(content: str, source: str) -> Document:
    """构造带 source 元数据的测试文档。"""
    return Document(page_content=content, metadata={"source": source, "file_name": source.rsplit("/", 1)[-1]})


def _manager(tmp_path, seed: bool = True) -> VectorStoreManager:
    """构造指向独立临时目录 / 独立集合的向量存储管理器。"""
    manager = VectorStoreManager(
        collection_name="hyb" + uuid.uuid4().hex[:10],
        persist_directory=str(tmp_path),
        embeddings=_NgramEmbeddings(),
    )
    if seed:
        for index, text in enumerate(CORPUS):
            manager.add_documents([_doc(text, f"/tmp/topic_{index}.txt")])
    return manager


# ---------------------------------------------------------------------- #
# 初始化与参数校验
# ---------------------------------------------------------------------- #
def test_unknown_fusion_strategy_raises(tmp_path):
    """不支持的融合策略应抛出 ValueError。"""
    manager = _manager(tmp_path, seed=False)

    with pytest.raises(ValueError, match="不支持的融合策略"):
        HybridRetriever(manager, fusion_strategy="naive", bm25_corpus=["x"])


def test_invalid_top_k_raises(tmp_path):
    """top_k 小于 1 应抛出 ValueError。"""
    manager = _manager(tmp_path, seed=False)

    with pytest.raises(ValueError, match="top_k 必须大于 0"):
        HybridRetriever(manager, top_k=0, bm25_corpus=["x"])


def test_mismatched_corpus_and_metadatas_raises(tmp_path):
    """bm25_metadatas 与 bm25_corpus 长度不一致应抛出 ValueError。"""
    manager = _manager(tmp_path, seed=False)

    with pytest.raises(ValueError, match="必须与 bm25_corpus 长度"):
        HybridRetriever(
            manager,
            bm25_corpus=["a", "b"],
            bm25_metadatas=[{"source": "/f/a.txt"}],
        )


# ---------------------------------------------------------------------- #
# 语料自动从向量库构建
# ---------------------------------------------------------------------- #
def test_bm25_corpus_auto_built_from_vector_store(tmp_path):
    """未提供 bm25_corpus 时，应自动从向量库拉取全部文档构建索引。"""
    manager = _manager(tmp_path, seed=True)

    retriever = HybridRetriever(manager, top_k=6)

    assert retriever._bm25 is not None
    assert len(retriever._bm25_corpus) == len(CORPUS)
    # BM25 路能命中含 FastAPI 的文档
    hits = retriever.get_relevant_documents("FastAPI web HTTP 服务")
    assert any("FastAPI" in doc.page_content for doc in hits)


# ---------------------------------------------------------------------- #
# 加权融合
# ---------------------------------------------------------------------- #
def test_weighted_fusion_end_to_end_returns_top_k(tmp_path):
    """加权融合：返回条数等于 top_k，分数降序且包含关键词命中文档。"""
    manager = _manager(tmp_path, seed=True)
    retriever = HybridRetriever(
        manager,
        fusion_strategy="weighted",
        weights=(0.6, 0.4),
        top_k=3,
    )

    docs = retriever.get_relevant_documents("LangChain framework LLM applications 链式调用")
    scored = retriever.get_relevant_documents_with_scores("LangChain framework LLM applications 链式调用")

    assert len(docs) == 3
    assert len(scored) == 3
    # 综合分数应降序排列
    assert [score for _, score in scored] == sorted(
        (score for _, score in scored), reverse=True
    )
    assert any("LangChain" in doc.page_content for doc in docs)


def test_weighted_fusion_dedups_and_merges_both_engines(tmp_path):
    """加权融合：同一文档去重，且合并来自两路的文档。"""
    manager = _manager(tmp_path, seed=False)
    doc_a = _doc("A 共享文档内容", "/tmp/a.txt")
    doc_b = _doc("B 仅 BM25 命中内容", "/tmp/b.txt")
    retriever = HybridRetriever(
        manager,
        bm25_corpus=["A 共享文档内容", "B 仅 BM25 命中内容"],
        bm25_metadatas=[{"source": "/tmp/a.txt"}, {"source": "/tmp/b.txt"}],
        weights=(0.5, 0.5),
        top_k=10,
    )

    # 向量路返回 A；BM25 路返回 A + B
    docs = retriever._weighted_fusion([(doc_a, 0.9)], [(doc_a, 0.8), (doc_b, 1.0)])

    contents = [doc.page_content for doc in docs]
    # A 被去重合并（0.5*0.9 + 0.5*0.8 = 0.85）高于仅 BM25 的 B（0.5）
    assert contents == ["A 共享文档内容", "B 仅 BM25 命中内容"]


# ---------------------------------------------------------------------- #
# RRF 融合
# ---------------------------------------------------------------------- #
def test_rrf_end_to_end_returns_results(tmp_path):
    """RRF 融合端到端：返回 top_k 条且结果非空。"""
    manager = _manager(tmp_path, seed=True)
    retriever = HybridRetriever(
        manager,
        fusion_strategy="rrf",
        rrf_k=60,
        top_k=3,
    )

    docs = retriever.get_relevant_documents("ChromaDB vector database 向量检索")
    scored = retriever.get_relevant_documents_with_scores("ChromaDB vector database 向量检索")

    assert len(docs) == 3
    assert len(scored) == 3
    assert [score for _, score in scored] == sorted(
        (score for _, score in scored), reverse=True
    )
    assert any("ChromaDB" in doc.page_content for doc in docs)


def test_rrf_ordering_differs_from_weighted(tmp_path):
    """RRF 与加权融合对同一组结果排序不同（算法语义差异）。

    构造：向量路高分单边命中 X、双边命中 Z（两侧分数中等）；
    加权偏好高分的 X，RRF 偏好双路均排前的 Z。
    """
    manager = _manager(tmp_path, seed=False)
    doc_x = _doc("X vector-only high score", "/tmp/x.txt")
    doc_z = _doc("Z appears in both lists", "/tmp/z.txt")
    corpus = ["X vector-only high score", "Z appears in both lists"]
    metadatas = [{"source": "/tmp/x.txt"}, {"source": "/tmp/z.txt"}]

    weighted_hr = HybridRetriever(manager, bm25_corpus=corpus, bm25_metadatas=metadatas, top_k=10)
    rrf_hr = HybridRetriever(manager, bm25_corpus=corpus, bm25_metadatas=metadatas, fusion_strategy="rrf", top_k=10)

    vec_results = [(doc_x, 0.99), (doc_z, 0.49)]
    bm25_results = [(doc_z, 0.49)]

    weighted_first = weighted_hr._weighted_fusion(vec_results, bm25_results)[0]
    rrf_first = rrf_hr._rrf_fusion(vec_results, bm25_results)[0]

    assert weighted_first.page_content == doc_x.page_content
    assert rrf_first.page_content == doc_z.page_content
    assert weighted_first.page_content != rrf_first.page_content


# ---------------------------------------------------------------------- #
# 空结果回退
# ---------------------------------------------------------------------- #
def test_vector_empty_falls_back_to_bm25(tmp_path):
    """向量路无结果时，应直接返回 BM25 检索结果。"""
    manager = _manager(tmp_path, seed=False)  # 向量库为空
    retriever = HybridRetriever(
        manager,
        bm25_corpus=CORPUS,
        bm25_metadatas=[{"source": f"/tmp/topic_{index}.txt"} for index in range(len(CORPUS))],
        top_k=5,
    )

    docs = retriever.get_relevant_documents("Okapi frequency 检索算法")

    assert 0 < len(docs) <= 5
    assert "Okapi" in docs[0].page_content  # BM25 第一命中为关键词最密集文档


def test_both_empty_returns_empty(tmp_path):
    """向量与 BM25 均无结果时返回空列表且不抛异常。"""
    manager = _manager(tmp_path, seed=False)

    retriever = HybridRetriever(manager, top_k=3)

    assert retriever.get_relevant_documents("完全不存在的词汇xyz") == []
    assert retriever.get_relevant_documents_with_scores("完全不存在的词汇xyz") == []


# ---------------------------------------------------------------------- #
# 索引刷新
# ---------------------------------------------------------------------- #
def test_refresh_index_after_adding_document(tmp_path):
    """向量库新增文档后，refresh_bm25_index() 应能检索到新文档。"""
    manager = _manager(tmp_path, seed=True)
    retriever = HybridRetriever(manager, top_k=6)

    # 新文档加入前：检索不到 Kalido 相关内容
    before = retriever.get_relevant_documents("Kalido unique token")
    assert not any("Kalido" in doc.page_content for doc in before)

    # 新增含独特 token 的文档
    new_doc = _doc("Kalido is the unique keyword for refresh test. 刷新验证独特标记。", "/tmp/new.txt")
    manager.add_documents([new_doc])

    # 刷新 BM25 索引后再检索
    retriever.refresh_bm25_index()
    after = retriever.get_relevant_documents("Kalido unique token")

    assert any("Kalido" in doc.page_content for doc in after)
