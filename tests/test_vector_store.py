"""ChromaDB 向量存储 CRUD 单元测试。

使用本地持久化模式 + 确定性假 Embeddings（避免真实网络/模型下载），
每个用例使用独立的 collection 名与临时目录。
"""

import hashlib
import uuid

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from retrievers import VectorStoreManager


class _FakeEmbeddings(Embeddings):
    """确定性伪 Embeddings：相同文本 → 相同向量（用于相似度验证）。"""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        digest = hashlib.md5(text.encode("utf-8")).digest()
        return [((digest[i] % 255) / 255.0) * 2 - 1 for i in range(8)]


def _manager(tmp_path, collection_name: str | None = None) -> VectorStoreManager:
    """构造指向独立临时目录 / 独立集合的向量存储管理器。"""
    name = collection_name or ("c" + uuid.uuid4().hex[:10])
    return VectorStoreManager(
        collection_name=name,
        persist_directory=str(tmp_path),
        embeddings=_FakeEmbeddings(),
    )


def _doc(text: str, file_name: str = "test.txt", **extra) -> Document:
    """构造测试文档。"""
    return Document(
        page_content=text,
        metadata={
            "source": f"/tmp/{file_name}",
            "file_name": file_name,
            "file_type": "txt",
            "page_number": None,
            **extra,
        },
    )


def test_init_local_mode_and_empty_stats(tmp_path):
    """本地模式初始化成功，空集合统计信息正确。"""
    manager = _manager(tmp_path)

    stats = manager.get_collection_stats()

    assert stats["collection_name"] == manager.collection_name
    assert stats["document_count"] == 0
    assert stats["embedding_dimension"] is None


def test_add_single_document_increases_count(tmp_path):
    """添加单个文档后 count 增加；ID 复用元数据 id 或自动生成。"""
    manager = _manager(tmp_path)
    doc_id = "doc_single_001"

    added = manager.add_documents([_doc("NexusRAG 文档加载器测试内容。", id=doc_id)])

    assert added == [doc_id]
    assert manager.get_collection_stats()["document_count"] == 1

    # 不提供 id 时自动生成 32 位十六进制 ID
    auto = manager.add_documents([_doc("自动生成 ID 的内容")])
    assert len(auto[0]) == 32
    assert manager.get_collection_stats()["document_count"] == 2


def test_batch_add_many_documents(tmp_path):
    """批量添加 >=10 条文档，全部写入成功。"""
    manager = _manager(tmp_path)
    documents = [_doc(f"批量文档第 {i} 条内容。", file_name="batch.txt") for i in range(12)]

    added = manager.add_documents(documents)

    assert len(added) == 12
    stats = manager.get_collection_stats()
    assert stats["document_count"] == 12
    assert stats["embedding_dimension"] == 8


def test_delete_documents_by_id(tmp_path):
    """按 ID 删除文档。"""
    manager = _manager(tmp_path)
    first = manager.add_documents([_doc("第一条内容")])
    second = manager.add_documents([_doc("第二条内容")])
    assert manager.get_collection_stats()["document_count"] == 2

    result = manager.delete_documents(first)

    assert result is True
    assert manager.get_collection_stats()["document_count"] == 1
    remain = manager.similarity_search("第二条内容", k=1)
    assert remain[0].metadata["file_name"] == "test.txt"
    assert second[0] != first[0]


def test_delete_by_filter(tmp_path):
    """按元数据条件删除（file_name=test.txt）。"""
    manager = _manager(tmp_path)
    manager.add_documents([_doc(f"目标文档 {i}", file_name="test.txt") for i in range(3)])
    manager.add_documents([_doc(f"其他文档 {i}", file_name="other.txt") for i in range(2)])

    deleted = manager.delete_by_filter({"file_name": "test.txt"})

    assert deleted == 3
    assert manager.get_collection_stats()["document_count"] == 2
    remain = manager.similarity_search("其他文档", k=5)
    assert all(doc.metadata["file_name"] == "other.txt" for doc in remain)


def test_similarity_search_returns_results(tmp_path):
    """相似度检索应返回匹配文档，精确匹配排最前。"""
    manager = _manager(tmp_path)
    text = "向量数据库用于相似度检索与召回。"
    manager.add_documents(
        [
            _doc(text, file_name="a.txt"),
            _doc("完全无关的另一段内容。", file_name="b.txt"),
        ]
    )

    results = manager.similarity_search(text, k=2)

    assert len(results) >= 1
    assert results[0].page_content == text
    assert results[0].metadata["file_name"] == "a.txt"

    scored = manager.similarity_search_with_score(text, k=2)
    assert isinstance(scored[0], tuple)
    doc, score = scored[0]
    assert doc.page_content == text
    assert 0.0 <= score <= 1.0


def test_update_document_changes_content(tmp_path):
    """更新文档后，检索内容应反映新文本。"""
    manager = _manager(tmp_path)
    doc_id = "doc_to_update"
    manager.add_documents([_doc("旧版本内容描述。", id=doc_id)])

    ok = manager.update_document(
        doc_id,
        "更新后的全新内容说明。",
        {"file_name": "updated.txt"},
    )

    assert ok is True
    assert manager.get_collection_stats()["document_count"] == 1
    results = manager.similarity_search("更新后的全新内容说明。", k=1)
    assert results[0].page_content == "更新后的全新内容说明。"
    assert results[0].metadata["file_name"] == "updated.txt"


def test_similarity_search_with_threshold(tmp_path):
    """阈值检索：精确命中（相似度≈1）返回，明显不相关内容被过滤。"""
    manager = _manager(tmp_path)
    text = "只有完全一致的内容才会命中高阈值。"
    manager.add_documents([_doc(text)])

    hits = manager.similarity_search_with_threshold(text, k=4, threshold=0.99)
    assert any(doc.page_content == text for doc in hits)

    unrelated = manager.similarity_search_with_threshold(
        "与之毫无关联的查询词xyz", k=4, threshold=0.99
    )
    assert len(unrelated) == 0


def test_nested_dict_metadata_is_json_serialized(tmp_path):
    """嵌套 dict 元数据应自动序列化为 JSON 字符串后入库。"""
    manager = _manager(tmp_path)
    manager.add_documents(
        [_doc("嵌套元数据测试。", file_name="nested.txt", extra={"tags": ["a", "b"], "n": 1})]
    )

    results = manager.similarity_search("嵌套元数据测试。", k=1)

    assert len(results) == 1
    raw = results[0].metadata["extra"]
    assert isinstance(raw, str)
    assert '"tags"' in raw


def test_batch_add_200_documents(tmp_path):
    """一次批量添加 200 条文档，全部写入且计数正确。"""
    manager = _manager(tmp_path)
    documents = [_doc(f"批量文档第 {i} 条。", file_name="big_batch.txt") for i in range(200)]

    added = manager.add_documents(documents)

    assert len(added) == 200
    assert len(set(added)) == 200  # 无重复 ID
    assert manager.get_collection_stats()["document_count"] == 200


def test_add_texts_returns_ids_and_increments_count(tmp_path):
    """add_texts 应返回等长 ID 列表并正确入库。"""
    manager = _manager(tmp_path)
    texts = ["第一段文本", "第二段文本"]
    metadatas = [{"file_name": "a.txt"}, {"file_name": "b.txt"}]

    ids = manager.add_texts(texts, metadatas=metadatas)

    assert len(ids) == 2
    assert manager.get_collection_stats()["document_count"] == 2
    results = manager.similarity_search("第二段文本", k=1)
    assert results[0].metadata["file_name"] == "b.txt"


def test_add_empty_list_returns_empty(tmp_path):
    """空文档列表应返回空 ID 列表，不抛异常。"""
    manager = _manager(tmp_path)

    assert manager.add_documents([]) == []
    assert manager.get_collection_stats()["document_count"] == 0


def test_delete_missing_id_returns_true(tmp_path):
    """删除不存在的 ID 应返回 True 且不影响集合。"""
    manager = _manager(tmp_path)
    manager.add_documents([_doc("存在的文档")])

    ok = manager.delete_documents(["no_such_id_000"])

    assert ok is True
    assert manager.get_collection_stats()["document_count"] == 1


def test_clear_collection_then_reuse(tmp_path):
    """clear_collection 清空后集合仍可正常复用与检索。"""
    manager = _manager(tmp_path)
    manager.add_documents([_doc(f"文档 {i}") for i in range(5)])
    assert manager.get_collection_stats()["document_count"] == 5

    cleared = manager.clear_collection()
    assert cleared is True
    assert manager.get_collection_stats()["document_count"] == 0

    # 清空后重建：再次添加与检索应正常工作
    manager.add_documents([_doc("清空后重新写入的内容")])
    assert manager.get_collection_stats()["document_count"] == 1
    results = manager.similarity_search("清空后重新写入的内容", k=1)
    assert len(results) == 1
