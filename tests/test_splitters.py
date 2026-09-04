"""文本分块策略单元测试。"""

import pytest
from langchain_core.documents import Document

from loaders.splitters import split_documents


def _make_document(text: str, file_name: str = "sample.txt") -> Document:
    """构造带统一元数据的测试文档。"""
    return Document(
        page_content=text,
        metadata={
            "source": f"/tmp/{file_name}",
            "file_name": file_name,
            "file_type": "txt",
            "page_number": None,
        },
    )


def test_recursive_splits_long_text_into_multiple_chunks():
    """使用 recursive 策略分割长文本，返回的块数应大于 1。"""
    # 约 1400 字符、无分隔符的中文文本
    doc = _make_document("测试文档内容。" * 200)
    docs = split_documents([doc], strategy="recursive", chunk_size=500, chunk_overlap=50)

    assert len(docs) > 1
    assert all(d.metadata["chunk_strategy"] == "recursive" for d in docs)
    # 块大小接近目标 chunk_size（允许保留分隔符导致的少量上浮）
    assert all(d.metadata["chunk_size"] <= 600 for d in docs)


def test_recursive_chunks_carry_index_and_source():
    """每个子块都应包含 chunk_index / chunk_total / source 等元数据。"""
    doc = _make_document("NexusRAG 是一个多智能体 RAG 平台。" * 100)
    docs = split_documents([doc], strategy="recursive", chunk_size=200, chunk_overlap=20)

    assert len(docs) > 1
    for chunk in docs:
        metadata = chunk.metadata
        assert "source" in metadata
        assert isinstance(metadata["chunk_index"], int)
        assert metadata["chunk_total"] == len(docs)
        assert metadata["chunk_strategy"] == "recursive"
        assert metadata["parent_source"] == metadata["source"]


def test_semantic_strategy_runs_with_mocked_chunker(monkeypatch):
    """semantic 策略在依赖被 mock 的情况下应正常执行、不抛异常。"""
    import loaders.splitters as splitters

    # 1) 让依赖检查通过
    monkeypatch.setattr(splitters, "_semantic_dependencies_ready", lambda: True)

    # 2) 用假 chunker 替换真实的 SemanticChunker 构造
    class _FakeChunker:
        def split_text(self, text: str) -> list[str]:  # noqa: ARG002
            return ["这是语义分割后的第一段。", "这是语义分割后的第二段。"]

    monkeypatch.setattr(splitters, "_create_semantic_chunker", lambda **kw: _FakeChunker())

    doc = _make_document("第一句内容。第二句内容。第三句内容。")
    docs = split_documents(
        [doc],
        strategy="semantic",
        chunk_size=500,
        chunk_overlap=50,
        embedding_model="all-MiniLM-L6-v2",
    )

    assert len(docs) == 2
    assert all(d.metadata["chunk_strategy"] == "semantic" for d in docs)
    assert [d.metadata["chunk_index"] for d in docs] == [0, 1]


def test_semantic_degrades_to_recursive_without_dependencies(monkeypatch):
    """未安装 sentence-transformers 时，semantic 应自动降级为 recursive。"""
    import loaders.splitters as splitters

    monkeypatch.setattr(splitters, "_semantic_dependencies_ready", lambda: False)

    doc = _make_document("降级测试内容。" * 200)
    docs = split_documents([doc], strategy="semantic", chunk_size=500, chunk_overlap=50)

    assert len(docs) > 0
    assert all(d.metadata["chunk_strategy"] == "recursive" for d in docs)


def test_markdown_strategy_splits_by_headers(tmp_path):
    """Markdown 文档应按标题结构切分，并保留标题层级元数据。"""
    md_text = (
        "# 第一章\n\n第一章正文内容，用于验证 Markdown 结构分割。\n\n"
        "## 1.1 小节\n\n小节正文。\n\n"
        "# 第二章\n\n第二章正文。\n"
    )
    doc = _make_document(md_text, file_name="guide.md")
    doc.metadata["file_type"] = "md"

    docs = split_documents([doc], strategy="markdown", chunk_size=500, chunk_overlap=50)

    assert len(docs) >= 2
    assert all(d.metadata["chunk_strategy"] == "markdown" for d in docs)
    assert any("H1" in d.metadata for d in docs)


def test_markdown_degrades_for_non_md_documents():
    """非 Markdown 文档强行选择 markdown 策略时应降级为 recursive。"""
    doc = _make_document("普通文本内容。" * 200)  # file_type = txt

    docs = split_documents([doc], strategy="markdown", chunk_size=500, chunk_overlap=50)

    assert len(docs) > 0
    assert all(d.metadata["chunk_strategy"] == "recursive" for d in docs)


def test_unknown_strategy_raises_value_error():
    """未知策略名应抛出 ValueError。"""
    doc = _make_document("内容")

    with pytest.raises(ValueError, match="未知的分块策略"):
        split_documents([doc], strategy="char-level")


def test_short_single_sentence_stays_one_chunk():
    """单句短文本无需切分，应返回恰好 1 个块。"""
    doc = _make_document("一句话内容，不足以触发切分。")

    docs = split_documents([doc], strategy="recursive", chunk_size=500, chunk_overlap=50)

    assert len(docs) == 1
    assert docs[0].page_content == "一句话内容，不足以触发切分。"
    assert docs[0].metadata["chunk_index"] == 0
    assert docs[0].metadata["chunk_total"] == 1


def test_long_code_block_is_split():
    """纯代码/无自然语言标点的超长文本仍应按行递归切分。"""
    code = "def func():  # 注释行\n    return 1\n" * 150  # 约 3000 字符
    doc = _make_document(code)

    docs = split_documents([doc], strategy="recursive", chunk_size=500, chunk_overlap=50)

    assert len(docs) > 1
    assert all(d.page_content.strip() for d in docs)
    # 还原验证：按块内容累加不应丢失文本（重叠不计，仅验证非空）
    assert sum(len(d.page_content) for d in docs) >= len(code)


def test_recursive_chunks_inherit_original_metadata():
    """子块应继承原始文档的全部自定义元数据（如 author），并新增分块字段。"""
    doc = _make_document("NexusRAG 元数据继承验证。" * 80)
    doc.metadata["author"] = "nexus-team"
    doc.metadata["custom_key"] = 42

    docs = split_documents([doc], strategy="recursive", chunk_size=200, chunk_overlap=20)

    assert len(docs) > 1
    for chunk in docs:
        assert chunk.metadata["author"] == "nexus-team"
        assert chunk.metadata["custom_key"] == 42
        assert chunk.metadata["source"] == doc.metadata["source"]
        assert isinstance(chunk.metadata["chunk_index"], int)


def test_empty_documents_returns_empty():
    """空文档列表应直接返回空结果，不抛异常。"""
    assert split_documents([], strategy="recursive") == []


def test_whitespace_only_documents_returns_empty():
    """仅含空白字符的文档应被跳过并返回空结果。"""
    doc = _make_document("   \n\n  ")

    docs = split_documents([doc], strategy="recursive")

    assert docs == []


def test_markdown_long_section_is_sub_split(tmp_path):
    """Markdown 中单个标题下的超长内容应触发递归二次切分。"""
    long_body = ("长段落正文内容，用于测试二次切分。" * 200) + "\n"
    md_text = f"# 大章节\n\n{long_body}\n"
    doc = _make_document(md_text, file_name="long.md")
    doc.metadata["file_type"] = "md"

    docs = split_documents([doc], strategy="markdown", chunk_size=300, chunk_overlap=30)

    assert len(docs) > 1
    assert all(d.metadata["chunk_strategy"] == "markdown" for d in docs)
    assert any("H1" in d.metadata for d in docs)
