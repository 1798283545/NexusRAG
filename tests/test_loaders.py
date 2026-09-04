"""文档加载器工厂单元测试。"""

import os

import pytest

from loaders.loader_factory import LoaderError, LoaderFactory, UnsupportedFormatError


def test_load_txt(tmp_path):
    """加载真实存在的 .txt 文件，验证内容与统一元数据。"""
    file_path = tmp_path / "sample.txt"
    file_path.write_text("Hello NexusRAG\n第二行内容", encoding="utf-8")

    docs = LoaderFactory.get_loader(str(file_path)).load()

    assert len(docs) == 1
    doc = docs[0]
    assert "Hello NexusRAG" in doc.page_content
    metadata = doc.metadata
    assert metadata["source"] == os.path.abspath(str(file_path))
    assert metadata["file_name"] == "sample.txt"
    assert metadata["file_type"] == "txt"
    assert metadata["creation_date"]
    assert metadata["modification_date"]


def test_load_pdf(tmp_path):
    """加载 .pdf 文件（用 pypdf 动态生成真实 PDF），验证分页元数据。"""
    pytest.importorskip("pypdf")
    from pypdf import PdfWriter

    file_path = tmp_path / "sample.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.add_blank_page(width=612, height=792)
    with open(file_path, "wb") as f:
        writer.write(f)

    docs = LoaderFactory.get_loader(str(file_path)).load()

    assert len(docs) == 2
    assert docs[0].metadata["file_type"] == "pdf"
    assert [doc.metadata["page_number"] for doc in docs] == [1, 2]


def test_load_docx(tmp_path):
    """加载 .docx 文件（python-docx 动态生成），验证正文内容。"""
    pytest.importorskip("docx")
    from docx import Document as WordDocument

    file_path = tmp_path / "sample.docx"
    word_document = WordDocument()
    word_document.add_paragraph("Hello NexusRAG Word")
    word_document.save(str(file_path))

    docs = LoaderFactory.get_loader(str(file_path)).load()

    assert len(docs) == 1
    assert "Hello NexusRAG Word" in docs[0].page_content
    assert docs[0].metadata["file_type"] == "docx"


def test_load_markdown(tmp_path):
    """加载 .md 文件，验证 Markdown 原文读取。"""
    file_path = tmp_path / "sample.md"
    file_path.write_text("# NexusRAG\n\n这是一个 Markdown 文档。", encoding="utf-8")

    docs = LoaderFactory.get_loader(str(file_path)).load()

    assert len(docs) == 1
    assert "# NexusRAG" in docs[0].page_content
    assert docs[0].metadata["file_type"] == "md"


def test_unsupported_extension(tmp_path):
    """不支持的扩展名（.jpg）应抛出 UnsupportedFormatError。"""
    file_path = tmp_path / "image.jpg"
    file_path.write_bytes(b"fake image bytes")

    with pytest.raises(UnsupportedFormatError):
        LoaderFactory.get_loader(str(file_path))


def test_missing_file(tmp_path):
    """文件不存在应抛出 FileNotFoundError。"""
    missing = tmp_path / "not_exist.pdf"

    with pytest.raises(FileNotFoundError):
        LoaderFactory.get_loader(str(missing))


def test_corrupted_pdf_raises_loader_error(tmp_path):
    """内容非法的 PDF 应抛出 LoaderError。"""
    pytest.importorskip("pypdf")
    file_path = tmp_path / "broken.pdf"
    file_path.write_bytes(b"this is not a pdf")

    loader = LoaderFactory.get_loader(str(file_path))
    with pytest.raises(LoaderError):
        loader.load()


def test_empty_txt_returns_empty(tmp_path):
    """空 TXT 文件应优雅降级为空列表，而非返回空内容文档。"""
    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")

    docs = LoaderFactory.get_loader(str(file_path)).load()

    assert docs == []


def test_empty_markdown_returns_empty(tmp_path):
    """空 Markdown 文件应返回空列表。"""
    file_path = tmp_path / "empty.md"
    file_path.write_text("   \n\n  ", encoding="utf-8")

    docs = LoaderFactory.get_loader(str(file_path)).load()

    assert docs == []


def test_large_txt_file_loaded(tmp_path):
    """超大文件（>10MB）应能完整加载，不丢失内容。"""
    file_path = tmp_path / "large.txt"
    content = "NexusRAG 大文件压测内容行。\n" * 400_000  # 约 12MB
    file_path.write_text(content, encoding="utf-8")

    docs = LoaderFactory.get_loader(str(file_path)).load()

    assert len(docs) >= 1
    # 拼接所有块后应还原完整内容（兼容不同版本 TextLoader 的分块行为）
    assert "".join(doc.page_content for doc in docs) == content
    assert all(doc.metadata["file_name"] == "large.txt" for doc in docs)


def test_metadata_integrity(tmp_path):
    """所有加载器返回的统一元数据应包含完整的标准字段集。"""
    standard_keys = {
        "source",
        "file_name",
        "file_type",
        "page_number",
        "section",
        "creation_date",
        "modification_date",
    }
    file_path = tmp_path / "meta.md"
    file_path.write_text("# 标题\n正文内容。", encoding="utf-8")

    doc = LoaderFactory.get_loader(str(file_path)).load()[0]

    assert standard_keys <= set(doc.metadata.keys())
    assert doc.metadata["file_name"] == "meta.md"
    assert doc.metadata["source"] == os.path.abspath(str(file_path))


def test_invalid_utf8_markdown_raises_loader_error(tmp_path):
    """编码非法的 Markdown 文件应被封装为 LoaderError（而非裸 UnicodeDecodeError）。"""
    file_path = tmp_path / "broken.md"
    file_path.write_bytes(b"\xff\xfe\x00\x01 invalid utf-8 bytes")

    loader = LoaderFactory.get_loader(str(file_path))
    with pytest.raises(LoaderError):
        loader.load()
