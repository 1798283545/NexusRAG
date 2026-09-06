"""RAG 链端到端测试。

覆盖「加载 → 分割 → 向量化 → 检索 → 生成」完整流程。
为避免依赖外部 API / 模型下载，测试注入：
1. 确定性 1-hot 伪 Embeddings（含 "LangChain" 的文本与含该词的查询高度相似）；
2. 哑 LLM（invoke / stream 均返回固定文本，无需 OpenAI Key）。
"""

import uuid

from langchain_core.embeddings import Embeddings

from chains import RAGChain
from retrievers import VectorStoreManager


class _KeywordEmbeddings(Embeddings):
    """8 维 1-hot 伪嵌入。

    含 "LangChain" 的文本映射到维度 0，否则映射到维度 1。
    维度 0 与维度 1 正交 → 余弦相似度为 0（可验证低分过滤）。
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        vector = [0.0] * 8
        vector[0 if "LangChain" in text else 1] = 1.0
        return vector


class _FakeLLM:
    """哑 LLM：invoke 返回固定回答，stream 逐 token 产出。"""

    def __init__(self, answer: str = "根据上下文生成的测试回答。", stream_tokens: list[str] | None = None) -> None:
        self._answer = answer
        self._tokens = stream_tokens if stream_tokens is not None else list(answer)

    def invoke(self, messages: list) -> str:
        """忽略消息，直接返回固定回答。"""
        return self._answer

    def stream(self, messages: list):
        """逐 token 产出回答。"""
        yield from self._tokens


class _RaisingLLM:
    """invoke / stream 直接抛错的 LLM（模拟服务不可用）。"""

    def invoke(self, messages: list) -> str:
        raise RuntimeError("mock LLM outage")

    def stream(self, messages: list):
        raise RuntimeError("mock LLM outage")


class _BrokenStreamLLM:
    """流式输出中途抛错的 LLM。"""

    def stream(self, messages: list):
        yield "部分输出"
        raise RuntimeError("mock mid-stream failure")


def _build_chain(tmp_path, answer: str | None = None, llm=None) -> tuple[RAGChain, VectorStoreManager]:
    """构造指向独立临时目录 / 独立集合的完整 RAG 链。"""
    manager = VectorStoreManager(
        collection_name="c" + uuid.uuid4().hex[:10],
        persist_directory=str(tmp_path),
        embeddings=_KeywordEmbeddings(),
    )
    if llm is None:
        llm = _FakeLLM(answer=answer or "根据上下文生成的测试回答。")
    chain = RAGChain(
        manager,
        llm=llm,
        chunk_size=100,
        chunk_overlap=20,
        k=4,
    )
    return chain, manager


def test_full_rag_workflow(tmp_path):
    """完整 RAG 流程：加载 → 分割 → 索引 → 查询 → 删除。"""
    test_file = tmp_path / "test.txt"
    test_file.write_text("LangChain是一个用于构建LLM应用的框架。它提供了链式调用、工具集成等功能。")

    chain, _ = _build_chain(tmp_path)

    # 3. 处理文档
    count = chain.process_document(str(test_file))
    assert count > 0
    assert chain.get_document_status(str(test_file))["exists"] is True

    # 4. 查询
    result = chain.query("LangChain是什么？")
    assert "answer" in result
    assert "source_documents" in result
    assert len(result["source_documents"]) > 0
    # 高分命中：置信度应为 1.0，且回答附带 Sources 溯源
    assert result["confidence"] > 0.5
    assert "Sources:" in result["answer"]

    # 5. 清理
    assert chain.delete_document(str(test_file)) is True
    status = chain.get_document_status(str(test_file))
    assert status["exists"] is False
    assert status["chunk_count"] == 0


def test_process_documents_batch_stats(tmp_path):
    """批量处理：成功文件统计、总块数、失败文件隔离。"""
    file_a = tmp_path / "a.txt"
    file_b = tmp_path / "b.txt"
    bad_file = tmp_path / "unsupported.xyz"
    file_a.write_text("LangChain 支持链式调用与工具集成。")
    file_b.write_text("LangChain 也支持向量存储与检索增强生成。")
    bad_file.write_text("不支持格式")

    chain, _ = _build_chain(tmp_path)
    stats = chain.process_documents([str(file_a), str(file_b), str(bad_file)])

    assert stats["total_files"] == 3
    assert stats["success_files"] == 2
    assert stats["total_chunks"] > 0
    assert len(stats["failed_files"]) == 1


def test_query_rerank_filters_low_similarity(tmp_path):
    """低分检索结果被阈值（默认 0.5）过滤，来源仅保留相关文档。"""
    doc_a = tmp_path / "relevant.txt"
    doc_b = tmp_path / "irrelevant.txt"
    doc_a.write_text("LangChain 是构建 LLM 应用的框架。")
    doc_b.write_text("股票市场今天震荡收跌，与检索主题完全无关。")

    chain, _ = _build_chain(tmp_path)
    chain.process_documents([str(doc_a), str(doc_b)])

    result = chain.query("LangChain 是什么？")
    sources = result["source_documents"]

    assert len(sources) > 0
    # 重排后只保留相关文档（irrelevant.txt 的相似度为 0，被过滤）
    for source in sources:
        assert source["file_name"] == "relevant.txt"
        assert source["score"] > 0.5


def test_stream_query_yields_tokens_then_sources(tmp_path):
    """流式问答：逐 token 产出文本，结束后产出 sources 事件。"""
    test_file = tmp_path / "doc.txt"
    test_file.write_text("LangChain 通过链式调用串联多个组件。")

    chain, _ = _build_chain(tmp_path, answer="这是流式回答。")
    chain.process_document(str(test_file))

    events = list(chain.stream_query("LangChain 是什么？"))
    tokens = [event for event in events if isinstance(event, str)]
    final = events[-1]

    assert len(tokens) > 0
    assert isinstance(final, dict)
    assert final["type"] == "sources"
    assert "answer" in final
    assert len(final["source_documents"]) > 0


def test_query_history_parameter_reserved(tmp_path):
    """history 参数为预留接口：传入时不影响单轮问答结果。"""
    test_file = tmp_path / "doc.txt"
    test_file.write_text("LangChain 提供丰富的组件生态。")

    chain, _ = _build_chain(tmp_path)
    chain.process_document(str(test_file))

    result = chain.query(
        "LangChain 是什么？",
        history=[{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！"}],
    )
    assert "answer" in result
    assert len(result["source_documents"]) > 0


def test_query_llm_failure_falls_back_to_retrieval(tmp_path):
    """LLM 调用失败时 query 应自动降级为纯检索回答并标记 llm_error。"""
    test_file = tmp_path / "doc.txt"
    test_file.write_text("LangChain 提供文档处理与检索问答能力。")

    chain, _ = _build_chain(tmp_path, llm=_RaisingLLM())
    chain.process_document(str(test_file))

    result = chain.query("LangChain 是什么？")

    assert "llm_error" in result
    assert "mock LLM outage" in result["llm_error"]
    assert "已降级为纯检索" in result["answer"]
    assert "Sources:" in result["answer"]
    assert len(result["source_documents"]) > 0


def test_query_with_empty_collection_returns_no_sources(tmp_path):
    """向量库为空时查询：source_documents 为空、confidence 为 0，不抛异常。"""
    chain, _ = _build_chain(tmp_path)

    result = chain.query("LangChain 是什么？")

    assert result["source_documents"] == []
    assert result["confidence"] == 0.0
    assert "answer" in result


def test_delete_then_query_returns_no_sources(tmp_path):
    """删除文档后再次查询：来源应为空，且删除状态可确认。"""
    test_file = tmp_path / "doc.txt"
    test_file.write_text("LangChain 是一套用于构建 LLM 应用的框架。")

    chain, _ = _build_chain(tmp_path)
    chain.process_document(str(test_file))
    assert chain.get_document_status(str(test_file))["exists"] is True

    assert chain.delete_document(str(test_file)) is True
    result = chain.query("LangChain 是什么？")

    assert chain.get_document_status(str(test_file))["exists"] is False
    assert result["source_documents"] == []


def test_process_mixed_format_documents(tmp_path):
    """多文档混合处理：txt + md 同时入库，问答可引用两种来源。"""
    txt_file = tmp_path / "note.txt"
    md_file = tmp_path / "guide.md"
    txt_file.write_text("LangChain 是构建 LLM 应用的核心框架。")
    md_file.write_text("# 指南\nLangChain 提供了丰富的组件与集成生态。")

    chain, _ = _build_chain(tmp_path)
    stats = chain.process_documents([str(txt_file), str(md_file)])

    assert stats["total_files"] == 2
    assert stats["success_files"] == 2
    assert stats["failed_files"] == []

    result = chain.query("LangChain 是什么？")
    file_names = {src["file_name"] for src in result["source_documents"]}
    assert file_names == {"note.txt", "guide.md"}


def test_stream_query_falls_back_on_mid_stream_failure(tmp_path):
    """流式问答中途失败：已产出 token 保留，最终以 error 事件终止。"""
    test_file = tmp_path / "doc.txt"
    test_file.write_text("LangChain 支持流式输出。")

    chain, _ = _build_chain(tmp_path, llm=_BrokenStreamLLM())
    chain.process_document(str(test_file))

    events = list(chain.stream_query("LangChain 是什么？"))
    tokens = [event["content"] for event in events if event["type"] == "token"]
    assert tokens == ["部分输出"]
    last = events[-1]
    assert last["type"] == "error"
    assert "mock mid-stream failure" in last["detail"]
