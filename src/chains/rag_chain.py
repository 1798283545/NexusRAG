"""最简 RAG 链模块。

串联 Day 2（文档加载）→ Day 3（文本分割）→ Day 4（向量存储）→ LLM 生成，
提供单个/批量文档索引、问答、流式问答与文档状态管理等能力。

Requires: pip install langchain-openai（使用默认 ChatOpenAI 时需要）
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Generator, List, Optional, Tuple

from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel

from config import settings
from loaders import LoaderFactory
from loaders import split_documents
from retrievers import VectorStoreManager

logger = logging.getLogger(__name__)

#: 问答 Prompt 模板（带检索上下文）
QA_PROMPT_TEMPLATE: str = """你是一个专业的文档分析助手。请基于以下上下文回答用户的问题。

<上下文>
{context}
</上下文>

用户问题：{question}

要求：
1. 如果上下文不足以回答问题，请明确告知"根据现有文档无法回答此问题"，不要编造信息。
2. 如果回答中引用了具体文档，请引用来源（文件名称或页码）。
3. 回答要简洁、准确、有条理。

回答："""

#: 摘要 Prompt 模板（预留接口，文档处理后的摘要生成暂未启用）
SUMMARY_PROMPT_TEMPLATE: str = ""

#: 低分检索结果过滤阈值（重排）
_RERANK_THRESHOLD: float = 0.5


class RAGChain:
    """最简 RAG 问答链。

    封装完整的 RAG 流程：文档加载 → 分割 → 向量化入库 → 检索 → 生成。

    Attributes:
        vector_store_manager: 向量存储管理器（Day 4 实现）。
        k: 默认检索返回的 Top-K 文档数。
        chunk_strategy / chunk_size / chunk_overlap: 文档分割参数。
        rerank_threshold: 检索结果低分过滤阈值。
    """

    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        llm: Optional[BaseLanguageModel] = None,
        embedding_model: Optional[str] = None,
        k: int = 4,
        chunk_strategy: str = "recursive",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> None:
        """初始化 RAG 链。

        Args:
            vector_store_manager: 向量存储管理器（Day 4 实现）。
            llm: LLM 实例。默认使用 ChatOpenAI，模型与密钥从环境变量读取；
                也可传入任意实现了 ``invoke`` / ``stream`` 的对象（便于测试）。
            embedding_model: 可选的语义分割用嵌入模型名。
            k: 默认检索返回的文档数量。
            chunk_strategy: 文档分割策略（recursive / semantic / markdown）。
            chunk_size: 分割目标块大小（字符数）。
            chunk_overlap: 分割块重叠字符数。

        Raises:
            TypeError: vector_store_manager 不是 VectorStoreManager 实例。
        """
        if not isinstance(vector_store_manager, VectorStoreManager):
            raise TypeError("vector_store_manager 必须是 VectorStoreManager 实例")
        self.vector_store_manager: VectorStoreManager = vector_store_manager
        self.embedding_model: Optional[str] = embedding_model
        self.k: int = int(k or settings.DEFAULT_K)
        self.chunk_strategy: str = chunk_strategy or settings.DEFAULT_STRATEGY
        self.chunk_size: int = int(chunk_size or settings.DEFAULT_CHUNK_SIZE)
        self.chunk_overlap: int = int(chunk_overlap or settings.DEFAULT_CHUNK_OVERLAP)
        self.rerank_threshold: float = _RERANK_THRESHOLD
        self._llm: Optional[BaseLanguageModel] = llm
        self._llm_cached: Optional[BaseLanguageModel] = None
        self._qa_prompt = ChatPromptTemplate.from_template(QA_PROMPT_TEMPLATE)
        logger.info(
            "RAGChain 初始化完成（collection=%s, strategy=%s, k=%d）",
            vector_store_manager.collection_name,
            self.chunk_strategy,
            self.k,
        )

    # ------------------------------------------------------------------ #
    # LLM 解析
    # ------------------------------------------------------------------ #
    def _get_llm(self) -> BaseLanguageModel:
        """获取 LLM 实例（惰性构建默认 ChatOpenAI，并缓存）。"""
        if self._llm is not None:
            return self._llm
        if self._llm_cached is not None:
            return self._llm_cached
        if not settings.OPENAI_API_KEY:
            raise ValueError(
                "缺少 OPENAI_API_KEY 配置（请检查 .env），且构造时未注入 llm 实例"
            )
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError(
                "未安装 langchain-openai，请执行: pip install langchain-openai"
            ) from exc
        self._llm_cached = ChatOpenAI(
            model=settings.MODEL_NAME,
            temperature=settings.TEMPERATURE,
            max_tokens=settings.MAX_TOKENS,
            api_key=settings.OPENAI_API_KEY,
        )
        return self._llm_cached

    @staticmethod
    def _as_text(response: Any) -> str:
        """将 LLM 返回（字符串 / AIMessage / chunk 列表）归一化为文本。"""
        if isinstance(response, str):
            return response
        content = getattr(response, "content", response)
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and "text" in item:
                    parts.append(str(item["text"]))
            return "".join(parts)
        return str(content)

    @staticmethod
    def _fallback_text(context: str) -> str:
        """构造 LLM 不可用时的纯检索降级回答文本。"""
        if not context or context == "（未检索到相关文档内容）":
            return "根据现有文档无法回答此问题。"
        return (
            "（LLM 服务暂不可用，已降级为纯检索结果，以下为检索到的相关文档片段，请自行参考）\n\n"
            f"{context}"
        )

    def _generate(self, context: str, query: str) -> Tuple[str, Optional[str]]:
        """调用 LLM 生成回答；失败时自动降级为纯检索文本。

        Args:
            context: 检索组装得到的上下文文本。
            query: 用户问题。

        Returns:
            ``(回答文本, llm_error)`` 二元组；正常时为 ``(文本, None)``，
            LLM 调用失败时为 ``(纯检索降级文本, 错误信息字符串)``。
        """
        try:
            llm = self._get_llm()
            messages = self._qa_prompt.format_messages(context=context, question=query)
            raw = llm.invoke(messages)
        except Exception as exc:  # noqa: BLE001 - 失败自动降级为纯检索
            logger.error("LLM 生成失败，自动降级为纯检索回答: %s", exc)
            return self._fallback_text(context), str(exc)
        return self._as_text(raw).strip(), None

    # ------------------------------------------------------------------ #
    # 文档处理（索引）
    # ------------------------------------------------------------------ #
    def process_document(self, file_path: str) -> int:
        """加载并索引单个文档。

        执行：LoaderFactory 加载 → split_documents 分割 → add_documents 入库。

        Args:
            file_path: 文档路径。

        Returns:
            入库的文档块数量（>0 表示成功）。

        Raises:
            FileNotFoundError: 文件不存在。
            RuntimeError: 加载 / 分割 / 入库任一环节失败。
        """
        canonical = os.path.abspath(file_path)
        logger.info("开始处理文档: %s", canonical)

        try:
            documents = LoaderFactory.get_loader(canonical).load()
        except FileNotFoundError:
            # 保持文档约定的语义：文件不存在直接上抛，便于上层精确处理
            raise
        except Exception as exc:  # noqa: BLE001 - 其余错误包装为清晰异常
            logger.error("文档加载失败: %s（%s）", canonical, exc)
            raise RuntimeError(f"文档加载失败: {canonical}: {exc}") from exc

        split_kwargs: Dict[str, Any] = {}
        if self.chunk_strategy == "semantic" and self.embedding_model:
            split_kwargs["embedding_model"] = self.embedding_model
        try:
            chunks = split_documents(
                documents,
                strategy=self.chunk_strategy,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                **split_kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("文档分割失败: %s（%s）", canonical, exc)
            raise RuntimeError(f"文档分割失败: {canonical}: {exc}") from exc

        try:
            added = self.vector_store_manager.add_documents(chunks)
        except Exception as exc:  # noqa: BLE001
            logger.error("向量入库失败: %s（%s）", canonical, exc)
            raise RuntimeError(f"向量入库失败: {canonical}: {exc}") from exc

        logger.info("文档处理完成: %s，共 %d 个文本块", canonical, len(added))
        return len(added)

    def process_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """批量处理文档。

        Args:
            file_paths: 文档路径列表。

        Returns:
            统计字典：``total_files``（文件总数）、``success_files``（成功数）、
            ``total_chunks``（总块数）、``failed_files``（失败文件列表）。
        """
        total_chunks = 0
        failed_files: List[str] = []
        for file_path in file_paths:
            try:
                total_chunks += self.process_document(file_path)
            except Exception as exc:  # noqa: BLE001 - 单个失败不中断批量
                logger.error("批量处理中断于 %s: %s", file_path, exc)
                failed_files.append(file_path)
        success_files = len(file_paths) - len(failed_files)
        logger.info(
            "批量处理完成: 共 %d 个文件，成功 %d 个，入库 %d 块",
            len(file_paths),
            success_files,
            total_chunks,
        )
        return {
            "total_files": len(file_paths),
            "success_files": success_files,
            "total_chunks": total_chunks,
            "failed_files": failed_files,
        }

    # ------------------------------------------------------------------ #
    # 文档状态管理
    # ------------------------------------------------------------------ #
    def delete_document(self, file_path: str) -> bool:
        """删除某文档对应的全部向量块（按 source 元数据匹配）。

        Args:
            file_path: 文档路径。

        Returns:
            是否存在被删除的向量块（True 表示删除了 ≥1 块）。
        """
        canonical = os.path.abspath(file_path)
        try:
            deleted = self.vector_store_manager.delete_by_filter({"source": canonical})
        except Exception as exc:  # noqa: BLE001
            logger.error("删除文档向量失败: %s（%s）", canonical, exc)
            return False
        logger.info("文档向量删除完成: %s，删除 %d 块", canonical, deleted)
        return deleted > 0

    def get_document_status(self, file_path: str) -> Dict[str, Any]:
        """查询文档的索引状态。

        Args:
            file_path: 文档路径。

        Returns:
            ``{"exists": bool, "chunk_count": int}`` 字典。
        """
        canonical = os.path.abspath(file_path)
        chunk_count = self.vector_store_manager.count_by_metadata({"source": canonical})
        return {"exists": chunk_count > 0, "chunk_count": chunk_count}

    # ------------------------------------------------------------------ #
    # 检索与生成
    # ------------------------------------------------------------------ #
    def _retrieve(
        self,
        query: str,
        k: int,
        filter: Optional[Dict[str, Any]],
        threshold: Optional[float],
    ) -> List[Tuple[Document, float]]:
        """检索 Top-K 并按相似度阈值重排过滤。"""
        try:
            scored = self.vector_store_manager.similarity_search_with_score(
                query, k=k, filter=filter
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("向量检索失败: %s（%s）", query, exc)
            raise RuntimeError(f"向量检索失败: {exc}") from exc
        cutoff = self.rerank_threshold if threshold is None else threshold
        kept = [(doc, score) for doc, score in scored if score > cutoff]
        logger.info(
            "检索完成: 召回 %d 条，重排后保留 %d 条（阈值=%s）",
            len(scored),
            len(kept),
            cutoff,
        )
        return kept

    def _prepare(self, query: str, k: Optional[int], filter: Optional[Dict[str, Any]],
                 threshold: Optional[float]) -> Tuple[str, List[Dict[str, Any]], List[Tuple[Document, float]]]:
        """内部：检索 → 组装上下文 → 生成来源清单（query/stream 共用）。"""
        documents = self._retrieve(query, k if k is not None else self.k, filter, threshold)
        sources: List[Dict[str, Any]] = []
        context_parts: List[str] = []
        for index, (doc, score) in enumerate(documents, start=1):
            context_parts.append(f"[{index}] {doc.page_content}")
            file_name = doc.metadata.get("file_name", os.path.basename(str(doc.metadata.get("source", ""))))
            page_number = doc.metadata.get("page_number")
            sources.append(
                {
                    "file_name": file_name,
                    "page_number": page_number,
                    "source": doc.metadata.get("source"),
                    "score": score,
                    "content": doc.page_content,
                }
            )
        context = "\n\n".join(context_parts) if context_parts else "（未检索到相关文档内容）"
        return context, sources, documents

    def _attach_sources(self, answer: str, sources: List[Dict[str, Any]]) -> str:
        """在回答末尾附加 Sources 溯源信息。"""
        if not sources:
            return answer
        seen = set()
        lines: List[str] = []
        for src in sources:
            key = (str(src["file_name"]), src["page_number"])
            if key in seen:
                continue
            seen.add(key)
            if src["page_number"] is not None:
                lines.append(f"- {src['file_name']}（第 {src['page_number']} 页）")
            else:
                lines.append(f"- {src['file_name']}")
        return f"{answer}\n\nSources:\n" + "\n".join(lines)

    def query(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
        threshold: Optional[float] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """执行一次 RAG 问答。

        Args:
            query: 用户问题。
            k: 检索条数（默认使用实例 k）。
            filter: 可选的元数据过滤条件。
            threshold: 低分过滤阈值（默认 0.5）。
            history: 对话历史（预留接口，Day 6 多轮对话接入）。

        Returns:
            包含 ``answer``（含 Sources 溯源）、``source_documents``
            （来源文档列表）与 ``confidence``（平均相似度置信度）的字典；
            LLM 调用失败时会自动降级为纯检索回答，并附带 ``llm_error`` 字段。

        Raises:
            RuntimeError: 检索阶段失败（LLM 生成失败不抛出，自动降级）。
        """
        if history:
            logger.debug("已接收 %d 条对话历史（多轮记忆将在 Day 6 接入）", len(history))
        context, sources, _ = self._prepare(query, k, filter, threshold)
        answer_text, llm_error = self._generate(context, query)

        answer = self._attach_sources(answer_text, sources)
        confidence = (
            float(sum(src["score"] for src in sources) / len(sources)) if sources else 0.0
        )
        logger.info("问答完成: 引用 %d 个来源，置信度=%.3f", len(sources), confidence)
        result: Dict[str, Any] = {
            "answer": answer,
            "source_documents": sources,
            "confidence": confidence,
        }
        if llm_error is not None:
            result["llm_error"] = llm_error
        return result

    @staticmethod
    def _serialize_source_meta(doc: Document, score: float) -> Dict[str, Any]:
        """将单条检索结果的 metadata 归一化为可 JSON 序列化的来源信息。"""
        meta = dict(doc.metadata or {})
        file_name = meta.get("file_name") or os.path.basename(
            str(meta.get("source", "") or "")
        )
        if file_name:
            meta.setdefault("file_name", file_name)
        meta["score"] = round(float(score), 4)
        return meta

    def stream_query(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
        threshold: Optional[float] = None,
        history: Optional[List[Dict[str, str]]] = None,
        yield_sources: bool = True,
        yield_thinking: bool = True,
    ) -> Generator[Dict[str, Any], None, None]:
        """流式 RAG 问答（SSE 事件源，支持打字机效果与元信息）。

        依次产出可直接序列化为 SSE ``data:`` 帧的事件字典：

        - ``{"type": "thinking", "content": str, "step": int}``：检索 / 生成等
          阶段提示（``yield_thinking=False`` 时跳过）；
        - ``{"type": "sources", "sources": [{content, metadata}]}``：生成前
          发送的检索来源（``yield_sources=False`` 时跳过）；
        - ``{"type": "token", "content": str}``：每个流式文本块（打字机）；
        - ``{"type": "done", "total_tokens": int, "execution_time": float}``
          或 ``{"type": "error", "error": str, "detail": str}``：终止事件。

        失败降级语义：

        - 检索失败：发送 thinking 说明后，仍以空来源继续调用 LLM 生成；
        - LLM 中途失败：先产出已生成的 token，再发送 ``error`` 事件终止。

        Args:
            query: 用户问题。
            k: 检索条数（默认使用实例 k）。
            filter: 可选的元数据过滤条件。
            threshold: 低分过滤阈值（默认 0.5）。
            history: 对话历史（预留接口，多轮记忆由上层注入后透传）。
            yield_sources: 是否产出 sources 事件。
            yield_thinking: 是否产出 thinking 阶段提示事件。

        Yields:
            事件字典（键与 SSE ``data:`` 帧保持一致）。

        Note:
            ``execution_time`` 只累计生成器真正执行的耗时，自动剔除流式传输中
            客户端打字延迟 / 网络等待造成的挂起时间。
        """
        if history:
            logger.debug("已接收 %d 条对话历史（多轮记忆由上层注入）", len(history))

        # 计时基准：任何一次 yield 恢复后都重设，从而把客户端等待时间剔除
        step = 0
        active = 0.0
        _resume = [time.perf_counter()]

        def _acc() -> float:
            """返回自上次恢复（或生成器启动）以来实际执行的秒数。"""
            return time.perf_counter() - _resume[0]

        if yield_thinking:
            step += 1
            active += _acc()
            yield {"type": "thinking", "content": "正在检索相关文档…", "step": step}
            _resume[0] = time.perf_counter()

        context = "（未检索到相关文档内容）"
        sources: List[Dict[str, Any]] = []
        documents: List[Tuple[Document, float]] = []
        try:
            context, sources, documents = self._prepare(
                query, k if k is not None else self.k, filter, threshold
            )
        except Exception as exc:  # noqa: BLE001 - 检索失败降级为无上下文生成
            logger.exception("流式检索失败，尝试无上下文直接生成: %s", exc)
            if yield_thinking:
                step += 1
                active += _acc()
                yield {
                    "type": "thinking",
                    "content": f"检索失败（{exc}），将尝试直接生成回答（来源为空）。",
                    "step": step,
                }
                _resume[0] = time.perf_counter()

        if yield_sources:
            payload = [
                {
                    "content": doc.page_content,
                    "metadata": self._serialize_source_meta(doc, score),
                }
                for doc, score in documents
            ]
            active += _acc()
            yield {"type": "sources", "sources": payload}
            _resume[0] = time.perf_counter()

        tokens = 0
        try:
            llm = self._get_llm()
            messages = self._qa_prompt.format_messages(context=context, question=query)
            if yield_thinking:
                step += 1
                active += _acc()
                yield {
                    "type": "thinking",
                    "content": (
                        f"已获取 {len(sources)} 个相关文档，正在调用大模型生成回答…"
                        if sources
                        else "未检索到相关文档，正在调用大模型尝试回答…"
                    ),
                    "step": step,
                }
                _resume[0] = time.perf_counter()
            for chunk in llm.stream(messages):
                token = self._as_text(chunk)
                if not token:
                    continue
                tokens += 1
                active += _acc()
                yield {"type": "token", "content": token}
                _resume[0] = time.perf_counter()
        except Exception as exc:  # noqa: BLE001 - 流式中断以 error 事件终止
            logger.error("流式 LLM 生成失败: %s", exc)
            active += _acc()
            yield {
                "type": "error",
                "error": "llm_stream_error",
                "detail": f"{exc}（已流式输出 {tokens} 个 token）",
            }
            _resume[0] = time.perf_counter()
            return

        active += _acc()
        yield {
            "type": "done",
            "total_tokens": tokens,
            "execution_time": round(active, 4),
        }
