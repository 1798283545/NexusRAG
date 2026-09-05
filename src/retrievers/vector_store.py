"""ChromaDB 向量存储封装模块。

提供 :class:`VectorStoreManager`，对 ChromaDB 进行文档向量的增、删、改、查
（CRUD）完整封装，支持本地持久化模式与远程服务模式两种部署形态。
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config import settings

logger = logging.getLogger(__name__)

#: 单次批量提交的最大条数（防止内存溢出）
_BATCH_SIZE: int = 1000

#: ChromaDB 元数据仅支持的类型集合
_SCALAR_TYPES: Tuple[type, ...] = (str, int, float, bool)


class VectorStoreManager:
    """ChromaDB 向量存储管理器（单例）。

    封装文档向量的增、删、改、查操作。支持两种连接模式:

    1. **本地持久化模式**: 指定 ``persist_directory``（默认 ``./chroma_data``）；
    2. **远程服务模式**: 指定 ``host`` + ``port``（例如 docker 中的 chromadb）。

    若同时指定 ``persist_directory`` 与 ``host``/``port``，优先使用
    ``host`` + ``port``。首次创建时会自动 ``get_or_create_collection``，
    并用 ``collection.count()`` 做连接健康检查。

    同一个 ``collection_name`` 在整个进程中只初始化一次（类变量缓存单例）。
    """

    #: collection_name -> 已创建的实例（单例缓存）
    _instances: Dict[str, "VectorStoreManager"] = {}

    def __new__(
        cls,
        collection_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
        persist_directory: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        embeddings: Optional[Embeddings] = None,
    ) -> "VectorStoreManager":
        """按 collection_name 复用单例。"""
        name = collection_name or settings.COLLECTION_NAME
        if name in cls._instances:
            logger.warning(
                "集合 %r 已存在 VectorStoreManager 单例，直接复用（忽略本次其余参数）",
                name,
            )
            return cls._instances[name]
        instance = super().__new__(cls)
        cls._instances[name] = instance
        return instance

    def __init__(
        self,
        collection_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
        persist_directory: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        embeddings: Optional[Embeddings] = None,
    ) -> None:
        """初始化向量存储管理器。

        未显式传入的参数将从全局配置（``config.settings``）读取默认值，
        以支持通过 .env 文件统一调整，避免硬编码。

        Args:
            collection_name: ChromaDB 集合名，默认取配置 ``COLLECTION_NAME``。
            embedding_model: Embedding 模型名。``text-embedding-*`` 前缀使用
                OpenAIEmbeddings，其余（如 ``BAAI/bge-small-zh-v1.5``）使用
                HuggingFaceEmbeddings；默认取配置 ``EMBEDDING_MODEL``。
            persist_directory: 本地持久化目录，默认取配置 ``CHROMA_PERSIST_DIR``；
                host/port 均未指定时生效。
            host: 远程 ChromaDB 主机地址；与 port 同时指定时优先于本地模式。
            port: 远程 ChromaDB 端口。
            embeddings: 可选的 Embeddings 实例（用于测试注入 / 自定义模型）。

        Raises:
            ConnectionError: ChromaDB 连接或健康检查（count）失败。
        """
        if getattr(self, "_initialized", False):
            return

        collection_name = collection_name or settings.COLLECTION_NAME
        embedding_model = embedding_model or settings.EMBEDDING_MODEL
        persist_directory = persist_directory or settings.CHROMA_PERSIST_DIR
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self._embeddings: Embeddings = embeddings or self._build_embeddings(embedding_model)

        # 连接模式解析：host 与 port 同时存在优先远程模式
        self._remote_mode: bool = host is not None
        self._host: Optional[str] = host
        self._port: int = int(port) if port is not None else 8000
        self._persist_directory: Optional[str] = None if self._remote_mode else (
            persist_directory or "./chroma_data"
        )

        try:
            self._client = self._connect()
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            # 健康检查：触发一次实际调用验证连接可用
            self._collection.count()
        except ConnectionError:
            raise
        except Exception as exc:  # noqa: BLE001 - 统一转为 ConnectionError
            VectorStoreManager._instances.pop(collection_name, None)
            logger.error("初始化 ChromaDB 失败（collection=%s）: %s", collection_name, exc)
            raise ConnectionError(
                f"无法连接 ChromaDB（collection={collection_name}）: {exc}"
            ) from exc

        logger.info(
            "ChromaDB 初始化完成（collection=%s，mode=%s）",
            collection_name,
            "remote" if self._remote_mode else "local",
        )
        self._initialized = True

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    def _connect(self) -> Any:
        """按当前模式创建 ChromaDB 客户端。"""
        if self._remote_mode:
            logger.info("连接远程 ChromaDB: %s:%s", self._host, self._port)
            return chromadb.HttpClient(host=self._host, port=self._port)
        persist_path = str(self._persist_directory)
        os.makedirs(persist_path, exist_ok=True)
        logger.info("连接本地 ChromaDB（持久化目录=%s）", persist_path)
        return chromadb.PersistentClient(path=persist_path)

    @staticmethod
    def _build_embeddings(embedding_model: str) -> Embeddings:
        """根据模型名构建 Embeddings 实例。"""
        if embedding_model.lower().startswith("text-embedding"):
            from langchain_community.embeddings import OpenAIEmbeddings

            logger.info("使用 OpenAIEmbeddings（模型=%s）", embedding_model)
            return OpenAIEmbeddings(model=embedding_model)
        from langchain_community.embeddings import HuggingFaceEmbeddings

        logger.info("使用 HuggingFaceEmbeddings（模型=%s）", embedding_model)
        return HuggingFaceEmbeddings(model_name=embedding_model)

    @staticmethod
    def _sanitize_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """将元数据清洗为 ChromaDB 支持的扁平类型。

        ChromaDB 元数据仅支持 str/int/float/bool；``None`` 值被丢弃；
        dict/list 等嵌套结构会被 JSON 序列化为字符串；其余类型转为字符串。
        """
        if not metadata:
            return {}
        sanitized: Dict[str, Any] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, _SCALAR_TYPES) and not isinstance(value, (dict, list)):
                sanitized[key] = value
            elif isinstance(value, (dict, list, tuple)):
                sanitized[key] = json.dumps(value, ensure_ascii=False)
            else:
                sanitized[key] = str(value)
        return sanitized

    @staticmethod
    def _to_similarity(distance: float) -> float:
        """将 Chroma 余弦距离转换为相似度（0-1 区间，越高越相似）。"""
        return max(-1.0, min(1.0, 1.0 - float(distance)))

    def _add_batch(
        self,
        ids: List[str],
        texts: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> List[str]:
        """提交一批数据（自动计算向量）。"""
        vectors = self._embeddings.embed_documents(texts)
        self._collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=vectors,
        )
        return list(ids)

    def _search(
        self,
        query: str,
        k: int,
        where: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """内部检索：向量化 query 并从集合中查询，返回行字典列表。

        结果按距离升序（相似度降序）排列。
        """
        if self._collection.count() == 0:
            return []
        query_vector = self._embeddings.embed_query(query)
        result = self._collection.query(
            query_embeddings=[query_vector],
            n_results=max(k, 1),
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )
        rows: List[Dict[str, Any]] = []
        ids = (result.get("ids") or [[]])[0] or []
        documents = (result.get("documents") or [[]])[0] or []
        metadatas = (result.get("metadatas") or [[]])[0] or []
        distances = (result.get("distances") or [[]])[0] or []
        for index, record_id in enumerate(ids):
            rows.append(
                {
                    "id": record_id,
                    "content": documents[index] if index < len(documents) else "",
                    "metadata": (
                        dict(metadatas[index])
                        if index < len(metadatas) and metadatas[index]
                        else {}
                    ),
                    "distance": (
                        float(distances[index]) if index < len(distances) else 0.0
                    ),
                }
            )
        return rows

    # ------------------------------------------------------------------ #
    # 写入：增
    # ------------------------------------------------------------------ #
    def add_documents(self, documents: List[Document]) -> List[str]:
        """批量添加文档（自动分批，每批不超过 1000 条）。

        Args:
            documents: 待添加的 Document 列表。

        Returns:
            添加成功的文档 ID 列表。若文档元数据中存在 ``id`` 字段则复用，
            否则自动生成 ``uuid4().hex`` 形式的 32 位字符串 ID。
        """
        added_ids: List[str] = []
        total = len(documents)
        for start in range(0, total, _BATCH_SIZE):
            batch = documents[start : start + _BATCH_SIZE]
            ids: List[str] = []
            texts: List[str] = []
            metadatas: List[Dict[str, Any]] = []
            for doc in batch:
                record_id = doc.metadata.get("id")
                ids.append(str(record_id) if record_id is not None else uuid.uuid4().hex)
                texts.append(doc.page_content)
                metadatas.append(self._sanitize_metadata(doc.metadata))
            added_ids.extend(self._add_batch(ids, texts, metadatas))
        logger.info("add_documents 完成: 共添加 %d 条文档", total)
        return added_ids

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """批量添加纯文本（内部包装为 Document 后调用 add_documents）。

        Args:
            texts: 文本内容列表。
            metadatas: 与 texts 等长的元数据列表（可为 None）。

        Returns:
            添加成功的文档 ID 列表。
        """
        documents: List[Document] = []
        for index, text in enumerate(texts):
            metadata = metadatas[index] if metadatas else {}
            documents.append(Document(page_content=text, metadata=metadata or {}))
        return self.add_documents(documents)

    # ------------------------------------------------------------------ #
    # 写入：删
    # ------------------------------------------------------------------ #
    def delete_documents(self, ids: List[str]) -> bool:
        """按 ID 批量删除文档。

        Args:
            ids: 待删除的文档 ID 列表。

        Returns:
            删除操作是否成功。
        """
        try:
            self._collection.delete(ids=list(ids))
        except Exception as exc:  # noqa: BLE001
            logger.error("按 ID 删除失败: %s（%s）", ids, exc)
            return False
        logger.info("delete_documents 完成: 删除 %d 条", len(ids))
        return True

    def delete_by_filter(self, filter: Dict[str, Any]) -> int:
        """按元数据条件删除文档。

        Args:
            filter: ChromaDB where 过滤条件，如 ``{"file_name": "report.pdf"}``。

        Returns:
            实际删除的文档数量。
        """
        matched = self._collection.get(where=filter or None, include=[])
        ids = matched.get("ids") or []
        if ids:
            self._collection.delete(ids=ids)
        logger.info("delete_by_filter 完成: 按条件删除 %d 条", len(ids))
        return len(ids)

    def count_by_metadata(self, filter: Dict[str, Any]) -> int:
        """按元数据条件统计文档数量（只读，不删除）。

        Args:
            filter: ChromaDB where 过滤条件，如 ``{"source": "/path/to/file.pdf"}``。

        Returns:
            匹配的文档数量。
        """
        matched = self._collection.get(where=filter or None, include=[])
        return len(matched.get("ids") or [])

    def get_all_documents(self) -> List[Document]:
        """导出集合中的全部文档（正文 + 元数据）。

        先取全部 ID，再按 ``_BATCH_SIZE`` 分批读取，避免一次性加载过多
        文档导致内存占用过高，供 BM25 等基于全量语料的功能使用。

        Returns:
            集合中全部文档的 Document 列表（顺序不保证稳定）。
        """
        ids: List[str] = self._collection.get(include=[])["ids"] or []
        documents: List[Document] = []
        for start in range(0, len(ids), _BATCH_SIZE):
            chunk = ids[start : start + _BATCH_SIZE]
            if not chunk:
                continue
            result = self._collection.get(ids=chunk, include=["documents", "metadatas"])
            texts = result.get("documents") or []
            metadatas = result.get("metadatas") or []
            for index, text in enumerate(texts):
                raw_meta = metadatas[index] if index < len(metadatas) else None
                documents.append(
                    Document(
                        page_content=text or "",
                        metadata={**(raw_meta or {})},
                    )
                )
        logger.info(
            "get_all_documents 完成: 集合 %s 共导出 %d 条文档",
            self.collection_name,
            len(documents),
        )
        return documents

    def clear_collection(self) -> bool:
        """清空集合中的所有文档（用于测试或重置）。"""
        try:
            remaining = self._collection.get(include=[])["ids"] or []
            for start in range(0, len(remaining), _BATCH_SIZE):
                chunk = remaining[start : start + _BATCH_SIZE]
                if chunk:
                    self._collection.delete(ids=chunk)
        except Exception as exc:  # noqa: BLE001
            logger.error("清空集合失败: %s", exc)
            return False
        logger.info("clear_collection 完成: 清空集合 %s", self.collection_name)
        return True

    # ------------------------------------------------------------------ #
    # 写入：改
    # ------------------------------------------------------------------ #
    def update_document(
        self,
        document_id: str,
        new_text: str,
        new_metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """更新文档（先删除旧记录，再用相同 ID 写入新内容）。

        Args:
            document_id: 目标文档 ID。
            new_text: 新文本内容。
            new_metadata: 新元数据（可为 None，表示清空元数据）。

        Returns:
            更新是否成功。
        """
        try:
            self._collection.delete(ids=[document_id])
            metadata = self._sanitize_metadata(new_metadata)
            self._add_batch([document_id], [new_text], [metadata])
        except Exception as exc:  # noqa: BLE001
            logger.error("更新文档失败（id=%s）: %s", document_id, exc)
            return False
        logger.info("update_document 完成（id=%s）", document_id)
        return True

    # ------------------------------------------------------------------ #
    # 查询：查
    # ------------------------------------------------------------------ #
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """相似度检索。

        Args:
            query: 查询文本。
            k: 返回结果条数。
            filter: 可选的元数据过滤条件（ChromaDB where 语法）。

        Returns:
            按相似度降序排列的 Document 列表。
        """
        rows = self._search(query, k, filter)
        return [
            Document(page_content=row["content"], metadata=row["metadata"]) for row in rows
        ]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """带相似度分数的检索。

        Args:
            query: 查询文本。
            k: 返回结果条数。
            filter: 可选的元数据过滤条件。

        Returns:
            ``(Document, similarity)`` 列表，similarity 为 0-1 的余弦相似度
            （越高越相似），结果按相似度降序排列。
        """
        rows = self._search(query, k, filter)
        return [
            (
                Document(page_content=row["content"], metadata=row["metadata"]),
                self._to_similarity(row["distance"]),
            )
            for row in rows
        ]

    def similarity_search_with_threshold(
        self,
        query: str,
        k: int = 4,
        threshold: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """仅返回相似度大于阈值的文档。

        Args:
            query: 查询文本。
            k: 候选结果条数（先取 Top-k 再过滤）。
            threshold: 相似度阈值（0-1），低于该值的文档不返回。
            filter: 可选的元数据过滤条件。

        Returns:
            相似度严格大于阈值、按相似度降序排列的 Document 列表。
        """
        rows = self._search(query, k, filter)
        kept = [row for row in rows if self._to_similarity(row["distance"]) > threshold]
        return [
            Document(page_content=row["content"], metadata=row["metadata"]) for row in kept
        ]

    def get_relevant_documents_with_metadata(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """返回含文档内容与完整元数据的字典列表（便于 API 输出 JSON）。

        Args:
            query: 查询文本。
            k: 返回结果条数。
            filter: 可选的元数据过滤条件。

        Returns:
            字典列表，每项包含 ``content`` / ``metadata`` / ``document_id`` /
            ``similarity`` 四个字段。
        """
        rows = self._search(query, k, filter)
        return [
            {
                "content": row["content"],
                "metadata": row["metadata"],
                "document_id": row["id"],
                "similarity": self._to_similarity(row["distance"]),
            }
            for row in rows
        ]

    # ------------------------------------------------------------------ #
    # 统计
    # ------------------------------------------------------------------ #
    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息。

        Returns:
            包含 ``document_count`` / ``collection_name`` /
            ``embedding_dimension`` 的字典。集合为空时维度为 None。
        """
        count = self._collection.count()
        dimension: Optional[int] = None
        if count > 0:
            peeked = self._collection.peek(limit=1)
            embeddings = peeked.get("embeddings") or []
            if embeddings and embeddings[0] is not None:
                dimension = len(embeddings[0])
        return {
            "document_count": count,
            "collection_name": self.collection_name,
            "embedding_dimension": dimension,
        }
