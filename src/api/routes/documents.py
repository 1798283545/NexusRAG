"""文档管理接口。

提供文件上传（单 / 批量）、文档列表、按文档删除与清空集合等能力。
索引写入交给 ``RAGChain.process_document``；文件先落盘到系统临时目录，
索引完成后立即清理，避免把临时文件混入项目目录。
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from api.dependencies import get_rag_chain
from api.schemas import (
    DocumentDeleteResponse,
    DocumentItem,
    DocumentListResponse,
    DocumentUploadResponse,
)

router = APIRouter()
logger = logging.getLogger(__name__)


def _ensure_unique(document_name: str, rag_chain: Any) -> None:
    """同名文档重新上传前，先清理旧向量块（幂等更新语义）。"""
    vsm = getattr(rag_chain, "vector_store_manager", None)
    delete = getattr(vsm, "delete_by_filter", None)
    if callable(delete):
        try:
            delete({"file_name": document_name})
        except Exception as exc:  # noqa: BLE001 - 去重失败不阻断上传
            logger.warning("同名文档清理失败（%s）: %s", document_name, exc)


def _save_tmp(upload: UploadFile) -> Path:
    """把上传文件流写入临时文件并返回其路径（保留扩展名）。"""
    suffix = Path(upload.filename or "").suffix
    fd, tmp_name = tempfile.mkstemp(prefix="nexusrag_upload_", suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as buffer:
            shutil.copyfileobj(upload.file, buffer)
    except Exception:
        os.unlink(tmp_name)
        raise
    return Path(tmp_name)


def _process_upload(file: UploadFile, rag_chain: Any) -> DocumentUploadResponse:
    """单个上传文件的索引全流程。"""
    document_name = Path(file.filename or "").name
    if not document_name:
        raise HTTPException(status_code=400, detail="文件名不能为空")
    file_type = Path(document_name).suffix.lower().lstrip(".")
    _ensure_unique(document_name, rag_chain)

    tmp_path: Path | None = None
    try:
        tmp_path = _save_tmp(file)
        chunk_count = int(rag_chain.process_document(str(tmp_path)))
    except FileNotFoundError as exc:
        logger.warning("文档不存在: %s", exc)
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - 索引失败归并为 500 可读错误
        logger.exception("文档 %s 索引失败", document_name)
        raise HTTPException(status_code=500, detail=f"文档处理失败：{exc}") from exc
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:  # pragma: no cover - 清理失败无需影响响应
                pass

    return DocumentUploadResponse(
        file_name=document_name,
        file_type=file_type,
        chunk_count=chunk_count,
        document_id=str(tmp_path) if tmp_path else document_name,
        status="success",
    )


@router.post("/upload", response_model=DocumentUploadResponse, summary="上传并索引单个文档")
def upload_document(
    file: UploadFile = File(...),
    rag_chain: Any = Depends(get_rag_chain),
) -> DocumentUploadResponse:
    """上传一个文档并完成加载、分割、向量化入库。"""
    return _process_upload(file, rag_chain)


@router.post("/upload-multiple", response_model=List[DocumentUploadResponse], summary="批量上传文档")
def upload_multiple_documents(
    files: List[UploadFile] = File(...),
    rag_chain: Any = Depends(get_rag_chain),
) -> List[DocumentUploadResponse]:
    """一次上传并索引多个文档（逐个处理，首个失败即中断并报错）。"""
    responses: List[DocumentUploadResponse] = []
    for item in files:
        responses.append(_process_upload(item, rag_chain))
    return responses


@router.get("/list", response_model=DocumentListResponse, summary="文档列表")
def list_documents(rag_chain: Any = Depends(get_rag_chain)) -> DocumentListResponse:
    """按文档聚合返回已入库文档的元信息（文件名、块数、来源等）。"""
    try:
        documents = rag_chain.vector_store_manager.get_all_documents()
    except Exception as exc:  # noqa: BLE001
        logger.exception("读取文档列表失败")
        raise HTTPException(status_code=500, detail=f"读取文档列表失败：{exc}") from exc

    grouped: Dict[str, Dict[str, Any]] = {}
    for doc in documents:
        meta = doc.metadata or {}
        name = str(meta.get("file_name") or Path(str(meta.get("source", ""))).name or "unknown")
        entry = grouped.setdefault(
            name,
            {
                "name": name,
                "file_type": str(meta.get("file_type") or Path(name).suffix.lstrip(".")),
                "chunk_count": 0,
                "created_at": meta.get("modification_date"),
                "source": str(meta.get("source", "")),
            },
        )
        entry["chunk_count"] += 1

    items = [
        DocumentItem(
            id=str(item["source"] or item["name"]),
            name=item["name"],
            file_type=item["file_type"],
            chunk_count=item["chunk_count"],
            created_at=item["created_at"],
            source=str(item["source"] or ""),
        )
        for item in sorted(grouped.values(), key=lambda x: str(x["name"]))
    ]
    return DocumentListResponse(documents=items)


@router.delete("/{document_name}", response_model=DocumentDeleteResponse, summary="删除单个文档")
def delete_document(
    document_name: str,
    rag_chain: Any = Depends(get_rag_chain),
) -> DocumentDeleteResponse:
    """删除指定文档的全部向量块（按文件名匹配；也支持直接传来源路径）。"""
    vsm = rag_chain.vector_store_manager
    deleted = 0
    try:
        # 优先按文件名精确匹配；若传入的是完整来源路径再按 source 匹配一次
        deleted += int(vsm.delete_by_filter({"file_name": document_name}) or 0)
        if document_name != Path(document_name).name:
            deleted += int(vsm.delete_by_filter({"source": document_name}) or 0)
    except Exception as exc:  # noqa: BLE001
        logger.exception("删除文档 %s 失败", document_name)
        raise HTTPException(status_code=500, detail=f"删除文档失败：{exc}") from exc
    if deleted <= 0:
        raise HTTPException(status_code=404, detail=f"未找到文档：{document_name}")
    return DocumentDeleteResponse(deleted_count=deleted)


@router.delete("/", response_model=DocumentDeleteResponse, summary="清空全部文档")
def delete_all_documents(rag_chain: Any = Depends(get_rag_chain)) -> DocumentDeleteResponse:
    """清空知识库中的全部向量数据。"""
    vsm = rag_chain.vector_store_manager
    try:
        before = int(vsm.get_collection_stats().get("document_count") or 0)
        ok = bool(vsm.clear_collection())
    except Exception as exc:  # noqa: BLE001
        logger.exception("清空知识库失败")
        raise HTTPException(status_code=500, detail=f"清空知识库失败：{exc}") from exc
    if not ok:
        raise HTTPException(status_code=500, detail="清空知识库失败")
    return DocumentDeleteResponse(deleted_count=before)
