"""健康检查接口。

返回服务与核心组件的可用状态（vector_store / llm / bm25_index）。
组件探测采用惰性构建并捕获异常，任意组件不可用时整体返回 ``degraded``。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

from api.dependencies import get_llm, get_vector_store
from api.schemas import HealthResponse
from config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

VERSION = "0.1.0"


def _check(name: str) -> bool:
    """探测单个组件是否可用（True=可用，False=不可用）。"""
    try:
        if name == "vector_store":
            get_vector_store()
        elif name == "llm":
            if not settings.OPENAI_API_KEY:
                return False
            get_llm()
        return True
    except Exception as exc:  # noqa: BLE001 - 组件故障仅标记不可用，不中断探测
        logger.warning("组件 %s 探测失败: %s", name, exc)
        return False


@router.get("", response_model=HealthResponse, summary="健康检查")
@router.get("/", response_model=HealthResponse, include_in_schema=False)
def health_check() -> HealthResponse:
    """服务与核心组件健康检查。"""
    vector_store_ok = _check("vector_store")
    llm_ok = _check("llm")
    components = {
        "vector_store": vector_store_ok,
        # 当前 RAG 链以向量库为底座，BM25 混合索引可用性视作随向量库就绪
        "bm25_index": vector_store_ok,
        "llm": llm_ok,
    }
    status = "healthy" if all(components.values()) else "degraded"
    return HealthResponse(status=status, version=VERSION, components=components)
