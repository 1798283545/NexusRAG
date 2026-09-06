"""NexusRAG API 异常处理。

为应用注册统一 JSON 错误处理器：业务异常 / 文件缺失返回结构化
:class:`ErrorResponse` 载荷；HTTPException 与校验错误保持 FastAPI 既有语义，
仅统一为 JSON 输出。
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.schemas import ErrorResponse

logger = logging.getLogger(__name__)


def _dump(model: ErrorResponse) -> Dict[str, Any]:
    """兼容 pydantic v1/v2 的序列化辅助。"""
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()  # type: ignore[attr-defined]  # pragma: no cover - v1 兼容


def _json(status_code: int, error: str, detail: str = "") -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=_dump(ErrorResponse(error=error, detail=detail or None)),
    )


def setup_exception_handlers(app: FastAPI) -> None:
    """注册统一异常处理器到 FastAPI 应用。"""

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        """请求参数校验失败 → 422 JSON（保持默认状态码语义）。"""
        del request
        errors = []
        for item in exc.errors():
            loc = ".".join(str(part) for part in item.get("loc", []))
            errors.append(f"{loc}: {item.get('msg', '')}" if loc else str(item.get("msg", "")))
        return _json(422, "validation_error", "；".join(errors))

    @app.exception_handler(StarletteHTTPException)
    async def _http_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        """HTTP 业务异常 → 保持状态码的 JSON 输出。"""
        del request
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        return _json(exc.status_code, "http_error", detail)

    @app.exception_handler(FileNotFoundError)
    async def _file_not_found(request: Request, exc: FileNotFoundError) -> JSONResponse:
        """文件不存在 → 404。"""
        del request
        logger.warning("文件不存在: %s", exc)
        return _json(404, "file_not_found", str(exc))

    @app.exception_handler(Exception)
    async def _generic_handler(request: Request, exc: Exception) -> JSONResponse:
        """兜底：未预期异常 → 500（并记录完整堆栈便于排查）。"""
        del request
        logger.exception("未处理的 API 异常: %s", exc)
        return _json(500, "internal_server_error", str(exc))
