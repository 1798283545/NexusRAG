"""NexusRAG API 服务入口。

第四阶段 Day 1-3：以应用工厂 :func:`create_app` 组装 FastAPI 应用 ——
统一注册 CORS、全部业务路由（health / documents / chat / agents / workflows）
与统一异常处理器，并暴露 ``nexusrag-serve`` 命令使用的 :func:`run` 启动入口。
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import exceptions
from api.routes import (
    agents as agents_routes,
    chat as chat_routes,
    documents as documents_routes,
    health as health_routes,
    workflows as workflows_routes,
)
from config import settings

logger = logging.getLogger(__name__)

#: 服务版本（与 pyproject.toml / schemas.HealthResponse 保持一致）
VERSION = "0.1.0"

#: (router, 前缀) —— 所有业务路由统一挂载到 /api 命名空间
_ROUTERS = (
    (health_routes.router, "/api/health"),
    (documents_routes.router, "/api/documents"),
    (chat_routes.router, "/api/chat"),
    (agents_routes.router, "/api/agents"),
    (workflows_routes.router, "/api/workflows"),
)


def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用（工厂函数，便于测试与多实例部署）。

    - 文档地址：``/api/docs``（Swagger UI）与 ``/api/redoc``；
    - 跨域来源取自 ``settings.API_CORS_ORIGINS``（逗号分隔，默认放行）；
    - 全部路由经 :data:`_ROUTERS` 注册后套用统一异常处理。
    """
    app = FastAPI(
        title="NexusRAG",
        description="基于 LangChain + LangGraph 的多智能体 RAG 平台 REST API",
        version=VERSION,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    origins = settings.api_cors_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        # "*" 与 credentials 不能同时生效；显式列出白名单时才允许携带凭据
        allow_credentials=origins != ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    for router, prefix in _ROUTERS:
        app.include_router(router, prefix=prefix)

    exceptions.setup_exception_handlers(app)
    return app


#: 模块级应用实例（供 uvicorn / 测试直接引用）
app = create_app()


def run(application: Optional[FastAPI] = None) -> None:
    """启动 API 服务（供 ``nexusrag-serve`` 命令调用）。

    Args:
        application: 可选的应用实例；缺省使用模块级 :data:`app`。
    """
    import uvicorn

    uvicorn.run(
        application or app,
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
    )
