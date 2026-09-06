"""API 业务路由包。

各子模块导出独立 ``router``，由上层 ``api.main`` 统一挂载到
``/api/health``、``/api/documents``、``/api/chat``、``/api/agents``、
``/api/workflows`` 前缀。
"""

from .agents import router as agents_router
from .chat import router as chat_router
from .documents import router as documents_router
from .health import router as health_router
from .workflows import router as workflows_router

__all__ = [
    "health_router",
    "documents_router",
    "chat_router",
    "agents_router",
    "workflows_router",
]
