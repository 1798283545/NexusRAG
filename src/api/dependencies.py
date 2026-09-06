"""NexusRAG API 依赖注入。

提供 FastAPI ``Depends`` 依赖：LLM / 向量存储 / RAG 链 / 智能体 / 工作流等
全局共享实例。重量级第三方模块（langchain、chromadb 等）一律在函数体内
**惰性导入**，保证仅导入本包（如 ``from api.main import app``）时无需安装
全套 AI 依赖，避免冒烟测试与应用启动被无关依赖拖垮。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict

from config import settings

__all__ = [
    "get_settings",
    "get_llm",
    "get_vector_store",
    "get_rag_chain",
    "get_agents",
    "get_workflow",
    "clear_caches",
]


def get_settings():
    """返回全局配置单例（不缓存对象本身，仅作便捷依赖）。"""
    return settings


@lru_cache()
def get_llm():
    """构建并缓存全局 LLM 实例（来自 .env 的模型 / 密钥配置）。"""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.MODEL_NAME,
        temperature=settings.TEMPERATURE,
        max_tokens=settings.MAX_TOKENS,
        api_key=settings.OPENAI_API_KEY,
    )


@lru_cache()
def get_vector_store():
    """构建并缓存向量存储管理器（远程 ChromaDB 优先，否则本地持久化）。"""
    from retrievers import VectorStoreManager

    host = settings.CHROMA_HOST or None
    return VectorStoreManager(
        collection_name=settings.COLLECTION_NAME,
        persist_directory=settings.CHROMA_PERSIST_DIR,
        host=host,
        port=settings.CHROMA_PORT,
    )


@lru_cache()
def get_rag_chain():
    """构建并缓存 RAG 链（向量存储 + LLM）。"""
    from chains import RAGChain

    return RAGChain(
        vector_store_manager=get_vector_store(),
        llm=get_llm(),
        k=settings.DEFAULT_K,
        chunk_size=settings.DEFAULT_CHUNK_SIZE,
        chunk_overlap=settings.DEFAULT_CHUNK_OVERLAP,
    )


@lru_cache()
def get_agents() -> Dict[str, Any]:
    """构建并缓存全部专业智能体 + Supervisor（含 Supervisor 键）。"""
    from agents import AgentFactory

    return AgentFactory.create_all_with_supervisor(
        llm=get_llm(),
        rag_chain=get_rag_chain(),
        workspace_dir=settings.AGENT_WORKSPACE_DIR,
    )


@lru_cache()
def get_workflow():
    """构建并缓存 LangGraph 多智能体工作流执行器。"""
    from workflows import MultiAgentWorkflow

    agents = get_agents()
    supervisor = agents["supervisor"]
    specialists = {key: agent for key, agent in agents.items() if key != "supervisor"}
    return MultiAgentWorkflow(
        supervisor=supervisor,
        agents=specialists,
        enable_hitl=settings.WORKFLOW_ENABLE_HITL,
        max_iterations=settings.WORKFLOW_MAX_ITERATIONS,
    )


def clear_caches() -> None:
    """清空全部依赖缓存（测试隔离 / 热重载配置使用）。"""
    for cached in (get_llm, get_vector_store, get_rag_chain, get_agents, get_workflow):
        cached.cache_clear()
