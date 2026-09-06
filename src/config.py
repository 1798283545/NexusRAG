"""NexusRAG 全局配置模块。

从环境变量（.env 文件）读取配置，未设置时使用内置默认值。
"""

import os
from typing import List

from dotenv import load_dotenv

load_dotenv()


class Settings:
    """应用配置（与 .env 联动）。

    Attributes:
        CHROMA_HOST: ChromaDB 主机地址。
        CHROMA_PORT: ChromaDB 服务端口。
        CHROMA_PERSIST_DIR: 本地持久化目录（host 为空时生效）。
        COLLECTION_NAME: 默认向量集合名。
        EMBEDDING_MODEL: 默认 Embedding 模型。
        MEMORY_TYPE: 默认对话记忆类型（buffer / buffer_window / summary）。
        MEMORY_WINDOW_SIZE: 记忆保留的最近消息条数。
        MEMORY_SUMMARY_MAX_TOKENS: 摘要模式的最大 Token 约束。
        MEMORY_RETURN_MESSAGES: 记忆变量返回消息对象而非文本。
        RERANKER_MODEL: 默认 Cross-Encoder 重排模型。
        RERANKER_DEVICE: 重排推理设备（cpu / cuda / auto）。
        RERANKER_BATCH_SIZE: 重排批量推理大小。
        RERANKER_MAX_LENGTH: 重排输入序列最大长度。
        RERANKER_USE_FP16: 是否对重排启用半精度（仅 GPU）。
        RERANKER_THRESHOLD: 重排结果阈值过滤（0 表示不过滤）。
        AGENT_WORKSPACE_DIR: 智能体工作目录（限制文件/图表工具范围）。
        AGENT_VERBOSE: 智能体是否打印详细执行日志。
        AGENT_MAX_ITERATIONS: 单个智能体最大推理迭代次数。
        AGENT_CODE_TIMEOUT: 代码执行工具的单次超时（秒）。
        SUPERVISOR_ENABLE_PARALLEL: Supervisor 是否并行执行独立子任务。
        SUPERVISOR_MAX_PLANNING_ATTEMPTS: 任务规划失败后的最大重试次数。
        SUPERVISOR_SUBTASK_TIMEOUT: 单个子任务执行超时（秒）。
        SUPERVISOR_VERBOSE: Supervisor 是否打印详细执行日志。
        WORKFLOW_ENABLE_HITL: 是否启用人工审核（Human-in-the-Loop）。
        WORKFLOW_MAX_ITERATIONS: 工作流最大循环迭代次数（防死循环）。
        WORKFLOW_SUBTASK_TIMEOUT: 工作流内单个子任务执行超时（秒）。
        WORKFLOW_CHECKPOINT_DIR: LangGraph 检查点持久化目录。
        API_HOST: API 服务监听地址。
        API_PORT: API 服务监听端口。
        API_RELOAD: 开发期是否启用 uvicorn 热重载。
        API_CORS_ORIGINS: 允许跨域的来源（逗号分隔，"*" 表示全部放行）。
    """

    CHROMA_HOST: str = os.getenv("CHROMA_HOST", "localhost")
    CHROMA_PORT: int = int(os.getenv("CHROMA_PORT", "8000"))
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "nexusrag_docs")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")

    # RAG 默认参数
    DEFAULT_K: int = int(os.getenv("DEFAULT_K", "4"))
    DEFAULT_CHUNK_SIZE: int = int(os.getenv("DEFAULT_CHUNK_SIZE", "500"))
    DEFAULT_CHUNK_OVERLAP: int = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "50"))
    DEFAULT_STRATEGY: str = os.getenv("DEFAULT_STRATEGY", "recursive")

    # LLM 配置
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.3"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "2048"))

    # 混合检索配置
    HYBRID_FUSION_STRATEGY: str = os.getenv("HYBRID_FUSION_STRATEGY", "weighted")
    HYBRID_WEIGHT_VECTOR: float = float(os.getenv("HYBRID_WEIGHT_VECTOR", "0.5"))
    HYBRID_WEIGHT_BM25: float = float(os.getenv("HYBRID_WEIGHT_BM25", "0.5"))
    HYBRID_RRF_K: int = int(os.getenv("HYBRID_RRF_K", "60"))
    HYBRID_TOP_K: int = int(os.getenv("HYBRID_TOP_K", "10"))

    # 对话记忆配置
    MEMORY_TYPE: str = os.getenv("MEMORY_TYPE", "buffer_window")
    MEMORY_WINDOW_SIZE: int = int(os.getenv("MEMORY_WINDOW_SIZE", "10"))
    MEMORY_SUMMARY_MAX_TOKENS: int = int(os.getenv("MEMORY_SUMMARY_MAX_TOKENS", "200"))
    MEMORY_RETURN_MESSAGES: bool = os.getenv("MEMORY_RETURN_MESSAGES", "true").lower() == "true"

    # Reranker 配置（Cross-Encoder 重排）
    RERANKER_MODEL: str = os.getenv(
        "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    RERANKER_DEVICE: str = os.getenv("RERANKER_DEVICE", "cpu")
    RERANKER_BATCH_SIZE: int = int(os.getenv("RERANKER_BATCH_SIZE", "32"))
    RERANKER_MAX_LENGTH: int = int(os.getenv("RERANKER_MAX_LENGTH", "512"))
    RERANKER_USE_FP16: bool = os.getenv("RERANKER_USE_FP16", "false").lower() == "true"
    # 重排阈值过滤（0 表示不过滤，供调用方读取后传入 filter_by_threshold）
    RERANKER_THRESHOLD: float = float(os.getenv("RERANKER_THRESHOLD", "0.0"))

    # 智能体（Agent）配置
    AGENT_WORKSPACE_DIR: str = os.getenv("AGENT_WORKSPACE_DIR", "./workspace")
    AGENT_VERBOSE: bool = os.getenv("AGENT_VERBOSE", "true").lower() == "true"
    AGENT_MAX_ITERATIONS: int = int(os.getenv("AGENT_MAX_ITERATIONS", "5"))
    AGENT_CODE_TIMEOUT: int = int(os.getenv("AGENT_CODE_TIMEOUT", "30"))

    # Supervisor（监督者）配置
    SUPERVISOR_ENABLE_PARALLEL: bool = os.getenv("SUPERVISOR_ENABLE_PARALLEL", "true").lower() == "true"
    SUPERVISOR_MAX_PLANNING_ATTEMPTS: int = int(os.getenv("SUPERVISOR_MAX_PLANNING_ATTEMPTS", "3"))
    SUPERVISOR_SUBTASK_TIMEOUT: float = float(os.getenv("SUPERVISOR_SUBTASK_TIMEOUT", "60"))
    SUPERVISOR_VERBOSE: bool = os.getenv("SUPERVISOR_VERBOSE", "true").lower() == "true"

    # LangGraph 工作流配置
    WORKFLOW_ENABLE_HITL: bool = os.getenv("WORKFLOW_ENABLE_HITL", "true").lower() == "true"
    WORKFLOW_MAX_ITERATIONS: int = int(os.getenv("WORKFLOW_MAX_ITERATIONS", "10"))
    WORKFLOW_SUBTASK_TIMEOUT: float = float(os.getenv("WORKFLOW_SUBTASK_TIMEOUT", "60"))
    WORKFLOW_CHECKPOINT_DIR: str = os.getenv("WORKFLOW_CHECKPOINT_DIR", "./checkpoints")

    # API 服务配置
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_RELOAD: bool = os.getenv("API_RELOAD", "false").lower() == "true"
    API_CORS_ORIGINS: str = os.getenv("API_CORS_ORIGINS", "*")

    # SSE 流式响应配置
    SSE_BUFFER_SIZE: int = int(os.getenv("SSE_BUFFER_SIZE", "1"))
    SSE_CHUNK_DELAY: float = float(os.getenv("SSE_CHUNK_DELAY", "0.01"))
    SSE_TIMEOUT: int = int(os.getenv("SSE_TIMEOUT", "120"))
    SSE_ENABLE_THINKING: bool = os.getenv("SSE_ENABLE_THINKING", "true").lower() == "true"

    @property
    def api_cors_origins(self) -> List[str]:
        """解析后的跨域白名单（逗号分隔 → 列表；空值回退为 ``["*"]``）。"""
        origins = [item.strip() for item in self.API_CORS_ORIGINS.split(",") if item.strip()]
        return origins or ["*"]


#: 全局单例，供各模块直接引用
settings = Settings()
