"""NexusRAG 全局配置模块。

从环境变量（.env 文件）读取配置，未设置时使用内置默认值。
"""

import os

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


#: 全局单例，供各模块直接引用
settings = Settings()
