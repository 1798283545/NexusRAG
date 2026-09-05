"""NexusRAG 对话记忆模块。

提供多会话对话记忆管理器，为多轮 RAG 对话提供 buffer / buffer_window /
summary 三种记忆策略，并预留与 RAGChain 的集成辅助方法。
"""

from .conversation_memory import ConversationMemoryManager

__all__ = ["ConversationMemoryManager"]
