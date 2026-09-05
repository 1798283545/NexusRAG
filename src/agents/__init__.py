"""NexusRAG 多智能体模块。

第三阶段（多智能体与工作流）基础层：定义所有专业智能体的抽象基类
（:class:`BaseAgent`），并提供四个开箱即用的专业智能体与注册工厂：

- :class:`RAGAgent`：知识库检索与问答；
- :class:`CodeAgent`：代码执行 / 数据分析 / 图表生成；
- :class:`WebSearchAgent`：联网搜索与信息整合；
- :class:`SummarizerAgent`：文档摘要与要点提炼；
- :class:`SupervisorAgent`：任务拆解 / 调度 / 整合的监督者；
- :class:`AgentFactory`：按名称创建 / 批量创建智能体与 Supervisor。
"""

from agents.agent_factory import AgentFactory
from agents.base_agent import BaseAgent
from agents.code_agent import CodeAgent
from agents.rag_agent import RAGAgent
from agents.summarizer_agent import SummarizerAgent
from agents.supervisor import SupervisorAgent
from agents.web_agent import WebSearchAgent

__all__ = [
    "BaseAgent",
    "RAGAgent",
    "CodeAgent",
    "WebSearchAgent",
    "SummarizerAgent",
    "SupervisorAgent",
    "AgentFactory",
]
