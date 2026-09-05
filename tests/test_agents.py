"""第三阶段：多智能体与工作流 —— 智能体 / 工具单元测试。

覆盖以下场景（全部离线确定性，不依赖真实模型 / 网络）：

1. 每个 Agent 初始化且工具列表非空；
2. RAGAgent 检索与问答（伪造“向量库已有测试文档”）；
3. CodeAgent 执行简单 Python 代码（``print("hello")``）；
4. WebSearchAgent 搜索（mock duckduckgo-search 模块，跳过真实联网）；
5. SummarizerAgent 摘要：长度小于原文且包含关键信息；
6. 中间步骤（``intermediate_steps``）与工具使用记录（``tools_used``）。
"""

from __future__ import annotations

import os
import sys
import types
from typing import Any, List, Optional

import pytest
from langchain_core.agents import AgentAction
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from agents import (
    AgentFactory,
    BaseAgent,
    CodeAgent,
    RAGAgent,
    SummarizerAgent,
    WebSearchAgent,
)
from agents.base_agent import BaseAgent as _BaseAgentAlias
from tools import (
    DocumentComparisonTool,
    DocumentQATool,
    DocumentSummaryTool,
    FileOperationTool,
    KeyPointExtractionTool,
    RetrievalTool,
    VisualizationTool,
    WebSearchTool,
)
from tools.python_tool import PythonREPLTool

# --------------------------------------------------------------------------- #
# 测试替身（stub）：不触碰真实 LLM / ChromaDB / 网络
# --------------------------------------------------------------------------- #
_ORIGINAL_DOC = (
    "NexusRAG 是一个基于 LangChain 与 LangGraph 的多智能体 RAG 平台。"
    "平台提供文档加载、分块、向量化、混合检索、重排以及对话记忆等能力。"
    "在第三阶段，平台引入了 RAGAgent、CodeAgent、WebSearchAgent 与 SummarizerAgent，"
    "为后续 Supervisor 的任务分发做好准备。\n"
    "关于项目 Alpha 的推进情况，最终关键结论是：核心指标环比增长 45%，"
    "整体风险可控，建议在下个季度扩大投入。\n"
    "此外，多智能体之间的协作通过统一结果结构（output / intermediate_steps / "
    "tools_used）进行标准化，便于上游统一编排。"
)


class _FakeVectorStore:
    """伪造向量库：无论查询什么，都返回固定测试文档。"""

    def __init__(self, text: str, source: str) -> None:
        self._text = text
        self._source = source

    def similarity_search_with_score(self, query: str, k: int = 4) -> List[Any]:
        doc = Document(page_content=self._text, metadata={"source": self._source})
        return [(doc, 0.95)]


class _FakeRagChain:
    """伪造 RAG 链：提供 vector_store_manager 与 query() 两处接口。"""

    def __init__(self, text: str = _ORIGINAL_DOC, source: str = "test.md") -> None:
        self.vector_store_manager = _FakeVectorStore(text, source)

    def query(self, query: str, k: int = 4) -> dict:
        return {
            "answer": f"根据测试文档，问题「{query}」的答案是：关键结论为指标增长 45%。",
            "source_documents": [
                Document(page_content=self.vector_store_manager._text, metadata={"source": "test.md"})
            ],
            "confidence": 0.93,
        }


class _FakeLLM:
    """确定性 LLM 替身：仅支持 invoke，不支持 bind_tools（触发直答降级）。

    按系统提示词内容返回预设文本，保证摘要 / 要点 / 对比工具的断言稳定。
    """

    def __init__(self) -> None:
        self.calls: List[Any] = []

    def invoke(self, message: Any) -> str:
        self.calls.append(message)
        system, user = "", ""
        if isinstance(message, str):
            user = message
        else:
            for msg in message:
                text = str(getattr(msg, "content", ""))
                if isinstance(msg, SystemMessage):
                    system += text
                else:
                    user += text
        if "文档摘要专家" in system:
            return (
                "本报告给出关键结论：项目 Alpha 核心指标环比增长 45%，"
                "整体风险可控，建议下季度扩大投入。"
            )
        if "信息提炼专家" in system:
            return "• 核心指标环比增长 45%\n• 整体风险可控\n• 建议扩大投入"
        if "文档对比专家" in system:
            return "相同点：均基于 LangChain 构建。\n差异点：RAGAgent 侧重检索，CodeAgent 侧重执行。"
        return "已收到任务。"


class _EchoTool(BaseTool):
    """最小工具替身（用于 BaseAgent 执行链路测试）。"""

    name: str = "echo_tool"
    description: str = "原样返回输入。"

    def _run(self, query: str, **kwargs: Any) -> str:
        return f"echo: {query}"

    async def _arun(self, query: str, **kwargs: Any) -> str:  # pragma: no cover
        return self._run(query, **kwargs)


class _DummyAgent(BaseAgent):
    """只用于测试的 BaseAgent 具体子类。"""

    def _get_default_tools(self) -> List[BaseTool]:
        return [_EchoTool()]

    def _get_system_prompt(self) -> str:
        return "你是测试智能体。"


class _FakeExecutor:
    """伪造 AgentExecutor：返回固定中间步骤，用于验证结果记录逻辑。"""

    def __init__(self) -> None:
        self.invoked_inputs: Optional[dict] = None

    def invoke(self, inputs: dict) -> dict:
        self.invoked_inputs = inputs
        step = (AgentAction(tool="echo_tool", tool_input="ping", log=""), "echo: ping")
        return {"output": "测试输出", "intermediate_steps": [step]}


@pytest.fixture
def fake_llm() -> _FakeLLM:
    return _FakeLLM()


@pytest.fixture
def fake_chain() -> _FakeRagChain:
    return _FakeRagChain()


# --------------------------------------------------------------------------- #
# 1. Agent 初始化：工具列表非空 + 基本信息
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("factory", "expected_name", "expected_tools"),
    [
        (lambda llm, chain: RAGAgent(llm=llm, rag_chain=chain), "RAGAgent",
         {"retriever", "document_qa"}),
        (lambda llm, chain: CodeAgent(llm=llm), "CodeAgent",
         {"python_executor", "pandas_analyzer", "visualizer", "file_operator"}),
        (lambda llm, chain: WebSearchAgent(llm=llm), "WebSearchAgent",
         {"web_search", "web_content_extractor", "information_synthesizer"}),
        (lambda llm, chain: SummarizerAgent(llm=llm, rag_chain=chain), "SummarizerAgent",
         {"document_summarizer", "keypoint_extractor", "document_comparer"}),
    ],
)
def test_agent_initialization_with_nonempty_tools(
    factory, expected_name: str, expected_tools: set, fake_llm, fake_chain
) -> None:
    agent = factory(fake_llm, fake_chain)
    assert isinstance(agent, BaseAgent)
    assert agent.name == expected_name
    assert agent.description
    assert agent.tools, "工具列表不应为空"
    assert {tool.name for tool in agent.tools} == expected_tools
    description_text = agent.get_tools_description()
    for tool in agent.tools:
        assert tool.name in description_text


# --------------------------------------------------------------------------- #
# 2. RAGAgent：检索工具 + 文档问答工具
# --------------------------------------------------------------------------- #
def test_rag_retrieval_tool_returns_sources(fake_chain) -> None:
    tool = RetrievalTool(rag_chain=fake_chain)
    output = tool._run("多智能体 RAG", k=2)
    assert "test.md" in output
    assert "NexusRAG" in output
    assert "分数" in output


def test_rag_retrieval_tool_empty_store_returns_graceful_text() -> None:
    empty_store = _FakeRagChain(text="", source="")
    tool = RetrievalTool(rag_chain=empty_store)
    assert "未检索到相关文档内容" in tool._run("任何查询")


def test_rag_qa_tool_answers_with_sources(fake_chain) -> None:
    agent = RAGAgent(llm=_FakeLLM(), rag_chain=fake_chain)
    qa_tool = next(t for t in agent.tools if t.name == "document_qa")
    output = qa_tool._run("项目 Alpha 进展如何", k=2)
    assert "关键结论" in output
    assert "引用 1 个来源" in output
    assert "置信度" in output


# --------------------------------------------------------------------------- #
# 3. CodeAgent：Python 代码执行 + 文件工具路径防护
# --------------------------------------------------------------------------- #
def test_python_repl_tool_executes_print() -> None:
    output = PythonREPLTool()._run("print('hello')")
    assert "[stdout]" in output
    assert "hello" in output
    assert "[exit_code] 0" in output


def test_file_operation_within_workspace(tmp_path) -> None:
    tool = FileOperationTool(workspace_dir=str(tmp_path))
    result = tool._run("write", "notes.txt", content="测试内容")
    assert "已写入" in result
    assert tool._run("read", "notes.txt") == "测试内容"
    listed = tool._run("list", ".")
    assert "notes.txt" in listed


def test_file_operation_rejects_path_escape(tmp_path) -> None:
    tool = FileOperationTool(workspace_dir=str(tmp_path))
    with pytest.raises(ValueError, match="路径越界"):
        tool._run("read", "../secret.txt")


def test_visualization_tool_generates_image(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    tool = VisualizationTool(workspace_dir=str(tmp_path))
    code = (
        "import matplotlib.pyplot as plt\n"
        "plt.plot([1, 2, 3], [1, 4, 9])\n"
        "plt.savefig('chart.png')\n"
    )
    output = tool._run(code, output_name="chart.png")
    assert "已生成图片" in output
    assert "chart.png" in output


# --------------------------------------------------------------------------- #
# 4. WebSearchAgent：mock 底层搜索模块（不真实联网）
# --------------------------------------------------------------------------- #
def test_web_search_tool_mocked(monkeypatch) -> None:
    fake_module = types.ModuleType("duckduckgo_search")

    class _FakeDDGS:
        def __enter__(self):
            return self

        def __exit__(self, *args: Any) -> bool:
            return False

        def text(self, keywords: str, max_results: int = 5) -> List[dict]:
            return [
                {
                    "title": "NexusRAG 介绍",
                    "body": "基于 LangChain 的多智能体 RAG 平台",
                    "href": "https://example.com/nexusrag",
                }
            ]

    fake_module.DDGS = _FakeDDGS
    monkeypatch.setitem(sys.modules, "duckduckgo_search", fake_module)

    tool = WebSearchTool(search_api="duckduckgo", max_results=3)
    output = tool._run("NexusRAG")
    assert "NexusRAG 介绍" in output
    assert "example.com" in output
    assert "摘要" in output


def test_web_search_tool_unsupported_api() -> None:
    tool = WebSearchTool(search_api="google", max_results=3)
    assert "暂不支持" in tool._run("hello")


# --------------------------------------------------------------------------- #
# 5. SummarizerAgent：摘要短于原文且保留关键信息
# --------------------------------------------------------------------------- #
def test_summarizer_agent_summary_is_shorter_and_keeps_key_info(
    fake_llm, fake_chain
) -> None:
    agent = SummarizerAgent(llm=fake_llm, rag_chain=fake_chain, max_length=200)
    summarizer = next(t for t in agent.tools if t.name == "document_summarizer")
    output = summarizer._run("项目 Alpha 关键结论", max_length=120)
    assert "【摘要】" in output
    summary_body = output.split("【摘要】")[1].split("【相关来源】")[0]
    assert "关键结论" in summary_body
    assert len(summary_body) < len(_ORIGINAL_DOC)
    assert "test.md" in output


def test_keypoint_extraction_tool_returns_bullets(fake_llm, fake_chain) -> None:
    tool = KeyPointExtractionTool(rag_chain=fake_chain, llm=fake_llm)
    output = tool._run("项目 Alpha 要点", max_points=3)
    assert "•" in output
    assert "45%" in output


def test_document_comparison_tool(fake_llm, fake_chain) -> None:
    tool = DocumentComparisonTool(rag_chain=fake_chain, llm=fake_llm)
    output = tool._run(query_a="RAGAgent", query_b="CodeAgent")
    assert "相同点" in output
    assert "差异点" in output


# --------------------------------------------------------------------------- #
# 6. 工具调用的中间步骤记录验证（BaseAgent.run 结果结构）
# --------------------------------------------------------------------------- #
def test_base_agent_run_records_intermediate_steps_and_tools_used() -> None:
    agent = _DummyAgent(
        name="DummyAgent", description="测试", llm=_FakeLLM(), verbose=False
    )
    fake_executor = _FakeExecutor()
    agent._agent_executor = fake_executor  # 注入伪造执行器，跳过真实 langchain

    result = agent.run("帮我跑一下")
    assert result["output"] == "测试输出"
    assert len(result["intermediate_steps"]) == 1
    assert result["tools_used"] == ["echo_tool"]
    assert fake_executor.invoked_inputs == {"input": "帮我跑一下"}


def test_base_agent_falls_back_to_direct_mode_when_llm_lacks_bind_tools(
    fake_llm,
) -> None:
    agent = _DummyAgent(
        name="DummyAgent", description="测试", llm=fake_llm, verbose=False
    )
    # 直答模式下应返回输出且不产生任何工具调用记录
    result = agent.run("你好")
    assert result["output"]
    assert result["intermediate_steps"] == []
    assert result["tools_used"] == []


# --------------------------------------------------------------------------- #
# 7. AgentFactory：创建与批量注册
# --------------------------------------------------------------------------- #
def test_factory_create_known_agents(fake_llm, fake_chain) -> None:
    assert isinstance(
        AgentFactory.create("rag", llm=fake_llm, rag_chain=fake_chain), RAGAgent
    )
    assert isinstance(AgentFactory.create("Code", llm=fake_llm), CodeAgent)
    assert isinstance(AgentFactory.create("WEB", llm=fake_llm), WebSearchAgent)
    assert isinstance(
        AgentFactory.create("summarizer", llm=fake_llm, rag_chain=fake_chain),
        SummarizerAgent,
    )


def test_factory_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="未知"):
        AgentFactory.create("unknown_agent", llm=_FakeLLM())


def test_factory_get_all_agents_with_and_without_rag_chain(
    fake_llm, fake_chain, caplog
) -> None:
    full = AgentFactory.get_all_agents(llm=fake_llm, rag_chain=fake_chain)
    assert set(full) == {"rag", "code", "web", "summarizer"}
    assert all(isinstance(a, BaseAgent) for a in full.values())

    # 未提供 rag_chain：RAG / Summarizer 应被跳过并告警
    partial = AgentFactory.get_all_agents(llm=fake_llm)
    assert set(partial) == {"code", "web"}


def test_base_agent_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseAgent(name="x", description="y", llm=_FakeLLM())  # type: ignore[abstract]
