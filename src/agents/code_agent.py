"""代码分析专业智能体：负责执行代码、数据分析与图表生成。"""

from __future__ import annotations

from typing import Any, List, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

from agents.base_agent import BaseAgent
from config import settings
from tools.analysis_tool import DataAnalysisTool
from tools.file_tool import FileOperationTool
from tools.python_tool import PythonREPLTool
from tools.visualization_tool import VisualizationTool


class CodeAgent(BaseAgent):
    """代码执行 / 数据分析 / 图表生成智能体。"""

    def __init__(
        self,
        llm: BaseLanguageModel,
        workspace_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """初始化 CodeAgent。

        Args:
            llm: 大语言模型实例。
            workspace_dir: 工作目录（限制所有文件操作范围），
                默认取配置 ``AGENT_WORKSPACE_DIR``。
            **kwargs: 透传给 BaseAgent 的其余参数。
        """
        self.workspace_dir: str = workspace_dir or settings.AGENT_WORKSPACE_DIR
        super().__init__(
            name="CodeAgent",
            description=(
                "负责执行 Python 代码、数据分析、图表生成和文件操作。"
                "适用场景：计算、数据可视化、代码生成。"
            ),
            llm=llm,
            **kwargs,
        )

    def _get_default_tools(self) -> List[BaseTool]:
        """返回默认工具：代码执行 / pandas 分析 / 图表生成 / 文件操作。"""
        return [
            PythonREPLTool(),
            DataAnalysisTool(),
            VisualizationTool(workspace_dir=self.workspace_dir),
            FileOperationTool(workspace_dir=self.workspace_dir),
        ]

    def _get_system_prompt(self) -> str:
        return """你是代码分析专业智能体，专门负责执行代码和数据处理任务。

你的工作流程：
1. 当用户需要计算、数据处理时，使用 PythonREPLTool
2. 当用户需要图表生成时，使用 VisualizationTool
3. 注意：所有文件操作限制在 workspace 目录内
4. 执行完代码后，解释结果并总结关键发现"""
