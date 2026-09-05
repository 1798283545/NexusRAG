"""pandas 数据分析工具。

在隔离子进程中执行 pandas 分析代码（自动注入 ``import pandas as pd``），
返回 stdout / stderr，供 CodeAgent 进行描述统计与聚合分析。
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from tools.python_tool import execute_python


class _AnalysisArgs(BaseModel):
    code: str = Field(
        description="使用 pandas 编写的数据分析代码（变量名统一为 df），需 print() 输出关键结果"
    )
    csv_path: str = Field(
        default="", description="可选的 CSV 文件路径（workspace 内），会自动读取为 df"
    )


class DataAnalysisTool(BaseTool):
    """使用 pandas 对数据执行描述统计 / 聚合等分析。"""

    name: str = "pandas_analyzer"
    description: str = (
        "使用 pandas 进行数据分析（describe、groupby 聚合、merge 等）。"
        "传入分析代码与可选 CSV 路径，返回运行输出。"
    )
    args_schema: type[BaseModel] = _AnalysisArgs
    code_timeout: int = 60

    def _run(self, code: str, csv_path: str = "", **kwargs) -> str:
        preamble: list[str] = ["import pandas as pd", "import numpy as np"]
        if csv_path:
            # 路径由调用方保证在 workspace 内
            preamble.append(f"df = pd.read_csv({csv_path!r})")
        result = execute_python(code, timeout=self.code_timeout, preamble="\n".join(preamble))
        parts = []
        if result["stdout"]:
            parts.append(f"[stdout]\n{result['stdout']}")
        if result["stderr"]:
            parts.append(f"[stderr]\n{result['stderr']}")
        parts.append(f"[exit_code] {result['exit_code']}")
        return "\n".join(parts)

    async def _arun(self, code: str, csv_path: str = "", **kwargs) -> str:  # pragma: no cover
        return self._run(code, csv_path=csv_path, **kwargs)
