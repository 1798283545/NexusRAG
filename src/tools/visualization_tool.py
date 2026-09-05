"""图表生成工具（matplotlib / seaborn）。

在 workspace 内以 ``Agg`` 后端执行绘图代码，将图片保存为文件，
并汇报本次运行新增的图片文件。供 CodeAgent 生成数据可视化。
"""

from __future__ import annotations

import os
from typing import List

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from config import settings
from tools.python_tool import execute_python

#: 认可的图片后缀
_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".svg", ".pdf")


class _VisualArgs(BaseModel):
    code: str = Field(
        description="使用 matplotlib / seaborn 绘图的完整 Python 代码；需调用 plt.savefig('文件名.png') 保存图片"
    )
    output_name: str = Field(
        default="chart.png", description="期望生成的图片文件名（默认 chart.png）"
    )


class VisualizationTool(BaseTool):
    """在 workspace 内运行绘图代码并保存 / 汇报图片文件。"""

    name: str = "visualizer"
    description: str = (
        "使用 matplotlib / seaborn 生成图表，保存为图片文件并返回文件路径。"
        "适用：柱状图、折线图、散点图等数据可视化。"
    )
    args_schema: type[BaseModel] = _VisualArgs
    code_timeout: int = 60
    workspace_dir: str = "./workspace"

    def _snapshot_images(self) -> set[str]:
        """收集 workspace 内已有的图片文件绝对路径。"""
        found: set[str] = set()
        if not os.path.isdir(self.workspace_dir):
            return found
        for name in os.listdir(self.workspace_dir):
            full = os.path.join(self.workspace_dir, name)
            if os.path.isfile(full) and name.lower().endswith(_IMAGE_SUFFIXES):
                found.add(full)
        return found

    def _run(self, code: str, output_name: str = "chart.png", **kwargs) -> str:
        os.makedirs(self.workspace_dir, exist_ok=True)
        before = self._snapshot_images()
        preamble = "\n".join(
            [
                "import matplotlib",
                "matplotlib.use('Agg')",
                "import matplotlib.pyplot as plt",
                "import seaborn as sns",
            ]
        )
        result = execute_python(
            code, timeout=self.code_timeout, cwd=self.workspace_dir, preamble=preamble
        )
        created = self._snapshot_images() - before
        lines: List[str] = []
        if created:
            lines.append(f"已生成图片：{', '.join(sorted(created))}")
        else:
            lines.append("未检测到新生成的图片，请确认代码中调用了 plt.savefig() 且文件名以图片后缀结尾。")
        if result["stdout"]:
            lines.append(f"[stdout]\n{result['stdout']}")
        if result["stderr"]:
            lines.append(f"[stderr]\n{result['stderr']}")
        lines.append(f"[exit_code] {result['exit_code']}")
        return "\n".join(lines)

    async def _arun(self, code: str, output_name: str = "chart.png", **kwargs) -> str:  # pragma: no cover
        return self._run(code, output_name=output_name, **kwargs)

    def __init__(self, workspace_dir: str | None = None, **kwargs) -> None:
        kwargs["workspace_dir"] = workspace_dir or settings.AGENT_WORKSPACE_DIR
        super().__init__(**kwargs)
