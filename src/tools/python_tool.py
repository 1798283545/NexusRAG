"""Python 代码执行工具。

在子进程隔离环境中执行 Python 代码（``sys.executable -c``），
设置超时与受限环境变量，返回 stdout / stderr，供 CodeAgent 使用。
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Dict, Optional

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

#: 子进程运行受限环境（关闭用户 site 包、字节码缓存与缓冲）
_RESTRICTED_ENV = {
    "PYTHONNOUSERSITE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONUNBUFFERED": "1",
}


def execute_python(
    code: str,
    timeout: int = 30,
    cwd: Optional[str] = None,
    preamble: str = "",
) -> Dict[str, str]:
    """在子进程中执行代码，返回 ``{stdout, stderr, exit_code}``。"""
    script = f"{preamble}\n{code}" if preamble else code
    env = dict(os.environ)
    env.update(_RESTRICTED_ENV)
    try:
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            env=env,
        )
        return {
            "stdout": (completed.stdout or "").strip(),
            "stderr": (completed.stderr or "").strip(),
            "exit_code": str(completed.returncode),
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": f"执行超时（>{timeout}s）", "exit_code": "-1"}
    except Exception as exc:  # noqa: BLE001 - 启动失败也要可观测
        return {"stdout": "", "stderr": f"执行启动失败: {exc}", "exit_code": "-1"}


class _PythonArgs(BaseModel):
    code: str = Field(description="待执行的 Python 代码，请以 print() 输出结果")


class PythonREPLTool(BaseTool):
    """在隔离子进程中执行 Python 代码，返回 stdout / stderr。"""

    name: str = "python_executor"
    description: str = (
        "在隔离环境中执行一段 Python 代码并返回运行输出。"
        "适用：数学计算、字符串处理、数据分析脚本等。代码需自包含并通过 print 输出结果。"
    )
    args_schema: type[BaseModel] = _PythonArgs
    code_timeout: int = 30

    def _run(self, code: str, **kwargs) -> str:
        result = execute_python(code, timeout=self.code_timeout)
        parts = []
        if result["stdout"]:
            parts.append(f"[stdout]\n{result['stdout']}")
        if result["stderr"]:
            parts.append(f"[stderr]\n{result['stderr']}")
        parts.append(f"[exit_code] {result['exit_code']}")
        return "\n".join(parts)

    async def _arun(self, code: str, **kwargs) -> str:  # pragma: no cover - 同步实现即可
        return self._run(code, **kwargs)
