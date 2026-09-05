"""workspace 文件操作工具。

仅允许在 ``workspace_dir`` 内读写 / 列出文件；对传入路径做规范化
并校验其位于 workspace 内，防止路径穿越逃逸。
"""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from config import settings


class _FileArgs(BaseModel):
    action: Literal["list", "read", "write", "delete"] = Field(description="操作类型")
    path: str = Field(description="相对 workspace 的文件路径（不允许 .. 逃逸）")
    content: str = Field(default="", description="write 操作时的文件内容")


class FileOperationTool(BaseTool):
    """在 workspace 内进行文件读写 / 列表 / 删除（含路径逃逸防护）。"""

    name: str = "file_operator"
    description: str = (
        "在 workspace 工作目录内操作文件：list 列出目录、read 读取文本、"
        "write 写入文本、delete 删除文件。path 一律使用相对路径。"
    )
    args_schema: type[BaseModel] = _FileArgs
    workspace_dir: str = "./workspace"

    def _resolve(self, path: str) -> str:
        """将相对路径规范化为 workspace 内绝对路径，越界则抛 ValueError。"""
        root = os.path.abspath(self.workspace_dir)
        target = os.path.abspath(os.path.join(root, path))
        if target != root and not target.startswith(root + os.sep):
            raise ValueError(f"路径越界：{path!r} 不在 workspace 目录内")
        return target

    def _run(self, action: str, path: str, content: str = "", **kwargs) -> str:
        target = self._resolve(path)
        if action == "list":
            if not os.path.isdir(target):
                return "（目录不存在）"
            names = sorted(os.listdir(target))
            return "\n".join(names) if names else "（空目录）"
        if action == "read":
            if not os.path.isfile(target):
                return f"（文件不存在：{path}）"
            with open(target, "r", encoding="utf-8", errors="replace") as handle:
                return handle.read()
        if action == "write":
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "w", encoding="utf-8") as handle:
                handle.write(content)
            return f"已写入 {len(content)} 字符到 {path}"
        if action == "delete":
            if os.path.isfile(target):
                os.remove(target)
                return f"已删除 {path}"
            return f"（文件不存在：{path}）"
        return f"不支持的操作：{action}"

    async def _arun(self, action: str, path: str, content: str = "", **kwargs) -> str:  # pragma: no cover
        return self._run(action, path, content=content, **kwargs)

    def __init__(self, workspace_dir: str | None = None, **kwargs) -> None:
        kwargs["workspace_dir"] = workspace_dir or settings.AGENT_WORKSPACE_DIR
        super().__init__(**kwargs)
