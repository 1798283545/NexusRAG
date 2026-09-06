"""``nexusrag-ui`` 控制台命令入口。

定位仓库内的 ``ui/streamlit_app.py`` 并把当前工作目录切换到 ``ui``
（确保 ``ui/.streamlit/config.toml`` 主题与上传限制生效），随后交给
Streamlit CLI 启动。端口 / 监听地址可通过环境变量覆盖：

- ``NEXUSRAG_UI_HOST``（默认 0.0.0.0）
- ``NEXUSRAG_UI_PORT``（默认 8501）
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _locate_app() -> Path:
    """定位 ui/streamlit_app.py：环境变量 > 源码仓库根 > 当前工作目录。"""
    override = os.getenv("NEXUSRAG_UI_APP", "").strip()
    if override:
        candidate = Path(override)
        if candidate.is_file():
            return candidate
    # src/api/ui_launcher.py → <repo>/src/api → <repo>/src → <repo>（可编辑安装）
    repo_root = Path(__file__).resolve().parents[2]
    for root in (repo_root, Path.cwd()):
        app_path = root / "ui" / "streamlit_app.py"
        if app_path.is_file():
            return app_path
    raise SystemExit(f"找不到前端入口文件（ui/streamlit_app.py），请从仓库目录运行或设置 NEXUSRAG_UI_APP")


def main() -> None:
    try:
        from streamlit.web import cli as stcli
    except ImportError as exc:  # pragma: no cover - 依赖缺失时的友好提示
        raise SystemExit("缺少 streamlit，请先安装依赖：uv sync 或 pip install streamlit") from exc

    app_path = _locate_app()

    # 切到 app 所在目录再启动，使 .streamlit/config.toml 生效
    os.chdir(str(app_path.parent))

    host = os.getenv("NEXUSRAG_UI_HOST", "0.0.0.0")
    port = os.getenv("NEXUSRAG_UI_PORT", "8501")
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        host,
        "--server.port",
        port,
    ]
    sys.exit(stcli.main())


if __name__ == "__main__":  # pragma: no cover
    main()
