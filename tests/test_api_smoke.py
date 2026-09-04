"""API 服务入口冒烟测试。

确保 FastAPI 应用可正常导入并暴露 `run()` 启动函数（覆盖发布入口）。
"""

from api.main import app, run


def test_api_app_metadata():
    """应用应携带正确的标题与版本号。"""
    assert app.title == "NexusRAG"
    assert app.version == "0.1.0"


def test_run_launches_uvicorn(monkeypatch):
    """调用 run() 应委托给 uvicorn.run（monkeypatch 避免真正监听端口）。"""
    calls: dict = {}

    class _FakeUvicorn:
        @staticmethod
        def run(*args, **kwargs):
            calls["args"] = args
            calls["kwargs"] = kwargs

    monkeypatch.setitem(__import__("sys").modules, "uvicorn", _FakeUvicorn)

    run()

    assert calls["kwargs"]["host"] == "0.0.0.0"
    assert calls["kwargs"]["port"] == 8000
    assert calls["args"][0] is app
