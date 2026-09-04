"""NexusRAG API 服务入口。"""

from fastapi import FastAPI

app = FastAPI(title="NexusRAG", version="0.1.0")


def run() -> None:
    """启动 API 服务（供 nexusrag-serve 命令调用）。"""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
