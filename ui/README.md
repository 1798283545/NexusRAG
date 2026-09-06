# NexusRAG Streamlit 前端

第四阶段 Day 5-7 的可视化操作界面，基于 [Streamlit](https://streamlit.io)，
通过 REST / SSE 调用后端 API，提供：文档管理、流式对话（打字机效果 +
来源 / 思考过程展示）、智能体控制、多智能体工作流（含 HITL 审核）。

## 目录结构

```text
ui/
├── streamlit_app.py        # 主应用（5 个页签）
├── utils/
│   ├── __init__.py
│   └── api_client.py       # 后端 REST / SSE 客户端封装
├── .streamlit/
│   └── config.toml         # 主题 / 服务端配置
└── README.md
```

## 环境准备

1. 安装依赖（含 streamlit / httpx）：

   ```bash
   uv sync                 # 或 pip install -e .
   ```

2. 先启动后端 API（默认 `http://127.0.0.1:8000`）：

   ```bash
   nexusrag-serve
   # 或：uv run uvicorn api.main:app --reload --port 8000
   ```

## 启动前端

方式一：通过项目脚本命令（推荐，自动应用 ui/.streamlit/config.toml）：

```bash
nexusrag-ui
# 可选环境变量：NEXUSRAG_UI_HOST / NEXUSRAG_UI_PORT
```

方式二：手动运行：

```bash
cd ui
streamlit run streamlit_app.py
# 浏览器打开 http://localhost:8501
```

## 后端地址配置

默认地址解析顺序：环境变量 `NEXUSRAG_API_URL` > `src/config.py` 的
`API_HOST/API_PORT` > `http://127.0.0.1:8000`。也可在界面左侧「后端连接」
中手动填写 API 地址并点击「连接 / 刷新」。

```bash
export NEXUSRAG_API_URL=http://192.168.1.10:8000
```

## 功能说明

| 页签 | 说明 |
| --- | --- |
| 🏠 首页 | 后端健康、组件状态、文档统计、智能体状态概览 |
| 💬 对话 | SSE 流式打字机效果；展开显示思考过程与参考来源；按会话 ID 多轮记忆 |
| 📁 文档管理 | 上传 / 多选上传、按类型统计、删除单个文档、清空知识库 |
| 🤖 智能体 | 手动触发 supervisor / rag / code / web / summarizer |
| ⚙️ 工作流 | 运行多智能体工作流，展示计划与执行追踪；HITL 暂停时提供批准 / 拒绝 |

> 注意：工作流是否启用人工审核由后端 `WORKFLOW_ENABLE_HITL` 全局配置决定。
