# NexusRAG 🧠

> 基于 LangChain + LangGraph 构建的多智能体文档协作平台

## 📖 目录

- [项目简介](#项目简介)
- [核心功能](#核心功能)
- [技术栈](#技术栈)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [API 文档](#api-文档)
- [开发计划](#开发计划)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

---

## 🎯 项目简介

**动机**：企业知识散落在 PDF、Word、Markdown、TXT 等异构文档中，通用问答模型无法感知内部资料，而零散拼装的 RAG 脚本又难以维护、难以扩展。

**功能**：NexusRAG 是一个面向文档的知识问答与协作平台，提供「文档加载 → 智能分块 → 向量化入库 → 语义检索 → 大模型生成」的完整 RAG 流水线，并规划了多智能体协作与人机协同审核能力。

**展示**：阶段一已打通端到端链路——`python scripts/quick_start.py` 即可体验「丢入一份文档、提出一个问题、得到带来源溯源的回答」的完整流程（见[快速开始](#快速开始)）。

## ⚡ 核心功能

以「3+1」能力矩阵规划产品能力：

| 能力 | 说明 | 当前状态 |
| :--- | :--- | :--- |
| 🧭 自主规划中枢 | 由 LangGraph 驱动的多智能体编排，自动拆解复杂任务 | 阶段三规划中 |
| 📚 企业级知识库 | 多格式文档加载、智能分块、ChromaDB 向量存储、来源溯源 | ✅ 阶段一已落地 |
| 🤝 人机协作流 | 回答生成后支持人工审核、批注与采纳，构建可闭环的知识沉淀 | 阶段三规划中 |
| 🔍 白盒化观测 | 检索分数、引用来源、置信度全程可查，回答可追溯、可审计 | ✅ 阶段一已落地 |

## 🛠️ 技术栈

| 组件 | 技术选型 |
| :--- | :--- |
| LLM 框架 | LangChain 0.3.x + LangGraph |
| 向量数据库 | ChromaDB |
| Embedding | BAAI/bge-small-zh-v1.5 |
| Web 框架 | FastAPI |
| 前端 | Streamlit |
| 包管理 | uv / Poetry |
| 部署 | Docker + Docker Compose |

## 🚀 快速开始

### 环境要求

- Python 3.11+
- Docker（可选，用于启动远程 ChromaDB）

### 1. 克隆项目

```bash
git clone https://github.com/yourusername/nexusrag.git
cd nexusrag
```

### 2. 安装依赖（使用 conda + pip）

```bash
conda create -n nexusrag python=3.11 -y
conda activate nexusrag
pip install -e .
pip install pytest pytest-cov   # 开发 / 测试依赖
```

> 使用 uv 亦可：`uv sync --dev`

### 3. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 填入你的 OPENAI_API_KEY
```

> 可选：如需远程 ChromaDB，先执行 `docker compose up -d chromadb`，再在 `.env` 中设置 `CHROMA_HOST` / `CHROMA_PORT`。

### 4. 运行单元测试（验证安装成功）

```bash
pytest tests/ -v
```

### 5. 快速体验（处理第一个文档）

```bash
python scripts/quick_start.py
```

运行 `python scripts/check_health.py` 可一键体检各核心模块是否就绪。

## 📂 项目结构

```
NexusRAG/
├── src/                        # 核心源码
│   ├── loaders/                # 文档加载与分块（Day 2-3）
│   │   ├── loader_factory.py   #   加载器工厂：PDF / Word / Markdown / TXT
│   │   └── splitters.py        #   分块策略：recursive / semantic / markdown
│   ├── retrievers/             # 检索层（Day 4）
│   │   └── vector_store.py     #   ChromaDB 向量存储封装（CRUD + 阈值检索）
│   ├── chains/                 # 链式编排（Day 5）
│   │   └── rag_chain.py        #   RAG 问答链（索引 / 检索 / 生成 / 溯源）
│   ├── agents/                 # 多智能体（阶段三）
│   ├── tools/                  # 智能体工具集（阶段三）
│   ├── memory/                 # 对话记忆（阶段二）
│   ├── workflows/              # 工作流编排（阶段三）
│   ├── api/                    # FastAPI 服务（阶段四）
│   │   └── routes/
│   └── config.py               # 全局配置（.env 联动）
├── tests/                      # 单元测试（loaders / splitters / vector_store / rag_chain / api）
├── scripts/
│   ├── quick_start.py          # 一键体验完整 RAG 流程
│   └── check_health.py         # 模块健康检查
├── docs/                       # 设计文档
├── ui/                         # 前端（Streamlit，阶段四）
├── pyproject.toml              # 项目元数据与依赖
├── docker-compose.yml          # ChromaDB + API 容器编排
├── Dockerfile
├── .env.example                # 环境变量示例
└── README.md
```

## 📚 API 文档

Web API（FastAPI）规划于阶段四实现。上线后自动生成的交互式文档可通过以下地址访问（预留）：

```
http://localhost:8000/docs
http://localhost:8000/redoc
```

当前阶段建议以 `RAGChain`（`src/chains/rag_chain.py`）作为编程式调用入口：

```python
from chains import RAGChain
from retrievers import VectorStoreManager

vsm = VectorStoreManager(collection_name="demo")
rag = RAGChain(vsm, k=4)
rag.process_document("report.pdf")
result = rag.query("报告的核心结论是什么？")
print(result["answer"])        # 含 Sources 溯源
print(result["source_documents"])
```

## 🗺️ 开发路线图

- ☑ **阶段一（第 1 周）：基础架构与 RAG 核心**
  - Day 2 文档加载器 / Day 3 智能分块 / Day 4 向量存储 / Day 5 RAG 链 / Day 6-7 测试与文档收尾
- □ **阶段二（第 2 周）：混合检索与对话记忆**
  - Day 8-9 混合检索（向量 + BM25）/ Day 10 结果重排（Cross-Encoder）/ Day 11 多轮对话记忆
- □ **阶段三（第 3-4 周）：多智能体与工作流**
  - 自主规划中枢、工具调用、人机协作审核流
- □ **阶段四（第 4-5 周）：API 服务与前端界面**
  - FastAPI 接口层、Streamlit 可视化、白盒观测面板
- □ **阶段五（第 5 周末）：测试、部署与文档**
  - 全量测试、Docker 发布、用户文档

## 🤝 贡献指南

欢迎 Issue 和 PR！请参考 [CONTRIBUTING.md](CONTRIBUTING.md)（规划中）。提交代码前请确保：

```bash
pytest tests/ -v                                  # 全部通过
pytest tests/ --cov=src --cov-report=html         # 核心模块覆盖率 > 80%
```

## 📄 许可证

MIT License
