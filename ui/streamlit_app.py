"""NexusRAG 前端主应用（Streamlit，第四阶段 Day 5-7）。

提供五个页签：

- 首页 / 仪表盘：后端健康、文档统计、智能体状态概览；
- 对话：SSE 流式打字机效果 + 思考过程 + 来源文档 + 多轮会话管理；
- 文档管理：上传 / 删除 / 清空知识库；
- 智能体：手动触发各专业智能体并查看输出；
- 工作流：执行多智能体 LangGraph 工作流（含 HITL 人工审核恢复）。

所有后端调用经 ``ui.utils.api_client.APIClient``，运行方式见 ui/README.md。
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

#: streamlit 只把脚本所在目录加入 sys.path；把仓库根目录也加入，便于导入 ui 包
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

from ui.utils.api_client import APIClient, default_base_url  # noqa: E402

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #
_MAX_HISTORY = 100  # 界面最多保留的消息条数，避免长会话内存膨胀

AGENT_OPTIONS: Dict[str, str] = {
    "supervisor": "🧭 Supervisor（任务总控编排）",
    "rag": "📚 RAG（知识库检索问答）",
    "code": "💻 Code（代码执行 / 数据分析）",
    "web": "🌐 Web（联网搜索）",
    "summarizer": "📝 Summarizer（文档总结）",
}

UPLOAD_TYPES = ["pdf", "docx", "txt", "md", "html"]

_CSS = """
<style>
    .app-title {
        background: linear-gradient(90deg, #1F6FEB, #4CAF50);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.6rem;
        font-weight: 700;
    }
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.4rem;
    }
    .stButton > button, .stDownloadButton > button {
        border-radius: 10px;
    }
</style>
"""


# --------------------------------------------------------------------------- #
# 客户端缓存与全局状态
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def _cached_client(base_url: str) -> APIClient:
    return APIClient(base_url=base_url)


def _client() -> APIClient:
    return _cached_client(st.session_state["api_base_url"])


def _try_health(client: APIClient) -> Optional[Dict[str, Any]]:
    """健康探测（失败返回 None，不抛异常）。"""
    try:
        return client.health_check()
    except Exception:  # noqa: BLE001
        return None


def _init_state() -> None:
    if "api_base_url" not in st.session_state:
        st.session_state["api_base_url"] = default_base_url()
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = uuid.uuid4().hex
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "wf_session" not in st.session_state:
        st.session_state["wf_session"] = ""
    if "wf_snapshot" not in st.session_state:
        st.session_state["wf_snapshot"] = None


# --------------------------------------------------------------------------- #
# 来源 / 消息展示工具
# --------------------------------------------------------------------------- #
def _source_title(src: Dict[str, Any]) -> str:
    """从来源条目（流式 SSE 或非流式）中提取可读标题。"""
    meta = src.get("metadata") or {}
    title = (
        meta.get("file_name")
        or meta.get("source")
        or src.get("file_name")
        or src.get("source")
        or "未知来源"
    )
    return str(title)


def _source_meta(src: Dict[str, Any]) -> Dict[str, Any]:
    """统一来源条目元信息（兼容流式 metadata 与非流式扁平结构）。"""
    meta = src.get("metadata") or {}
    return {
        "page_number": meta.get("page_number", src.get("page_number")),
        "score": meta.get("score", src.get("score")),
    }


def _render_sources_block(sources: List[Dict[str, Any]], max_items: int = 5) -> None:
    """在可折叠展开区中渲染来源文档列表。"""
    if not sources:
        return
    with st.expander(f"📚 参考来源（{len(sources)} 条）", expanded=False):
        for src in sources[:max_items]:
            content = str(src.get("content") or src.get("page_content") or "").strip()
            meta = _source_meta(src)
            header = _source_title(src)
            if meta["page_number"] is not None:
                header += f" · 第 {meta['page_number']} 页"
            if meta["score"] is not None:
                try:
                    header += f" · 相关度 {float(meta['score']):.2f}"
                except (TypeError, ValueError):
                    pass
            st.markdown(f"**{header}**")
            st.caption(content[:240] + ("…" if len(content) > 240 else ""))


def _render_message(msg: Dict[str, Any]) -> None:
    """渲染单条历史消息（user / assistant，附带来源）。"""
    role = msg.get("role", "assistant")
    content = str(msg.get("content") or "")
    if role == "assistant" and not content:
        return
    with st.chat_message(role):
        st.markdown(content if content else "_(空回复)_")
        sources = msg.get("sources") or []
        if sources:
            _render_sources_block(list(sources))


# --------------------------------------------------------------------------- #
# 首页 / 仪表盘
# --------------------------------------------------------------------------- #
def _render_dashboard(client: APIClient) -> None:
    st.markdown("## 🏠 系统概览")

    health = _try_health(client)
    docs: List[Dict[str, Any]] = []
    try:
        docs = client.list_documents()
    except Exception as exc:  # noqa: BLE001
        st.error(f"获取文档列表失败：{exc}")

    try:
        agent_status = client.get_agent_status().get("agents", {})
    except Exception:  # noqa: BLE001
        agent_status = {}

    total_chunks = sum(int(d.get("chunk_count") or 0) for d in docs)
    ready_agents = sum(1 for v in agent_status.values() if v == "ready")

    c1, c2, c3, c4 = st.columns(4)
    backend_status = health.get("status", "offline") if health else "offline"
    c1.metric("后端状态", backend_status)
    c2.metric("文档数", len(docs))
    c3.metric("向量块总数", total_chunks)
    c4.metric("可用智能体", f"{ready_agents}/{len(agent_status) or 5}")

    with st.expander("🔌 组件健康", expanded=False):
        if health is None:
            st.warning("无法连接后端，请检查服务与 API 地址。")
        else:
            for name, ok in (health.get("components") or {}).items():
                st.markdown(f"{'✅' if ok else '❌'} **{name}**：{'正常' if ok else '不可用'}")
            st.caption(f"版本：{health.get('version', '-')}")

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("#### 🤖 智能体状态")
        if not agent_status:
            st.caption("暂无智能体信息")
        for key, val in agent_status.items():
            icon = "🟢" if val == "ready" else "🔴"
            st.markdown(f"{icon} {AGENT_OPTIONS.get(key, key)}：`{val}`")
    with col_right:
        st.markdown("#### 📄 最近文档")
        if not docs:
            st.info("暂无文档，请到「文档管理」上传。")
        else:
            table = [
                {
                    "文档": d.get("name"),
                    "类型": d.get("file_type"),
                    "块数": int(d.get("chunk_count") or 0),
                    "来源": str(d.get("source") or ""),
                }
                for d in docs
            ]
            st.dataframe(table[:8], use_container_width=True, hide_index=True)
            if len(docs) > 8:
                st.caption(f"…共 {len(docs)} 个文档，完整列表见「文档管理」")


# --------------------------------------------------------------------------- #
# 对话（SSE 流式 + 降级）
# --------------------------------------------------------------------------- #
def _stream_chat(client: APIClient, prompt: str) -> Dict[str, Any]:
    """执行一次 SSE 流式对话，渲染打字机效果，返回可存入历史的消息字典。"""
    full_parts: List[str] = []
    sources: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {}
    error_info: Optional[str] = None
    fallback = False

    with st.chat_message("assistant"):
        status_slot = st.empty()       # 顶部：thinking 状态提示
        sources_slot = st.container()  # 中部：来源文档
        answer_slot = st.empty()       # 下部：打字机文本

        try:
            for event in client.chat_stream(
                prompt,
                session_id=st.session_state["session_id"],
                use_memory=True,
            ):
                etype = event.get("type")
                if etype == "session":
                    sid = event.get("session_id")
                    if sid:
                        st.session_state["session_id"] = sid
                elif etype == "thinking":
                    text = str(event.get("content") or "").strip()
                    if text:
                        status_slot.caption("🧠 " + text)
                elif etype == "sources":
                    status_slot.empty()
                    got = list(event.get("sources") or [])
                    if got:
                        sources = got
                        with sources_slot:
                            _render_sources_block(sources)
                elif etype == "token":
                    token = str(event.get("content") or "")
                    if token:
                        full_parts.append(token)
                        answer_slot.markdown("".join(full_parts) + "▌")
                elif etype == "done":
                    stats = {
                        "total_tokens": event.get("total_tokens"),
                        "execution_time": event.get("execution_time"),
                    }
                elif etype == "error":
                    error_info = str(event.get("detail") or event.get("error") or "生成中断")
                    status_slot.empty()
                    status_slot.caption("❌ " + error_info)
        except Exception as exc:  # noqa: BLE001 - 连接中断 / 服务异常
            error_info = str(exc)

        content = "".join(full_parts).strip()
        if content:
            answer_slot.markdown(content)

        # 出错且未产出任何内容 → 降级为非流式问答
        if error_info and not content:
            status_slot.empty()
            fallback = True
            with st.spinner("流式连接异常，正在尝试非流式问答…"):
                try:
                    resp = client.chat(
                        prompt,
                        session_id=st.session_state["session_id"],
                        use_memory=True,
                    )
                    content = str(resp.get("answer") or "").strip()
                    sources = list(resp.get("sources") or [])
                except Exception as exc:  # noqa: BLE001
                    st.error(f"❌ 后端不可用：{exc}")
            if content:
                answer_slot.markdown(content)
                if sources:
                    with sources_slot:
                        _render_sources_block(sources)
        elif error_info and content:
            st.caption(f"⚠️ 响应中断：{error_info}")

        if stats.get("total_tokens") is not None:
            note = f"⏱ 完成：{stats.get('total_tokens')} tokens"
            if stats.get("execution_time") is not None:
                note += f" · 耗时 {float(stats['execution_time']):.2f}s"
            st.caption(note)

    result: Dict[str, Any] = {"role": "assistant", "content": content, "sources": sources}
    result["meta"] = {"fallback": fallback}
    if error_info:
        result["meta"]["error"] = error_info
    return result


def _render_chat_tab(client: APIClient) -> None:
    st.markdown("## 💬 智能对话")
    header_col, clear_col = st.columns([6, 1])
    header_col.caption(f"会话 ID：{st.session_state['session_id']}")
    if clear_col.button("🗑️ 清空", use_container_width=True, help="仅清空界面消息，不删除后端记忆"):
        st.session_state["messages"] = []
        st.rerun()

    for msg in st.session_state["messages"][-_MAX_HISTORY:]:
        _render_message(msg)

    prompt = st.chat_input("请输入您的问题，Enter 发送…")
    if not prompt:
        return

    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    result = _stream_chat(client, prompt)
    st.session_state["messages"].append(result)
    st.session_state["messages"] = st.session_state["messages"][-_MAX_HISTORY:]


# --------------------------------------------------------------------------- #
# 文档管理
# --------------------------------------------------------------------------- #
def _render_documents_tab(client: APIClient) -> None:
    st.markdown("## 📁 文档管理")

    docs: List[Dict[str, Any]] = []
    try:
        docs = client.list_documents()
    except Exception as exc:  # noqa: BLE001
        st.error(f"获取文档列表失败：{exc}")

    left, right = st.columns([3, 2])
    with left:
        uploaded = st.file_uploader(
            "选择要上传的文档",
            type=UPLOAD_TYPES,
            accept_multiple_files=True,
            help="支持 pdf / docx / txt / md / html，将自动完成加载、分割与向量化。",
        )
        if st.button("📤 上传并索引", type="primary", disabled=not uploaded):
            for f in uploaded:
                try:
                    with st.spinner(f"正在处理 {f.name}…"):
                        result = client.upload_document(f.name, f.getvalue())
                    st.toast(f"✅ {f.name}：入库 {result.get('chunk_count', 0)} 个文本块")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"❌ {f.name} 上传失败：{exc}")
            st.rerun()
    with right:
        st.markdown("#### 📊 统计")
        total_chunks = sum(int(d.get("chunk_count") or 0) for d in docs)
        by_type: Dict[str, int] = {}
        for d in docs:
            t = d.get("file_type") or "unknown"
            by_type[t] = by_type.get(t, 0) + 1
        st.metric("文档总数", len(docs))
        st.metric("向量块总数", total_chunks)
        if by_type:
            st.caption("　".join(f"{t}: {n}" for t, n in by_type.items()))
        with st.expander("⚠️ 清空知识库（危险操作）", expanded=False):
            agree = st.checkbox("我确认清空全部向量数据")
            if st.button("清空全部文档", disabled=not agree):
                try:
                    resp = client.clear_documents()
                    st.success(f"已清空 {resp.get('deleted_count', 0)} 个向量块")
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"清空失败：{exc}")

    st.markdown("#### 📄 已索引文档")
    if not docs:
        st.info("暂无文档，请先上传。")
        return

    for doc in docs:
        name = doc.get("name") or "(unnamed)"
        c1, c2, c3, c4, c5 = st.columns([4, 1, 1, 2, 1])
        c1.markdown(f"**{name}**")
        c2.markdown(f"`{doc.get('file_type') or '-'}`")
        c3.markdown(f"{doc.get('chunk_count') or 0} 块")
        c4.caption(str(doc.get("created_at") or doc.get("source") or "")[:40])
        if c5.button("删除", key=f"del_{name}"):
            try:
                resp = client.delete_document(name)
                st.toast(f"🗑️ 已删除 {name}（{resp.get('deleted_count', 0)} 块）")
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"删除失败：{exc}")
    st.caption("提示：同名文档重新上传会先清理旧向量块（幂等更新）。")


# --------------------------------------------------------------------------- #
# 智能体控制
# --------------------------------------------------------------------------- #
def _render_agents_tab(client: APIClient) -> None:
    st.markdown("## 🤖 智能体控制")
    st.caption("手动触发各专业智能体，查看执行结果与使用的工具。")

    label_to_key = {label: key for key, label in AGENT_OPTIONS.items()}
    choice = st.selectbox("选择智能体", list(AGENT_OPTIONS.values()))
    agent_type = label_to_key[choice]

    query = st.text_area(
        "输入指令",
        height=110,
        placeholder="例如：帮我总结知识库中关于 LangChain 的内容",
    )
    if st.button("▶️ 执行", type="primary", disabled=not query.strip()):
        with st.spinner(f"正在执行 {agent_type}…"):
            try:
                result = client.run_agent(agent_type, query)
            except Exception as exc:  # noqa: BLE001
                st.error(f"❌ 执行失败：{exc}")
                return
        st.success(
            f"✅ 执行完成 · {agent_type} · 耗时 {float(result.get('execution_time') or 0):.2f}s"
        )
        st.markdown("### 📤 输出")
        st.markdown(str(result.get("output") or "_(无输出)_"))
        tools = list(result.get("tools_used") or [])
        if tools:
            st.markdown("### 🔧 使用的工具")
            st.markdown("　".join(f"`{t}`" for t in tools))


# --------------------------------------------------------------------------- #
# 工作流
# --------------------------------------------------------------------------- #
def _compact_item(item: Any) -> str:
    """把计划 / 追踪条目压缩成一行可读文本。"""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        parts: List[str] = []
        for key in ("step", "agent", "tool", "status", "task", "description", "query"):
            value = item.get(key)
            if value is None:
                continue
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            parts.append(f"{key}={value}")
        return " · ".join(parts) if parts else json.dumps(item, ensure_ascii=False)
    return str(item)


def _render_items(title: str, items: Any) -> None:
    if not items:
        return
    with st.expander(f"{title}（{len(items)}）", expanded=False):
        for index, item in enumerate(items, start=1):
            st.markdown(f"{index}. {_compact_item(item)}")


def _render_workflow_result(client: APIClient, snap: Dict[str, Any]) -> None:
    status = snap.get("status")
    if status == "done":
        st.success("✅ 工作流执行完成")
        final_answer = str(snap.get("final_answer") or "")
        if final_answer:
            st.markdown("### 🤖 最终回答")
            st.info(final_answer)
        _render_items("📋 执行计划", snap.get("plan"))
        _render_items("🔍 执行追踪", snap.get("execution_trace"))
        if snap.get("error"):
            st.caption(f"⚠️ 附带错误信息：{snap.get('error')}")
        st.caption(f"⏱ 耗时 {float(snap.get('execution_time') or 0):.2f}s")
    elif status == "awaiting_review":
        st.warning("⚠️ 工作流暂停，等待人工审核（Human-in-the-Loop）")
        pending = snap.get("pending_action") or {}
        if pending:
            with st.expander("📌 待审核操作", expanded=True):
                st.json(pending)

        feedback = st.text_input(
            "审核意见",
            value="approve",
            help="approve=批准继续；reject=拒绝；或直接输入修改说明。",
            key="wf_feedback",
        )
        left, right = st.columns(2)
        review: Optional[str] = None
        if left.button("✅ 批准并继续", use_container_width=True, type="primary"):
            review = feedback.strip() or "approve"
        if right.button("❌ 拒绝", use_container_width=True):
            review = "reject"

        if review:
            try:
                new_snap = client.resume_workflow(snap.get("session_id"), review)
                st.session_state["wf_snapshot"] = new_snap
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"❌ 恢复工作流失败：{exc}")
    elif status == "error":
        st.error(f"❌ 工作流执行出错：{snap.get('error') or '未知错误'}")
    else:
        st.warning(f"工作流状态：{status}")

    if st.button("🧹 清除本次工作流结果"):
        st.session_state["wf_snapshot"] = None
        st.session_state["wf_session"] = ""
        st.rerun()


def _render_workflow_tab(client: APIClient) -> None:
    st.markdown("## ⚙️ 多智能体工作流")
    st.caption("执行完整的多智能体协作工作流（规划 → 执行 → 整合），展示计划与过程。")

    task = st.text_area(
        "复杂任务描述",
        height=110,
        placeholder="例如：分析知识库中的销售文档，联网搜索最新行业趋势，并生成一份对比报告",
    )
    st.caption(
        "人工审核（HITL）由服务端 WORKFLOW_ENABLE_HITL 全局配置决定；"
        "若任务在审核节点暂停，将自动展示审批入口。"
    )

    if st.button("🚀 启动工作流", type="primary", disabled=not task.strip()):
        try:
            with st.spinner("工作流执行中…"):
                snap = client.run_workflow(
                    task, session_id=st.session_state["wf_session"] or None
                )
        except Exception as exc:  # noqa: BLE001
            st.error(f"❌ 工作流启动失败：{exc}")
            return
        st.session_state["wf_session"] = snap.get("session_id") or ""
        st.session_state["wf_snapshot"] = snap
        st.rerun()

    snap = st.session_state.get("wf_snapshot")
    if not snap:
        st.info("尚未执行工作流。")
        return

    st.caption(f"工作流会话：{snap.get('session_id') or '-'}")
    if st.button("🔄 查询最新状态"):
        try:
            st.session_state["wf_snapshot"] = client.workflow_status(snap["session_id"])
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(f"状态查询失败：{exc}")
    _render_workflow_result(client, st.session_state["wf_snapshot"])


# --------------------------------------------------------------------------- #
# 侧边栏
# --------------------------------------------------------------------------- #
def _render_sidebar(client: APIClient) -> None:
    st.markdown('<div class="app-title">🧠 NexusRAG</div>', unsafe_allow_html=True)
    st.caption("多智能体文档协作平台")

    st.markdown("---")
    st.markdown("#### 🔌 后端连接")
    api_url = st.text_input(
        "API 地址",
        value=st.session_state["api_base_url"],
        key="sidebar_api_url",
    )
    if st.button("连接 / 刷新", use_container_width=True):
        url = api_url.strip().rstrip("/") or default_base_url()
        if url != st.session_state["api_base_url"]:
            st.session_state["api_base_url"] = url
            _cached_client.clear()
        st.rerun()

    health = _try_health(client)
    if health is None:
        st.error("⚠️ 无法连接后端")
    else:
        status = health.get("status", "unknown")
        if status == "healthy":
            st.success(f"✅ 系统健康 · v{health.get('version', '-')}")
        else:
            st.warning(f"⚠️ 系统降级（{status}）")
        for name, ok in (health.get("components") or {}).items():
            st.markdown(f"{'✅' if ok else '❌'} `{name}`")

    st.markdown("---")
    st.markdown("#### 🤖 智能体")
    try:
        agent_status = client.get_agent_status().get("agents", {})
    except Exception:  # noqa: BLE001
        agent_status = {}
    if not agent_status:
        st.caption("暂无智能体信息")
    for key, val in agent_status.items():
        icon = "🟢" if val == "ready" else "🔴"
        st.markdown(f"{icon} {AGENT_OPTIONS.get(key, key)}：`{val}`")

    st.markdown("---")
    with st.expander("💬 会话管理"):
        sid = st.text_input(
            "会话 ID", value=st.session_state["session_id"], key="sidebar_session_id"
        )
        left, right = st.columns(2)
        if left.button("切换会话", use_container_width=True):
            st.session_state["session_id"] = (sid or "").strip() or uuid.uuid4().hex
            st.session_state["messages"] = []
            st.rerun()
        if right.button("🆕 新会话", use_container_width=True):
            st.session_state["session_id"] = uuid.uuid4().hex
            st.session_state["messages"] = []
            st.rerun()
        st.caption("切换会话会清空界面历史；后端按 session_id 记忆多轮对话。")

    st.markdown("---")
    st.caption("NexusRAG · v0.1.0")


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(
        page_title="NexusRAG · 多智能体文档协作平台",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_CSS, unsafe_allow_html=True)

    _init_state()
    client = _client()

    with st.sidebar:
        _render_sidebar(client)

    tab_home, tab_chat, tab_docs, tab_agents, tab_wf = st.tabs(
        ["🏠 首页", "💬 对话", "📁 文档管理", "🤖 智能体", "⚙️ 工作流"]
    )
    with tab_home:
        _render_dashboard(client)
    with tab_chat:
        _render_chat_tab(client)
    with tab_docs:
        _render_documents_tab(client)
    with tab_agents:
        _render_agents_tab(client)
    with tab_wf:
        _render_workflow_tab(client)


if __name__ == "__main__":
    main()
