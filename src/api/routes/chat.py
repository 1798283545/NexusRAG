"""对话接口（含 SSE 流式）。

基于 RAGChain 提供普通问答与流式问答；会话状态通过
``ConversationMemoryManager`` 以 ``session_id`` 维度在进程内保存
（buffer / buffer_window / summary 策略），支持多轮对话记忆。

``POST /api/chat/stream`` 遵循 SSE 事件契约，逐帧发送：

- ``session``  —— 流开始，携带会话 ID；
- ``thinking``  —— 检索 / 加载历史等阶段提示（可折叠展示）；
- ``sources``   —— 生成前发送已检索到的引用文档；
- ``token``     —— 逐 token 输出（打字机效果，支持缓冲批量）；
- ``done``      —— 正常结束（携带 total_tokens / execution_time）；
- ``error``     —— 流式中断 / 超时 / 服务异常时终止。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, AsyncIterator, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from api.dependencies import get_rag_chain, get_settings
from api.schemas import ChatRequest, ChatResponse

router = APIRouter()
logger = logging.getLogger(__name__)


def _memory_for(session_id: str, use_memory: bool):
    """构建（或复用）会话记忆管理器；use_memory=False 返回 None。"""
    if not use_memory:
        return None
    # ConversationMemoryManager 内部按 session_id 缓存单例，直接构造即可复用
    from memory import ConversationMemoryManager

    settings = get_settings()
    return ConversationMemoryManager(
        max_window_size=settings.MEMORY_WINDOW_SIZE,
        memory_type=settings.MEMORY_TYPE,
        session_id=session_id,
    )


def _encode(payload: Dict[str, Any]) -> str:
    """将事件字典编码为 SSE 帧（``data: {...}`` + 空行结尾）。"""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _to_sources(result: Dict[str, Any]) -> list:
    """把链返回的来源文档归一化为可 JSON 序列化结构。"""
    return list(result.get("source_documents") or [])


# --------------------------------------------------------------------------- #
# 非流式问答
# --------------------------------------------------------------------------- #
@router.post("/", response_model=ChatResponse, summary="普通问答")
def chat(
    request: ChatRequest,
    rag_chain: Any = Depends(get_rag_chain),
) -> ChatResponse:
    """一次 RAG 问答；携带 session_id 时自动读写多轮记忆。"""
    session_id = request.session_id or uuid.uuid4().hex
    memory = _memory_for(session_id, request.use_memory)

    if memory is not None:
        memory.add_user_message(request.query)

    started = time.perf_counter()
    try:
        result = rag_chain.query(query=request.query, k=request.k)
    except Exception as exc:  # noqa: BLE001 - 检索/生成失败返回 500
        logger.exception("对话问答失败")
        raise HTTPException(status_code=500, detail=f"问答执行失败：{exc}") from exc
    elapsed = time.perf_counter() - started

    answer = str(result.get("answer", ""))
    if memory is not None:
        try:
            memory.add_ai_message(answer)
        except Exception:  # noqa: BLE001 - 记忆写入失败不阻断回答
            logger.exception("记忆写入失败（session=%s）", session_id)

    return ChatResponse(
        answer=answer,
        sources=_to_sources(result),
        session_id=session_id,
        execution_time=round(elapsed, 3),
    )


# --------------------------------------------------------------------------- #
# SSE 流式问答
# --------------------------------------------------------------------------- #
@router.post("/stream", summary="流式问答（SSE 打字机）")
async def chat_stream(
    request: ChatRequest,
    http_request: Request,
    rag_chain: Any = Depends(get_rag_chain),
) -> StreamingResponse:
    """以 Server-Sent Events 流式返回：session → sources/thinking → token → done。

    特性：

    - 逐 token 打字机效果（帧间隔由 ``SSE_CHUNK_DELAY`` 控制）；
    - ``SSE_BUFFER_SIZE`` > 1 时批量打包 token，减少网络往返；
    - 生成前发送 ``sources``（来源）与 ``thinking``（思考过程）事件；
    - 支持客户端断开检测与 ``SSE_TIMEOUT`` 超时终止；
    - 记忆在完成后写入；断连时尽力保存已生成的部分内容。
    """
    session_id = request.session_id or uuid.uuid4().hex
    cfg = get_settings()
    memory = _memory_for(session_id, request.use_memory)

    async def generate() -> AsyncIterator[str]:
        # 计时与缓冲参数
        buffer_size = max(1, int(cfg.SSE_BUFFER_SIZE))
        chunk_delay = max(0.0, float(cfg.SSE_CHUNK_DELAY))
        timeout_s = max(0.0, float(cfg.SSE_TIMEOUT))
        started = time.perf_counter()
        collected: List[str] = []
        pending: List[str] = []
        saw_done = False
        saw_error = False

        async def flush_pending() -> AsyncIterator[str]:
            """把缓存的 token 按 buffer_size 分组发送（可选打字延迟）。"""
            while pending:
                group = pending[:buffer_size]
                del pending[:buffer_size]
                yield _encode({"type": "token", "content": "".join(group)})
                if chunk_delay > 0:
                    await asyncio.sleep(chunk_delay)

        def save_ai(text: str) -> None:
            """将 AI 回答写入记忆；失败仅记录日志，不中断流式输出。"""
            if memory is None or not text:
                return
            try:
                memory.add_ai_message(text)
            except Exception:  # noqa: BLE001
                logger.exception("流式回答写入记忆失败（session=%s）", session_id)

        try:
            # 1. session 事件：告知前端会话 ID
            yield _encode({"type": "session", "session_id": session_id})
            logger.info("SSE 流开始：session=%s, query_length=%d", session_id, len(request.query))

            # 2. 记忆：先报告历史概况，再写入本轮用户问题
            if memory is not None:
                try:
                    history_count = len(memory.get_messages_windowed())
                except Exception:  # noqa: BLE001
                    history_count = 0
                if cfg.SSE_ENABLE_THINKING and history_count > 0:
                    yield _encode(
                        {
                            "type": "thinking",
                            "content": f"已加载对话历史，共 {history_count} 条历史消息",
                            "step": 0,
                        }
                    )
                try:
                    memory.add_user_message(request.query)
                except Exception:  # noqa: BLE001
                    logger.exception("用户消息写入记忆失败（session=%s）", session_id)

            # 3. 逐事件转发 RAG 链的流式输出
            for event in rag_chain.stream_query(
                query=request.query,
                k=request.k,
                yield_sources=True,
                yield_thinking=bool(cfg.SSE_ENABLE_THINKING),
            ):
                # 3.1 客户端断开：保存已生成的部分后终止
                if await http_request.is_disconnected():
                    logger.info("SSE 客户端断开：session=%s", session_id)
                    save_ai("".join(collected).strip())
                    return

                # 3.2 超时保护：超过 SSE_TIMEOUT 发送 error 帧终止
                if timeout_s > 0 and (time.perf_counter() - started) > timeout_s:
                    logger.warning("SSE 超时终止：session=%s（%.1fs）", session_id, timeout_s)
                    yield _encode(
                        {
                            "type": "error",
                            "error": "stream_timeout",
                            "detail": f"流式响应超过 {timeout_s:g}s 未完成，已终止。",
                        }
                    )
                    saw_error = True
                    return

                event_type = event.get("type")
                if event_type == "token":
                    chunk = str(event.get("content") or "")
                    if chunk:
                        collected.append(chunk)
                    pending.append(chunk)
                    if len(pending) >= buffer_size:
                        async for frame in flush_pending():
                            yield frame
                    continue

                # 非 token 事件：先冲刷已缓存 token，保证事件顺序正确
                if pending:
                    async for frame in flush_pending():
                        yield frame

                if event_type == "done":
                    saw_done = True
                    # 记忆落盘先于 done 帧：保证「完成」事件到达前回答已持久化
                    save_ai("".join(collected).strip())
                    yield _encode(event)
                elif event_type == "error":
                    saw_error = True
                    yield _encode(event)
                elif event_type == "thinking":
                    if cfg.SSE_ENABLE_THINKING:
                        yield _encode(event)
                else:  # sources 及未知事件一律透传
                    yield _encode(event)

            # 4. 兜底：链路未给出终止事件时补发 done
            if not saw_done and not saw_error:
                async for frame in flush_pending():
                    yield frame
                elapsed = round(time.perf_counter() - started, 3)
                save_ai("".join(collected).strip())
                yield _encode(
                    {
                        "type": "done",
                        "total_tokens": len(collected),
                        "execution_time": elapsed,
                    }
                )
                saw_done = True

            logger.info(
                "SSE 流完成：session=%s, tokens=%d, time=%.3fs",
                session_id,
                len(collected),
                time.perf_counter() - started,
            )
        except asyncio.CancelledError:  # 连接关闭，由框架正常终止
            logger.info("SSE 流被取消：session=%s", session_id)
            raise
        except Exception as exc:  # noqa: BLE001 - 兜底错误帧
            logger.exception("SSE 流异常（session=%s）", session_id)
            try:
                yield _encode({"type": "error", "error": "stream_error", "detail": str(exc)})
            except Exception:  # pragma: no cover - 客户端已断开时忽略发送失败
                pass

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 Nginx 等代理缓冲
        },
    )
