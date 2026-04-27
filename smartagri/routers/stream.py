"""
SSE 推送端点：MQTT 有新传感器数据时，立即推给所有已连接的浏览器客户端。
"""
import asyncio
import json
import logging

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

log = logging.getLogger(__name__)
router = APIRouter()

_clients: list[asyncio.Queue] = []
_loop: asyncio.AbstractEventLoop | None = None


def _get_loop() -> asyncio.AbstractEventLoop | None:
    return _loop


def broadcast(node_id: str, fields: dict):
    """由 mqtt_bridge 线程调用，把新数据推给所有 SSE 客户端。"""
    if _loop is None or not _clients:
        return
    payload = json.dumps({"node_id": node_id, **fields})
    for q in list(_clients):
        try:
            _loop.call_soon_threadsafe(q.put_nowait, payload)
        except Exception:
            pass


@router.get("/api/stream")
async def sse_stream():
    global _loop
    _loop = asyncio.get_event_loop()

    q: asyncio.Queue = asyncio.Queue()
    _clients.append(q)
    log.info(f"SSE 客户端已连接，当前 {len(_clients)} 个")

    async def generator():
        try:
            # 先发一个心跳，让浏览器确认连接建立
            yield "event: ping\ndata: ok\n\n"
            while True:
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=25)
                    yield f"data: {payload}\n\n"
                except asyncio.TimeoutError:
                    # 每 25s 发一次心跳防止连接超时
                    yield "event: ping\ndata: ok\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            try:
                _clients.remove(q)
            except ValueError:
                pass
            log.info(f"SSE 客户端断开，剩余 {len(_clients)} 个")

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
