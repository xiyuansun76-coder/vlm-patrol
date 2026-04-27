"""VLM-Patrol local service proxy router."""

import logging
import os

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
import httpx

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/vlp", tags=["vlm-patrol"])

VLP = os.environ.get("VLP_SERVICE_URL", "http://127.0.0.1:8765")


async def _proxy_get(path: str) -> JSONResponse:
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.get(f"{VLP}{path}")
            return JSONResponse(r.json(), status_code=r.status_code)
    except httpx.RequestError as e:
        return JSONResponse({"error": "VLM-Patrol service unavailable", "detail": str(e)}, 503)


async def _proxy_post(path: str, body: bytes = b"") -> JSONResponse:
    try:
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post(f"{VLP}{path}", content=body,
                             headers={"Content-Type": "application/json"})
            return JSONResponse(r.json(), status_code=r.status_code)
    except httpx.RequestError as e:
        return JSONResponse({"error": "VLM-Patrol service unavailable", "detail": str(e)}, 503)


# Dashboard
@router.get("/config")
async def vlp_config():
    return await _proxy_get("/api/config")


@router.get("/patrol/status")
async def vlp_patrol_status():
    return await _proxy_get("/api/patrol/status")


@router.post("/patrol/start")
async def vlp_patrol_start(request: Request):
    body = await request.body()
    return await _proxy_post("/api/patrol/start", body)


@router.post("/patrol/stop")
async def vlp_patrol_stop():
    return await _proxy_post("/api/patrol/stop")


@router.get("/patrol/history")
async def vlp_patrol_history():
    return await _proxy_get("/api/patrol/history")


# Agent
@router.get("/agent/status")
async def vlp_agent_status():
    return await _proxy_get("/api/agent/status")


@router.post("/agent/start")
async def vlp_agent_start():
    return await _proxy_post("/api/agent/start")


@router.post("/agent/stop")
async def vlp_agent_stop():
    return await _proxy_post("/api/agent/stop")


@router.post("/agent/analyze")
async def vlp_agent_analyze():
    return await _proxy_post("/api/agent/analyze")


@router.post("/agent/auto-care")
async def vlp_auto_care(enable: bool = True):
    return await _proxy_post(f"/api/agent/auto-care?enable={enable}")


# YOLO
@router.get("/yolo/status")
async def vlp_yolo_status():
    return await _proxy_get("/api/yolo/status")


@router.post("/yolo/train")
async def vlp_yolo_train():
    return await _proxy_post("/api/yolo/train")


# PTZ
@router.get("/ptz/status")
async def vlp_ptz_status():
    return await _proxy_get("/api/ptz/status")


@router.get("/ptz/snapshot")
async def vlp_ptz_snapshot():
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"{VLP}/api/ptz/snapshot")
            if r.status_code == 200:
                return Response(content=r.content, media_type="image/jpeg",
                                headers={"Cache-Control": "no-store"})
            return JSONResponse({"error": "snapshot failed"}, r.status_code)
    except httpx.RequestError as e:
        return JSONResponse({"error": str(e)}, 503)


# Chat (setup assistant)
@router.post("/chat")
async def vlp_chat(request: Request):
    body = await request.body()
    return await _proxy_post("/api/chat", body)


# Config save
@router.post("/config")
async def vlp_config_save(request: Request):
    body = await request.body()
    return await _proxy_post("/api/config", body)
