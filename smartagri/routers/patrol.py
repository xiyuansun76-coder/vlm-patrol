"""Patrol proxy router — forwards to VLM-Patrol service."""

import json
import logging
import os

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
import httpx

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/patrol", tags=["patrol"])

VLP_SERVICE = os.environ.get("VLP_SERVICE_URL", "http://127.0.0.1:8765")
PATROL_HISTORY_DIR = os.environ.get("PATROL_HISTORY_DIR", "patrol_history")


async def _proxy(method: str, path: str, body: bytes = b"") -> JSONResponse:
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            if method == "GET":
                r = await client.get(f"{VLP_SERVICE}{path}")
            elif method == "POST":
                r = await client.post(f"{VLP_SERVICE}{path}", content=body,
                                       headers={"Content-Type": "application/json"})
            elif method == "PUT":
                r = await client.put(f"{VLP_SERVICE}{path}", content=body,
                                      headers={"Content-Type": "application/json"})
            else:
                return JSONResponse({"error": f"Unsupported method: {method}"}, 400)
            return JSONResponse(r.json(), status_code=r.status_code)
    except httpx.RequestError as e:
        log.error("Patrol service connection failed: %s", e)
        return JSONResponse({"error": "Patrol service unavailable", "detail": str(e)}, 503)


@router.post("/start")
async def start(request: Request):
    body = await request.body()
    return await _proxy("POST", "/api/patrol/start", body or b'{"strategy": "vlm_active"}')


@router.post("/stop")
async def stop():
    return await _proxy("POST", "/api/patrol/stop")


@router.get("/status")
async def status():
    return await _proxy("GET", "/api/patrol/status")


@router.put("/plant/{plant_id}")
async def update_plant(plant_id: str, request: Request):
    body = await request.body()
    return await _proxy("PUT", f"/api/patrol/plant/{plant_id}", body)


@router.put("/plant/{plant_id}/annotate")
async def annotate(plant_id: str, request: Request):
    body = await request.body()
    return await _proxy("PUT", f"/api/patrol/plant/{plant_id}/annotate", body)


@router.get("/history")
async def history():
    """List patrol history sessions."""
    sessions = []
    if os.path.isdir(PATROL_HISTORY_DIR):
        for d in sorted(os.listdir(PATROL_HISTORY_DIR), reverse=True):
            meta_path = os.path.join(PATROL_HISTORY_DIR, d, "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                    sessions.append({
                        "session_id": meta.get("session_id", d),
                        "strategy": meta.get("strategy", "unknown"),
                        "start_time": meta.get("start_time"),
                        "end_time": meta.get("end_time"),
                        "duration_s": meta.get("duration_s"),
                        "overview_plants": meta.get("overview_plants", 0),
                        "confirmed_plants": meta.get("confirmed_plants", 0),
                        "rejected": meta.get("rejected", 0),
                        "status": meta.get("status"),
                        "ptz_travel_deg": meta.get("ptz_travel_deg", 0),
                    })
    return sessions


@router.get("/history/{session_id}")
async def history_detail(session_id: str):
    """Get single patrol session detail."""
    meta_path = os.path.join(PATROL_HISTORY_DIR, session_id, "meta.json")
    if not os.path.exists(meta_path):
        return JSONResponse({"error": "Session not found"}, 404)
    with open(meta_path) as f:
        return json.load(f)
