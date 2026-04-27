import os
import base64
import json as _json
import logging
import re

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response
import httpx

router = APIRouter(prefix="/api/hik", tags=["hikvision"])
_log = logging.getLogger(__name__)

# All configurable via environment
_CAM = os.environ.get("HIK_CAM_URL", "http://192.168.1.100")
_AUTH = httpx.DigestAuth(
    os.environ.get("HIK_CAM_USER", "admin"),
    os.environ.get("HIK_CAM_PASS", ""),
)
_GO2RTC = os.environ.get("GO2RTC_URL", "http://127.0.0.1:1984")
_OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
_VLM_MODEL = os.environ.get("VLM_MODEL", "qwen3-vl:8b")


async def _cam_get(path: str) -> httpx.Response:
    async with httpx.AsyncClient(auth=_AUTH, timeout=6) as client:
        return await client.get(f"{_CAM}{path}")


async def _cam_put(path: str, xml: str) -> httpx.Response:
    async with httpx.AsyncClient(auth=_AUTH, timeout=6) as client:
        return await client.put(
            f"{_CAM}{path}",
            content=xml.encode(),
            headers={"Content-Type": "application/xml"},
        )


@router.get("/snapshot")
async def snapshot():
    """Return latest JPEG snapshot from the Hikvision camera."""
    try:
        r = await _cam_get("/ISAPI/Streaming/channels/101/picture")
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Camera returned {r.status_code}")
        return Response(
            content=r.content,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store, no-cache"},
        )
    except httpx.RequestError as e:
        raise HTTPException(status_code=504, detail=str(e))


@router.post("/ptz")
async def ptz(
    pan:  int = Query(0, ge=-100, le=100),
    tilt: int = Query(0, ge=-100, le=100),
    zoom: int = Query(0, ge=-100, le=100),
):
    """Send PTZ continuous move command. Use 0/0/0 to stop."""
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<PTZData version="2.0" xmlns="http://www.hikvision.com/ver20/XMLSchema">'
        f"<pan>{pan}</pan><tilt>{tilt}</tilt><zoom>{zoom}</zoom>"
        "</PTZData>"
    )
    try:
        r = await _cam_put("/ISAPI/PTZCtrl/channels/1/continuous", xml)
        return {"ok": r.status_code < 300}
    except httpx.RequestError as e:
        raise HTTPException(status_code=504, detail=str(e))


@router.post("/webrtc")
async def webrtc_proxy(request: Request, src: str = Query("hik_ptz")):
    """Proxy WebRTC SDP exchange to go2rtc, avoiding CORS issues."""
    body = await request.body()
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(
                f"{_GO2RTC}/api/webrtc?src={src}",
                content=body,
                headers={"Content-Type": "application/sdp"},
            )
        return Response(
            content=r.content,
            media_type="application/sdp",
            status_code=r.status_code,
        )
    except httpx.RequestError as e:
        raise HTTPException(status_code=504, detail=str(e))


@router.get("/frame.jpeg")
async def frame_jpeg(src: str = Query("hik_ptz")):
    """Proxy a single JPEG frame from go2rtc."""
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.get(f"{_GO2RTC}/api/frame.jpeg?src={src}")
        return Response(
            content=r.content,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )
    except httpx.RequestError as e:
        raise HTTPException(status_code=504, detail=str(e))


async def _get_sensor_data() -> dict:
    """Get latest sensor data from database."""
    try:
        from smartagri import database as db
        rows = db.query_latest()
        sensor = {}
        for r in rows:
            raw = r.get("raw_json")
            if raw:
                d = _json.loads(raw) if isinstance(raw, str) else raw
                for key in ("moisture", "temperature", "air_temperature",
                            "air_humidity", "co2", "nitrogen", "phosphorus", "potassium"):
                    if key in d and d[key]:
                        mapped = {"moisture": "soil_moisture", "air_humidity": "humidity"}.get(key, key)
                        sensor[mapped] = d[key]
        return sensor
    except Exception as e:
        _log.warning("Failed to get sensor data: %s", e)
        return {}


async def _vlm_diagnose(image_bytes: bytes, sensor_data: dict = None) -> dict:
    """Call Ollama VLM for plant diagnosis."""
    b64 = base64.b64encode(image_bytes).decode()

    sensor_context = ""
    if sensor_data:
        parts = []
        mapping = {"soil_moisture": ("soil moisture", "%"), "temperature": ("soil temp", "C"),
                    "air_temperature": ("air temp", "C"), "humidity": ("humidity", "%"),
                    "co2": ("CO2", "ppm")}
        for k, (label, unit) in mapping.items():
            if k in sensor_data:
                parts.append(f"{label}={sensor_data[k]}{unit}")
        if parts:
            sensor_context = (
                f"\n\nEnvironmental sensor readings: {', '.join(parts)}. "
                "Consider these when assessing plant health and recommending actions."
            )

    prompt = (
        "/no_think\nAnalyze this greenhouse plant image. "
        "Identify the species, health status "
        "(healthy/mild_stress/stressed/severe_stress), "
        "confidence (0.0-1.0), and describe ALL visual symptoms in detail. "
        "If any stress signs exist, suggest probable cause and recommended action."
        f"{sensor_context}\n\n"
        'Output as JSON: {"species": "...", "health": "...", '
        '"confidence": 0.9, "symptoms": "...", "cause": "...", "action": "..."}'
    )

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(_OLLAMA, json={
                "model": _VLM_MODEL,
                "messages": [{"role": "user", "content": prompt, "images": [b64]}],
                "stream": False,
            })
            r.raise_for_status()
            msg = r.json()["message"]
            text = msg.get("content", "") or msg.get("thinking", "")

            m = re.search(r'\{.*\}', text, re.DOTALL)
            if m:
                return _json.loads(m.group(0))
            return {"species": "unknown", "health": "unknown", "symptoms": text[:200]}
    except Exception as e:
        _log.error("VLM diagnosis failed: %s", e)
        return {"error": str(e)}


@router.post("/diagnose")
async def diagnose(mode: str = Query("both", regex="^(vision|multi|both)$")):
    """Plant health diagnosis using VLM + optional sensor data."""
    try:
        r = await _cam_get("/ISAPI/Streaming/channels/101/picture")
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail="Camera capture failed")
        image = r.content
    except httpx.RequestError as e:
        raise HTTPException(status_code=504, detail=f"Camera error: {e}")

    result = {"timestamp": __import__("datetime").datetime.now().isoformat()}

    if mode in ("vision", "both"):
        result["vision_only"] = await _vlm_diagnose(image)
    if mode in ("multi", "both"):
        sensor = await _get_sensor_data()
        result["sensor_data"] = sensor
        result["multi_modal"] = await _vlm_diagnose(image, sensor)

    return result
