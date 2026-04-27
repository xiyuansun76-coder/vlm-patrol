from fastapi import APIRouter, HTTPException, Header, Query
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timezone, timedelta
import database as db
import mqtt_bridge
import control_state

router = APIRouter(prefix="/api", tags=["devices"])

KNOWN = {
    "greenhouse_controller": "温室控制器",
    "greenhouse_zigbee":     "温室环境",
    "30aea49c97d8":          "土壤传感器",
    "rpi_camera":            "摄像头",
}

@router.get("/devices")
def devices():
    now    = datetime.now(timezone.utc)
    result = []
    cache  = mqtt_bridge.device_status
    for node_id, name in KNOWN.items():
        entry = cache.get(node_id, {})
        last  = entry.get("last_seen")
        online = False
        if last:
            try:
                delta = now - datetime.fromisoformat(last)
                online = delta < timedelta(seconds=120)
            except Exception:
                pass
        result.append({
            "node_id":   node_id,
            "name":      name,
            "online":    online,
            "last_seen": last,
            "fields":    entry.get("fields", {}),
        })
    return result


class ControlCmd(BaseModel):
    action:      str
    enable:      bool
    duration_sec: Optional[int] = None
    source:      str = "api"

VALID_ACTIONS = {"light", "pump", "curtain", "fan"}

@router.post("/control")
def control(cmd: ControlCmd, authorization: Optional[str] = Header(None)):
    from config import API_TOKEN
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    if cmd.action not in VALID_ACTIONS:
        raise HTTPException(status_code=400, detail=f"Unknown action: {cmd.action}")
    mqtt_bridge.publish_control(cmd.action, cmd.enable, cmd.duration_sec)
    control_state.set_state(cmd.action, cmd.enable)
    db.insert_control_log(cmd.action, cmd.enable, cmd.duration_sec, cmd.source)
    return {"ok": True}

@router.get("/control/state")
def get_control_state():
    return control_state.get_all()

@router.get("/control/logs")
def control_logs(limit: int = Query(50, le=200)):
    return db.query_control_logs(limit)
