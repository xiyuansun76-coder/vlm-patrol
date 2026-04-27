from fastapi import APIRouter, HTTPException, Header, Query
from fastapi.responses import Response
from typing import Optional
import base64
import database as db
import mqtt_bridge

router = APIRouter(prefix="/api/camera", tags=["camera"])

@router.get("/history")
def history(limit: int = Query(6, le=10)):
    return db.query_snapshots(limit)

@router.get("/image/{snap_id}")
def image(snap_id: int):
    snap = db.query_snapshot_by_id(snap_id)
    if not snap:
        raise HTTPException(status_code=404, detail="图片不存在")
    try:
        img_bytes = base64.b64decode(snap["image_b64"])
        return Response(content=img_bytes, media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=86400"})
    except Exception:
        raise HTTPException(status_code=500, detail="图片解码失败")

@router.get("/latest")
def latest():
    snap = db.query_latest_snapshot()
    if not snap:
        raise HTTPException(status_code=404, detail="暂无图像")
    return {"timestamp": snap["timestamp"], "image": snap["image_b64"]}

@router.post("/capture")
def capture(authorization: Optional[str] = Header(None)):
    from config import API_TOKEN
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    mqtt_bridge.publish_capture()
    return {"ok": True}
