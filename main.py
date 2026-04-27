"""VLM-Patrol server entry point."""

import asyncio
import logging
import json
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Request, Form, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, RedirectResponse
import httpx
import uvicorn

from vlm_patrol.config import Config
from vlm_patrol.vlm import VLM
from vlm_patrol.yolo import YOLOManager
from vlm_patrol.patrol import Patrol
from vlm_patrol.agent import Agent
from vlm_patrol.setup_assistant import handle_setup_message

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
log = logging.getLogger("vlm-patrol")

# Load config
cfg = Config()
vlm = VLM(cfg)
yolo_mgr = YOLOManager(cfg)

# Initialize PTZ if configured
ptz = None
if cfg.ptz_enabled and cfg.ptz_url:
    from vlm_patrol.ptz import PTZController
    ptz = PTZController(cfg)
    log.info("PTZ enabled: %s", cfg.ptz_url)

patrol = Patrol(cfg, vlm, yolo_mgr, ptz=ptz)
agent = Agent(cfg, vlm, patrol)

app = FastAPI(title="VLM-Patrol", version="0.1.0")

# SmartAgri templates
templates_dir = Path(__file__).parent / "smartagri" / "templates"

# Login config from env
import os
_LOGIN_USER = os.environ.get("SMARTAGRI_USER", "admin")
_LOGIN_PASS = os.environ.get("SMARTAGRI_PASS", "admin")
_SESSION_SECRET = os.environ.get("SMARTAGRI_SESSION_SECRET", "vlm-patrol-session")

import hmac, hashlib
_SESSION_TOKEN = hmac.new(
    _SESSION_SECRET.encode(), b"authenticated", hashlib.sha256
).hexdigest()


def _valid_session(request: Request) -> bool:
    return request.cookies.get("sid") == _SESSION_TOKEN


# ── Pages ──

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    if not _valid_session(request):
        return RedirectResponse("/login", status_code=302)
    html = (templates_dir / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html, headers={"Cache-Control": "no-cache, no-store"})


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if _valid_session(request):
        return RedirectResponse("/", status_code=302)
    return HTMLResponse((templates_dir / "login.html").read_text(encoding="utf-8"))


@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    if username == _LOGIN_USER and password == _LOGIN_PASS:
        response = RedirectResponse("/", status_code=302)
        response.set_cookie(key="sid", value=_SESSION_TOKEN, httponly=True,
                            samesite="lax", max_age=86400 * 7)
        return response
    html = (templates_dir / "login.html").read_text(encoding="utf-8")
    html = html.replace("<!--ERROR_PLACEHOLDER-->",
                        '<p class="text-error text-xs text-center font-mono">Invalid credentials</p>')
    return HTMLResponse(html, status_code=401)


@app.post("/logout")
async def logout():
    response = RedirectResponse("/login", status_code=302)
    response.delete_cookie("sid")
    return response


# ── Patrol API ──

@app.post("/api/patrol/start")
async def patrol_start(data: dict = Body(default={})):
    """Start patrol. Accepts optional annotation to use as patrol base."""
    if patrol._running:
        return {"status": "already_running"}
    strategy = data.get("strategy") if data else None
    annotation = data.get("annotation") if data else None
    asyncio.create_task(patrol.run_once(strategy=strategy, annotation=annotation))
    return {"status": "started", "strategy": strategy or cfg.patrol_strategy,
            "from_annotation": annotation is not None}


@app.post("/api/patrol/stop")
async def patrol_stop():
    patrol.stop()
    return {"status": "stopped"}


@app.get("/api/patrol/status")
async def patrol_status():
    return patrol.get_status()


@app.get("/api/patrol/history")
async def patrol_history():
    return patrol.history


# ── PTZ API (optional, only works if PTZ configured) ──

@app.get("/api/ptz/status")
async def ptz_status():
    if not ptz:
        return {"enabled": False}
    az, el, zoom = await ptz.get_position()
    return {"enabled": True, "az": az, "el": el, "zoom": zoom,
            "travel": round(ptz.total_travel, 1)}


@app.post("/api/ptz/goto")
async def ptz_goto(az: int, el: int, zoom: int = 10):
    if not ptz:
        return {"error": "PTZ not configured"}
    ok = await ptz.goto(az, el, zoom)
    return {"status": "ok" if ok else "failed"}


@app.post("/api/ptz/home")
async def ptz_home():
    if not ptz:
        return {"error": "PTZ not configured"}
    ok = await ptz.go_home()
    return {"status": "ok" if ok else "failed"}


@app.get("/api/ptz/snapshot")
async def ptz_snapshot():
    if not ptz:
        return JSONResponse({"error": "PTZ not configured"}, 400)
    img = await ptz.snapshot()
    if img:
        from fastapi.responses import Response
        return Response(content=img, media_type="image/jpeg")
    return JSONResponse({"error": "snapshot failed"}, 500)


@app.get("/api/ptz/stream")
async def ptz_stream():
    """MJPEG stream — continuous snapshot polling as multipart stream."""
    if not ptz:
        return JSONResponse({"error": "PTZ not configured"}, 400)
    from fastapi.responses import StreamingResponse

    async def generate():
        while True:
            try:
                img = await ptz.snapshot()
                if img:
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n"
                           b"Content-Length: " + str(len(img)).encode() + b"\r\n\r\n"
                           + img + b"\r\n")
            except Exception:
                pass
            await asyncio.sleep(0.3)

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


# ── Agent API ──

@app.post("/api/agent/analyze")
async def agent_analyze():
    result = await agent.analyze_once()
    return result.to_dict()


@app.post("/api/agent/start")
async def agent_start():
    if agent._running:
        return {"status": "already_running"}
    asyncio.create_task(agent.run_continuous())
    return {"status": "started", "interval_minutes": cfg.agent_interval}


@app.post("/api/agent/stop")
async def agent_stop():
    agent.stop()
    return {"status": "stopped"}


@app.get("/api/agent/status")
async def agent_status():
    return agent.get_status()


@app.post("/api/agent/auto-care")
async def agent_auto_care(enable: bool = True):
    agent.auto_care = enable
    return {"auto_care": agent.auto_care}


@app.get("/api/agent/history")
async def agent_history():
    return [r.to_dict() for r in agent.history]


# ── Sensor API ──

@app.post("/api/sensor/push")
async def sensor_push(data: dict):
    """Push sensor data from external source."""
    agent.update_sensor_data(data)
    return {"status": "ok"}


# ── YOLO API ──

@app.get("/api/yolo/status")
async def yolo_status():
    return yolo_mgr.status()


@app.post("/api/yolo/detect")
async def yolo_detect(file: UploadFile = File(...)):
    """Run YOLO detection on uploaded image."""
    import tempfile
    content = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    detections = yolo_mgr.detect(tmp_path)
    Path(tmp_path).unlink(missing_ok=True)
    return {"detections": detections}


@app.post("/api/yolo/train")
async def yolo_train(epochs: int = 100):
    if yolo_mgr._training:
        return {"status": "already_training"}
    asyncio.create_task(asyncio.get_event_loop().run_in_executor(
        None, yolo_mgr.train, epochs))
    return {"status": "training_started"}


# ── VLM API ──

@app.post("/api/vlm/detect")
async def vlm_detect(file: UploadFile = File(...)):
    """Run VLM grounding detection on uploaded image."""
    content = await file.read()
    plants = await vlm.grounding_detect(content)
    return {"plants": plants}


@app.post("/api/vlm/diagnose")
async def vlm_diagnose(file: UploadFile = File(...)):
    """Run VLM diagnosis on uploaded image."""
    content = await file.read()
    result = await vlm.diagnose(content, agent.sensor_data)
    return result


@app.post("/api/vlm/ask")
async def vlm_ask(prompt: str, file: UploadFile = File(None)):
    """General VLM query."""
    image = await file.read() if file else None
    result = await vlm.ask(prompt, image)
    return {"response": result}


# ── Panoramic Annotation API ──

_annotations_dir = Path(__file__).parent / "data" / "annotations"
_annotations_dir.mkdir(parents=True, exist_ok=True)


@app.post("/api/annotation/save")
async def annotation_save(data: dict):
    """Save panoramic annotation (VLM result + human edits)."""
    ann_id = data.get("id") or datetime.now().strftime("%Y%m%d_%H%M%S")
    ann_file = _annotations_dir / f"{ann_id}.json"
    ann_file.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    return {"status": "ok", "id": ann_id}


@app.get("/api/annotation/list")
async def annotation_list():
    """List all saved annotations."""
    items = []
    for f in sorted(_annotations_dir.glob("*.json"), reverse=True):
        try:
            d = json.loads(f.read_text())
            items.append({
                "id": f.stem,
                "timestamp": d.get("timestamp", f.stem),
                "plant_count": len(d.get("plants", [])),
            })
        except Exception:
            pass
    return items


@app.get("/api/annotation/{ann_id}")
async def annotation_get(ann_id: str):
    """Get a saved annotation by ID."""
    ann_file = _annotations_dir / f"{ann_id}.json"
    if not ann_file.exists():
        return JSONResponse({"error": "not found"}, 404)
    return json.loads(ann_file.read_text())


@app.delete("/api/annotation/{ann_id}")
async def annotation_delete(ann_id: str):
    """Delete an annotation."""
    ann_file = _annotations_dir / f"{ann_id}.json"
    if ann_file.exists():
        ann_file.unlink()
    return {"status": "ok"}


@app.post("/api/annotation/detect")
async def annotation_detect():
    """Take panorama snapshot and run VLM grounding detection."""
    image = None
    if ptz:
        await ptz.go_home()
        image = await ptz.snapshot()
    if not image and cfg.camera_snapshot_url:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(cfg.camera_snapshot_url)
                if resp.status_code == 200:
                    image = resp.content
        except Exception:
            pass
    if not image:
        return JSONResponse({"error": "No camera available"}, 400)

    import base64
    img_w = cfg.ptz_img_w if cfg.ptz_enabled else 1920
    img_h = cfg.ptz_img_h if cfg.ptz_enabled else 1080
    plants = await vlm.grounding_detect(image, img_w, img_h)
    img_b64 = base64.b64encode(image).decode()
    ann_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        "id": ann_id,
        "timestamp": datetime.now().isoformat(),
        "image": img_b64,
        "width": img_w, "height": img_h,
        "plants": plants,
        "classes": cfg.classes,
    }
    # Auto-save
    ann_file = _annotations_dir / f"{ann_id}.json"
    save_data = {**result}
    save_data.pop("image", None)  # don't save image in JSON, too large
    ann_file.write_text(json.dumps(save_data, ensure_ascii=False, indent=2))
    # Save image separately
    img_file = _annotations_dir / f"{ann_id}.jpg"
    img_file.write_bytes(image)

    return result


@app.get("/api/annotation/{ann_id}/image")
async def annotation_image(ann_id: str):
    """Get annotation image."""
    from fastapi.responses import Response
    img_file = _annotations_dir / f"{ann_id}.jpg"
    if not img_file.exists():
        return JSONResponse({"error": "not found"}, 404)
    return Response(content=img_file.read_bytes(), media_type="image/jpeg")


# ── Dataset Console API ──

@app.get("/api/dataset/list")
async def dataset_list(split: str = "train"):
    """List all images in a dataset split with their label status."""
    img_dir = yolo_mgr.images_dir / split
    lbl_dir = yolo_mgr.labels_dir / split
    yolo_mgr._ensure_dirs()
    items = []
    for img_file in sorted(img_dir.glob("*.*"), reverse=True):
        if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        lbl_file = lbl_dir / (img_file.stem + ".txt")
        label_count = 0
        if lbl_file.exists():
            label_count = len([l for l in lbl_file.read_text().strip().split("\n") if l.strip()])
        items.append({
            "filename": img_file.name,
            "split": split,
            "labels": label_count,
            "size": img_file.stat().st_size,
        })
    return {"items": items, "total": len(items), "classes": cfg.classes}


@app.get("/api/dataset/image/{split}/{filename}")
async def dataset_image(split: str, filename: str):
    """Serve a dataset image."""
    from fastapi.responses import Response
    img_file = yolo_mgr.images_dir / split / filename
    if not img_file.exists():
        return JSONResponse({"error": "not found"}, 404)
    ct = "image/jpeg" if img_file.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    return Response(content=img_file.read_bytes(), media_type=ct)


@app.get("/api/dataset/labels/{split}/{filename}")
async def dataset_labels(split: str, filename: str):
    """Get YOLO labels for an image. Returns normalized coords."""
    stem = Path(filename).stem
    lbl_file = yolo_mgr.labels_dir / split / (stem + ".txt")
    labels = []
    if lbl_file.exists():
        for line in lbl_file.read_text().strip().split("\n"):
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                cls_name = cfg.classes[cls_id] if cls_id < len(cfg.classes) else str(cls_id)
                labels.append({
                    "class_id": cls_id, "class_name": cls_name,
                    "cx": cx, "cy": cy, "w": w, "h": h,
                })
    return {"labels": labels, "classes": cfg.classes}


@app.post("/api/dataset/labels/{split}/{filename}")
async def dataset_labels_save(split: str, filename: str, data: dict):
    """Save updated YOLO labels for an image."""
    yolo_mgr._ensure_dirs()
    stem = Path(filename).stem
    lbl_file = yolo_mgr.labels_dir / split / (stem + ".txt")
    labels = data.get("labels", [])
    lines = []
    for lbl in labels:
        lines.append(f"{lbl['class_id']} {lbl['cx']:.6f} {lbl['cy']:.6f} {lbl['w']:.6f} {lbl['h']:.6f}")
    lbl_file.write_text("\n".join(lines))
    return {"status": "ok", "count": len(lines)}


@app.delete("/api/dataset/image/{split}/{filename}")
async def dataset_image_delete(split: str, filename: str):
    """Delete an image and its labels from the dataset."""
    img_file = yolo_mgr.images_dir / split / filename
    stem = Path(filename).stem
    lbl_file = yolo_mgr.labels_dir / split / (stem + ".txt")
    if img_file.exists():
        img_file.unlink()
    if lbl_file.exists():
        lbl_file.unlink()
    return {"status": "ok"}


@app.post("/api/dataset/move/{split}/{filename}")
async def dataset_move(split: str, filename: str, data: dict):
    """Move an image between train/val splits."""
    target = data.get("target", "val" if split == "train" else "train")
    yolo_mgr._ensure_dirs()
    src_img = yolo_mgr.images_dir / split / filename
    dst_img = yolo_mgr.images_dir / target / filename
    stem = Path(filename).stem
    src_lbl = yolo_mgr.labels_dir / split / (stem + ".txt")
    dst_lbl = yolo_mgr.labels_dir / target / (stem + ".txt")
    import shutil
    if src_img.exists():
        shutil.move(str(src_img), str(dst_img))
    if src_lbl.exists():
        shutil.move(str(src_lbl), str(dst_lbl))
    return {"status": "ok", "moved_to": target}


# ── Plant Knowledge Base API ──

PLANT_KNOWLEDGE = {
    "strawberry": {
        "name": "草莓", "latin": "Fragaria × ananassa", "icon": "local_florist",
        "category": "浆果",
        "description": "多年生草本植物，果实鲜红多汁，富含维生素C。",
        "growing_conditions": {
            "temperature": "15-25°C（最适20-22°C）",
            "humidity": "60-80%",
            "soil_ph": "5.5-6.5（微酸性）",
            "light": "全日照，每天6-8小时",
            "soil_moisture": "保持湿润，避免积水",
        },
        "common_issues": [
            {"name": "灰霉病", "symptom": "果实和叶片出现灰色霉层", "solution": "通风降湿，及时摘除病果"},
            {"name": "白粉病", "symptom": "叶片表面白色粉状物", "solution": "喷洒硫磺制剂，降低湿度"},
            {"name": "叶斑病", "symptom": "叶片出现褐色斑点", "solution": "清除病叶，改善通风"},
            {"name": "红蜘蛛", "symptom": "叶片背面有细小红色虫体", "solution": "喷水增湿，生物防治"},
        ],
        "care_tips": "定期修剪匍匐茎，花期授粉辅助可提高产量。注意轮作，避免连作障碍。",
    },
    "celery": {
        "name": "芹菜", "latin": "Apium graveolens", "icon": "eco",
        "category": "叶菜",
        "description": "伞形科草本植物，茎叶均可食用，富含膳食纤维和矿物质。",
        "growing_conditions": {
            "temperature": "15-20°C（耐寒，不耐高温）",
            "humidity": "70-90%",
            "soil_ph": "6.0-7.0",
            "light": "半日照到全日照",
            "soil_moisture": "喜湿润，需经常浇水",
        },
        "common_issues": [
            {"name": "软腐病", "symptom": "茎基部水渍状腐烂", "solution": "降低湿度，避免机械伤"},
            {"name": "斑枯病", "symptom": "叶片出现黄褐色枯斑", "solution": "及时摘除病叶，喷药防治"},
            {"name": "蚜虫", "symptom": "嫩叶卷曲变形", "solution": "黄板诱杀，生物防治"},
        ],
        "care_tips": "定期培土软化茎部可提高品质。高温期注意遮阳降温。",
    },
    "chive": {
        "name": "韭菜", "latin": "Allium tuberosum", "icon": "grass",
        "category": "叶菜",
        "description": "百合科多年生草本，叶片扁平，具有独特辛香味。",
        "growing_conditions": {
            "temperature": "12-24°C（耐寒力强）",
            "humidity": "60-70%",
            "soil_ph": "5.5-7.0",
            "light": "全日照或半日照",
            "soil_moisture": "适中，避免水涝",
        },
        "common_issues": [
            {"name": "灰霉病", "symptom": "叶尖枯黄，湿度大时有灰色霉层", "solution": "通风降湿，减少浇水"},
            {"name": "韭蛆", "symptom": "植株矮小发黄，根部有蛆虫", "solution": "灌根处理，轮作换茬"},
            {"name": "疫病", "symptom": "叶片水渍状暗绿色病斑", "solution": "排水防涝，药剂防治"},
        ],
        "care_tips": "每次收割后追肥浇水促进再生。一般可收割3-4茬后翻新种植。",
    },
    "coriander": {
        "name": "香菜", "latin": "Coriandrum sativum", "icon": "spa",
        "category": "香料",
        "description": "伞形科一年生草本，全株有特殊芳香，是常用调味菜。",
        "growing_conditions": {
            "temperature": "17-20°C（不耐高温）",
            "humidity": "60-70%",
            "soil_ph": "6.2-6.8",
            "light": "全日照，短日照促进营养生长",
            "soil_moisture": "保持湿润",
        },
        "common_issues": [
            {"name": "猝倒病", "symptom": "幼苗茎基部缢缩倒伏", "solution": "控制浇水，种子消毒"},
            {"name": "叶斑病", "symptom": "叶片出现圆形褐色斑", "solution": "清除病叶，药剂喷雾"},
            {"name": "蚜虫", "symptom": "叶片卷曲、变黄", "solution": "黄板诱杀，喷洒皂水"},
        ],
        "care_tips": "播种前浸种12-24小时可提高发芽率。高温季节易抽薹，注意遮阳。",
    },
    "rose": {
        "name": "月季/玫瑰", "latin": "Rosa spp.", "icon": "local_florist",
        "category": "花卉",
        "description": "蔷薇科灌木植物，花色丰富，是世界著名观赏花卉。",
        "growing_conditions": {
            "temperature": "18-25°C（适应性强）",
            "humidity": "50-70%",
            "soil_ph": "5.5-7.0（微酸至中性）",
            "light": "全日照，每天6小时以上",
            "soil_moisture": "见干见湿",
        },
        "common_issues": [
            {"name": "黑斑病", "symptom": "叶片出现黑色圆斑，逐渐扩大", "solution": "及时摘除病叶，药剂防护"},
            {"name": "白粉病", "symptom": "嫩叶和花蕾覆盖白色粉状物", "solution": "通风透光，硫磺制剂"},
            {"name": "蚜虫", "symptom": "嫩梢和花蕾聚集大量蚜虫", "solution": "喷洒皂水，瓢虫天敌"},
            {"name": "红蜘蛛", "symptom": "叶片失绿发黄", "solution": "喷水增湿，阿维菌素"},
        ],
        "care_tips": "每年冬季修剪整形，花后及时修剪残花促进二次开花。注意防寒越冬。",
    },
    "blueberry": {
        "name": "蓝莓", "latin": "Vaccinium spp.", "icon": "nutrition",
        "category": "浆果",
        "description": "杜鹃花科灌木，果实蓝紫色，富含花青素，被誉为超级食物。",
        "growing_conditions": {
            "temperature": "18-25°C（需一定低温休眠）",
            "humidity": "60-80%",
            "soil_ph": "4.0-5.5（强酸性！）",
            "light": "全日照",
            "soil_moisture": "保持湿润，排水良好",
        },
        "common_issues": [
            {"name": "根腐病", "symptom": "植株萎蔫，根系发黑腐烂", "solution": "改善排水，避免积水"},
            {"name": "灰霉病", "symptom": "果实腐烂，覆盖灰色霉层", "solution": "及时采收，通风降湿"},
            {"name": "缺铁黄化", "symptom": "新叶脉间黄化", "solution": "调低土壤pH，施用硫酸亚铁"},
        ],
        "care_tips": "必须使用酸性基质（pH 4.0-5.5）。建议覆盖松针或锯末保持酸性。需要冷积温满足休眠。",
    },
}


@app.get("/api/knowledge")
async def knowledge_list():
    """Return plant knowledge base."""
    return {"plants": PLANT_KNOWLEDGE, "classes": cfg.classes}


@app.get("/api/knowledge/{plant_name}")
async def knowledge_get(plant_name: str):
    """Get knowledge for a specific plant."""
    if plant_name in PLANT_KNOWLEDGE:
        return PLANT_KNOWLEDGE[plant_name]
    return JSONResponse({"error": "plant not found"}, 404)


# ── Chat API (setup assistant + VLM fallback) ──

def _save_config_to_yaml():
    """Save current cfg state to config.yaml."""
    import yaml
    yaml_data = {
        "llm": {"url": cfg.llm_url, "model": cfg.llm_model},
        "camera": {
            "snapshot_url": cfg.camera_snapshot_url,
            "stream_url": cfg.camera_stream_url,
            "ptz": {
                "enabled": cfg.ptz_enabled, "url": cfg.ptz_url,
                "user": cfg.ptz_user, "pass": cfg.ptz_pass,
                "fov_h_deg": cfg.ptz_fov_h, "fov_v_deg": cfg.ptz_fov_v,
                "image_width": cfg.ptz_img_w, "image_height": cfg.ptz_img_h,
                "home_az": cfg.ptz_home_az, "home_el": cfg.ptz_home_el,
                "wide_zoom": cfg.ptz_wide_zoom, "close_zoom": cfg.ptz_close_zoom,
            },
        },
        "sensor": {"url": cfg.sensor_url},
        "actuator": {"url": cfg.actuator_url},
        "mqtt": {
            "enabled": cfg.mqtt_enabled, "broker": cfg.mqtt_broker, "port": cfg.mqtt_port,
            "sensor_topic": cfg.mqtt_sensor_topic, "control_topic": cfg.mqtt_control_topic,
            "status_topic": cfg.mqtt_status_topic,
        },
        "classes": cfg.classes,
        "yolo": {
            "model_path": cfg.yolo_model_path, "data_dir": str(cfg.yolo_data_dir),
            "auto_train": cfg.yolo_auto_train, "train_threshold": cfg.yolo_train_threshold,
        },
        "patrol": {
            "enabled": cfg.patrol_enabled, "interval_minutes": cfg.patrol_interval,
            "strategy": cfg.patrol_strategy,
        },
        "agent": {
            "auto_analysis": cfg.agent_auto_analysis, "interval_minutes": cfg.agent_interval,
        },
        "server": {"host": cfg.server_host, "port": cfg.server_port},
    }
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    log.info("Config saved via setup assistant")


@app.post("/api/chat")
async def chat(data: dict):
    """Chat endpoint: setup assistant first, then VLM fallback."""
    message = data.get("message", "").strip()
    if not message:
        return {"role": "system", "text": "Please enter a message."}

    # Try setup assistant first
    setup_response = await handle_setup_message(message, cfg, _save_config_to_yaml)
    if setup_response:
        return {"role": "assistant", "text": setup_response, "is_setup": True}

    # Fallback to VLM general query
    try:
        result = await vlm.ask(message)
        return {"role": "agent", "text": result, "is_setup": False}
    except Exception as e:
        return {"role": "system", "text": f"LLM error: {e}", "is_setup": False}


# ── Config API ──

@app.get("/api/config")
async def get_config():
    """Return current config (no secrets)."""
    return {
        "llm": {"url": cfg.llm_url, "model": cfg.llm_model},
        "camera": {"snapshot_url": cfg.camera_snapshot_url, "stream_url": cfg.camera_stream_url},
        "sensor_url": cfg.sensor_url,
        "actuator_url": cfg.actuator_url,
        "mqtt": {
            "broker": cfg.mqtt_broker, "port": cfg.mqtt_port,
            "sensor_topic": cfg.mqtt_sensor_topic,
            "control_topic": cfg.mqtt_control_topic,
            "status_topic": cfg.mqtt_status_topic,
        },
        "ptz_enabled": cfg.ptz_enabled,
        "ptz_url": cfg.ptz_url,
        "ptz_user": cfg.ptz_user,
        "ptz_pass": cfg.ptz_pass,
        "ptz_img_w": cfg.ptz_img_w,
        "ptz_img_h": cfg.ptz_img_h,
        "ptz_fov_h": cfg.ptz_fov_h,
        "ptz_fov_v": cfg.ptz_fov_v,
        "classes": cfg.classes,
        "patrol": {"enabled": cfg.patrol_enabled, "interval": cfg.patrol_interval, "strategy": cfg.patrol_strategy},
        "agent": {"auto_analysis": cfg.agent_auto_analysis, "interval": cfg.agent_interval},
        "yolo": {**yolo_mgr.status(), "data_dir": str(cfg.yolo_data_dir)},
        "server_host": cfg.server_host,
        "server_port": cfg.server_port,
    }


@app.post("/api/config")
async def save_config(data: dict):
    """Save config to config.yaml and apply to running instance."""
    import yaml

    # Build YAML structure
    llm = data.get("llm", {})
    camera = data.get("camera", {})
    sensor = data.get("sensor", {})
    actuator = data.get("actuator", {})
    mqtt_cfg = data.get("mqtt", {})
    ptz_cfg = data.get("ptz", {})
    yolo = data.get("yolo", {})
    patrol_cfg = data.get("patrol", {})
    agent_cfg = data.get("agent", {})
    server = data.get("server", {})

    yaml_data = {
        "llm": {
            "url": llm.get("url", cfg.llm_url),
            "model": llm.get("model", cfg.llm_model),
        },
        "camera": {
            "snapshot_url": camera.get("snapshot_url", cfg.camera_snapshot_url),
            "stream_url": camera.get("stream_url", cfg.camera_stream_url),
            "ptz": {
                "enabled": ptz_cfg.get("enabled", cfg.ptz_enabled),
                "url": ptz_cfg.get("url", cfg.ptz_url),
                "fov_h_deg": ptz_cfg.get("fov_h_deg", cfg.ptz_fov_h),
                "fov_v_deg": ptz_cfg.get("fov_v_deg", cfg.ptz_fov_v),
                "image_width": ptz_cfg.get("image_width", cfg.ptz_img_w),
                "image_height": ptz_cfg.get("image_height", cfg.ptz_img_h),
                "user": ptz_cfg.get("user", cfg.ptz_user),
                "pass": ptz_cfg.get("pass", cfg.ptz_pass),
                "home_az": cfg.ptz_home_az,
                "home_el": cfg.ptz_home_el,
                "wide_zoom": cfg.ptz_wide_zoom,
                "close_zoom": cfg.ptz_close_zoom,
            },
        },
        "sensor": {
            "url": sensor.get("url", cfg.sensor_url),
        },
        "actuator": {
            "url": actuator.get("url", cfg.actuator_url),
        },
        "mqtt": {
            "enabled": bool(mqtt_cfg.get("broker", cfg.mqtt_broker)),
            "broker": mqtt_cfg.get("broker", cfg.mqtt_broker),
            "port": mqtt_cfg.get("port", cfg.mqtt_port),
            "sensor_topic": mqtt_cfg.get("sensor_topic", cfg.mqtt_sensor_topic),
            "control_topic": mqtt_cfg.get("control_topic", cfg.mqtt_control_topic),
            "status_topic": mqtt_cfg.get("status_topic", cfg.mqtt_status_topic),
        },
        "classes": data.get("classes", cfg.classes),
        "yolo": {
            "model_path": yolo.get("model_path", cfg.yolo_model_path),
            "data_dir": yolo.get("data_dir", str(cfg.yolo_data_dir)),
            "auto_train": yolo.get("auto_train", cfg.yolo_auto_train),
            "train_threshold": yolo.get("train_threshold", cfg.yolo_train_threshold),
        },
        "patrol": {
            "enabled": patrol_cfg.get("enabled", cfg.patrol_enabled),
            "interval_minutes": patrol_cfg.get("interval_minutes", cfg.patrol_interval),
            "strategy": patrol_cfg.get("strategy", cfg.patrol_strategy),
        },
        "agent": {
            "auto_analysis": agent_cfg.get("auto_analysis", cfg.agent_auto_analysis),
            "interval_minutes": agent_cfg.get("interval_minutes", cfg.agent_interval),
        },
        "server": {
            "host": server.get("host", cfg.server_host),
            "port": server.get("port", cfg.server_port),
        },
    }

    # Save to config.yaml
    config_path = Path(__file__).parent / "config.yaml"
    try:
        with open(config_path, "w") as f:
            yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    except Exception as e:
        return {"status": "error", "error": str(e)}

    # Apply to running instance (no restart needed for most settings)
    cfg.llm_url = yaml_data["llm"]["url"]
    cfg.llm_model = yaml_data["llm"]["model"]
    cfg.camera_snapshot_url = yaml_data["camera"]["snapshot_url"]
    cfg.camera_stream_url = yaml_data["camera"]["stream_url"]
    cfg.sensor_url = yaml_data["sensor"]["url"]
    cfg.actuator_url = yaml_data["actuator"]["url"]
    cfg.classes = yaml_data["classes"]
    cfg.yolo_model_path = yaml_data["yolo"]["model_path"]
    cfg.yolo_data_dir = Path(yaml_data["yolo"]["data_dir"])
    cfg.yolo_auto_train = yaml_data["yolo"]["auto_train"]
    cfg.yolo_train_threshold = yaml_data["yolo"]["train_threshold"]
    cfg.ptz_enabled = yaml_data["camera"]["ptz"]["enabled"]
    cfg.ptz_url = yaml_data["camera"]["ptz"]["url"]
    cfg.ptz_fov_h = yaml_data["camera"]["ptz"]["fov_h_deg"]
    cfg.ptz_fov_v = yaml_data["camera"]["ptz"]["fov_v_deg"]
    cfg.ptz_img_w = yaml_data["camera"]["ptz"]["image_width"]
    cfg.ptz_img_h = yaml_data["camera"]["ptz"]["image_height"]
    cfg.ptz_user = yaml_data["camera"]["ptz"]["user"]
    cfg.ptz_pass = yaml_data["camera"]["ptz"]["pass"]
    cfg.patrol_enabled = yaml_data["patrol"]["enabled"]
    cfg.patrol_interval = yaml_data["patrol"]["interval_minutes"]
    cfg.patrol_strategy = yaml_data["patrol"]["strategy"]
    cfg.agent_auto_analysis = yaml_data["agent"]["auto_analysis"]
    cfg.agent_interval = yaml_data["agent"]["interval_minutes"]

    # Re-initialize PTZ controller if settings changed
    global ptz
    if cfg.ptz_enabled and cfg.ptz_url:
        from vlm_patrol.ptz import PTZController
        ptz = PTZController(cfg)
        patrol.ptz = ptz
        log.info("PTZ re-initialized: %s", cfg.ptz_url)
    else:
        ptz = None
        patrol.ptz = None

    # Apply MQTT config and reconnect
    cfg.mqtt_enabled = yaml_data["mqtt"]["enabled"]
    cfg.mqtt_broker = yaml_data["mqtt"]["broker"]
    cfg.mqtt_port = yaml_data["mqtt"]["port"]
    cfg.mqtt_sensor_topic = yaml_data["mqtt"]["sensor_topic"]
    cfg.mqtt_control_topic = yaml_data["mqtt"]["control_topic"]
    cfg.mqtt_status_topic = yaml_data["mqtt"]["status_topic"]
    _start_mqtt()

    # Apply API key if provided (not saved to yaml, stays in env/.env)
    api_key = llm.get("api_key", "")
    if api_key:
        cfg.llm_api_key = api_key
        import os
        os.environ["LLM_API_KEY"] = api_key

    log.info("Config saved and applied")
    return {"status": "ok"}


# ── WebSocket for real-time updates ──

ws_clients: list[WebSocket] = []


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    ws_clients.append(ws)
    try:
        while True:
            data = await ws.receive_text()
            # handle incoming sensor pushes via WebSocket
            try:
                msg = json.loads(data)
                if msg.get("type") == "sensor":
                    agent.update_sensor_data(msg.get("data", {}))
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        ws_clients.remove(ws)


async def broadcast(msg: dict):
    """Broadcast message to all WebSocket clients."""
    text = json.dumps(msg, ensure_ascii=False)
    for ws in ws_clients[:]:
        try:
            await ws.send_text(text)
        except Exception:
            ws_clients.remove(ws)


# ── MQTT sensor bridge ──

_mqtt_client = None
_latest_sensor: dict = {}


def _start_mqtt():
    """Connect to MQTT broker and forward sensor data to WebSocket clients."""
    global _mqtt_client
    if _mqtt_client:
        try:
            _mqtt_client.disconnect()
        except Exception:
            pass

    broker = getattr(cfg, 'mqtt_broker', '') or ''
    if not broker:
        log.info("MQTT not configured, skipping")
        return

    try:
        import paho.mqtt.client as paho_mqtt
    except ImportError:
        log.warning("paho-mqtt not installed, MQTT disabled")
        return

    topics = getattr(cfg, 'mqtt_topics', []) or []
    port = getattr(cfg, 'mqtt_port', 1883) or 1883
    control_topic = getattr(cfg, 'mqtt_control_topic', '')
    sensor_topic = getattr(cfg, 'mqtt_sensor_topic', '')

    # Collect all topics to subscribe
    sub_topics = set(topics)
    if sensor_topic:
        sub_topics.add(sensor_topic)
    if not sub_topics:
        sub_topics.add("#")  # subscribe to all if no specific topic

    def on_connect(client, userdata, flags, rc, properties=None):
        log.info("MQTT connected to %s:%d (rc=%d)", broker, port, rc)
        for t in sub_topics:
            client.subscribe(t)
            log.info("MQTT subscribed: %s", t)

    def on_message(client, userdata, msg):
        global _latest_sensor
        try:
            payload = json.loads(msg.payload.decode())
        except Exception:
            payload = {"raw": msg.payload.decode()}

        # Normalize fields
        normalized = {}
        key_map = {
            "temp": "temperature", "temperature": "temperature",
            "hum": "humidity", "humidity": "humidity", "moisture": "moisture",
            "air_temperature": "air_temperature", "air_humidity": "air_humidity",
            "soil_temperature": "soil_temperature", "soil_moisture": "soil_moisture",
            "co2": "co2", "ph": "pH",
            "nitrogen": "nitrogen", "n": "nitrogen",
            "phosphorus": "phosphorus", "p": "phosphorus",
            "potassium": "potassium", "k": "potassium",
        }
        if isinstance(payload, dict):
            for k, v in payload.items():
                nk = key_map.get(k.lower(), k)
                normalized[nk] = v

        _latest_sensor.update(normalized)
        agent.update_sensor_data(_latest_sensor)

        # Push to WebSocket clients
        ws_msg = json.dumps({
            "type": "sensor",
            "topic": msg.topic,
            "data": normalized,
            "all": _latest_sensor,
        }, ensure_ascii=False)
        for ws in ws_clients[:]:
            try:
                asyncio.run_coroutine_threadsafe(ws.send_text(ws_msg), asyncio.get_event_loop())
            except Exception:
                pass

    client = paho_mqtt.Client(paho_mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    try:
        client.connect(broker, port, keepalive=60)
        client.loop_start()
        _mqtt_client = client
        log.info("MQTT client started: %s:%d", broker, port)
    except Exception as e:
        log.warning("MQTT connection failed: %s", e)


@app.get("/api/sensor/latest")
async def sensor_latest_all():
    """Return latest sensor data from all sources."""
    return _latest_sensor


@app.post("/api/control/send")
async def control_send(data: dict):
    """Send control command via MQTT."""
    command = data.get("command", "")
    if not command:
        return {"error": "no command"}
    topic = getattr(cfg, 'mqtt_control_topic', '') or 'control/command'
    if _mqtt_client and _mqtt_client.is_connected():
        _mqtt_client.publish(topic, command)
        return {"status": "sent", "topic": topic, "command": command}
    return {"error": "MQTT not connected"}


# ── Startup ──

@app.on_event("startup")
async def startup():
    log.info("VLM-Patrol starting on %s:%d", cfg.server_host, cfg.server_port)
    log.info("LLM: %s (%s)", cfg.llm_url, cfg.llm_model)
    log.info("Camera: %s", cfg.camera_snapshot_url or "(not configured)")
    log.info("Classes: %s", cfg.classes)

    _start_mqtt()

    if cfg.patrol_enabled:
        log.info("Auto patrol enabled (every %d min)", cfg.patrol_interval)
        asyncio.create_task(patrol.run_continuous())

    if cfg.agent_auto_analysis:
        log.info("Auto analysis enabled (every %d min)", cfg.agent_interval)
        asyncio.create_task(agent.run_continuous())


if __name__ == "__main__":
    uvicorn.run("main:app", host=cfg.server_host, port=cfg.server_port, reload=False)
