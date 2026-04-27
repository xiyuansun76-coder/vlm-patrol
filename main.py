"""VLM-Patrol server entry point."""

import asyncio
import logging
import json
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

from vlm_patrol.config import Config
from vlm_patrol.vlm import VLM
from vlm_patrol.yolo import YOLOManager
from vlm_patrol.patrol import Patrol
from vlm_patrol.agent import Agent

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

# Serve static frontend
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ── Pages ──

@app.get("/")
async def index():
    return FileResponse(str(static_dir / "index.html"))


# ── Patrol API ──

@app.post("/api/patrol/start")
async def patrol_start(strategy: str = None):
    """Start patrol. strategy: single, sweep, vlm_active (default from config)."""
    if patrol._running:
        return {"status": "already_running"}
    asyncio.create_task(patrol.run_once(strategy=strategy))
    return {"status": "started", "strategy": strategy or cfg.patrol_strategy}


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


@app.get("/api/sensor/latest")
async def sensor_latest():
    return agent.sensor_data


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


# ── Config API ──

@app.get("/api/config")
async def get_config():
    """Return current config (no secrets)."""
    return {
        "llm": {"url": cfg.llm_url, "model": cfg.llm_model},
        "camera": {"snapshot_url": cfg.camera_snapshot_url, "stream_url": cfg.camera_stream_url},
        "sensor_url": cfg.sensor_url,
        "actuator_url": cfg.actuator_url,
        "ptz_enabled": cfg.ptz_enabled,
        "ptz_url": cfg.ptz_url,
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


# ── Startup ──

@app.on_event("startup")
async def startup():
    log.info("VLM-Patrol starting on %s:%d", cfg.server_host, cfg.server_port)
    log.info("LLM: %s (%s)", cfg.llm_url, cfg.llm_model)
    log.info("Camera: %s", cfg.camera_snapshot_url or "(not configured)")
    log.info("Classes: %s", cfg.classes)

    if cfg.patrol_enabled:
        log.info("Auto patrol enabled (every %d min)", cfg.patrol_interval)
        asyncio.create_task(patrol.run_continuous())

    if cfg.agent_auto_analysis:
        log.info("Auto analysis enabled (every %d min)", cfg.agent_interval)
        asyncio.create_task(agent.run_continuous())


if __name__ == "__main__":
    uvicorn.run("main:app", host=cfg.server_host, port=cfg.server_port, reload=False)
