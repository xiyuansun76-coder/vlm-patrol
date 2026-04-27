"""Setup assistant — handles configuration through chat conversation.

Detects user intent from chat messages and executes config operations:
- Camera/PTZ setup and testing
- LLM API configuration and testing
- YOLO setup
- Sensor/actuator configuration
- Guided initial setup wizard
"""

import logging
import re
import httpx

log = logging.getLogger(__name__)


async def test_camera(url: str) -> dict:
    """Test if a camera URL returns an image."""
    try:
        async with httpx.AsyncClient(timeout=8) as c:
            r = await c.get(url)
            if r.status_code == 200 and len(r.content) > 1000:
                content_type = r.headers.get("content-type", "")
                if "image" in content_type or r.content[:3] == b'\xff\xd8\xff':
                    return {"ok": True, "size": len(r.content), "type": content_type}
            return {"ok": False, "error": f"HTTP {r.status_code}, {len(r.content)} bytes"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


async def test_ptz(url: str, user: str, pwd: str) -> dict:
    """Test if a PTZ camera responds to ISAPI."""
    try:
        async with httpx.AsyncClient(auth=httpx.DigestAuth(user, pwd), timeout=8) as c:
            r = await c.get(url.rstrip("/") + "/ISAPI/PTZCtrl/channels/1/status")
            if r.status_code == 200 and "<azimuth>" in r.text:
                az = re.search(r"<azimuth>(\d+)</azimuth>", r.text)
                el = re.search(r"<elevation>(\d+)</elevation>", r.text)
                return {"ok": True, "az": az.group(1) if az else "?",
                        "el": el.group(1) if el else "?"}
            return {"ok": False, "error": f"HTTP {r.status_code}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


async def test_llm(url: str, model: str, api_key: str = "") -> dict:
    """Test if LLM API responds."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(url, json={
                "model": model,
                "messages": [{"role": "user", "content": "Reply with just 'ok'"}],
                "max_tokens": 10,
            }, headers=headers)
            if r.status_code == 200:
                data = r.json()
                reply = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                return {"ok": True, "reply": reply[:50]}
            return {"ok": False, "error": f"HTTP {r.status_code}: {r.text[:200]}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


async def test_yolo() -> dict:
    """Test if ultralytics is available."""
    try:
        import ultralytics
        return {"ok": True, "version": ultralytics.__version__}
    except ImportError:
        return {"ok": False, "error": "ultralytics not installed"}


# ── Intent detection and response ──

SETUP_KEYWORDS = {
    "camera": ["摄像头", "camera", "相机", "监控", "球机", "ptz", "海康", "hikvision", "大华"],
    "llm": ["大模型", "llm", "模型", "ollama", "api", "nvidia", "openai", "qwen"],
    "yolo": ["yolo", "检测", "训练", "detect", "train", "模型训练"],
    "sensor": ["传感器", "sensor", "mqtt", "温度", "湿度", "土壤"],
    "actuator": ["执行器", "actuator", "浇水", "通风", "灌溉", "water", "vent"],
    "setup": ["配置", "设置", "setup", "初始化", "开始", "帮我", "怎么用", "如何"],
    "status": ["状态", "status", "测试", "test", "检查", "check", "连接"],
    "patrol": ["巡检", "patrol", "巡逻", "扫描", "scan"],
}


def detect_intent(message: str) -> list[str]:
    """Detect setup-related intents from user message."""
    msg = message.lower()
    intents = []
    for intent, keywords in SETUP_KEYWORDS.items():
        for kw in keywords:
            if kw in msg:
                intents.append(intent)
                break
    return intents


def extract_ip(text: str) -> str | None:
    m = re.search(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})', text)
    return m.group(1) if m else None


def extract_port(text: str) -> int | None:
    m = re.search(r':(\d{4,5})', text)
    return int(m.group(1)) if m else None


def extract_url(text: str) -> str | None:
    m = re.search(r'(https?://\S+)', text)
    return m.group(1) if m else None


async def handle_setup_message(message: str, cfg, save_config_fn) -> str | None:
    """
    Process a chat message for setup intent.
    Returns a response string if it's a setup message, None otherwise.
    """
    intents = detect_intent(message)
    if not intents:
        return None

    msg = message.lower()
    ip = extract_ip(message)
    url = extract_url(message)

    # ── Full setup wizard ──
    if "setup" in intents and not any(i in intents for i in ["camera", "llm", "yolo"]):
        status_parts = []

        # Check LLM
        llm_result = await test_llm(cfg.llm_url, cfg.llm_model, cfg.llm_api_key)
        if llm_result["ok"]:
            status_parts.append(f"✅ LLM connected: {cfg.llm_model} @ {cfg.llm_url}")
        else:
            status_parts.append(f"❌ LLM not connected: {llm_result['error']}\n   → Configure LLM URL and model. Example: 'set llm url http://localhost:11434/v1/chat/completions'")

        # Check camera
        if cfg.camera_snapshot_url:
            cam_result = await test_camera(cfg.camera_snapshot_url)
            if cam_result["ok"]:
                status_parts.append(f"✅ Camera connected: {cfg.camera_snapshot_url} ({cam_result['size']} bytes)")
            else:
                status_parts.append(f"❌ Camera not responding: {cam_result['error']}")
        else:
            status_parts.append("❌ Camera not configured\n   → Tell me your camera IP, e.g. 'connect camera 10.168.1.239'")

        # Check PTZ
        if cfg.ptz_enabled:
            ptz_result = await test_ptz(cfg.ptz_url, cfg.ptz_user, cfg.ptz_pass)
            if ptz_result["ok"]:
                status_parts.append(f"✅ PTZ connected: az={ptz_result['az']} el={ptz_result['el']}")
            else:
                status_parts.append(f"❌ PTZ not responding: {ptz_result['error']}")
        else:
            status_parts.append("ℹ️ PTZ not enabled (optional)")

        # Check YOLO
        yolo_result = await test_yolo()
        if yolo_result["ok"]:
            status_parts.append(f"✅ YOLO ready: ultralytics {yolo_result['version']}")
        else:
            status_parts.append(f"❌ YOLO not installed (will auto-install on first use)")

        status_parts.append(f"\nClasses: {', '.join(cfg.classes)}")
        status_parts.append(f"Patrol: {'enabled' if cfg.patrol_enabled else 'disabled'} ({cfg.patrol_strategy})")

        return "📋 System Status:\n\n" + "\n".join(status_parts)

    # ── Camera setup ──
    if "camera" in intents:
        # User provides IP → auto-configure Hikvision
        if ip:
            # Test as Hikvision PTZ first
            ptz_result = await test_ptz(f"http://{ip}", cfg.ptz_user, cfg.ptz_pass)
            if ptz_result["ok"]:
                cfg.camera_snapshot_url = f"http://{ip}/ISAPI/Streaming/channels/101/picture"
                cfg.ptz_enabled = True
                cfg.ptz_url = f"http://{ip}"
                save_config_fn()
                return (f"✅ Detected Hikvision PTZ camera at {ip}\n"
                        f"   Position: az={ptz_result['az']} el={ptz_result['el']}\n"
                        f"   Snapshot URL set to: {cfg.camera_snapshot_url}\n"
                        f"   PTZ control enabled\n\n"
                        f"   You can now run patrol with 'vlm_active' strategy!")

            # Try as generic snapshot URL
            for path in ["/snapshot", "/capture", "/ISAPI/Streaming/channels/101/picture",
                         "/cgi-bin/snapshot.cgi", "/jpg/image.jpg"]:
                test_url = f"http://{ip}{path}"
                cam_result = await test_camera(test_url)
                if cam_result["ok"]:
                    cfg.camera_snapshot_url = test_url
                    save_config_fn()
                    return (f"✅ Camera found at {test_url} ({cam_result['size']} bytes)\n"
                            f"   Snapshot URL configured.\n"
                            f"   PTZ not detected — using 'single' patrol strategy.")

            return (f"❌ Could not connect to camera at {ip}\n"
                    f"   Tried common snapshot paths but none responded.\n"
                    f"   Please provide the full snapshot URL, e.g.:\n"
                    f"   'set camera http://{ip}/your/snapshot/path'")

        # User provides full URL
        if url:
            cam_result = await test_camera(url)
            if cam_result["ok"]:
                cfg.camera_snapshot_url = url
                save_config_fn()
                return f"✅ Camera configured: {url} ({cam_result['size']} bytes)"
            else:
                return f"❌ Camera URL not responding: {cam_result['error']}"

        # General camera help
        if cfg.camera_snapshot_url:
            cam_result = await test_camera(cfg.camera_snapshot_url)
            status = "connected" if cam_result["ok"] else f"error: {cam_result['error']}"
            return (f"📷 Current camera: {cfg.camera_snapshot_url} ({status})\n"
                    f"   PTZ: {'enabled' if cfg.ptz_enabled else 'disabled'}\n\n"
                    f"   To change: tell me the camera IP or URL\n"
                    f"   Example: 'connect camera 192.168.1.100'")
        else:
            return ("📷 No camera configured.\n\n"
                    "   Tell me your camera IP and I'll auto-detect the type:\n"
                    "   Example: 'connect camera 192.168.1.100'\n"
                    "   Or provide a snapshot URL: 'set camera http://...'")

    # ── LLM setup ──
    if "llm" in intents:
        if url or ip:
            new_url = url or f"http://{ip}:11434/v1/chat/completions"
            # Extract model name if mentioned
            model = cfg.llm_model
            for m in ["qwen3-vl:8b", "qwen2.5-vl:7b", "llava", "gemma", "llama"]:
                if m.split(":")[0] in msg or m.split("-")[0] in msg:
                    model = m
                    break
            result = await test_llm(new_url, model, cfg.llm_api_key)
            if result["ok"]:
                cfg.llm_url = new_url
                cfg.llm_model = model
                save_config_fn()
                return (f"✅ LLM connected!\n"
                        f"   URL: {new_url}\n"
                        f"   Model: {model}\n"
                        f"   Test reply: {result['reply']}")
            else:
                return (f"❌ LLM connection failed: {result['error']}\n"
                        f"   URL tried: {new_url}\n"
                        f"   Model: {model}\n\n"
                        f"   Make sure Ollama is running: 'ollama serve'\n"
                        f"   And the model is pulled: 'ollama pull {model}'")

        # General LLM help
        result = await test_llm(cfg.llm_url, cfg.llm_model, cfg.llm_api_key)
        status = "connected" if result["ok"] else f"error: {result['error']}"
        return (f"🤖 Current LLM: {cfg.llm_model} @ {cfg.llm_url} ({status})\n\n"
                f"   To change:\n"
                f"   • Local Ollama: 'set llm http://localhost:11434/v1/chat/completions'\n"
                f"   • NVIDIA cloud: 'set llm https://integrate.api.nvidia.com/v1/chat/completions'\n"
                f"   • Set API key in .env file: LLM_API_KEY=your-key")

    # ── YOLO setup ──
    if "yolo" in intents:
        yolo_result = await test_yolo()
        if "install" in msg or "安装" in msg:
            try:
                import subprocess, sys
                subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics", "-q"])
                return "✅ YOLO (ultralytics) installed successfully!"
            except Exception as e:
                return f"❌ YOLO install failed: {e}"

        yolo_status = cfg.__dict__
        return (f"🔍 YOLO Status:\n"
                f"   Installed: {'yes (' + yolo_result['version'] + ')' if yolo_result['ok'] else 'no (auto-installs on first use)'}\n"
                f"   Model: {cfg.yolo_model_path or 'default (yolo11s.pt)'}\n"
                f"   Data dir: {cfg.yolo_data_dir}\n"
                f"   Auto-train: {'on' if cfg.yolo_auto_train else 'off'} (threshold: {cfg.yolo_train_threshold} images)\n\n"
                f"   To install now: 'install yolo'\n"
                f"   To change model: edit config.yaml → yolo.model_path")

    # ── Sensor setup ──
    if "sensor" in intents:
        if url or ip:
            new_url = url or f"http://{ip}/sensors"
            cfg.sensor_url = new_url
            save_config_fn()
            return f"✅ Sensor URL set to: {new_url}\n   The agent will fetch sensor data from this URL."

        return ("📡 Sensor Configuration:\n"
                f"   Current URL: {cfg.sensor_url or '(not set)'}\n\n"
                f"   3 ways to provide sensor data:\n"
                f"   1. HTTP push: POST /api/sensor/push with JSON body\n"
                f"   2. WebSocket: send {{\"type\":\"sensor\",\"data\":{{...}}}}\n"
                f"   3. HTTP pull: set sensor URL, agent fetches periodically\n\n"
                f"   Example: 'set sensor http://192.168.1.50/api/sensors'")

    # ── Actuator setup ──
    if "actuator" in intents:
        if url or ip:
            new_url = url or f"http://{ip}/control"
            cfg.actuator_url = new_url
            save_config_fn()
            return f"✅ Actuator URL set to: {new_url}"

        return ("🔧 Actuator Configuration:\n"
                f"   Current URL: {cfg.actuator_url or '(not set)'}\n\n"
                f"   The agent sends care commands as HTTP POST:\n"
                f"   {{\"action\":\"water\",\"enable\":true,\"duration_sec\":300}}\n\n"
                f"   Example: 'set actuator http://192.168.1.200/control'")

    # ── Status check ──
    if "status" in intents:
        # Redirect to full setup check
        return await handle_setup_message("setup check", cfg, save_config_fn)

    # ── Patrol control ──
    if "patrol" in intents:
        if any(w in msg for w in ["start", "开始", "启动", "run"]):
            return "🔄 Use the Dashboard 'Run Patrol' button, or POST /api/patrol/start"
        if any(w in msg for w in ["vlm_active", "主动", "active"]):
            cfg.patrol_strategy = "vlm_active"
            save_config_fn()
            return "✅ Patrol strategy set to vlm_active (panorama → focus → diagnose)"
        if any(w in msg for w in ["sweep", "扫描", "grid"]):
            cfg.patrol_strategy = "sweep"
            save_config_fn()
            return "✅ Patrol strategy set to sweep (PTZ grid scan)"

        return (f"🔄 Patrol Configuration:\n"
                f"   Strategy: {cfg.patrol_strategy}\n"
                f"   Auto: {'on' if cfg.patrol_enabled else 'off'} (every {cfg.patrol_interval} min)\n\n"
                f"   Available strategies:\n"
                f"   • single — one snapshot (no PTZ needed)\n"
                f"   • sweep — PTZ grid scan\n"
                f"   • vlm_active — panorama → focus each plant → diagnose\n\n"
                f"   Say 'set patrol vlm_active' to change")

    return None
