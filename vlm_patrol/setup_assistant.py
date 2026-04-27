"""Setup assistant — handles configuration through chat conversation.

Uses a lightweight LLM (NVIDIA API or local) for natural language intent
detection, with keyword fallback when LLM is unavailable.

Supported intents:
- Camera/PTZ setup and testing
- LLM API configuration and testing
- YOLO setup
- Sensor/actuator configuration
- Guided initial setup wizard
- Patrol control
"""

import json
import logging
import os
import re
import httpx

log = logging.getLogger(__name__)

# ── Intent LLM configuration ──
# Uses a small fast model for intent classification only.
# Configurable via env vars or falls back to keyword matching.

INTENT_LLM_URL = os.environ.get(
    "INTENT_LLM_URL",
    "https://integrate.api.nvidia.com/v1/chat/completions"
)
INTENT_LLM_MODEL = os.environ.get("INTENT_LLM_MODEL", "meta/llama-3.1-8b-instruct")
INTENT_LLM_KEY = os.environ.get("INTENT_LLM_KEY", os.environ.get("LLM_API_KEY", ""))

INTENT_PROMPT = """You are an intent classifier for a plant monitoring system setup assistant.
Given a user message, extract the intent and any parameters.

Possible intents:
- camera: user wants to configure or check camera/PTZ (may include IP or URL)
- llm: user wants to configure or check the LLM/AI model API (may include URL, model name)
- yolo: user wants to check, install, or configure YOLO detection
- sensor: user wants to configure sensor data source (may include URL)
- actuator: user wants to configure actuator/control endpoint (may include URL)
- setup: user wants a general system status check or guided setup
- patrol: user wants to configure or control patrol (may include strategy name)
- none: message is not about system configuration at all

Reply with ONLY a JSON object, no other text:
{"intent": "camera", "ip": "192.168.1.100", "url": null, "model": null, "action": null, "extra": null}

Rules:
- intent: the primary intent (pick the most specific one)
- ip: extracted IP address if any, null otherwise
- url: extracted full URL (http/https) if any, null otherwise
- model: model name if mentioned (e.g. "qwen3-vl:8b", "llava"), null otherwise
- action: specific action like "install", "start", "stop", "test", "status", "vlm_active", "sweep", null if unclear
- extra: any other relevant extracted info, null if none

User message: """


async def _llm_detect_intent(message: str) -> dict | None:
    """Use a lightweight LLM to classify intent. Returns parsed dict or None on failure."""
    if not INTENT_LLM_KEY:
        return None

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {INTENT_LLM_KEY}",
    }
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(INTENT_LLM_URL, json={
                "model": INTENT_LLM_MODEL,
                "messages": [{"role": "user", "content": INTENT_PROMPT + message}],
                "max_tokens": 150,
                "temperature": 0,
            }, headers=headers)
            if r.status_code != 200:
                log.debug("Intent LLM returned %d", r.status_code)
                return None
            data = r.json()
            reply = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            # Extract JSON from reply
            m = re.search(r'\{[^}]+\}', reply)
            if m:
                parsed = json.loads(m.group())
                log.info("Intent LLM: %s → %s", message[:50], parsed.get("intent"))
                return parsed
    except Exception as e:
        log.debug("Intent LLM failed: %s", e)
    return None


# ── Keyword fallback ──

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


def _keyword_detect_intent(message: str) -> dict:
    """Fallback: detect intent via keyword matching."""
    msg = message.lower()
    intents = []
    for intent, keywords in SETUP_KEYWORDS.items():
        for kw in keywords:
            if kw in msg:
                intents.append(intent)
                break

    if not intents:
        return {"intent": "none"}

    # Pick most specific intent (prefer camera/llm/yolo over generic setup/status)
    specific = [i for i in intents if i not in ("setup", "status")]
    primary = specific[0] if specific else intents[0]

    # Extract params with regex
    ip_m = re.search(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})', message)
    url_m = re.search(r'(https?://\S+)', message)

    # Detect action keywords
    action = None
    for a, words in [
        ("install", ["install", "安装"]),
        ("start", ["start", "开始", "启动", "run"]),
        ("stop", ["stop", "停止"]),
        ("test", ["test", "测试", "检查"]),
        ("status", ["status", "状态", "check"]),
        ("vlm_active", ["vlm_active", "主动", "active"]),
        ("sweep", ["sweep", "扫描", "grid"]),
    ]:
        if any(w in msg for w in words):
            action = a
            break

    # Detect model name
    model = None
    for m in ["qwen3-vl:8b", "qwen2.5-vl:7b", "llava", "gemma", "llama"]:
        if m.split(":")[0] in msg or m.split("-")[0] in msg:
            model = m
            break

    return {
        "intent": primary,
        "all_intents": intents,
        "ip": ip_m.group(1) if ip_m else None,
        "url": url_m.group(1) if url_m else None,
        "model": model,
        "action": action,
        "extra": None,
    }


# ── Test functions ──

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


# ── Main handler ──

async def handle_setup_message(message: str, cfg, save_config_fn) -> str | None:
    """
    Process a chat message for setup intent.
    Uses LLM for intent detection (with keyword fallback).
    Returns a response string if it's a setup message, None otherwise.
    """
    # Try LLM-based intent detection first, fall back to keywords
    parsed = await _llm_detect_intent(message)
    if not parsed or parsed.get("intent") == "none":
        parsed = _keyword_detect_intent(message)

    intent = parsed.get("intent", "none")
    if intent == "none":
        return None

    ip = parsed.get("ip")
    url = parsed.get("url")
    model = parsed.get("model")
    action = parsed.get("action")
    msg = message.lower()

    # ── Full setup wizard ──
    if intent == "setup" or intent == "status":
        status_parts = []

        # Check LLM
        llm_result = await test_llm(cfg.llm_url, cfg.llm_model, cfg.llm_api_key)
        if llm_result["ok"]:
            status_parts.append(f"✅ LLM connected: {cfg.llm_model} @ {cfg.llm_url}")
        else:
            status_parts.append(f"❌ LLM not connected: {llm_result['error']}\n   → Tell me your LLM API URL to configure it")

        # Check camera
        if cfg.camera_snapshot_url:
            cam_result = await test_camera(cfg.camera_snapshot_url)
            if cam_result["ok"]:
                status_parts.append(f"✅ Camera connected: {cfg.camera_snapshot_url} ({cam_result['size']} bytes)")
            else:
                status_parts.append(f"❌ Camera not responding: {cam_result['error']}")
        else:
            status_parts.append("❌ Camera not configured\n   → Tell me your camera IP to auto-detect")

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
            status_parts.append("❌ YOLO not installed (will auto-install on first use)")

        status_parts.append(f"\nClasses: {', '.join(cfg.classes)}")
        status_parts.append(f"Patrol: {'enabled' if cfg.patrol_enabled else 'disabled'} ({cfg.patrol_strategy})")

        return "📋 System Status:\n\n" + "\n".join(status_parts)

    # ── Camera setup ──
    if intent == "camera":
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

            # Try common snapshot paths
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
                    f"   Please provide the full snapshot URL.")

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
                    f"   To change: tell me the camera IP or snapshot URL")
        else:
            return ("📷 No camera configured.\n\n"
                    "   Tell me your camera IP and I'll auto-detect the type.\n"
                    "   Or provide a snapshot URL directly.")

    # ── LLM setup ──
    if intent == "llm":
        if url or ip:
            new_url = url or f"http://{ip}:11434/v1/chat/completions"
            use_model = model or cfg.llm_model
            result = await test_llm(new_url, use_model, cfg.llm_api_key)
            if result["ok"]:
                cfg.llm_url = new_url
                cfg.llm_model = use_model
                save_config_fn()
                return (f"✅ LLM connected!\n"
                        f"   URL: {new_url}\n"
                        f"   Model: {use_model}\n"
                        f"   Test reply: {result['reply']}")
            else:
                return (f"❌ LLM connection failed: {result['error']}\n"
                        f"   URL tried: {new_url}\n"
                        f"   Model: {use_model}\n\n"
                        f"   Make sure the LLM service is running and the model is available.")

        # General LLM help
        result = await test_llm(cfg.llm_url, cfg.llm_model, cfg.llm_api_key)
        status = "connected" if result["ok"] else f"error: {result['error']}"
        return (f"🤖 Current LLM: {cfg.llm_model} @ {cfg.llm_url} ({status})\n\n"
                f"   To change: tell me the LLM API URL\n"
                f"   Set API key via env: LLM_API_KEY=your-key")

    # ── YOLO setup ──
    if intent == "yolo":
        yolo_result = await test_yolo()
        if action == "install" or "install" in msg or "安装" in msg:
            try:
                import subprocess, sys
                subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics", "-q"])
                return "✅ YOLO (ultralytics) installed successfully!"
            except Exception as e:
                return f"❌ YOLO install failed: {e}"

        return (f"🔍 YOLO Status:\n"
                f"   Installed: {'yes (' + yolo_result['version'] + ')' if yolo_result['ok'] else 'no (auto-installs on first use)'}\n"
                f"   Model: {cfg.yolo_model_path or 'default (yolo11s.pt)'}\n"
                f"   Data dir: {cfg.yolo_data_dir}\n"
                f"   Auto-train: {'on' if cfg.yolo_auto_train else 'off'} (threshold: {cfg.yolo_train_threshold} images)\n\n"
                f"   Say 'install yolo' to install now.")

    # ── Sensor setup ──
    if intent == "sensor":
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
                f"   3. HTTP pull: tell me the sensor URL")

    # ── Actuator setup ──
    if intent == "actuator":
        if url or ip:
            new_url = url or f"http://{ip}/control"
            cfg.actuator_url = new_url
            save_config_fn()
            return f"✅ Actuator URL set to: {new_url}"

        return ("🔧 Actuator Configuration:\n"
                f"   Current URL: {cfg.actuator_url or '(not set)'}\n\n"
                f"   The agent sends care commands as HTTP POST:\n"
                f"   {{\"action\":\"water\",\"enable\":true,\"duration_sec\":300}}\n\n"
                f"   Tell me the actuator endpoint URL to configure it.")

    # ── Patrol control ──
    if intent == "patrol":
        if action in ("start",):
            return "🔄 Use the Dashboard 'Run Patrol' button, or POST /api/patrol/start"
        if action == "vlm_active":
            cfg.patrol_strategy = "vlm_active"
            save_config_fn()
            return "✅ Patrol strategy set to vlm_active (panorama → focus → diagnose)"
        if action == "sweep":
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
                f"   Tell me which strategy you'd like to use.")

    return None
