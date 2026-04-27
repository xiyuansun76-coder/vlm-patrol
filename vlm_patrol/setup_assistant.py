"""Setup assistant — handles configuration through chat conversation.

Security & performance:
1. API Key AES encryption at rest (Fernet)
2. Request rate limiting (per-minute cap)
3. Local Ollama preferred over cloud API
4. Intent cache (LRU) for repeated similar queries

Falls back to keyword matching when no LLM is available.
"""

import base64
import hashlib
import json
import logging
import os
import re
import time
from collections import OrderedDict
from pathlib import Path

import httpx

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# 1. API Key encryption (AES via Fernet)
# ═══════════════════════════════════════════════════════════

def _derive_fernet_key(passphrase: str) -> bytes:
    """Derive a Fernet-compatible key from a passphrase."""
    digest = hashlib.sha256(passphrase.encode()).digest()
    return base64.urlsafe_b64encode(digest)


def encrypt_api_key(api_key: str, passphrase: str = "") -> str:
    """Encrypt an API key. Returns base64 ciphertext."""
    try:
        from cryptography.fernet import Fernet
    except ImportError:
        log.warning("cryptography not installed — storing key as base64 only")
        return "b64:" + base64.b64encode(api_key.encode()).decode()

    secret = passphrase or os.environ.get("VLP_SECRET", "vlm-patrol-default-key")
    f = Fernet(_derive_fernet_key(secret))
    return "enc:" + f.encrypt(api_key.encode()).decode()


def decrypt_api_key(token: str, passphrase: str = "") -> str:
    """Decrypt an API key. Handles enc:, b64:, and plaintext."""
    if token.startswith("enc:"):
        try:
            from cryptography.fernet import Fernet
        except ImportError:
            log.error("cryptography not installed — cannot decrypt key")
            return ""
        secret = passphrase or os.environ.get("VLP_SECRET", "vlm-patrol-default-key")
        f = Fernet(_derive_fernet_key(secret))
        return f.decrypt(token[4:].encode()).decode()
    if token.startswith("b64:"):
        return base64.b64decode(token[4:]).decode()
    return token  # plaintext


# ═══════════════════════════════════════════════════════════
# 2. Rate limiter (token bucket)
# ═══════════════════════════════════════════════════════════

class RateLimiter:
    """Simple token-bucket rate limiter."""

    def __init__(self, max_calls: int = 20, period_sec: float = 60):
        self.max_calls = max_calls
        self.period = period_sec
        self._calls: list[float] = []

    def allow(self) -> bool:
        now = time.monotonic()
        self._calls = [t for t in self._calls if now - t < self.period]
        if len(self._calls) >= self.max_calls:
            return False
        self._calls.append(now)
        return True

    @property
    def remaining(self) -> int:
        now = time.monotonic()
        self._calls = [t for t in self._calls if now - t < self.period]
        return max(0, self.max_calls - len(self._calls))


_rate_limiter = RateLimiter(
    max_calls=int(os.environ.get("INTENT_RATE_LIMIT", "20")),
    period_sec=60,
)


# ═══════════════════════════════════════════════════════════
# 3. Intent cache (LRU)
# ═══════════════════════════════════════════════════════════

class LRUCache:
    """Simple LRU cache with TTL."""

    def __init__(self, maxsize: int = 128, ttl_sec: float = 300):
        self._cache: OrderedDict[str, tuple[float, dict]] = OrderedDict()
        self.maxsize = maxsize
        self.ttl = ttl_sec

    def _normalize_key(self, message: str) -> str:
        """Normalize message for cache lookup — lowercase, strip, collapse whitespace."""
        return re.sub(r'\s+', ' ', message.lower().strip())

    def get(self, message: str) -> dict | None:
        key = self._normalize_key(message)
        if key in self._cache:
            ts, val = self._cache[key]
            if time.monotonic() - ts < self.ttl:
                self._cache.move_to_end(key)
                log.debug("Intent cache hit: %s", key[:40])
                return val
            del self._cache[key]
        return None

    def put(self, message: str, result: dict):
        key = self._normalize_key(message)
        self._cache[key] = (time.monotonic(), result)
        if len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)


_intent_cache = LRUCache(
    maxsize=int(os.environ.get("INTENT_CACHE_SIZE", "128")),
    ttl_sec=float(os.environ.get("INTENT_CACHE_TTL", "300")),
)


# ═══════════════════════════════════════════════════════════
# 4. Intent LLM — local Ollama first, then cloud API
# ═══════════════════════════════════════════════════════════

# Cloud config (NVIDIA etc.)
INTENT_LLM_URL = os.environ.get(
    "INTENT_LLM_URL",
    "https://integrate.api.nvidia.com/v1/chat/completions",
)
INTENT_LLM_MODEL = os.environ.get("INTENT_LLM_MODEL", "meta/llama-3.1-8b-instruct")
_raw_key = os.environ.get("INTENT_LLM_KEY", os.environ.get("LLM_API_KEY", ""))
INTENT_LLM_KEY = decrypt_api_key(_raw_key) if _raw_key else ""

# Local Ollama config
LOCAL_INTENT_URL = os.environ.get(
    "LOCAL_INTENT_URL",
    "http://localhost:11434/v1/chat/completions",
)
LOCAL_INTENT_MODEL = os.environ.get("LOCAL_INTENT_MODEL", "qwen3:0.6b")

INTENT_PROMPT = """You are an intent classifier for a plant monitoring system.
Given a user message, extract the intent and parameters.

Possible intents:
- camera: configure or check camera/PTZ (may include IP or URL)
- llm: configure or check LLM/AI model API (may include URL, model name)
- yolo: check, install, or configure YOLO detection
- sensor: configure sensor data source (may include URL)
- actuator: configure actuator/control endpoint (may include URL)
- setup: general system status check or guided setup
- patrol: configure or control patrol (may include strategy name)
- none: not about system configuration

Reply with ONLY a JSON object:
{"intent": "camera", "ip": "192.168.1.100", "url": null, "model": null, "action": null, "extra": null}

Fields:
- intent: primary intent
- ip: IP address if present, null otherwise
- url: full URL (http/https) if present, null otherwise
- model: model name if mentioned, null otherwise
- action: "install"/"start"/"stop"/"test"/"status"/"vlm_active"/"sweep", null if unclear
- extra: other relevant info, null if none

User message: """


async def _try_local_ollama() -> bool:
    """Check if local Ollama is available."""
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.get("http://localhost:11434/api/tags")
            return r.status_code == 200
    except Exception:
        return False


async def _call_intent_llm(message: str, url: str, model: str, api_key: str = "") -> dict | None:
    """Call an OpenAI-compatible LLM for intent classification."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(url, json={
                "model": model,
                "messages": [{"role": "user", "content": INTENT_PROMPT + message}],
                "max_tokens": 150,
                "temperature": 0,
            }, headers=headers)
            if r.status_code != 200:
                log.debug("Intent LLM %s returned %d", url[:40], r.status_code)
                return None
            data = r.json()
            reply = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            m = re.search(r'\{[^}]+\}', reply)
            if m:
                parsed = json.loads(m.group())
                return parsed
    except Exception as e:
        log.debug("Intent LLM %s failed: %s", url[:40], e)
    return None


async def _llm_detect_intent(message: str) -> dict | None:
    """
    Detect intent via LLM. Priority:
    1. Cache hit → return immediately
    2. Local Ollama → fast, private, free
    3. Cloud API (NVIDIA) → fallback
    4. None → keyword fallback
    """
    # Check cache
    cached = _intent_cache.get(message)
    if cached is not None:
        return cached

    # Rate limit check
    if not _rate_limiter.allow():
        log.warning("Intent LLM rate limited (%d/%d calls used)",
                     _rate_limiter.max_calls - _rate_limiter.remaining,
                     _rate_limiter.max_calls)
        return None

    result = None

    # Try local Ollama first
    if await _try_local_ollama():
        result = await _call_intent_llm(message, LOCAL_INTENT_URL, LOCAL_INTENT_MODEL)
        if result:
            log.info("Intent (local %s): %s → %s",
                     LOCAL_INTENT_MODEL, message[:40], result.get("intent"))

    # Fall back to cloud API
    if not result and INTENT_LLM_KEY:
        result = await _call_intent_llm(message, INTENT_LLM_URL, INTENT_LLM_MODEL, INTENT_LLM_KEY)
        if result:
            log.info("Intent (cloud %s): %s → %s",
                     INTENT_LLM_MODEL, message[:40], result.get("intent"))

    # Cache the result
    if result:
        _intent_cache.put(message, result)

    return result


# ═══════════════════════════════════════════════════════════
# Keyword fallback
# ═══════════════════════════════════════════════════════════

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

    specific = [i for i in intents if i not in ("setup", "status")]
    primary = specific[0] if specific else intents[0]

    ip_m = re.search(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})', message)
    url_m = re.search(r'(https?://\S+)', message)

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


# ═══════════════════════════════════════════════════════════
# Test functions
# ═══════════════════════════════════════════════════════════

async def test_camera(url: str) -> dict:
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
    try:
        import ultralytics
        return {"ok": True, "version": ultralytics.__version__}
    except ImportError:
        return {"ok": False, "error": "ultralytics not installed"}


# ═══════════════════════════════════════════════════════════
# Main handler
# ═══════════════════════════════════════════════════════════

async def handle_setup_message(message: str, cfg, save_config_fn) -> str | None:
    """
    Process a chat message for setup intent.
    Priority: cache → local LLM → cloud LLM → keywords.
    Returns a response string if it's a setup message, None otherwise.
    """
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
    if intent in ("setup", "status"):
        status_parts = []

        llm_result = await test_llm(cfg.llm_url, cfg.llm_model, cfg.llm_api_key)
        if llm_result["ok"]:
            status_parts.append(f"✅ LLM connected: {cfg.llm_model} @ {cfg.llm_url}")
        else:
            status_parts.append(f"❌ LLM not connected: {llm_result['error']}\n   → Tell me your LLM API URL to configure it")

        if cfg.camera_snapshot_url:
            cam_result = await test_camera(cfg.camera_snapshot_url)
            if cam_result["ok"]:
                status_parts.append(f"✅ Camera connected: {cfg.camera_snapshot_url} ({cam_result['size']} bytes)")
            else:
                status_parts.append(f"❌ Camera not responding: {cam_result['error']}")
        else:
            status_parts.append("❌ Camera not configured\n   → Tell me your camera IP to auto-detect")

        if cfg.ptz_enabled:
            ptz_result = await test_ptz(cfg.ptz_url, cfg.ptz_user, cfg.ptz_pass)
            if ptz_result["ok"]:
                status_parts.append(f"✅ PTZ connected: az={ptz_result['az']} el={ptz_result['el']}")
            else:
                status_parts.append(f"❌ PTZ not responding: {ptz_result['error']}")
        else:
            status_parts.append("ℹ️ PTZ not enabled (optional)")

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
        if action == "start":
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
