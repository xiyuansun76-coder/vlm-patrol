"""Configuration loader — single source of truth."""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

_DEFAULT_CONFIG = Path(__file__).parent.parent / "config.yaml"

# VLM output class name normalization
CLASS_ALIASES = {
    "strawberry": "strawberry", "草莓": "strawberry", "草莓苗": "strawberry", "fragaria": "strawberry",
    "celery": "celery", "芹菜": "celery", "西芹": "celery", "芹菜幼苗": "celery", "apium": "celery",
    "chive": "chive", "chives": "chive", "韭菜": "chive", "葱": "chive", "香葱": "chive", "allium": "chive",
    "coriander": "coriander", "香菜": "coriander", "芫荽": "coriander", "cilantro": "coriander",
    "rose": "rose", "rosa": "rose", "月季": "rose", "玫瑰": "rose",
    "blueberry": "blueberry", "蓝莓": "blueberry", "vaccinium": "blueberry",
}


def normalize_class(name: str, classes: list[str]) -> str | None:
    """Normalize a VLM-output class name to standard label. Returns None if no match."""
    low = name.lower().strip()
    # direct alias match
    std = CLASS_ALIASES.get(low)
    if std and std in classes:
        return std
    # fuzzy: check if any alias is substring
    for alias, std_name in CLASS_ALIASES.items():
        if (alias in low or low in alias) and std_name in classes:
            return std_name
    # direct match against class list
    if low in classes:
        return low
    return None


class Config:
    """Loads config.yaml + .env, provides typed access."""

    def __init__(self, config_path: str | Path | None = None):
        path = Path(config_path) if config_path else _DEFAULT_CONFIG
        if path.exists():
            with open(path) as f:
                self._data = yaml.safe_load(f) or {}
        else:
            self._data = {}

        # LLM
        llm = self._data.get("llm", {})
        self.llm_url = llm.get("url", "http://localhost:11434/v1/chat/completions")
        self.llm_model = llm.get("model", "qwen3-vl:8b")
        self.llm_api_key = os.environ.get("LLM_API_KEY", "")

        # Camera
        cam = self._data.get("camera", {})
        self.camera_snapshot_url = cam.get("snapshot_url", "")
        self.camera_stream_url = cam.get("stream_url", "")

        # PTZ (optional)
        ptz = cam.get("ptz", {})
        self.ptz_enabled = ptz.get("enabled", False)
        self.ptz_url = ptz.get("url", "")
        self.ptz_user = os.environ.get("PTZ_USER", "admin")
        self.ptz_pass = os.environ.get("PTZ_PASS", "")
        self.ptz_fov_h = ptz.get("fov_h_deg", 55)
        self.ptz_fov_v = ptz.get("fov_v_deg", 32)
        self.ptz_img_w = ptz.get("image_width", 1920)
        self.ptz_img_h = ptz.get("image_height", 1080)
        self.ptz_home_az = ptz.get("home_az", 2800)
        self.ptz_home_el = ptz.get("home_el", 350)
        self.ptz_wide_zoom = ptz.get("wide_zoom", 10)
        self.ptz_close_zoom = ptz.get("close_zoom", 25)

        # Sensor
        sensor = self._data.get("sensor", {})
        self.sensor_url = sensor.get("url", "")

        # Actuator (optional — HTTP endpoint that receives care commands)
        actuator = self._data.get("actuator", {})
        self.actuator_url = actuator.get("url", "")
        mqtt = sensor.get("mqtt", {})
        self.mqtt_enabled = mqtt.get("enabled", False)
        self.mqtt_broker = mqtt.get("broker", "localhost")
        self.mqtt_port = mqtt.get("port", 1883)
        self.mqtt_topics = mqtt.get("topics", [])

        # YOLO
        yolo = self._data.get("yolo", {})
        self.yolo_model_path = yolo.get("model_path", "")
        self.yolo_data_dir = Path(yolo.get("data_dir", "./data"))
        self.yolo_auto_train = yolo.get("auto_train", True)
        self.yolo_train_threshold = yolo.get("train_threshold", 50)

        # Classes
        self.classes = self._data.get("classes", [
            "strawberry", "celery", "chive", "coriander", "rose", "blueberry"
        ])

        # Patrol
        patrol = self._data.get("patrol", {})
        self.patrol_enabled = patrol.get("enabled", False)
        self.patrol_interval = patrol.get("interval_minutes", 60)
        self.patrol_strategy = patrol.get("strategy", "single")

        # Agent
        agent = self._data.get("agent", {})
        self.agent_auto_analysis = agent.get("auto_analysis", False)
        self.agent_interval = agent.get("interval_minutes", 30)

        # Server
        srv = self._data.get("server", {})
        self.server_host = srv.get("host", "0.0.0.0")
        self.server_port = srv.get("port", 8765)

    def normalize_class(self, name: str) -> str | None:
        return normalize_class(name, self.classes)
