"""
MQTT → SQLite 桥接线程
订阅所有相关 topic，解析后写库。
"""
import json
import logging
import threading
from datetime import datetime

import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion

from config import BROKER_HOST, BROKER_PORT
import database as db
import cloud_sync
import control_state

log = logging.getLogger(__name__)

# 最新设备状态缓存（供 /api/devices 接口使用）
device_status: dict = {}
_lock = threading.Lock()

# 外部可注入的回调（用于 WebSocket 推送等）
on_sensor_update = None
on_camera_update = None


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _parse_fields(raw: dict) -> dict:
    """把 JSON 字段统一转成 float 存库。"""
    FIELD_MAP = {
        "temperature":      "temperature",
        "moisture":         "moisture",
        "pH": "ph", "ph":  "ph",
        "nitrogen":         "nitrogen",
        "phosphorus":       "phosphorus",
        "potassium":        "potassium",
        "air_temperature":  "air_temperature",
        "air_humidity":     "air_humidity",
        "soil_temperature": "soil_temperature",
        "soil_moisture":    "soil_moisture",
        "co2":              "co2",
        "light":            "light",
    }
    return {v: raw[k] for k, v in FIELD_MAP.items() if k in raw}


def _on_connect(client, userdata, flags, reason_code, properties):
    if reason_code.is_failure:
        log.error(f"MQTT 连接失败: {reason_code}")
        return
    log.info(f"MQTT 已连接 {BROKER_HOST}:{BROKER_PORT}")
    for topic in ("zigbee/sensors", "sensor/soil", "sensors", "greenhouse/status",
                  "greenhouse/camera", "greenhouse/control"):
        client.subscribe(topic, qos=0)
        log.info(f"已订阅 {topic}")


def _on_message(client, userdata, msg):
    topic   = msg.topic
    payload = msg.payload.decode("utf-8", errors="ignore").strip()
    try:
        raw = json.loads(payload)
    except json.JSONDecodeError:
        # 处理 app 发出的 "pump/on" 纯字符串格式
        if topic == "greenhouse/control" and "/" in payload:
            parts = payload.split("/")
            if len(parts) == 2 and parts[1] in ("on", "off"):
                control_state.set_state(parts[0], parts[1] == "on")
                if on_sensor_update:
                    on_sensor_update("__control__", control_state.get_all())
        return

    ts = _now()

    # ── 控制指令（同步 app 发出的状态）─────────────────────────
    if topic == "greenhouse/control":
        # app 发 "pump/on" 格式；web 经服务器转发也用同格式
        if "action" in raw and "enable" in raw:
            control_state.set_state(raw["action"], bool(raw["enable"]))
        if on_sensor_update:
            on_sensor_update("__control__", control_state.get_all())
        return

    # ── 摄像头图像 ────────────────────────────────────────────────
    if topic == "greenhouse/camera":
        if "image" in raw:
            db.insert_snapshot(ts, raw["image"])
            with _lock:
                device_status["rpi_camera"] = {"last_seen": ts, "online": True}
            log.info("摄像头图像已存库")
            if on_camera_update:
                on_camera_update(ts)
        elif raw.get("status") == "online":
            with _lock:
                device_status["rpi_camera"] = {"last_seen": ts, "online": True}
        return

    # ── ESP32 控制器状态 ──────────────────────────────────────────
    if topic == "greenhouse/status":
        node_id = "greenhouse_controller"
        fields  = _parse_fields(raw)
        record  = {"timestamp": ts, "node_id": node_id, "topic": topic,
                   "raw_json": payload, **fields}
        db.insert_reading(record)
        cloud_sync.push_reading(record)
        with _lock:
            device_status[node_id] = {"last_seen": ts, "online": True, "fields": fields}
        if on_sensor_update:
            on_sensor_update(node_id, fields)
        return

    # ── 传感器数据（zigbee/sensors 及其他）────────────────────────
    fallbacks = {
        "zigbee/sensors": "greenhouse_zigbee",
        "sensor/soil":    "30aea49c97d8",
    }
    fallback = fallbacks.get(topic, topic.replace("/", "_"))
    node_id = raw.get("nodeID") or raw.get("node_id") or fallback
    fields  = _parse_fields(raw)
    record  = {"timestamp": ts, "node_id": node_id, "topic": topic,
               "raw_json": payload, **fields}
    db.insert_reading(record)
    cloud_sync.push_reading(record)
    with _lock:
        device_status[node_id] = {"last_seen": ts, "online": True, "fields": fields}
    log.debug(f"[{node_id}] 已写库: {fields}")
    if on_sensor_update:
        on_sensor_update(node_id, fields)


def _on_disconnect(client, userdata, disconnect_flags, reason_code, properties):
    log.warning(f"MQTT 断开: {reason_code}，等待重连…")


_client: mqtt.Client | None = None


def get_client() -> mqtt.Client:
    return _client


def publish_control(action: str, enable: bool, duration_sec: int | None = None):
    if _client is None:
        raise RuntimeError("MQTT 客户端未初始化")
    # ESP32 只认 "pump/on" 字符串格式
    cmd_str = f"{action}/{'on' if enable else 'off'}"
    _client.publish("greenhouse/control", cmd_str, qos=1)
    log.info(f"已发布控制指令: {cmd_str}")


def publish_capture():
    if _client is None:
        raise RuntimeError("MQTT 客户端未初始化")
    _client.publish("greenhouse/camera/capture", "snap", qos=1)
    log.info("已触发拍照")


def start(block=False):
    global _client
    _client = mqtt.Client(
        callback_api_version=CallbackAPIVersion.VERSION2,
        client_id="smartagri_server",
        clean_session=True,
    )
    _client.on_connect    = _on_connect
    _client.on_message    = _on_message
    _client.on_disconnect = _on_disconnect

    _client.connect_async(BROKER_HOST, BROKER_PORT, keepalive=60)
    if block:
        _client.loop_forever()
    else:
        _client.loop_start()
        log.info("MQTT bridge 已在后台启动")
