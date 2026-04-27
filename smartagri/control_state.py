"""全局继电器状态，由 MQTT greenhouse/control 消息维护。"""
import threading

_lock  = threading.Lock()
_state: dict[str, bool] = {
    "light": False, "pump": False, "curtain": False, "fan": False,
}

def set_state(action: str, enable: bool):
    with _lock:
        _state[action] = enable

def get_all() -> dict:
    with _lock:
        return dict(_state)

def init_from_db():
    """启动时从 control_logs 恢复每个设备的最新状态。"""
    import database as db
    conn = db.get_conn()
    with _lock:
        for action in list(_state.keys()):
            row = conn.execute(
                "SELECT enable FROM control_logs WHERE action=? ORDER BY id DESC LIMIT 1",
                (action,)
            ).fetchone()
            if row is not None:
                _state[action] = bool(row["enable"])
