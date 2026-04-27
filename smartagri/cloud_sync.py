"""
阶段二：Supabase 云端同步

- 每次有新传感器数据时，异步推送到 Supabase
- 推送失败时写入 pending_sync 表，网络恢复后批量补传
- 图像数据不推云端（体积大，只存本地）
"""
import json
import logging
import threading
import urllib.request
import urllib.error
from datetime import datetime

from config import SUPABASE_URL, SUPABASE_KEY
import database as db

log = logging.getLogger(__name__)

_enabled = bool(SUPABASE_URL and SUPABASE_KEY)
_headers = {
    "Content-Type": "application/json",
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Prefer": "return=minimal",
}


def _post(endpoint: str, payload: dict) -> bool:
    """发送一条记录到 Supabase REST API，成功返回 True。"""
    if not _enabled:
        return True
    url = f"{SUPABASE_URL}/rest/v1/{endpoint}"
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers=_headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10):
            return True
    except urllib.error.HTTPError as e:
        log.warning(f"Supabase HTTP {e.code}: {e.read().decode()[:200]}")
    except Exception as e:
        log.warning(f"Supabase 推送失败: {e}")
    return False


def _save_pending(table: str, payload: dict):
    """推送失败时暂存到本地 pending_sync 表。"""
    conn = db.get_conn()
    conn.execute(
        "INSERT INTO pending_sync (table_name, payload, created) VALUES (?,?,?)",
        (table, json.dumps(payload), datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    )
    conn.commit()


def push_reading(record: dict):
    """异步推送传感器数据，不阻塞主线程。"""
    if not _enabled:
        return

    def _run():
        # 只推关键字段，排除 id 和 raw_json
        payload = {k: v for k, v in record.items()
                   if k not in ("id", "raw_json") and v is not None}
        if not _post("sensor_readings", payload):
            _save_pending("sensor_readings", payload)
            log.info("已暂存到 pending_sync")

    threading.Thread(target=_run, daemon=True).start()


def flush_pending():
    """补传积压的离线数据，启动时或网络恢复后调用。"""
    if not _enabled:
        return
    conn = db.get_conn()
    rows = conn.execute(
        "SELECT id, table_name, payload FROM pending_sync ORDER BY id LIMIT 100"
    ).fetchall()
    if not rows:
        return
    log.info(f"开始补传 {len(rows)} 条离线数据…")
    for row in rows:
        payload = json.loads(row["payload"])
        if _post(row["table_name"], payload):
            conn.execute("DELETE FROM pending_sync WHERE id=?", (row["id"],))
            conn.commit()
        else:
            break  # 网络仍不通，停止尝试
    log.info("补传完成")
