import sqlite3
import threading
from config import DB_PATH

_local = threading.local()

def get_conn():
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
    return _local.conn

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sensor_readings (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp        TEXT NOT NULL,
            node_id          TEXT NOT NULL,
            topic            TEXT,
            temperature      REAL,
            moisture         REAL,
            ph               REAL,
            nitrogen         REAL,
            phosphorus       REAL,
            potassium        REAL,
            air_temperature  REAL,
            air_humidity     REAL,
            soil_temperature REAL,
            soil_moisture    REAL,
            co2              REAL,
            light            REAL,
            raw_json         TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_ts   ON sensor_readings(timestamp);
        CREATE INDEX IF NOT EXISTS idx_node ON sensor_readings(node_id);

        CREATE TABLE IF NOT EXISTS camera_snapshots (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            image_b64 TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS control_logs (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            action    TEXT NOT NULL,
            enable    INTEGER NOT NULL,
            duration  INTEGER,
            source    TEXT DEFAULT 'api'
        );

        CREATE TABLE IF NOT EXISTS pending_sync (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            table_name TEXT NOT NULL,
            payload   TEXT NOT NULL,
            created   TEXT NOT NULL
        );
    """)
    conn.commit()
    conn.close()

def _f(val):
    """安全转 float，失败或为布尔值返回 None。"""
    if val is None or val == "" or val == "null":
        return None
    if isinstance(val, bool):   # false/true 是设备开关状态，不是数值
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None

def insert_reading(data: dict):
    conn = get_conn()
    conn.execute("""
        INSERT INTO sensor_readings
        (timestamp, node_id, topic,
         temperature, moisture, ph, nitrogen, phosphorus, potassium,
         air_temperature, air_humidity, soil_temperature, soil_moisture,
         co2, light, raw_json)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        data.get("timestamp"),
        data.get("node_id"),
        data.get("topic"),
        _f(data.get("temperature")),
        _f(data.get("moisture")),
        _f(data.get("ph") or data.get("pH")),
        _f(data.get("nitrogen")),
        _f(data.get("phosphorus")),
        _f(data.get("potassium")),
        _f(data.get("air_temperature")),
        _f(data.get("air_humidity")),
        _f(data.get("soil_temperature")),
        _f(data.get("soil_moisture")),
        _f(data.get("co2")),
        _f(data.get("light")),
        data.get("raw_json"),
    ))
    conn.commit()

def insert_snapshot(timestamp: str, image_b64: str):
    conn = get_conn()
    conn.execute("INSERT INTO camera_snapshots (timestamp, image_b64) VALUES (?,?)",
                 (timestamp, image_b64))
    # 只保留最近 10 张
    conn.execute("""
        DELETE FROM camera_snapshots WHERE id NOT IN (
            SELECT id FROM camera_snapshots ORDER BY id DESC LIMIT 10
        )
    """)
    conn.commit()

def insert_control_log(action: str, enable: bool, duration: int | None, source: str):
    from datetime import datetime
    conn = get_conn()
    conn.execute(
        "INSERT INTO control_logs (timestamp, action, enable, duration, source) VALUES (?,?,?,?,?)",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), action, int(enable), duration, source)
    )
    conn.commit()

def query_readings(node_id=None, limit=100, from_ts=None, to_ts=None):
    conn = get_conn()
    sql  = "SELECT * FROM sensor_readings WHERE 1=1"
    args = []
    if node_id:
        sql += " AND node_id = ?"; args.append(node_id)
    if from_ts:
        sql += " AND timestamp >= ?"; args.append(from_ts)
    if to_ts:
        sql += " AND timestamp <= ?"; args.append(to_ts)
    sql += " ORDER BY timestamp DESC LIMIT ?"
    args.append(limit)
    return [dict(r) for r in conn.execute(sql, args).fetchall()]

def query_latest():
    conn = get_conn()
    rows = conn.execute("""
        SELECT * FROM sensor_readings
        WHERE id IN (
            SELECT MAX(id) FROM sensor_readings GROUP BY node_id
        )
    """).fetchall()
    return [dict(r) for r in rows]

def query_latest_snapshot():
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM camera_snapshots ORDER BY timestamp DESC, id DESC LIMIT 1"
    ).fetchone()
    return dict(row) if row else None

def query_snapshots(limit=6):
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, timestamp FROM camera_snapshots ORDER BY timestamp DESC, id DESC LIMIT ?",
        (limit,)
    ).fetchall()
    return [dict(r) for r in rows]

def query_snapshot_by_id(snap_id):
    conn = get_conn()
    row = conn.execute(
        "SELECT id, timestamp, image_b64 FROM camera_snapshots WHERE id=?", (snap_id,)
    ).fetchone()
    return dict(row) if row else None

def query_control_logs(limit=50):
    conn = get_conn()
    return [dict(r) for r in conn.execute(
        "SELECT * FROM control_logs ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()]
