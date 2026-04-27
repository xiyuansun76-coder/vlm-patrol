from fastapi import APIRouter, Query
from typing import Optional
import database as db

router = APIRouter(prefix="/api/sensors", tags=["sensors"])

@router.get("/latest")
def latest():
    return db.query_latest()

@router.get("")
def history(
    node_id: Optional[str] = None,
    limit:   int           = Query(100, le=1000),
    from_ts: Optional[str] = None,
    to_ts:   Optional[str] = None,
):
    return db.query_readings(node_id=node_id, limit=limit, from_ts=from_ts, to_ts=to_ts)
