"""Patrol core — grab image → VLM detect → filter → collect → YOLO train → loop."""

import asyncio
import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

# Minimum VLM confidence to accept a pseudo-label
MIN_CONFIDENCE = 0.5
# Fraction of collected images routed to val split
VAL_RATIO = 0.2


@dataclass
class PlantRecord:
    id: str = ""
    type: str = ""
    health: str = ""
    confidence: float = 0.0
    bbox: list = field(default_factory=list)
    details: str = ""
    timestamp: str = ""

    def to_dict(self):
        return {
            "id": self.id, "type": self.type, "health": self.health,
            "confidence": self.confidence, "bbox": self.bbox,
            "details": self.details, "timestamp": self.timestamp,
        }


@dataclass
class PatrolSession:
    session_id: str = ""
    status: str = "idle"       # idle, running, completed, error
    started_at: str = ""
    plants: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    images_collected: int = 0
    yolo_detections: int = 0   # plants found by YOLO (after training)

    def to_dict(self):
        return {
            "session_id": self.session_id, "status": self.status,
            "started_at": self.started_at,
            "plants": [p.to_dict() for p in self.plants],
            "errors": self.errors, "images_collected": self.images_collected,
            "yolo_detections": self.yolo_detections,
        }


class Patrol:
    """Core patrol loop: snapshot → VLM detect → filter → collect → YOLO distill."""

    def __init__(self, config, vlm, yolo_mgr):
        self.cfg = config
        self.vlm = vlm
        self.yolo = yolo_mgr
        self.http = httpx.AsyncClient(timeout=15)
        self.session: PatrolSession | None = None
        self._running = False
        self._collect_count = 0  # total images collected, for train/val split
        self.history: list[dict] = []

    async def snapshot(self) -> bytes | None:
        """Grab image from camera_snapshot_url."""
        url = self.cfg.camera_snapshot_url
        if not url:
            log.warning("No camera_snapshot_url configured")
            return None
        try:
            resp = await self.http.get(url)
            resp.raise_for_status()
            return resp.content
        except Exception as e:
            log.error("Snapshot failed: %s", e)
            return None

    def _pick_split(self) -> str:
        """Route ~20% of images to val, rest to train."""
        self._collect_count += 1
        return "val" if random.random() < VAL_RATIO else "train"

    def _filter_labels(self, plants: list[dict]) -> list[dict]:
        """Filter pseudo-labels: known class + sufficient bbox area."""
        labels = []
        for p in plants:
            cls_name = p["type"]
            if cls_name not in self.cfg.classes:
                continue
            bbox = p["bbox"]
            # reject tiny boxes (likely noise)
            bw = bbox[2] - bbox[0]
            bh = bbox[3] - bbox[1]
            if bw < 10 or bh < 10:
                continue
            labels.append({
                "class_id": self.cfg.classes.index(cls_name),
                "bbox": bbox,
            })
        return labels

    async def run_once(self, sensor_data: dict = None) -> PatrolSession:
        """Single patrol cycle: snapshot → VLM detect → filter → collect → diagnose."""
        session = PatrolSession(
            session_id=str(uuid.uuid4())[:8],
            status="running",
            started_at=datetime.now().isoformat(),
        )
        self.session = session

        try:
            # 1. Grab image
            image = await self.snapshot()
            if not image:
                session.status = "error"
                session.errors.append("Failed to get camera image")
                return session

            img_w = self.cfg.ptz_img_w if self.cfg.ptz_enabled else 1920
            img_h = self.cfg.ptz_img_h if self.cfg.ptz_enabled else 1080

            # 2. Try YOLO first (fast, if model is trained)
            yolo_plants = []
            if self.yolo.model is not None:
                try:
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                        tmp.write(image)
                        tmp_path = tmp.name
                    yolo_plants = self.yolo.detect(tmp_path)
                    Path(tmp_path).unlink(missing_ok=True)
                    session.yolo_detections = len(yolo_plants)
                    log.info("YOLO detected %d plants", len(yolo_plants))
                except Exception as e:
                    log.warning("YOLO detect failed: %s", e)

            # 3. VLM grounding detection (always — for pseudo-label generation)
            vlm_plants = await self.vlm.grounding_detect(image, img_w, img_h)
            log.info("VLM detected %d plants", len(vlm_plants))

            # 4. Filter and collect for YOLO training
            labels = self._filter_labels(vlm_plants)
            if labels:
                split = self._pick_split()
                fname = f"patrol_{session.session_id}_{int(time.time())}.jpg"
                self.yolo.collect(image, fname, labels, split=split)
                session.images_collected += 1
                log.info("Collected %s → %s (%d labels)", fname, split, len(labels))

            # 5. Build plant records (merge VLM + YOLO results)
            for p in vlm_plants:
                record = PlantRecord(
                    id=str(uuid.uuid4())[:8],
                    type=p["type"],
                    health=p["health"],
                    bbox=p["bbox"],
                    details=p.get("description", ""),
                    timestamp=datetime.now().isoformat(),
                )
                session.plants.append(record)

            # 6. Auto-train check
            if self.yolo.should_train():
                log.info("Dataset threshold reached (%d images), triggering YOLO training",
                         self.yolo.dataset_size())
                asyncio.create_task(self._train_background())

            session.status = "completed"

        except Exception as e:
            log.error("Patrol error: %s", e)
            session.status = "error"
            session.errors.append(str(e))

        # Save to history
        self.history.append(session.to_dict())
        if len(self.history) > 100:
            self.history = self.history[-100:]

        self._running = False
        return session

    async def _train_background(self):
        """Run YOLO training in background thread."""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self.yolo.train)
        log.info("Background training result: %s", result)

    async def run_continuous(self, sensor_data: dict = None):
        """Run patrol in a loop at configured interval."""
        self._running = True
        log.info("Continuous patrol started (every %d min)", self.cfg.patrol_interval)
        while self._running:
            await self.run_once(sensor_data)
            await asyncio.sleep(self.cfg.patrol_interval * 60)

    def stop(self):
        self._running = False

    def get_status(self) -> dict:
        return {
            "running": self._running,
            "total_collected": self._collect_count,
            "session": self.session.to_dict() if self.session else None,
            "yolo": self.yolo.status(),
        }
