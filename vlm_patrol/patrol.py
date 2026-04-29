"""Patrol core — VLM-driven active perception:
  - vlm_active: panorama → VLM grounding → PTZ focus each plant → close-up diagnose
  - annotation: use pre-annotated plant positions → PTZ focus → diagnose
"""

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

MIN_CONFIDENCE = 0.5
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
    ptz_az: int = 0
    ptz_el: int = 0

    def to_dict(self):
        return {
            "id": self.id, "type": self.type, "health": self.health,
            "confidence": self.confidence, "bbox": self.bbox,
            "details": self.details, "timestamp": self.timestamp,
            "ptz_az": self.ptz_az, "ptz_el": self.ptz_el,
        }


@dataclass
class PatrolSession:
    session_id: str = ""
    status: str = "idle"
    strategy: str = "vlm_active"
    started_at: str = ""
    plants: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    images_collected: int = 0
    yolo_detections: int = 0
    ptz_travel: float = 0.0

    def to_dict(self):
        return {
            "session_id": self.session_id, "status": self.status,
            "strategy": self.strategy, "started_at": self.started_at,
            "plants": [p.to_dict() for p in self.plants],
            "errors": self.errors, "images_collected": self.images_collected,
            "yolo_detections": self.yolo_detections, "ptz_travel": round(self.ptz_travel, 1),
        }


class Patrol:
    """Patrol controller: VLM active perception + annotation-based."""

    def __init__(self, config, vlm, yolo_mgr, ptz=None):
        self.cfg = config
        self.vlm = vlm
        self.yolo = yolo_mgr
        self.ptz = ptz  # None if no PTZ camera
        self.http = httpx.AsyncClient(timeout=15)
        self.session: PatrolSession | None = None
        self._running = False
        self._collect_count = 0
        self.history: list[dict] = []

    async def snapshot(self) -> bytes | None:
        """Grab image — via PTZ ISAPI if available, otherwise HTTP GET."""
        if self.ptz:
            img = await self.ptz.snapshot()
            if img:
                return img
        url = self.cfg.camera_snapshot_url
        if not url:
            log.warning("No camera source configured")
            return None
        try:
            resp = await self.http.get(url)
            resp.raise_for_status()
            return resp.content
        except Exception as e:
            log.error("Snapshot failed: %s", e)
            return None

    def _pick_split(self) -> str:
        self._collect_count += 1
        return "val" if random.random() < VAL_RATIO else "train"

    def _filter_labels(self, plants: list[dict]) -> list[dict]:
        labels = []
        for p in plants:
            cls_name = p["type"]
            if cls_name not in self.cfg.classes:
                continue
            bbox = p["bbox"]
            if (bbox[2] - bbox[0]) < 10 or (bbox[3] - bbox[1]) < 10:
                continue
            labels.append({
                "class_id": self.cfg.classes.index(cls_name),
                "bbox": bbox,
            })
        return labels

    def _collect_image(self, image: bytes, plants: list[dict], session: PatrolSession):
        """Filter pseudo-labels and save for YOLO training."""
        labels = self._filter_labels(plants)
        if labels:
            split = self._pick_split()
            fname = f"patrol_{session.session_id}_{int(time.time())}_{random.randint(0,999):03d}.jpg"
            self.yolo.collect(image, fname, labels, split=split)
            session.images_collected += 1
            log.info("Collected %s → %s (%d labels)", fname, split, len(labels))

    # ── Strategy: vlm_active (panorama → focus → diagnose) ──

    async def _run_vlm_active(self, session: PatrolSession, sensor_data: dict = None):
        """VLM-Active: panorama → grounding → PTZ focus each plant → close-up diagnose."""
        if not self.ptz:
            log.warning("VLM-Active requires PTZ, skipping")
            session.errors.append("PTZ not configured")
            return

        self.ptz.reset_travel()
        img_w, img_h = self.cfg.ptz_img_w, self.cfg.ptz_img_h

        # 1. Go home for wide-angle panorama
        await self.ptz.go_home()
        ref_az, ref_el, _ = await self.ptz.get_position()
        panorama = await self.ptz.snapshot()
        if not panorama:
            session.errors.append("Panorama snapshot failed")
            return

        # 2. VLM grounding on panorama
        plants = await self.vlm.grounding_detect(panorama, img_w, img_h)
        log.info("VLM-Active: panorama detected %d plants", len(plants))

        # Collect panorama pseudo-labels
        self._collect_image(panorama, plants, session)

        # 3. Focus on each plant → close-up → diagnose
        for i, p in enumerate(plants):
            if not self._running:
                break

            bbox = p["bbox"]
            cls_name = p["type"]
            log.info("Focusing on plant %d/%d: %s at bbox %s", i + 1, len(plants), cls_name, bbox)

            # Move PTZ to plant
            await self.ptz.focus_on_bbox(bbox, ref_az, ref_el)
            closeup = await self.ptz.snapshot()
            if not closeup:
                session.errors.append(f"Close-up snapshot failed for plant {i}")
                continue

            # VLM diagnosis on close-up
            diag = await self.vlm.diagnose(closeup, sensor_data)

            # Collect close-up image with VLM label
            if diag.get("bbox") and diag.get("type") in self.cfg.classes:
                self._collect_image(closeup, [{
                    "type": diag["type"],
                    "bbox": diag["bbox"],
                }], session)

            cur_az, cur_el, _ = await self.ptz.get_position()
            session.plants.append(PlantRecord(
                id=str(uuid.uuid4())[:8],
                type=diag.get("type", cls_name),
                health=diag.get("health", p["health"]),
                confidence=diag.get("confidence", 0),
                bbox=bbox,
                details=diag.get("details", ""),
                timestamp=datetime.now().isoformat(),
                ptz_az=cur_az, ptz_el=cur_el,
            ))

        await self.ptz.go_home()
        session.ptz_travel = self.ptz.total_travel

    # ── Strategy: annotation-based (use pre-annotated plants) ──

    async def _run_from_annotation(self, session: PatrolSession,
                                    plants: list[dict],
                                    panorama: bytes | None = None,
                                    sensor_data: dict = None):
        """Run patrol using pre-annotated plant positions from panoramic annotation."""
        if not self.ptz:
            log.warning("Annotation-based patrol requires PTZ")
            session.errors.append("PTZ not configured")
            return

        self.ptz.reset_travel()
        img_w, img_h = self.cfg.ptz_img_w, self.cfg.ptz_img_h

        # Go home to get reference position
        await self.ptz.go_home()
        ref_az, ref_el, _ = await self.ptz.get_position()

        # Collect panorama pseudo-labels if image provided
        if panorama:
            self._collect_image(panorama, plants, session)

        log.info("Annotation patrol: %d plants to inspect", len(plants))

        # Focus on each annotated plant → close-up → diagnose
        for i, p in enumerate(plants):
            if not self._running:
                break

            bbox = p.get("bbox", [])
            cls_name = p.get("type", "unknown")
            if not bbox or len(bbox) < 4:
                continue

            log.info("Focusing on plant %d/%d: %s at bbox %s", i + 1, len(plants), cls_name, bbox)

            # Move PTZ to plant
            await self.ptz.focus_on_bbox(bbox, ref_az, ref_el)
            closeup = await self.ptz.snapshot()
            if not closeup:
                session.errors.append(f"Close-up snapshot failed for plant {i}")
                continue

            # VLM diagnosis on close-up
            diag = await self.vlm.diagnose(closeup, sensor_data)

            # Collect close-up image with VLM label
            if diag.get("bbox") and diag.get("type") in self.cfg.classes:
                self._collect_image(closeup, [{
                    "type": diag["type"],
                    "bbox": diag["bbox"],
                }], session)

            cur_az, cur_el, _ = await self.ptz.get_position()
            session.plants.append(PlantRecord(
                id=str(uuid.uuid4())[:8],
                type=diag.get("type", cls_name),
                health=diag.get("health", p.get("health", "unknown")),
                confidence=diag.get("confidence", 0),
                bbox=bbox,
                details=diag.get("details", ""),
                timestamp=datetime.now().isoformat(),
                ptz_az=cur_az, ptz_el=cur_el,
            ))

        await self.ptz.go_home()
        session.ptz_travel = self.ptz.total_travel

    # ── Main entry ──

    async def run_once(self, sensor_data: dict = None,
                       strategy: str = None,
                       annotation: dict = None) -> PatrolSession:
        """Run one patrol cycle with the specified strategy."""
        strategy = strategy or self.cfg.patrol_strategy
        session = PatrolSession(
            session_id=str(uuid.uuid4())[:8],
            status="running",
            strategy=strategy,
            started_at=datetime.now().isoformat(),
        )
        self.session = session
        self._running = True

        try:
            if annotation and annotation.get("plants"):
                await self._run_from_annotation(
                    session, annotation["plants"],
                    panorama=None, sensor_data=sensor_data)
            else:
                await self._run_vlm_active(session, sensor_data)

            # YOLO fast detect for comparison (if model trained)
            if self.yolo.model is not None:
                try:
                    image = await self.snapshot()
                    if image:
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                            tmp.write(image)
                            tmp_path = tmp.name
                        yolo_dets = self.yolo.detect(tmp_path)
                        session.yolo_detections = len(yolo_dets)
                        Path(tmp_path).unlink(missing_ok=True)
                except Exception as e:
                    log.warning("YOLO detect failed: %s", e)

            # Auto-train check
            if self.yolo.should_train():
                log.info("Dataset threshold reached (%d images), triggering YOLO training",
                         self.yolo.dataset_size())
                asyncio.create_task(self._train_background())

            session.status = "completed"

        except Exception as e:
            log.error("Patrol error: %s", e)
            session.status = "error"
            session.errors.append(str(e))

        self.history.append(session.to_dict())
        if len(self.history) > 100:
            self.history = self.history[-100:]

        self._running = False
        return session

    async def _train_background(self):
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self.yolo.train)
        log.info("Background training result: %s", result)

    async def run_continuous(self, sensor_data: dict = None):
        self._running = True
        log.info("Continuous patrol started (every %d min, strategy=%s)",
                 self.cfg.patrol_interval, self.cfg.patrol_strategy)
        while self._running:
            await self.run_once(sensor_data)
            await asyncio.sleep(self.cfg.patrol_interval * 60)

    def stop(self):
        self._running = False
        log.info("Patrol stopped")

    def get_status(self) -> dict:
        return {
            "running": self._running,
            "total_collected": self._collect_count,
            "session": self.session.to_dict() if self.session else None,
            "yolo": self.yolo.status(),
        }
