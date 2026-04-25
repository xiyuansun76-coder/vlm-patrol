"""YOLO detection and training — find in environment or auto-install."""

import logging
import shutil
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)

_yolo_module = None


def _ensure_ultralytics():
    """Find ultralytics in environment, or install it."""
    global _yolo_module
    if _yolo_module is not None:
        return _yolo_module

    try:
        import ultralytics
        _yolo_module = ultralytics
        log.info("Found ultralytics %s", ultralytics.__version__)
        return _yolo_module
    except ImportError:
        pass

    log.info("ultralytics not found, installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics", "-q"])
    import ultralytics
    _yolo_module = ultralytics
    log.info("Installed ultralytics %s", ultralytics.__version__)
    return _yolo_module


class YOLOManager:
    """Manages YOLO detection, data collection, and training."""

    def __init__(self, config):
        self.cfg = config
        self.data_dir = Path(config.yolo_data_dir)
        self.images_dir = self.data_dir / "images"
        self.labels_dir = self.data_dir / "labels"
        self.model = None
        self._training = False

    def _ensure_dirs(self):
        for d in [self.images_dir / "train", self.images_dir / "val",
                   self.labels_dir / "train", self.labels_dir / "val"]:
            d.mkdir(parents=True, exist_ok=True)

    def _load_model(self):
        if self.model is not None:
            return
        ul = _ensure_ultralytics()
        model_path = self.cfg.yolo_model_path
        if model_path and Path(model_path).exists():
            log.info("Loading YOLO from %s", model_path)
            self.model = ul.YOLO(model_path)
        else:
            log.info("Loading default yolo11s.pt")
            self.model = ul.YOLO("yolo11s.pt")

    def detect(self, image_path: str | Path, conf: float = 0.25) -> list[dict]:
        """Run detection on an image. Returns list of detections."""
        self._load_model()
        results = self.model(str(image_path), conf=conf, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = r.names.get(cls_id, str(cls_id))
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "class": cls_name,
                    "class_id": cls_id,
                    "confidence": float(box.conf[0]),
                    "bbox": [x1, y1, x2, y2],
                })
        return detections

    def collect(self, image_bytes: bytes, filename: str,
                labels: list[dict], split: str = "train") -> Path:
        """
        Save image + YOLO labels for training.
        labels: [{"class_id": 0, "bbox": [x1,y1,x2,y2]}] (pixel coords)
        Returns saved image path.
        """
        self._ensure_dirs()
        img_path = self.images_dir / split / filename
        img_path.write_bytes(image_bytes)

        # convert pixel bbox to YOLO normalized format
        import cv2
        import numpy as np
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return img_path
        h, w = img.shape[:2]

        label_name = Path(filename).stem + ".txt"
        label_path = self.labels_dir / split / label_name
        lines = []
        for lbl in labels:
            cls_id = lbl["class_id"]
            x1, y1, x2, y2 = lbl["bbox"]
            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        label_path.write_text("\n".join(lines))

        log.info("Collected %s with %d labels", filename, len(labels))
        return img_path

    def dataset_size(self) -> int:
        """Count training images."""
        self._ensure_dirs()
        train_dir = self.images_dir / "train"
        return len(list(train_dir.glob("*.jpg")) + list(train_dir.glob("*.png")))

    def should_train(self) -> bool:
        """Check if we have enough data and auto-train is enabled."""
        return (self.cfg.yolo_auto_train and
                self.dataset_size() >= self.cfg.yolo_train_threshold and
                not self._training)

    def train(self, epochs: int = 100) -> dict:
        """Train YOLO on collected data. Returns training results."""
        if self._training:
            return {"status": "already_training"}

        self._load_model()
        self._ensure_dirs()

        # write dataset.yaml
        dataset_yaml = self.data_dir / "dataset.yaml"
        dataset_yaml.write_text(
            f"path: {self.data_dir.resolve()}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"nc: {len(self.cfg.classes)}\n"
            f"names: {self.cfg.classes}\n"
        )

        self._training = True
        try:
            results = self.model.train(
                data=str(dataset_yaml),
                epochs=epochs,
                imgsz=640,
                batch=-1,
                project=str(self.data_dir / "runs"),
                name="train",
                exist_ok=True,
            )
            # update model to best weights
            best = self.data_dir / "runs" / "train" / "weights" / "best.pt"
            if best.exists():
                self.cfg.yolo_model_path = str(best)
                ul = _ensure_ultralytics()
                self.model = ul.YOLO(str(best))
                log.info("Model updated to %s", best)
            return {"status": "completed", "best_weights": str(best)}
        except Exception as e:
            log.error("Training failed: %s", e)
            return {"status": "error", "error": str(e)}
        finally:
            self._training = False

    def status(self) -> dict:
        return {
            "dataset_size": self.dataset_size(),
            "training": self._training,
            "model_path": self.cfg.yolo_model_path or "yolo11s.pt (default)",
            "auto_train": self.cfg.yolo_auto_train,
            "train_threshold": self.cfg.yolo_train_threshold,
        }
