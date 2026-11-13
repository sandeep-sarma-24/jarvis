"""YOLOE open-vocabulary detector + ByteTrack, run synchronously (callers use a worker thread).

ultralytics is an optional extra (AGPL-3.0). If it or the weights are missing, load() logs a
warning and returns False; nothing else in JARVIS depends on it.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

from app.config import DetectorConfig
from app.perception.types import Entity, Frame

logger = logging.getLogger(__name__)
WEIGHTS_DIR = Path("~/.cache/jarvis-models/w").expanduser()


class Detector:
    def __init__(self, cfg: DetectorConfig) -> None:
        self._cfg = cfg
        self._model = None
        self._device = cfg.device
        self.loaded = False

    def load(self) -> bool:
        """Load weights, bake the class prompts, warm up. Blocking; never raises."""
        try:
            os.environ.setdefault("YOLO_OFFLINE", "1")  # local-only: no hub checks or downloads
            from ultralytics import YOLO, settings
            from ultralytics.utils import WEIGHTS_DIR as ULTRA_DIR

            settings.update({"sync": False})  # ultralytics ships anonymous analytics on by default

            # The text encoder is otherwise fetched into the cwd; reuse the cached copy offline.
            for f in WEIGHTS_DIR.glob("mobileclip*.ts"):
                if not (ULTRA_DIR / f.name).exists():
                    ULTRA_DIR.mkdir(parents=True, exist_ok=True)
                    (ULTRA_DIR / f.name).symlink_to(f)
            name = self._cfg.model
            path = Path(name).expanduser()
            if not path.exists() and (WEIGHTS_DIR / name).exists():
                path = WEIGHTS_DIR / name
            model = YOLO(str(path if path.exists() else name))
            if self._cfg.classes:
                model.set_classes(self._cfg.classes, model.get_text_pe(self._cfg.classes))
            self._model = model
            blank = np.zeros((self._cfg.imgsz, self._cfg.imgsz, 3), np.uint8)
            try:
                self._run(blank)
            except Exception:
                if self._device == "cpu":
                    raise
                logger.warning("Detector: %s failed, falling back to cpu", self._device, exc_info=True)
                self._device = "cpu"
                self._run(blank)
            self.loaded = True
            logger.info("Detector ready — model=%s device=%s classes=%d", name, self._device, len(self._cfg.classes))
        except Exception as exc:
            logger.warning("Detector disabled: %r", exc)
            self._model = None
            self.loaded = False
        return self.loaded

    def _run(self, bgr: np.ndarray):
        return self._model.track(bgr, persist=True, conf=self._cfg.conf, imgsz=self._cfg.imgsz,
                                 device=self._device, verbose=False)[0]

    def detect(self, frame: Frame) -> list[Entity]:
        """Tracked entities in one rgb24 frame. Detections without a track id yet are skipped."""
        rgb = np.frombuffer(frame.data, np.uint8).reshape(frame.height, frame.width, 3)
        res = self._run(np.ascontiguousarray(rgb[:, :, ::-1]))
        boxes = res.boxes
        if boxes is None or boxes.id is None:
            return []
        ids, cls, conf, xyxy = boxes.id.int().tolist(), boxes.cls.int().tolist(), boxes.conf.tolist(), boxes.xyxyn.tolist()
        return [Entity(i, res.names[c], round(p, 3), tuple(round(v, 4) for v in b))
                for i, c, p, b in zip(ids, cls, conf, xyxy)]
