"""Inference helpers for YOLO ONNX (best.onnx)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

CLASS_NAMES = ["helmet", "licenseplate", "motorcyclist", "nohelmet"]
ALERT_CLASSES = {"nohelmet"}

ALERT_MESSAGES = {
    "nohelmet": "Phát hiện người đi xe KHÔNG đội mũ bảo hiểm!",
}


def load_model(model_path: str | Path) -> YOLO:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy model: {path.resolve()}\n"
            "Đặt file best.onnx vào thư mục dự án hoặc export từ best.pt:\n"
            '  from ultralytics import YOLO\n'
            '  YOLO("best.pt").export(format="onnx", imgsz=640)'
        )
    return YOLO(str(path), task="detect")


def predict_frame(
    model: YOLO,
    frame_bgr: np.ndarray,
    conf: float = 0.5,
) -> tuple[np.ndarray, list[dict[str, Any]], Counter]:
    """Run detection on one BGR frame. Returns annotated frame, detections, counts."""
    results = model(frame_bgr, conf=conf, verbose=False)[0]
    names = results.names or dict(enumerate(CLASS_NAMES))

    detections: list[dict[str, Any]] = []
    counts: Counter = Counter()

    if results.boxes is not None and len(results.boxes):
        for box in results.boxes:
            cls_id = int(box.cls.item())
            label = names.get(cls_id, CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id))
            score = float(box.conf.item())
            xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
            detections.append({"label": label, "conf": score, "box": xyxy})
            counts[label] += 1

    annotated = results.plot()
    return annotated, detections, counts


def get_alerts(counts: Counter) -> list[str]:
    """Return alert messages for dangerous classes found."""
    alerts = []
    for cls in ALERT_CLASSES:
        if counts.get(cls, 0) > 0:
            alerts.append(ALERT_MESSAGES.get(cls, f"Cảnh báo: phát hiện {cls}"))
    return alerts


def has_alert(counts: Counter) -> bool:
    return any(counts.get(c, 0) > 0 for c in ALERT_CLASSES)
