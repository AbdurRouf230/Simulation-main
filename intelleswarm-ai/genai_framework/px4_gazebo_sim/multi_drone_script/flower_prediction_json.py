"""One JSON file of FlowerDetector results for every captured mission image.

Layout:
{
  "drone0": {
    "frame_1": {"image": "/path/to.png", "prediction": [...]},
    "frame_2": {"image": "...", "prediction": [...]}
  },
  "drone1": { ... }
}
"""

from __future__ import annotations

import importlib.util
import json
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

_MULTI = Path(__file__).resolve().parent
_SIM = _MULTI.parent
_REPO = _SIM.parent.parent
JSON_PATH = _SIM / "flower_predictions.json"

_lock = threading.Lock()
_infer_lock = threading.Lock()
_detector = None
_detector_error: Optional[str] = None
_data: dict[str, Any] = {}


def _load_flower_detector_class():
    """Load FlowerDetector without executing assistive_pollination/__init__.py."""
    if str(_REPO) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(_REPO))
    path = _REPO / "assistive_pollination" / "models" / "flower_detector.py"
    spec = importlib.util.spec_from_file_location("merge_flower_detector", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.FlowerDetector


try:
    FlowerDetector = _load_flower_detector_class()
except Exception as exc:
    print(f"Warning: FlowerDetector not available: {exc}")
    FlowerDetector = None


def reset_prediction_log() -> Path:
    global _data
    with _lock:
        _data = {}
        JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        JSON_PATH.write_text("{}\n", encoding="utf-8")
    return JSON_PATH


def _write_unlocked() -> None:
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = JSON_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(_data, indent=2), encoding="utf-8")
    tmp.replace(JSON_PATH)


def get_detector():
    """Build the untrained FlowerDetector once (CPU). Returns None if import failed."""
    global _detector, _detector_error
    with _lock:
        if _detector is not None:
            return _detector
        if FlowerDetector is None:
            _detector_error = _detector_error or "FlowerDetector class not imported (install torch + opencv)"
            return None
        if _detector_error:
            return None
        try:
            model = FlowerDetector(img_size=(256, 256))
            model.eval()
            _detector = model
            return _detector
        except Exception as exc:
            _detector_error = str(exc)
            print(f"FlowerDetector construct failed: {exc}")
            return None


def detect_rgb(rgb: np.ndarray):
    """Thread-safe FlowerDetector.detect_flowers. Raises if model is unavailable."""
    detector = get_detector()
    if detector is None:
        raise RuntimeError(_detector_error or "FlowerDetector unavailable")
    with _infer_lock:
        return detector.detect_flowers(rgb)


def _detection_to_dict(det) -> dict[str, Any]:
    status = getattr(det, "pollination_status", None)
    status_val = status.value if hasattr(status, "value") else str(status)
    bbox = getattr(det, "bbox", None)
    return {
        "species": getattr(det, "species", "unknown"),
        "confidence": float(getattr(det, "confidence", 0.0)),
        "bbox": [int(x) for x in bbox] if bbox is not None else [],
        "pollination_status": status_val,
        "center_3d": list(det.center_3d) if getattr(det, "center_3d", None) else None,
        "pollen_source_quality": float(getattr(det, "pollen_source_quality", 0.0)),
        "pollen_need_urgency": float(getattr(det, "pollen_need_urgency", 0.0)),
    }


def _load_rgb(image_path: str) -> Optional[np.ndarray]:
    try:
        import cv2
        bgr = cv2.imread(image_path)
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        try:
            from PIL import Image
            return np.array(Image.open(image_path).convert("RGB"))
        except Exception:
            return None


def record_captured_image(drone_name: str, image_path: Optional[str]) -> None:
    """Run detect_flowers on a saved PNG and append frame_N under drone_name."""
    if not image_path:
        return
    started = time.time()
    frame: dict[str, Any] = {
        "image": str(image_path),
        "prediction": [],
        "model_ran": False,
    }
    detector = get_detector()
    if detector is None:
        frame["error"] = _detector_error or "FlowerDetector unavailable"
    else:
        rgb = _load_rgb(image_path)
        if rgb is None:
            frame["error"] = "could not read image"
        else:
            try:
                detections = detect_rgb(rgb)
                frame["prediction"] = [_detection_to_dict(d) for d in detections]
                frame["model_ran"] = True
                frame["image_size"] = [int(rgb.shape[1]), int(rgb.shape[0])]
            except Exception as exc:
                frame["error"] = str(exc)
    frame["inference_ms"] = round((time.time() - started) * 1000.0, 1)

    with _lock:
        drone = _data.setdefault(drone_name, {})
        key = f"frame_{len(drone) + 1}"
        drone[key] = frame
        _write_unlocked()
    print(f"[predict] {drone_name} {key} -> {JSON_PATH.name}  ran={frame['model_ran']} hits={len(frame['prediction'])}")
