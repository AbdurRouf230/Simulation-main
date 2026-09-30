#!/bin/bash
set -e
python3 -m pip install --user torch torchvision opencv-python-headless
python3 - <<'PY'
import sys
from pathlib import Path
root = Path("/mnt/e/Multi Drone/Merge folder/intelleswarm-ai")
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "genai_framework" / "px4_gazebo_sim" / "multi_drone_script"))
from flower_prediction_json import FlowerDetector, get_detector, detect_rgb
import numpy as np
print("FlowerDetector class:", FlowerDetector)
m = get_detector()
print("model:", type(m))
img = np.zeros((480, 640, 3), dtype=np.uint8)
img[:] = (20, 180, 40)
dets = detect_rgb(img)
print("dummy detect_flowers hits:", len(dets), "ok")
PY
