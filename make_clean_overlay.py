"""Regenerate a CLEAN detection overlay (no baked text) for the detection-zoom figure.

Background: paper/model_artifacts/input_sar_gray.png (640x640, no text).
Boxes: experiments/exp3_activity/detections/rotterdam_detections.json,
       frame E1_suez_2021_20210319.tif (n=300, boxes already in 640-space).
Draw ONLY rectangles (colour-coded by size); no scores, no title, no legend.
Output overwrites paper/model_artifacts/detection_overlay.png.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

PROJ = Path(__file__).resolve().parent
ART = PROJ / "model"

dets = json.load(open(PROJ / "experiments/exp3_activity/detections/rotterdam_detections.json"))
frame = next(e for e in dets if "20210319" in e["file"])
boxes = np.array(frame["boxes_xyxy"], dtype=float)
n_total = len(boxes)
sizes = np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
n_small = int((sizes < 32).sum())
print(f"frame {frame['file']}: n={n_total}, small<32px={n_small}")

bg = np.array(Image.open(ART / "input_sar_gray.png").convert("L"))
fig, ax = plt.subplots(figsize=(7.2, 7.2))
ax.imshow(bg, cmap="gray")
for b in boxes:
    c = "#D55E00" if max(b[2] - b[0], b[3] - b[1]) < 32 else "#56B4E9"
    ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                           edgecolor=c, facecolor="none", lw=0.8))
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
fig.savefig(str(ART / "detection_overlay.png"), dpi=120)
plt.close(fig)
print("clean overlay saved")
