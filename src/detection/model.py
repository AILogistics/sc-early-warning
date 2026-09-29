"""Load AMFDet-Lite (YOLOv8n + LAE + SPPFSENetV2 + EUCB).

Requires an Ultralytics YOLOv8 installation patched with the custom modules in
third_party/ultralytics_modules/ (registered in ultralytics/nn). Resolution order:
  1. $SC_ULTRALYTICS_ROOT environment variable
  2. an `ultralytics/` directory next to this repository (original project layout)
  3. the pip-installed ultralytics (if you patched it in place)
Weights resolution: model/best.pt inside this repository takes precedence.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_cand = []
if os.environ.get("SC_ULTRALYTICS_ROOT"):
    _cand.append(Path(os.environ["SC_ULTRALYTICS_ROOT"]))
_cand.append(REPO_ROOT.parent / "ultralytics")   # original project layout
MAIN_PROJ = next((c for c in _cand if (c / "ultralytics" / "nn").exists() or c.name == "ultralytics"), None)
MAIN_PROJ_STR = str(MAIN_PROJ) if MAIN_PROJ else ""

if MAIN_PROJ and MAIN_PROJ_STR not in sys.path:
    sys.path.insert(0, MAIN_PROJ_STR)
for _mod_name in list(sys.modules):
    if _mod_name == "ultralytics" or _mod_name.startswith("ultralytics."):
        del sys.modules[_mod_name]

from ultralytics import YOLO  # noqa: E402


AMFDET_WEIGHTS_CANDIDATES = [
    REPO_ROOT / "model" / "best.pt",
]


def find_amfdet_weights() -> Path:
    """返回第一个存在的 AMFDet 权重路径。"""
    for p in AMFDET_WEIGHTS_CANDIDATES:
        if p.exists():
            return p
    # 兜底:扫描 runs/detect/ 下任意 best.pt
    runs = MAIN_PROJ / "runs" / "detect"
    if runs.exists():
        for d in sorted(runs.iterdir(), reverse=True):
            w = d / "weights" / "best.pt"
            if w.exists():
                return w
    raise FileNotFoundError(f"No AMFDet weights found under {runs}")


def load_detector(weights_path: Path | None = None) -> YOLO:
    """加载 YOLOv8 检测器。"""
    if weights_path is None:
        weights_path = find_amfdet_weights()
    print(f"[detector] Loading {weights_path}", file=sys.stderr)
    return YOLO(str(weights_path))


if __name__ == "__main__":
    model = load_detector()
    print(f"[OK] Model loaded successfully")
    # 简单 smoke:打印模型参数量
    n_params = sum(p.numel() for p in model.model.parameters())
    print(f"[OK] Parameters: {n_params:,}")
