"""加载 AMFDet-Lite(YOLOv8n + LAE + SPPFSENetV2 + EUCB)。
复用主项目 ultralytics/ 框架(含 Addmodules 自定义模块注册)。

关键:必须在 `from ultralytics import YOLO` 之前把父项目根插入 sys.path[0],
否则 pip 安装的 ultralytics 会被优先加载,缺少 Addmodules 模块导致权重加载失败。

注意:parents[3] 指向父项目根 D:/Article/YOLOv8-SAR-Information/
(parents[2] 是 sc_early_warning/ 本身,不可用)
"""
from __future__ import annotations
import sys
from pathlib import Path

# 复用主项目的 ultralytics(含 Addmodules 自定义模块)
# model.py 路径: sc_early_warning/src/detection/model.py
#   parents[0]=src/detection/, [1]=src/, [2]=sc_early_warning/, [3]=父项目根
MAIN_PROJ = Path(__file__).resolve().parents[3]
MAIN_PROJ_STR = str(MAIN_PROJ)

# 必须在 import ultralytics 之前修改 sys.path
if MAIN_PROJ_STR not in sys.path:
    sys.path.insert(0, MAIN_PROJ_STR)
# 防御:若 pip 版 ultralytics 已被加载(例如父模块 import),强制清理
for _mod_name in list(sys.modules):
    if _mod_name == "ultralytics" or _mod_name.startswith("ultralytics."):
        del sys.modules[_mod_name]

from ultralytics import YOLO  # noqa: E402  (必须在 sys.path 修改后)


# AMFDet 权重候选路径(按训练完整性排序)
# 重要:improved_model_LAE_SPPFSENetV2_EUCB_full 只训练了 7 epoch(mAP50=0.69),
# 不是完整模型!正确选择是 ablation_Exp7_LAE_SPPF_EUCB_Full(357 epoch, mAP50=0.911)
AMFDET_WEIGHTS_CANDIDATES = [
    MAIN_PROJ / "runs" / "detect" / "ablation_Exp7_LAE_SPPF_EUCB_Full" / "weights" / "best.pt",
    MAIN_PROJ / "runs" / "ablation_final" / "baseline" / "weights" / "best.pt",
    # 备选(已确认未充分训练,仅作兜底):
    MAIN_PROJ / "runs" / "detect" / "improved_model_LAE_SPPFSENetV2_EUCB_full" / "weights" / "best.pt",
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
