"""双源融合检测器:光学主导 + SAR 补盲。

Task 2.3 / Exp 2: Multi-source satellite fusion detection.

选项 α(已选):SAR-trained AMFDet + 灰度伪 RGB 跨模态迁移
  - Sentinel-2 灰度港口切片 → 复制 3 通道 → 喂给 SAR-trained AMFDet
  - 不训练新模型,零额外 GPU 成本
  - 论文表述:"cross-modal transfer via grayscale pseudo-RGB channel replication"
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from ultralytics import YOLO  # noqa: E402 (model.py 处理 sys.path)

from .model import load_detector, find_amfdet_weights


class DualSourceDetector:
    """双源融合检测器。
    
    决策逻辑:
      1. 光学可用(云覆盖 < 30%):
         - 灰度 → 复制 3 通道 → SAR-trained AMFDet 推理
         - 标记 source='optical', confidence='high'
      2. SAR 可用:
         - 直接 SAR-trained AMFDet 推理(原始域)
         - 标记 source='sar', confidence='medium'
      3. 都不可用:
         - 返回空检测,confidence='low'
    
    注:选项 α 下,光学和 SAR 共用同一模型(跨模态迁移),
       只差"输入图像来源"和置信度标记。
    """
    
    def __init__(self, weights_path: Optional[Path] = None, conf: float = 0.25,
                 iou: float = 0.45, verbose: bool = False):
        """初始化检测器。
        
        Args:
            weights_path: AMFDet 权重路径(None 自动发现)
            conf: 置信度阈值
            iou: NMS IoU 阈值(港口密集场景用 0.35 更好)
            verbose: 推理时是否打印
        """
        self.model = load_detector(weights_path)
        self.conf = conf
        self.iou = iou
        self.verbose = verbose
    
    def _grayscale_to_pseudo_rgb(self, gray_img: np.ndarray) -> np.ndarray:
        """单通道灰度 → 3 通道伪 RGB(复制通道)。
        
        选项 α 核心操作:让 SAR-trained 模型处理光学输入。
        Sentinel-2 单波段(R/G/B/NIR 任选)灰度 → 复制 3 次 → pseudo-RGB。
        """
        if gray_img.ndim == 3 and gray_img.shape[2] == 3:
            return gray_img  # 已是 RGB,直接返回
        if gray_img.ndim == 2:
            return np.stack([gray_img] * 3, axis=-1)
        # 多波段取第一个或平均
        if gray_img.ndim == 3 and gray_img.shape[2] > 3:
            return gray_img[:, :, :3]
        return np.stack([gray_img[..., 0]] * 3, axis=-1)
    
    def _normalize_to_uint8(self, img: np.ndarray) -> np.ndarray:
        """任意数值范围归一化到 0-255 uint8(YOLO 输入要求)。"""
        if img.dtype == np.uint8:
            return img
        vmin, vmax = float(img.min()), float(img.max())
        if vmax - vmin < 1e-9:
            return np.zeros_like(img, dtype=np.uint8)
        return ((img - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    
    def detect(self, optical_img: Optional[np.ndarray] = None,
               sar_img: Optional[np.ndarray] = None,
               cloud_cover: float = 0.0) -> dict:
        """对单帧影像做双源融合检测。
        
        Args:
            optical_img: 光学影像(HxWxC 或 HxW,None 表示不可用)
            sar_img: SAR 影像(HxWxC 或 HxW,None 表示不可用)
            cloud_cover: 0-1,光学云覆盖率
        
        Returns:
            {boxes: [[x1,y1,x2,y2],...], source: str, confidence: str,
             n_ships: int, scores: [...]}
        """
        # 决策:光学优先(若云少),否则 SAR
        use_optical = (optical_img is not None) and (cloud_cover < 0.30)
        use_sar = sar_img is not None
        
        if use_optical:
            # 选项 α:灰度伪 RGB + SAR-trained 模型
            rgb = self._grayscale_to_pseudo_rgb(optical_img)
            rgb = self._normalize_to_uint8(rgb)
            source, confidence = "optical", "high"
        elif use_sar:
            rgb = self._grayscale_to_pseudo_rgb(sar_img)  # SAR 也可能多波段
            rgb = self._normalize_to_uint8(rgb)
            source, confidence = "sar", "medium"
        else:
            return {"boxes": [], "scores": [], "source": "interpolated",
                    "confidence": "low", "n_ships": 0}
        
        # YOLO 推理
        results = self.model(rgb, conf=self.conf, iou=self.iou, verbose=self.verbose)
        r = results[0]
        
        if r.boxes is None or len(r.boxes) == 0:
            return {"boxes": [], "scores": [], "source": source,
                    "confidence": confidence, "n_ships": 0}
        
        boxes = r.boxes.xyxy.cpu().numpy().tolist()
        scores = r.boxes.conf.cpu().numpy().tolist()
        return {
            "boxes": boxes,
            "scores": scores,
            "source": source,
            "confidence": confidence,
            "n_ships": len(boxes),
        }
    
    def detect_batch(self, frames: list[dict]) -> list[dict]:
        """批量检测多帧。
        
        Args:
            frames: [{optical_img, sar_img, cloud_cover, ...}, ...]
        
        Returns:
            检测结果列表(每帧一个 dict,继承原 frame 的元数据)
        """
        results = []
        for frame in frames:
            det = self.detect(
                optical_img=frame.get("optical_img"),
                sar_img=frame.get("sar_img"),
                cloud_cover=frame.get("cloud_cover", 0.0),
            )
            det.update({k: v for k, v in frame.items()
                        if k not in ("optical_img", "sar_img")})
            results.append(det)
        return results


if __name__ == "__main__":
    # Smoke test: 加载检测器,用合成数据验证管线
    print("Initializing DualSourceDetector...", file=sys.stderr)
    detector = DualSourceDetector(iou=0.35)  # 港口密集场景降低 IoU 阈值
    print("[OK] Detector ready", file=sys.stderr)
    
    # 合成测试:模拟光学(灰度)和 SAR 输入
    np.random.seed(42)
    fake_optical = np.random.rand(256, 256, 3) * 200 + 30  # 模拟光学反射率
    fake_sar = np.random.rand(256, 256) * (-5) - 10  # 模拟 SAR dB
    
    print("\n=== Test 1: Optical input (cloud_cover=0.1) ===", file=sys.stderr)
    r1 = detector.detect(optical_img=fake_optical, sar_img=fake_sar, cloud_cover=0.1)
    print(f"  source={r1['source']}, confidence={r1['confidence']}, n_ships={r1['n_ships']}",
          file=sys.stderr)
    
    print("\n=== Test 2: SAR fallback (cloud_cover=0.5) ===", file=sys.stderr)
    r2 = detector.detect(optical_img=fake_optical, sar_img=fake_sar, cloud_cover=0.5)
    print(f"  source={r2['source']}, confidence={r2['confidence']}, n_ships={r2['n_ships']}",
          file=sys.stderr)
    
    print("\n=== Test 3: Both unavailable ===", file=sys.stderr)
    r3 = detector.detect(optical_img=None, sar_img=None, cloud_cover=0.8)
    print(f"  source={r3['source']}, confidence={r3['confidence']}, n_ships={r3['n_ships']}",
          file=sys.stderr)
    
    print("\n[DONE] DualSourceDetector smoke test passed", file=sys.stderr)
