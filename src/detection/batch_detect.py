"""批量检测脚本(Task 3.1b):把 250m GeoTIFF 港口切片喂给 AMFDet。

方案 A(用户确认):250m 粗分辨率 + 上采样 + 放宽阈值,核心目标是船数趋势(PAI),
不是单船精确边界。论文里诚实写"coarse-resolution aggregate monitoring"。

设计:
  - 输入:250m GeoTIFF(VV+VH 或 B2/B3/B4/B8)
  - 预处理:波段选择 → 归一化 → 复制 3 通道 → 上采样到 640×640
  - 检测:AMFDet,conf=0.15(放宽,因低分辨率信号弱)
  - 输出:JSON(每帧船数 + 框 + 经纬度中心 + 元数据)

注意:GEE getDownloadURL 返回 ZIP(内含 .tif),需先解压。
"""
from __future__ import annotations
import sys
import io
import json
import zipfile
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import rasterio
from rasterio.transform import xy

# model.py 会处理 sys.path + sys.modules,从 detection.model 导入即可
from ..detection.model import load_detector

PROJ_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJ_ROOT / "data" / "processed"
OUTPUT_DIR = PROJ_ROOT / "experiments" / "exp3_activity" / "detections"

# 检测参数(方案 A:250m 粗分辨率适配)
CONF_THRESHOLD = 0.15  # 放宽(低分辨率信号弱)
IOU_THRESHOLD = 0.35   # 港口密集场景
INPUT_SIZE = 640       # AMFDet 训练尺寸


def open_geotiff_maybe_zip(tif_path: Path):
    """打开 GeoTIFF,自动处理 GEE 下载的 ZIP 包装格式。

    GEE getDownloadURL 返回 ZIP(含 .tif + 元数据)。文件头 'PK' = ZIP。
    直接 GeoTIFF 头 'II'/'MM'。
    """
    with open(tif_path, "rb") as f:
        magic = f.read(2)
    f.close()

    if magic == b"PK":
        # ZIP 包装:解压出 .tif,从内存读
        with zipfile.ZipFile(tif_path) as zf:
            tif_names = [n for n in zf.namelist() if n.endswith(".tif")]
            if not tif_names:
                raise ValueError(f"No .tif inside ZIP: {tif_path}")
            with zf.open(tif_names[0]) as tif_file:
                tif_bytes = tif_file.read()
        # rasterio 可以从文件对象读
        return rasterio.open(io.BytesIO(tif_bytes))
    else:
        # 直接 GeoTIFF
        return rasterio.open(tif_path)


def load_geotiff_as_array(tif_path: Path) -> tuple[np.ndarray, "rasterio.transform.Affine", str]:
    """加载 GeoTIFF(支持 ZIP 包装),返回 (array[H,W,C], transform, crs)。"""
    with open_geotiff_maybe_zip(tif_path) as src:
        arr = src.read()  # (bands, H, W)
        transform = src.transform
        crs = src.crs.to_string() if src.crs else "unknown"
    # 转 HWC
    if arr.ndim == 3:
        arr = np.transpose(arr, (1, 2, 0))  # (H, W, bands)
    return arr, transform, crs


def normalize_band(band: np.ndarray, dtype_out: type = np.uint8) -> np.ndarray:
    """单波段归一化:Sentinel-1 dB / Sentinel-2 reflectance → 0-255 uint8。"""
    band = band.astype(np.float32)
    # SAR (dB): 用百分位裁剪(-25 to 0 dB 常见)
    # 光学 (reflectance × 10000): 用 2-98 百分位
    vmin = float(np.percentile(band, 2))
    vmax = float(np.percentile(band, 98))
    if vmax - vmin < 1e-9:
        return np.zeros_like(band, dtype=dtype_out)
    norm = np.clip((band - vmin) / (vmax - vmin), 0, 1)
    return (norm * 255).astype(dtype_out)


def prepare_image_for_detection(arr: np.ndarray, source: str) -> np.ndarray:
    """把 GeoTIFF 数组转为 YOLO 输入格式(640×640 uint8 RGB)。

    Args:
        arr: (H, W, bands) numpy 数组
        source: 'sentinel1' 或 'sentinel2'

    Returns:
        (640, 640, 3) uint8 数组
    """
    # 波段选择 + 归一化
    if source == "sentinel1":
        # VV+VH,选 VV(主极化,船舶检测更强)
        if arr.shape[2] >= 1:
            vv = normalize_band(arr[:, :, 0])
        else:
            vv = normalize_band(arr[:, :, 0])
        rgb = np.stack([vv, vv, vv], axis=-1)  # 灰度伪 RGB(选项 α 一致)
    else:  # sentinel2
        # B,G,R,NIR = 索引 0,1,2,3 → 取 RGB = B2,B3,B4
        if arr.shape[2] >= 3:
            r = normalize_band(arr[:, :, 2])  # B4 Red
            g = normalize_band(arr[:, :, 1])  # B3 Green
            b = normalize_band(arr[:, :, 0])  # B2 Blue
            rgb = np.stack([r, g, b], axis=-1)
        else:
            g = normalize_band(arr[:, :, 0])
            rgb = np.stack([g, g, g], axis=-1)

    # 上采样到 640×640(用 OpenCV 或 PIL,这里简单用最近邻)
    from PIL import Image
    h, w = rgb.shape[:2]
    if (h, w) != (INPUT_SIZE, INPUT_SIZE):
        pil = Image.fromarray(rgb)
        pil = pil.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
        rgb = np.array(pil)

    return rgb


def boxes_to_lonlat(boxes_xyxy: np.ndarray, transform, original_h: int,
                     original_w: int, input_size: int = INPUT_SIZE) -> list[list[float]]:
    """把 640×640 坐标的框中心转回原始 GeoTIFF 经纬度。"""
    if len(boxes_xyxy) == 0:
        return []
    scale_x = original_w / input_size
    scale_y = original_h / input_size
    centers = []
    for box in boxes_xyxy:
        cx = (box[0] + box[2]) / 2 * scale_x
        cy = (box[1] + box[3]) / 2 * scale_y
        lon, lat = xy(transform, cy, cx)
        centers.append([float(lon), float(lat)])
    return centers


def detect_one_chip(model, tif_path: Path, source: str) -> dict:
    """检测单张 GeoTIFF。"""
    arr, transform, crs = load_geotiff_as_array(tif_path)
    original_h, original_w = arr.shape[:2]

    rgb = prepare_image_for_detection(arr, source)

    # YOLO 推理
    results = model(rgb, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
    r = results[0]

    if r.boxes is None or len(r.boxes) == 0:
        return {
            "file": tif_path.name,
            "source": source,
            "crs": crs,
            "image_size": [original_h, original_w],
            "n_ships": 0,
            "boxes_xyxy": [],
            "scores": [],
            "centers_lonlat": [],
        }

    boxes = r.boxes.xyxy.cpu().numpy()
    scores = r.boxes.conf.cpu().numpy().tolist()
    centers = boxes_to_lonlat(boxes, transform, original_h, original_w, INPUT_SIZE)

    return {
        "file": tif_path.name,
        "source": source,
        "crs": crs,
        "image_size": [original_h, original_w],
        "n_ships": len(boxes),
        "boxes_xyxy": boxes.tolist(),
        "scores": scores,
        "centers_lonlat": centers,
    }


def batch_detect_port(port_key: str, source: str = "both") -> dict:
    """批量检测某港口所有切片。"""
    model = load_detector()
    print(f"[detector] loaded", file=sys.stderr)

    results = []
    sources = ["sentinel1", "sentinel2"] if source == "both" else [source]
    for src in sources:
        src_dir = INPUT_DIR / port_key / src
        if not src_dir.exists():
            print(f"[skip] {src_dir} not found", file=sys.stderr)
            continue
        chips = sorted(src_dir.glob("*.tif"))
        print(f"[{port_key}/{src}] {len(chips)} chips to detect", file=sys.stderr)
        for i, chip in enumerate(chips):
            try:
                det = detect_one_chip(model, chip, src)
                results.append(det)
                if (i + 1) % 10 == 0:
                    print(f"  progress: {i+1}/{len(chips)}", file=sys.stderr)
            except Exception as e:
                print(f"  [WARN] {chip.name}: {str(e)[:80]}", file=sys.stderr)
                results.append({"file": chip.name, "source": src, "error": str(e)})

    # 保存
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / f"{port_key}_detections.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    n_ok = sum(1 for r in results if "error" not in r)
    total_ships = sum(r.get("n_ships", 0) for r in results if "error" not in r)
    print(f"[DONE] {port_key}: {n_ok} chips, {total_ships} total ships detected -> {out_file}",
          file=sys.stderr)
    return {"port": port_key, "chips": n_ok, "total_ships": total_ships}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("port", help="port key or 'all'")
    parser.add_argument("--source", default="both", choices=["sentinel1", "sentinel2", "both"])
    args = parser.parse_args()

    if args.port == "all":
        for port in ("shanghai", "los_angeles", "rotterdam"):
            batch_detect_port(port, args.source)
    else:
        batch_detect_port(args.port, args.source)


if __name__ == "__main__":
    main()
