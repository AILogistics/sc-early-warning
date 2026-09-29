"""Sentinel-1/2 港口时序影像 GEE 下载。

GEE 集合构造 + 可用性检查 + 缩略图导出。完整批量下载(完整 GeoTIFF)
在 Task 后续步骤,本模块负责:
  - 构造 S1/S2 ImageCollection(含云掩膜/极化筛选)
  - 检查 3 港口 × 2021-2024 的影像可用数
  - 缩略图快速预览(验证检测器输入格式)
"""
from __future__ import annotations
import sys
from pathlib import Path
import json

import ee
import yaml

from .ee_auth import init_ee

PROJ_ROOT = Path(__file__).resolve().parents[2]
PORTS_YAML = PROJ_ROOT / "configs" / "ports.yaml"
OUTPUT_DIR = PROJ_ROOT / "data" / "raw"


def load_ports_config() -> dict:
    with open(PORTS_YAML, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_port_roi(port_key: str, cfg: dict | None = None) -> ee.Geometry:
    """根据 bbox 构造 GEE 矩形 ROI。"""
    if cfg is None:
        cfg = load_ports_config()
    w, s, e, n = cfg["ports"][port_key]["bbox"]
    return ee.Geometry.Rectangle([w, s, e, n])


def _mask_s2_clouds(img: ee.Image) -> ee.Image:
    """Sentinel-2 QA60 云掩膜。"""
    qa = img.select("QA60")
    cloud = qa.bitwiseAnd(1 << 10).Or(qa.bitwiseAnd(1 << 11))
    return img.updateMask(cloud.Not())


def build_sentinel2_collection(roi: ee.Geometry, start: str, end: str,
                                max_cloud_pct: int = 60) -> ee.ImageCollection:
    """Sentinel-2 L2A,云掩膜。"""
    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(roi)
        .filterDate(start, end)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", max_cloud_pct))
    )
    return col.map(_mask_s2_clouds)


def build_sentinel1_collection(roi: ee.Geometry, start: str, end: str,
                                prefer_pass: str = "auto") -> ee.ImageCollection:
    """Sentinel-1 GRD,IW 模式,VV+VH 极化。

    Args:
        prefer_pass: "auto"(选数据量大的轨道方向)/ "ASCENDING" / "DESCENDING"
                     重要:同一时序必须用同一轨道方向,否则视角/几何不一致。
                     不同港口主轨道方向不同(上海主 ASC,鹿特丹主 DESC),auto 最稳。
    """
    base = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(roi)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
    )
    if prefer_pass == "auto":
        # 优先 DESCENDING,若 < 20 张则改 ASCENDING(避免 0 张的极端情况)
        desc = base.filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
        asc = base.filter(ee.Filter.eq("orbitProperties_pass", "ASCENDING"))
        n_desc = desc.size()
        n_asc = asc.size()
        # ee.Iterator 用不了简单 if;这里用 ee 对象的 boolean trick
        # 简化:直接用 ee.Algorithms.If
        chosen = ee.Algorithms.If(n_desc.gte(20), desc, asc)
        return ee.ImageCollection(chosen).select(["VV", "VH"])
    return base.filter(ee.Filter.eq("orbitProperties_pass", prefer_pass)).select(["VV", "VH"])


def check_port_availability(port_key: str) -> dict:
    """检查某港口在配置时间范围内的 S1/S2 可用影像数。"""
    cfg = load_ports_config()
    roi = get_port_roi(port_key, cfg)
    start, end = cfg["date_range"]["start"], cfg["date_range"]["end"]

    s1 = build_sentinel1_collection(roi, start, end)
    s2_clear = build_sentinel2_collection(roi, start, end, max_cloud_pct=30)   # 严格(可检测)
    s2_all = build_sentinel2_collection(roi, start, end, max_cloud_pct=60)     # 宽松(可下载)

    return {
        "port": port_key,
        "date_range": [start, end],
        "sentinel1": s1.size().getInfo(),
        "sentinel2_clear": s2_clear.size().getInfo(),    # < 30% 云,光学可用
        "sentinel2_all": s2_all.size().getInfo(),         # < 60% 云,总下载量
    }


def check_all_ports() -> list[dict]:
    """检查全部 3 港口的可用性,保存到 results/week1_data_availability.json。"""
    init_ee()
    cfg = load_ports_config()
    results = []
    for port_key in cfg["ports"]:
        print(f"[checking] {port_key}...", file=sys.stderr)
        try:
            r = check_port_availability(port_key)
            results.append(r)
            print(f"  S1={r['sentinel1']}, S2_clear={r['sentinel2_clear']}, S2_all={r['sentinel2_all']}",
                  file=sys.stderr)
        except Exception as e:
            print(f"  [ERROR] {e}", file=sys.stderr)
            results.append({"port": port_key, "error": str(e)})

    out_dir = PROJ_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "week1_data_availability.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[DONE] {out_file}", file=sys.stderr)
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("port", nargs="?", default="all",
                        help="port key (shanghai/los_angeles/rotterdam) or 'all'")
    args = parser.parse_args()

    init_ee()
    if args.port == "all":
        check_all_ports()
    else:
        r = check_port_availability(args.port)
        print(json.dumps(r, indent=2))
