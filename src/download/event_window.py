"""事件窗口批量下载脚本(Task 3.1)。

策略:每个 Tier 1 事件下载 -90 天到 +60 天窗口的 Sentinel-1/2 切片。
  - -90 天:捕获 Lead Time 预警期(SARIMA 基线建立 + 异常前置检测)
  - +60 天:捕获事件发展/恢复期
  - 窗口共 150 天,~6 天重访 → ~25 张/事件/卫星源

总数据量估算:
  6 事件 × 2 源(S1+S2)× 25 张 × ~5MB(降采样后)= ~1.5 GB
  (实际 GEE getDownloadURL 限制单次 32MB,需切片)

使用 GEE getDownloadURL 下载 GeoTIFF 到 data/processed/{port}/{source}/{date}.tif
"""
from __future__ import annotations
import sys
import json
import time
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import ee
import requests

# 子项目根:src/download/event_window.py → parents[2]
PROJ_ROOT = Path(__file__).resolve().parents[2]
EVENTS_YAML = PROJ_ROOT / "configs" / "events.yaml"
PORTS_YAML = PROJ_ROOT / "configs" / "ports.yaml"
OUT_DIR = PROJ_ROOT / "data" / "processed"

# 下载窗口:事件前 90 天 + 后 60 天 = 150 天
WINDOW_BEFORE_DAYS = 90
WINDOW_AFTER_DAYS = 60

# GEE 单次下载限制
MAX_PIXELS_PER_DOWNLOAD = 1e8  # 100M pixels(实际 GEE getDownloadURL 限制 50MB)
SCALE_S1 = 250  # SAR 降到 250m(港口监测用,船只计数趋势足够)
SCALE_S2 = 250  # 光学 250m

# 紧凑 bbox(港口核心区,降低单张影像尺寸)
# 从 ports.yaml 的 bbox 进一步裁剪,只保留主要港区
COMPACT_BBOXES = {
    "shanghai": [121.4, 30.5, 122.2, 31.2],         # 上海主港区 + 长江口锚地
    "los_angeles": [-118.27, 33.68, -118.10, 33.78],  # LA/LB 主港区
    "rotterdam": [3.95, 51.92, 4.40, 52.08],         # Europoort + Maasvlakte
}


def load_events() -> list[dict]:
    with open(EVENTS_YAML, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("tier1_disruptions", [])


def load_ports() -> dict:
    with open(PORTS_YAML, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["ports"]


def init_ee():
    from .ee_auth import init_ee as _init
    _init()


def build_event_window_roi(port_key: str, ports_cfg: dict) -> ee.Geometry:
    """事件窗口的港口 ROI(用紧凑 bbox,降到 100m 控制数据量)。

    原始 ports.yaml bbox 太大(0.8°×0.25°),30m 分辨率单张 1GB > 50MB GEE 限制。
    改用 COMPACT_BBOXES 紧凑核心港区 + 100m 分辨率。
    """
    if port_key in COMPACT_BBOXES:
        w, s, e, n = COMPACT_BBOXES[port_key]
    else:
        w, s, e, n = ports_cfg[port_key]["bbox"]
    return ee.Geometry.Rectangle([w, s, e, n])


def get_s1_collection(roi: ee.Geometry, start: str, end: str) -> ee.ImageCollection:
    """Sentinel-1 GRD IW(与 sentinel.py 的 auto orbit 逻辑一致)。"""
    base = (ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(roi).filterDate(start, end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV")))
    desc = base.filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
    asc = base.filter(ee.Filter.eq("orbitProperties_pass", "ASCENDING"))
    chosen = ee.Algorithms.If(desc.size().gte(20), desc, asc)
    return ee.ImageCollection(chosen).select(["VV", "VH"])


def get_s2_collection(roi: ee.Geometry, start: str, end: str) -> ee.ImageCollection:
    """Sentinel-2 L2A(云掩膜,< 30% 云)。"""
    def mask_clouds(img):
        qa = img.select("QA60")
        cloud = qa.bitwiseAnd(1 << 10).Or(qa.bitwiseAnd(1 << 11))
        return img.updateMask(cloud.Not())
    return (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(roi).filterDate(start, end)
            .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 30))
            .map(mask_clouds))


def download_image(img: ee.Image, out_path: Path, port_key: str, source: str,
                    scale: int = 30) -> bool:
    """用 getDownloadURL 下载单张 GeoTIFF。返回是否成功。"""
    try:
        url = img.getDownloadURL({
            "scale": scale,
            "fileFormat": "GeoTIFF",
            "maxPixels": MAX_PIXELS_PER_DOWNLOAD,
        })
        r = requests.get(url, timeout=120, stream=True)
        if r.status_code != 200:
            print(f"    [WARN] HTTP {r.status_code}", file=sys.stderr)
            return False
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        size_mb = out_path.stat().st_size / 1e6
        print(f"    [OK] {out_path.name} ({size_mb:.1f} MB)", file=sys.stderr)
        return True
    except Exception as e:
        print(f"    [FAIL] {out_path.name}: {str(e)[:80]}", file=sys.stderr)
        return False


def process_event_window(event: dict, ports_cfg: dict,
                          source: str = "both") -> dict:
    """处理单事件的下载窗口。"""
    eid = event["id"]
    port = event["port"]
    event_date = datetime.strptime(event["event_date"], "%Y-%m-%d")

    start_date = (event_date - timedelta(days=WINDOW_BEFORE_DAYS)).strftime("%Y-%m-%d")
    end_date = (event_date + timedelta(days=WINDOW_AFTER_DAYS)).strftime("%Y-%m-%d")

    print(f"\n[{eid}] {port} {event_date.date()}  window={start_date} to {end_date}",
          file=sys.stderr)

    roi = build_event_window_roi(port, ports_cfg)
    stats = {"event_id": eid, "port": port, "window": [start_date, end_date],
             "downloaded": {"sentinel1": 0, "sentinel2": 0},
             "failed": {"sentinel1": 0, "sentinel2": 0}}

    sources = ["sentinel1", "sentinel2"] if source == "both" else [source]
    for src in sources:
        print(f"  [{src}]", file=sys.stderr)
        try:
            if src == "sentinel1":
                col = get_s1_collection(roi, start_date, end_date)
                bands = ["VV", "VH"]
                scale = SCALE_S1
            else:
                col = get_s2_collection(roi, start_date, end_date)
                bands = ["B2", "B3", "B4", "B8"]  # B,G,R,NIR
                scale = SCALE_S2
        except Exception as e:
            print(f"    [COLLECTION ERROR] {e}", file=sys.stderr)
            continue

        # 获取所有影像的时间戳
        try:
            n_total = col.size().getInfo()
            print(f"    collection size: {n_total}", file=sys.stderr)
        except Exception as e:
            print(f"    [SIZE ERROR] {e}", file=sys.stderr)
            continue

        if n_total == 0:
            print(f"    [SKIP] no images in window", file=sys.stderr)
            continue

        # 迭代下载每张影像
        timestamps = col.aggregate_array("system:time_start").getInfo()
        img_list = col.toList(n_total)
        for i, ts_ms in enumerate(timestamps):
            img = ee.Image(img_list.get(i)).select(bands)
            date_str = datetime.utcfromtimestamp(ts_ms / 1000).strftime("%Y%m%d")
            out_path = OUT_DIR / port / src / f"{eid}_{date_str}.tif"
            if out_path.exists():
                print(f"    [SKIP exists] {out_path.name}", file=sys.stderr)
                stats["downloaded"][src] += 1
                continue
            ok = download_image(img, out_path, port, src, scale=scale)
            if ok:
                stats["downloaded"][src] += 1
            else:
                stats["failed"][src] += 1
            # GEE 礼貌等待(避免触发限流)
            time.sleep(0.5)

    print(f"  summary: S1 ok={stats['downloaded']['sentinel1']} fail={stats['failed']['sentinel1']}, "
          f"S2 ok={stats['downloaded']['sentinel2']} fail={stats['failed']['sentinel2']}",
          file=sys.stderr)
    return stats


def main():
    init_ee()
    events = load_events()
    ports_cfg = load_ports()
    print(f"Loaded {len(events)} Tier 1 events", file=sys.stderr)
    print(f"Window: -{WINDOW_BEFORE_DAYS}d to +{WINDOW_AFTER_DAYS}d around each event",
          file=sys.stderr)
    print(f"Output: {OUT_DIR}", file=sys.stderr)

    all_stats = []
    for event in events:
        try:
            stats = process_event_window(event, ports_cfg, source="both")
            all_stats.append(stats)
        except Exception as e:
            print(f"[EVENT ERROR] {event['id']}: {e}", file=sys.stderr)
            all_stats.append({"event_id": event["id"], "error": str(e)})
        # 事件间等待,避免 GEE 配额突发
        time.sleep(2)

    # 保存统计
    out_stats = PROJ_ROOT / "results" / "task3_data_download_stats.json"
    out_stats.parent.mkdir(parents=True, exist_ok=True)
    out_stats.write_text(json.dumps(all_stats, indent=2), encoding="utf-8")
    print(f"\n[DONE] Stats saved to {out_stats}", file=sys.stderr)

    # 总结
    total_s1 = sum(s.get("downloaded", {}).get("sentinel1", 0) for s in all_stats)
    total_s2 = sum(s.get("downloaded", {}).get("sentinel2", 0) for s in all_stats)
    total_fail = sum(sum(s.get("failed", {}).values()) for s in all_stats)
    print(f"\n========== TOTAL ==========", file=sys.stderr)
    print(f"Sentinel-1 downloaded: {total_s1}", file=sys.stderr)
    print(f"Sentinel-2 downloaded: {total_s2}", file=sys.stderr)
    print(f"Failed: {total_fail}", file=sys.stderr)


if __name__ == "__main__":
    main()
