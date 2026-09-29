"""Quiet-period imagery acquisition (revision upgrade: measured calm-period FPR).

Downloads Sentinel-1/2 at 250 m for five windows that contain NO curated Tier 1
event (buffered >= 1 month from every event window), reusing the exact download
machinery of src/download/event_window.py. Output naming Q{n}_{YYYYMMDD}.tif is
compatible with batch_detect.py and build_pai_series.py.

Quiet windows (all clear of event windows -90/+60):
  Rotterdam Q1: 2021-06-01..2021-10-31  (between E1 and E4)
  Rotterdam Q2: 2022-06-01..2022-10-31  (between E4 and E6)
  Shanghai  Q1: 2022-07-01..2022-12-31  (between E2 and E5)
  Shanghai  Q2: 2023-02-01..2023-06-30  (between E2 and E5)
  LA        Q1: 2022-01-01..2022-06-30  (after E3)
"""
from __future__ import annotations
import sys, json
from pathlib import Path
from datetime import datetime

PROJ_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ_ROOT))

from src.download.ee_auth import init_ee
from src.download.event_window import (
    build_event_window_roi, get_s1_collection, get_s2_collection,
    download_image, SCALE_S1, SCALE_S2, OUT_DIR,
)
import yaml
import ee

QUIET_WINDOWS = [
    {"id": "Q1_rotterdam_summer2021", "port": "rotterdam",  "start": "2021-06-01", "end": "2021-10-31"},
    {"id": "Q2_rotterdam_summer2022", "port": "rotterdam",  "start": "2022-06-01", "end": "2022-10-31"},
    {"id": "Q3_shanghai_h2_2022",     "port": "shanghai",   "start": "2022-07-01", "end": "2022-12-31"},
    {"id": "Q4_shanghai_h1_2023",     "port": "shanghai",   "start": "2023-02-01", "end": "2023-06-30"},
    {"id": "Q5_la_h1_2022",           "port": "los_angeles","start": "2022-01-01", "end": "2022-06-30"},
]

STATS_PATH = PROJ_ROOT / "results" / "quiet_acquisition_stats.json"


def main():
    init_ee()
    ports_cfg = yaml.safe_load((PROJ_ROOT / "configs" / "ports.yaml").read_text(encoding="utf-8"))["ports"]
    stats = []
    for qw in QUIET_WINDOWS:
        port, start, end = qw["port"], qw["start"], qw["end"]
        print(f"\n[{qw['id']}] {port} {start}..{end}", file=sys.stderr)
        roi = build_event_window_roi(port, ports_cfg)
        entry = {"id": qw["id"], "port": port, "window": [start, end],
                 "downloaded": {}, "failed": {}}
        for src, get_col, bands, scale in [
            ("sentinel1", get_s1_collection, ["VV", "VH"], SCALE_S1),
            ("sentinel2", get_s2_collection, ["B2", "B3", "B4", "B8"], SCALE_S2),
        ]:
            try:
                col = get_col(roi, start, end)
                n = col.size().getInfo()
            except Exception as e:
                print(f"  [{src}] collection error: {str(e)[:90]}", file=sys.stderr)
                entry["downloaded"][src], entry["failed"][src] = 0, 0
                continue
            print(f"  [{src}] collection size: {n}", file=sys.stderr)
            entry["downloaded"][src], entry["failed"][src] = 0, 0
            if n == 0:
                continue
            timestamps = col.aggregate_array("system:time_start").getInfo()
            img_list = col.toList(n)
            for i, ts_ms in enumerate(timestamps):
                img = ee.Image(img_list.get(i)).select(bands)
                date_str = datetime.utcfromtimestamp(ts_ms / 1000).strftime("%Y%m%d")
                out_path = OUT_DIR / port / src / f"{qw['id']}_{date_str}.tif"
                if out_path.exists():
                    entry["downloaded"][src] += 1
                    continue
                ok = download_image(img, out_path, port, src, scale=scale)
                if ok:
                    entry["downloaded"][src] += 1
                else:
                    entry["failed"][src] += 1
        stats.append(entry)
        STATS_PATH.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("\n[DONE] stats ->", STATS_PATH, file=sys.stderr)


if __name__ == "__main__":
    main()
