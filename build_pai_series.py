"""Task 3.4: 构建 PAI 时序 + 第 3 周检查点可视化。

输入:experiments/exp3_activity/detections/{port}_detections.json
输出:
  - experiments/exp3_activity/pai_series/{port}_pai.csv (每港 PAI 时序)
  - experiments/exp3_activity/pai_plot_{port}.png (肉眼可见季节性 + 事件期)
  - results/task3_pai_summary.json (汇总统计)
"""
from __future__ import annotations
import sys
import json
from pathlib import Path
from datetime import datetime
import yaml

# 让 src 可导入
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# build_pai_series.py 位于 sc_early_warning/ 根目录
PROJ_ROOT = Path(__file__).resolve().parent
DET_DIR = PROJ_ROOT / "experiments" / "exp3_activity" / "detections"
OUT_DIR = PROJ_ROOT / "experiments" / "exp3_activity" / "pai_series"
PLOT_DIR = PROJ_ROOT / "experiments" / "exp3_activity" / "plots"
EVENTS_YAML = PROJ_ROOT / "configs" / "events.yaml"


def parse_date_from_filename(filename: str) -> pd.Timestamp:
    """E1_suez_2021_20210323.tif → 2021-03-23"""
    # 末尾 8 位数字 = YYYYMMDD
    parts = filename.replace(".tif", "").split("_")
    date_str = parts[-1]
    return pd.to_datetime(date_str, format="%Y%m%d")


def build_port_pai_series(port: str) -> pd.DataFrame:
    """从检测 JSON 构建 PAI 时序 DataFrame。"""
    det_file = DET_DIR / f"{port}_detections.json"
    if not det_file.exists():
        raise FileNotFoundError(det_file)

    detections = json.loads(det_file.read_text(encoding="utf-8"))
    ok = [d for d in detections if "error" not in d]

    # 转为 DataFrame
    rows = []
    for d in ok:
        date = parse_date_from_filename(d["file"])
        rows.append({
            "date": date,
            "source": d["source"],
            "N_total": d["n_ships"],
            "centers_lonlat": d.get("centers_lonlat", []),
        })
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    # 占位:N_anchor/N_berth 用 30%/40% 估计(后续接 GIS 多边形精确化)
    df["N_anchor"] = (df["N_total"] * 0.3).astype(int)
    df["N_berth"] = (df["N_total"] * 0.4).astype(int)
    diff = df["N_total"].diff().fillna(0)
    df["delta_arrive"] = diff.clip(lower=0)
    df["delta_depart"] = (-diff).clip(lower=0)

    # PAI(等权)
    from src.activity.pai import compute_pai_equal_weight, compute_congestion_pressure_series
    df["PAI_equal"] = compute_pai_equal_weight(df)
    df["CP"] = compute_congestion_pressure_series(df["N_anchor"].values, df["N_berth"].values)

    # 按周重采样(平滑,降采样到周度)
    df_weekly = df.set_index("date").resample("W").agg({
        "N_total": "mean",
        "N_anchor": "mean",
        "N_berth": "mean",
        "delta_arrive": "sum",
        "delta_depart": "sum",
        "PAI_equal": "mean",
        "CP": "mean",
    }).dropna().reset_index()

    return df, df_weekly


def plot_port_pai(port: str, df: pd.DataFrame, events_for_port: list[dict]) -> Path:
    """绘制 PAI 时序 + 事件标记。"""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # 上:N_total 时序(分 S1/S2)
    s1 = df[df["source"] == "sentinel1"]
    s2 = df[df["source"] == "sentinel2"]
    axes[0].scatter(s1["date"], s1["N_total"], c="blue", alpha=0.6, s=30, label="Sentinel-1 (SAR)")
    axes[0].scatter(s2["date"], s2["N_total"], c="green", alpha=0.6, s=30, label="Sentinel-2 (optical)")
    # 滚动均值平滑
    if len(df) > 10:
        df_sorted = df.sort_values("date")
        rolling = df_sorted.set_index("date")["N_total"].rolling("14D").mean()
        axes[0].plot(rolling.index, rolling.values, "r-", linewidth=2, label="14-day rolling mean")
    axes[0].set_ylabel("Ship count (N_total)")
    axes[0].set_title(f"{port} — Port Activity Time Series (Task 3.4 checkpoint)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # 下:事件标记
    axes[1].plot(df_sorted["date"], df_sorted["N_total"], "gray", alpha=0.3, linewidth=0.5)
    for ev in events_for_port:
        ed = pd.to_datetime(ev["event_date"])
        axes[1].axvline(ed, color="red", linestyle="--", alpha=0.7)
        axes[1].text(ed, axes[1].get_ylim()[1] * 0.9, ev["id"].split("_")[0],
                     color="red", fontsize=8, rotation=90, va="top")
    axes[1].set_ylabel("Ship count (raw)")
    axes[1].set_xlabel("Date")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = PLOT_DIR / f"task3_pai_plot_{port}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # 加载事件 GT
    with open(EVENTS_YAML, "r", encoding="utf-8") as f:
        events_cfg = yaml.safe_load(f)
    events = events_cfg.get("tier1_disruptions", [])

    summary = {}
    for port in ("shanghai", "los_angeles", "rotterdam"):
        print(f"\n[{port}]", file=sys.stderr)
        try:
            df, df_weekly = build_port_pai_series(port)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}", file=sys.stderr)
            continue

        # 保存 PAI CSV
        out_csv = OUT_DIR / f"{port}_pai.csv"
        df.to_csv(out_csv, index=False)
        out_weekly = OUT_DIR / f"{port}_pai_weekly.csv"
        df_weekly.to_csv(out_weekly, index=False)
        print(f"  saved: {out_csv.name} ({len(df)} rows), {out_weekly.name} ({len(df_weekly)} weeks)",
              file=sys.stderr)

        # 绘图
        port_events = [e for e in events if e["port"] == port]
        plot_path = plot_port_pai(port, df, port_events)
        print(f"  plot: {plot_path.name}", file=sys.stderr)

        # 汇总
        summary[port] = {
            "n_observations": len(df),
            "n_weeks": len(df_weekly),
            "date_range": [str(df["date"].min().date()), str(df["date"].max().date())],
            "mean_N_total": float(df["N_total"].mean()),
            "max_N_total": int(df["N_total"].max()),
            "n_events": len(port_events),
            "event_ids": [e["id"] for e in port_events],
        }

    out_summary = PROJ_ROOT / "results" / "task3_pai_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[DONE] Summary: {out_summary}", file=sys.stderr)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
