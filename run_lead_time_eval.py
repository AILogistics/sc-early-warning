"""Task 5.1: 全 6 事件 Lead Time 真实数据评估。

论文 Contribution #1 的核心实验。
输入:
  - experiments/exp3_activity/pai_series/{port}_pai.csv(第 3 周生成)
  - configs/events.yaml(6 Tier 1 事件)
输出:
  - experiments/exp5_lead_time/lead_time_results.json
  - experiments/exp5_lead_time/comparison_table.json(含论文文字)
  - experiments/exp5_lead_time/lead_time_boxplot.png(Figure 9 候选)
"""
from __future__ import annotations
import sys
import json
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 子项目根
PROJ_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ_ROOT))

from src.warning.lead_time import (
    compute_lead_time, compute_lead_time_sensitivity, evaluate_all_events, bootstrap_ci
)
from src.warning.baselines import build_comparison_table, save_comparison_results


PAI_DIR = PROJ_ROOT / "experiments" / "exp3_activity" / "pai_series"
EVENTS_YAML = PROJ_ROOT / "configs" / "events.yaml"
OUT_DIR = PROJ_ROOT / "experiments" / "exp5_lead_time"

# Lead Time 用的活动列
ACTIVITY_COLUMNS = ["N_total", "N_anchor", "N_berth", "delta_arrive", "delta_depart"]


def load_port_pai(port: str) -> pd.DataFrame:
    """加载某港口的 PAI 时序 CSV。"""
    path = PAI_DIR / f"{port}_pai.csv"
    if not path.exists():
        raise FileNotFoundError(f"PAI series not found: {path}")
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def load_events() -> list[dict]:
    with open(EVENTS_YAML, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("tier1_disruptions", [])


def plot_lead_time_boxplot(lt_samples: list[float], out_path: Path) -> None:
    """Figure 9 候选:Lead Time 分布箱线图(论文卖点图)。"""
    if not lt_samples:
        print(f"[WARN] no LT samples to plot", file=sys.stderr)
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot([lt_samples], vert=True, patch_artist=True, widths=0.4,
                    boxprops=dict(facecolor="#4A90E2", alpha=0.7),
                    medianprops=dict(color="red", linewidth=2))
    # 目标线 10 天
    ax.axhline(y=10, color="green", linestyle="--", linewidth=1.5,
               label="Target: 10 days", alpha=0.7)
    ax.axhline(y=7, color="orange", linestyle=":", linewidth=1.5,
               label="Minimum: 7 days", alpha=0.7)
    ax.set_ylabel("Warning Lead Time (days)", fontsize=12)
    ax.set_title(f"Lead Time Distribution (n={len(lt_samples)} Tier 1 disruptions)",
                 fontsize=13)
    ax.legend(loc="upper right")
    # 散点叠加
    x = np.random.normal(1, 0.04, size=len(lt_samples))
    ax.scatter(x, lt_samples, alpha=0.7, color="black", zorder=3, s=40)
    # 均值标记
    mean_lt = np.mean(lt_samples)
    ax.scatter([1.3], [mean_lt], marker="D", color="red", s=100, zorder=4,
               label=f"Mean: {mean_lt:.1f}d")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {out_path.name}", file=sys.stderr)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    events = load_events()
    print(f"Loaded {len(events)} Tier 1 events", file=sys.stderr)

    # 加载 3 港口 PAI
    activity_by_port = {}
    for port in ("shanghai", "los_angeles", "rotterdam"):
        try:
            df = load_port_pai(port)
            activity_by_port[port] = df
            print(f"  [{port}] {len(df)} observations, "
                  f"{df['date'].min().date()} to {df['date'].max().date()}", file=sys.stderr)
        except FileNotFoundError as e:
            print(f"  [{port}] SKIP: {e}", file=sys.stderr)

    # ========== 主评估:全事件 Lead Time ==========
    print(f"\n========== Lead Time Evaluation ==========", file=sys.stderr)
    result = evaluate_all_events(activity_by_port, events, ACTIVITY_COLUMNS,
                                  percentiles=(95, 97.5, 99))

    # 打印 per-event 结果
    print(f"\nPer-event results:", file=sys.stderr)
    for e in result["per_event"]:
        eid = e.get("event_id", "?")
        status = e.get("status", "?")
        lt = e.get("lt_satellite")
        if lt is not None:
            print(f"  {eid:30s}: LT = {lt:6.1f} days  (t_sat={e.get('t_satellite')})  [{status}]",
                  file=sys.stderr)
        else:
            print(f"  {eid:30s}: {status}", file=sys.stderr)

    # 聚合
    agg = result["aggregate"]
    print(f"\nAggregate:", file=sys.stderr)
    if agg.get("n", 0) > 0:
        print(f"  Detected: {agg['n']}/{result['n_events_with_detection']} valid / "
              f"{result['n_events_total']} total", file=sys.stderr)
        print(f"  Mean LT: {agg['mean']:.1f} days", file=sys.stderr)
        print(f"  Median LT: {agg['median']:.1f} days", file=sys.stderr)
        print(f"  95% CI: [{agg['lower']:.1f}, {agg['upper']:.1f}]", file=sys.stderr)
        print(f"  Range: [{agg['min']:.0f}, {agg['max']:.0f}]", file=sys.stderr)
        # 核心成功标准
        if agg["mean"] >= 10:
            print(f"\n  ✅ CORE SUCCESS: Mean LT >= 10 days", file=sys.stderr)
        elif agg["mean"] >= 7:
            print(f"\n  ⚠️ MINIMUM MET: Mean LT 7-10 days", file=sys.stderr)
        else:
            print(f"\n  ❌ BELOW THRESHOLD: Mean LT < 7 days", file=sys.stderr)
    else:
        print(f"  ⚠️ No valid Lead Time detections", file=sys.stderr)

    # 保存主结果
    (OUT_DIR / "lead_time_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8"
    )

    # ========== Baseline 对比 ==========
    print(f"\n========== Baseline Comparison ==========", file=sys.stderr)
    comparison = build_comparison_table(activity_by_port, events, ACTIVITY_COLUMNS)
    save_comparison_results(comparison, OUT_DIR / "comparison_table.json")
    from src.warning.baselines import format_comparison_text
    paper_text = format_comparison_text(comparison)
    print(f"\nPaper text:\n  {paper_text}", file=sys.stderr)

    # ========== 绘图:Lead Time 箱线图(Figure 9) ==========
    print(f"\n========== Visualization ==========", file=sys.stderr)
    valid_lts = [e["lt_satellite"] for e in result["per_event"]
                 if e.get("lt_satellite") is not None and e["lt_satellite"] > 0]
    plot_lead_time_boxplot(valid_lts, OUT_DIR / "fig9_lead_time_boxplot.png")

    # ========== 核心成功标准总结 ==========
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"CORE SUCCESS CRITERIA (decides paper tier)", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    crit1 = agg.get("mean", 0) >= 10 if agg.get("n", 0) > 0 else False
    crit2_n_events = result["n_events_with_detection"]
    crit2 = crit2_n_events >= 6
    # PAI-吞吐量相关性需要外部数据(UNCTAD),这里先标记 TODO
    print(f"  [1] Lead Time >= 10 days: {'✅' if crit1 else '❌'} "
          f"(mean = {agg.get('mean', 'N/A')})", file=sys.stderr)
    print(f"  [2] Events >= 6: {'✅' if crit2 else '❌'} "
          f"(detected {crit2_n_events}/{result['n_events_total']})", file=sys.stderr)
    print(f"  [3] PAI-throughput r >= 0.7: TODO (needs UNCTAD data)", file=sys.stderr)


if __name__ == "__main__":
    main()
