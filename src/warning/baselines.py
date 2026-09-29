"""Baseline 对比(Task 4.3)。

审稿人强调:论文成败不能依赖 AIS(数据质量不稳定)。
策略:
  - Primary Baseline: News Monitoring(events.yaml 里的 public_recognition_date)
  - Secondary Baseline: AIS (where available;失败不阻断)

对比表(论文 Table 5):
  | Method         | Lead Time | P | R | F1 |
  | News (primary) | LT_news   |   |   |    |
  | AIS (optional) | LT_ais    |   |   |    |
  | Proposed (sat) | LT_sat    |   |   |    |
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from .lead_time import compute_lead_time


def news_baseline_lead_time(event: dict) -> dict:
    """新闻基线 LT:假定新闻 = 公众认知本身,LT_news = 0(定义性,无提前)。

    实际意义:新闻基线代表"事后报道",本身无 Lead Time。
    对比时,卫星 LT 相对新闻基线的"提前期"= LT_sat - LT_news = LT_sat。
    """
    return {
        "event_id": event.get("id"),
        "method": "news_baseline",
        "t_detection": event["public_recognition_date"],
        "lt_vs_public": 0,  # 新闻 = 公众认知本身
        "interpretation": "news reports = public recognition, no lead time by definition",
    }


def satellite_vs_news_comparison(satellite_lt_result: dict, event: dict) -> dict:
    """卫星相对新闻基线的优势 = 卫星 LT(因为新闻 LT=0)。

    Args:
        satellite_lt_result: compute_lead_time() 的输出
        event: 含 public_recognition_date

    Returns:
        {event_id, sat_lt, news_lt, advantage}
    """
    sat_lt = satellite_lt_result.get("lt_satellite")
    return {
        "event_id": satellite_lt_result.get("event_id"),
        "satellite_lt_days": sat_lt,
        "news_lt_days": 0,  # 新闻基线定义
        "satellite_advantage_days": sat_lt if sat_lt is not None else None,
        "satellite_status": satellite_lt_result.get("status"),
    }


def build_comparison_table(activity_by_port: dict, events: list[dict],
                            activity_columns: list[str] = None) -> dict:
    """构造完整对比表(论文 Table 5)。

    Args:
        activity_by_port: {port: DataFrame}
        events: 全部 Tier 1 事件
        activity_columns: Lead Time 用的特征列

    Returns:
        {
          per_event: [{event_id, port, sat_lt, news_lt, advantage, status}],
          aggregate: {mean_sat_lt, mean_advantage, n_events, n_detected}
        }
    """
    if activity_columns is None:
        activity_columns = ["N_total", "N_anchor", "N_berth", "delta_arrive", "delta_depart"]

    per_event = []
    for event in events:
        port = event["port"]
        df = activity_by_port.get(port)

        if df is None or len(df) < 10:
            per_event.append({
                "event_id": event["id"], "port": port,
                "status": "insufficient_data",
                "satellite_lt_days": None, "news_lt_days": 0,
                "satellite_advantage_days": None,
            })
            continue

        # 卫星 Lead Time
        sat_result = compute_lead_time(df, event, activity_columns, threshold_percentile=95.0)
        comparison = satellite_vs_news_comparison(sat_result, event)
        comparison["port"] = port
        comparison["event_date"] = event["event_date"]
        per_event.append(comparison)

    # 聚合
    valid = [e for e in per_event if e.get("satellite_lt_days") is not None
             and e["satellite_lt_days"] > 0]
    if valid:
        sat_lts = [e["satellite_lt_days"] for e in valid]
        aggregate = {
            "n_events_total": len(events),
            "n_events_detected": len(valid),
            "detection_rate": len(valid) / len(events),
            "mean_satellite_lt": float(np.mean(sat_lts)),
            "median_satellite_lt": float(np.median(sat_lts)),
            "std_satellite_lt": float(np.std(sat_lts, ddof=1)) if len(sat_lts) > 1 else 0.0,
            "min_satellite_lt": float(np.min(sat_lts)),
            "max_satellite_lt": float(np.max(sat_lts)),
        }
    else:
        aggregate = {
            "n_events_total": len(events),
            "n_events_detected": 0,
            "detection_rate": 0.0,
            "warning": "no valid Lead Time detections",
        }

    return {
        "per_event": per_event,
        "aggregate": aggregate,
        "baseline_strategy": {
            "primary": "news (= public_recognition_date from events.yaml)",
            "secondary": "AIS (where available, omitted if no data)",
            "note": "News baseline has LT=0 by definition (news = public recognition). "
                    "Satellite advantage = satellite LT.",
        },
    }


def format_comparison_text(comparison: dict) -> str:
    """生成论文-ready 的对比文字描述。"""
    agg = comparison["aggregate"]
    if agg.get("n_events_detected", 0) == 0:
        return ("No valid satellite Lead Time detections. "
                "Consider relaxing threshold or reviewing data quality.")

    mean_lt = agg["mean_satellite_lt"]
    n_det = agg["n_events_detected"]
    n_tot = agg["n_events_total"]
    rate = agg["detection_rate"]

    # 论文关键句(响应审稿):卫星提前期 vs 新闻基线
    text = (
        f"The proposed satellite-based framework detected {n_det}/{n_tot} "
        f"Tier 1 disruption events ({rate*100:.0f}% detection rate) "
        f"with a mean warning lead time of {mean_lt:.1f} days "
        f"before publicly recognized disruption events, "
        f"outperforming news-based monitoring (which by definition has 0 days lead). "
        f"Median satellite lead time: {agg['median_satellite_lt']:.1f} days "
        f"(range {agg['min_satellite_lt']:.0f}-{agg['max_satellite_lt']:.0f})."
    )
    return text


def save_comparison_results(comparison: dict, out_path: Path) -> None:
    """保存对比结果 + 论文文字。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "comparison": comparison,
        "paper_text": format_comparison_text(comparison),
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                        encoding="utf-8")
