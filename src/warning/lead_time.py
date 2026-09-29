"""Lead Time 量化算法(Task 4.2)— 论文 Contribution #1。

严格定义(响应审稿最担心第 1 点):
  t_satellite = 卫星指标首次超阈值的时刻(且 < event_date,避免事后诸葛)
  t_public    = 公开认知时刻(新闻首发 / 吞吐量下降 / AIS 异常)
  LT_satellite = t_public - t_satellite  [天]

多阈值敏感性(响应审稿必做 1):95/97.5/99 三个 Mahalanobis 阈值
Bootstrap 95% CI(响应审稿严格性)
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
from scipy.stats import chi2

from ..activity.mahalanobis import mahalanobis_distance


# ============================================================================
# PCI(Port Congestion Index)和基线
# ============================================================================

def compute_pci(series: pd.Series, baseline_window: int = 30) -> pd.Series:
    """Port Congestion Index = 当前 / 滚动中位数基线。

    PCI > 1.5 → 轻度拥堵
    PCI > 2.0 → 重度拥堵
    """
    baseline = series.rolling(window=baseline_window, min_periods=1).median()
    return series / (baseline + 1e-9)


def _first_anomaly_before_event(anomaly_flags: np.ndarray, dates, event_date) -> Optional[pd.Timestamp]:
    """找到事件发生前最后一次连续异常段的起点(避免事后诸葛)。

    双盲约束:t_satellite < event_date。
    取事件前最后一次 True 连续段的起点(不是单点)。
    """
    if hasattr(dates, "iloc"):
        getter = lambda i: dates.iloc[i]
    else:
        getter = lambda i: dates[i]
    event_ts = pd.Timestamp(event_date)

    # 找事件前的所有 True 索引
    pre_event_idx = []
    for i in range(len(anomaly_flags)):
        if anomaly_flags[i] == 1 and pd.Timestamp(getter(i)) < event_ts:
            pre_event_idx.append(i)

    if not pre_event_idx:
        return None

    # 取最后一段连续的起点
    last = pre_event_idx[-1]
    start = last
    while start > 0 and anomaly_flags[start - 1] == 1 and (start - 1) in pre_event_idx:
        start -= 1
    return pd.Timestamp(getter(start))


# ============================================================================
# Lead Time 量化(论文 Algorithm 1)
# ============================================================================

def compute_lead_time(activity_df: pd.DataFrame, event: dict,
                      activity_columns: list[str],
                      baseline_window: int = 30,
                      threshold_percentile: float = 95.0) -> dict:
    """单事件 Lead Time 计算。

    Algorithm 1(论文 §3.4):
      1. 构造多维活动特征 x(t) = [PAI 指标 + PCI]
      2. 计算 Mahalanobis D²
      3. 找 t_satellite = 事件前首次 D² > threshold
      4. LT_satellite = t_public - t_satellite

    Args:
        activity_df: 时间序列 DataFrame,需含 'date' 列 + activity_columns 列
        event: {id, event_date, public_recognition_date}
        activity_columns: 用于异常检测的多维特征列(如 ['N_total','N_anchor','N_berth'])
        baseline_window: PCI 基线窗口(天)
        threshold_percentile: Mahalanobis 阈值分位数

    Returns:
        {event_id, t_satellite, t_public, lt_satellite, threshold, status}
    """
    event_date = pd.Timestamp(event["event_date"])
    public_date = pd.Timestamp(event["public_recognition_date"])

    # 构造多维特征:活动指标 + PCI(拥堵信号)
    feature_df = activity_df[activity_columns].copy()
    # 加 PCI(N_total 的拥堵指数)作为补充维度
    if "N_total" in activity_df.columns:
        feature_df["PCI"] = compute_pci(activity_df["N_total"], baseline_window).values

    # Mahalanobis D²
    d2 = mahalanobis_distance(feature_df)
    threshold = chi2.ppf(threshold_percentile / 100.0, df=feature_df.shape[1])
    anomaly_flags = (d2 > threshold).astype(int)

    # 找 t_satellite(事件前首次异常)
    dates = activity_df["date"].values if "date" in activity_df.columns else activity_df.index
    t_satellite = _first_anomaly_before_event(anomaly_flags, dates, event_date)

    if t_satellite is None:
        return {
            "event_id": event.get("id"),
            "status": "no_detection",
            "t_satellite": None,
            "lt_satellite": None,
            "threshold_percentile": threshold_percentile,
        }

    lt_sat = (public_date - t_satellite).days
    return {
        "event_id": event.get("id"),
        "event_date": str(event_date.date()),
        "public_recognition_date": str(public_date.date()),
        "t_satellite": str(t_satellite.date()),
        "lt_satellite": lt_sat,
        "threshold_percentile": threshold_percentile,
        "status": "ok" if lt_sat > 0 else "late_warning",
    }


def compute_lead_time_sensitivity(activity_df: pd.DataFrame, event: dict,
                                    activity_columns: list[str],
                                    percentiles: tuple = (95, 97.5, 99)) -> dict:
    """三阈值敏感性(响应审稿必做 1)。

    95 / 97.5 / 99 三个阈值下的 Lead Time 稳定性。
    高阈值 → 更晚触发 → 更短 Lead Time(或 no_detection)。
    """
    results = {}
    for p in percentiles:
        results[p] = compute_lead_time(
            activity_df, event, activity_columns, threshold_percentile=p
        )
    return results


# ============================================================================
# Bootstrap 置信区间(响应审稿严格性)
# ============================================================================

def bootstrap_ci(lt_samples: list[float], n_boot: int = 1000,
                  confidence: float = 0.95, seed: int = 42) -> dict:
    """Bootstrap 95% CI(响应审稿统计严格性要求)。

    Args:
        lt_samples: 多事件的 LT 列表
        n_boot: bootstrap 次数
        confidence: 置信水平

    Returns:
        {mean, median, std, lower, upper, n}
    """
    if len(lt_samples) == 0:
        return {"mean": None, "n": 0, "warning": "no samples"}

    arr = np.array(lt_samples, dtype=float)
    rng = np.random.default_rng(seed)
    boot_means = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha = (1 - confidence) / 2
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "lower": float(np.percentile(boot_means, alpha * 100)),
        "upper": float(np.percentile(boot_means, (1 - alpha) * 100)),
        "n": len(arr),
        "n_boot": n_boot,
    }


# ============================================================================
# 批量评估(全事件)
# ============================================================================

def evaluate_all_events(activity_by_port: dict, events: list[dict],
                         activity_columns: list[str] = None,
                         percentiles: tuple = (95, 97.5, 99)) -> dict:
    """对全事件批量评估 Lead Time。

    Args:
        activity_by_port: {port_key: DataFrame}
        events: Tier 1 事件列表
        activity_columns: 用于异常检测的列(默认 ['N_total','N_anchor','N_berth','delta_arrive','delta_depart'])

    Returns:
        {per_event: [...], sensitivity: {...}, aggregate: {...}}
    """
    if activity_columns is None:
        activity_columns = ["N_total", "N_anchor", "N_berth", "delta_arrive", "delta_depart"]

    per_event = []
    sensitivity = {}
    for event in events:
        port = event["port"]
        df = activity_by_port.get(port)
        if df is None or len(df) < 10:
            per_event.append({"event_id": event["id"], "status": "insufficient_data",
                              "n_rows": len(df) if df is not None else 0})
            continue

        # 主阈值(95%)
        lt = compute_lead_time(df, event, activity_columns, threshold_percentile=95.0)
        per_event.append(lt)

        # 三阈值敏感性
        sensitivity[event["id"]] = compute_lead_time_sensitivity(
            df, event, activity_columns, percentiles=percentiles
        )

    # 聚合统计
    valid_lts = [e["lt_satellite"] for e in per_event
                 if e.get("lt_satellite") is not None and e["lt_satellite"] > 0]
    aggregate = bootstrap_ci(valid_lts) if valid_lts else {"n": 0, "warning": "no valid LT"}

    return {
        "per_event": per_event,
        "sensitivity": sensitivity,
        "aggregate": aggregate,
        "n_events_total": len(events),
        "n_events_with_detection": len(valid_lts),
    }
