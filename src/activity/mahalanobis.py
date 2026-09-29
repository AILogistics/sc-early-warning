"""Mahalanobis 多维异常评分(Task 4.1)。

替代单变量 z-score:考虑港口活动指标间的协方差结构,
更严谨地捕捉"多维同时异常"(锚泊+泊位+航道同时飙升才是真拥堵)。

D²(t) = (x(t) − μ)ᵀ Σ⁻¹ (x(t) − μ)
阈值用 χ² 分布的 95/97.5/99 分位数。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import chi2


def mahalanobis_distance(df: pd.DataFrame) -> np.ndarray:
    """计算每行的 Mahalanobis 距离平方 D²。

    Args:
        df: 多列活动指标 DataFrame(每行一时间点)

    Returns:
        D² 数组(长度 = len(df))
    """
    X = df.values.astype(float)
    mu = X.mean(axis=0)
    cov = np.cov(X.T)
    # 防奇异矩阵:加正则化
    cov += np.eye(cov.shape[0]) * 1e-6
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        # 奇异,用伪逆
        inv_cov = np.linalg.pinv(cov)
    diff = X - mu
    d2 = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)
    return d2


def detect_anomalies_mahalanobis(df: pd.DataFrame,
                                   threshold_percentile: float = 95.0) -> np.ndarray:
    """用 χ² 分布分位数作为阈值,标记异常。

    Args:
        df: 多列活动指标 DataFrame
        threshold_percentile: 95/97.5/99

    Returns:
        int 数组(1=异常, 0=正常),长度 = len(df)
    """
    d2 = mahalanobis_distance(df)
    n_features = df.shape[1]
    threshold = chi2.ppf(threshold_percentile / 100.0, df=n_features)
    return (d2 > threshold).astype(int)


def multi_threshold_sensitivity(df: pd.DataFrame,
                                  percentiles: tuple = (95, 97.5, 99)) -> dict:
    """多阈值敏感性分析(Task 4.2 阈值敏感性辅助)。

    审稿必做 1:Lead Time 不能依赖单一阈值。报告 3 个阈值的稳定性。

    Returns:
        {percentile: {threshold, n_anomalies, anomaly_rate, anomaly_indices}}
    """
    d2 = mahalanobis_distance(df)
    n_features = df.shape[1]
    results = {}
    for p in percentiles:
        threshold = chi2.ppf(p / 100.0, df=n_features)
        flags = (d2 > threshold).astype(int)
        results[p] = {
            "threshold": float(threshold),
            "n_anomalies": int(flags.sum()),
            "anomaly_rate": float(flags.mean()),
        }
    return results


def get_anomaly_dates(df: pd.DataFrame, dates,
                       threshold_percentile: float = 95.0) -> list:
    """返回异常日期列表(用于 Lead Time 算法的 t_satellite)。

    dates 可以是 pd.Series 或 pd.DatetimeIndex。
    """
    flags = detect_anomalies_mahalanobis(df, threshold_percentile)
    # 兼容 Series 和 DatetimeIndex
    if hasattr(dates, "iloc"):
        getter = lambda i: dates.iloc[i]
    else:
        getter = lambda i: dates[i]
    return [getter(i) for i in range(len(flags)) if flags[i] == 1]
