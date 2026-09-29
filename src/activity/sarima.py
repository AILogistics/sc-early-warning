"""SARIMA 季节性基线 + z-score 对照基线。

Task 3.3:Port Activity 时序异常检测基础。

两种基线(论文消融):
  - SARIMA: 捕捉趋势 + 季节性,残差更干净
  - z-score: 滚动均值/方差,简单对照基线

SARIMA 显著降低季节性误报(论文 §2.3.4 vs §2.3.5 消融对比)。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def fit_sarima_baseline(series: pd.Series, seasonal_period: int = 52,
                         order=(1, 1, 1),
                         seasonal_order=(1, 1, 1, None)) -> dict:
    """拟合 SARIMA 季节性基线。

    Args:
        series: 时间序列(必须 DatetimeIndex 或可排序 Index)
        seasonal_period: 季节周期
            - 52: 周度年季节(默认)
            - 12: 月度年季节
            - 7:  周内日度季节
        order: (p, d, q) 非季节 ARIMA 阶数
        seasonal_order: (P, D, Q, s);s 由 seasonal_period 填入

    Returns:
        {
          fitted: pd.Series,   # 拟合基线 ŝ(t)
          residual: pd.Series, # 残差 e(t) = series - fitted
          aic: float,          # AIC(模型选择)
          bic: float,          # BIC
          model: fitted model obj  # 用于 forecast
        }

    Raises:
        ValueError: 序列太短 / 含 NaN / seasonal_period 无效
    """
    # 输入校验
    if len(series) < 2 * seasonal_period:
        raise ValueError(
            f"Series too short: {len(series)} < 2*seasonal_period={2*seasonal_period}. "
            f"Need at least 2 full seasonal cycles for SARIMA."
        )
    if series.isna().any():
        raise ValueError("Series contains NaN. Please interpolate or drop first.")
    if seasonal_period < 2:
        raise ValueError(f"seasonal_period must be >= 2, got {seasonal_period}")

    # 填入季节周期
    so = (seasonal_order[0], seasonal_order[1], seasonal_order[2], seasonal_period)

    model = SARIMAX(
        series,
        order=order,
        seasonal_order=so,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False)

    return {
        "fitted": fit.fittedvalues,
        "residual": fit.resid,
        "aic": float(fit.aic),
        "bic": float(fit.bic),
        "model": fit,
    }


def forecast_sarima(fit_result: dict, steps: int) -> pd.Series:
    """用拟合的 SARIMA 做 steps 步预测。"""
    if "model" not in fit_result:
        raise KeyError("fit_result must contain 'model' key from fit_sarima_baseline")
    forecast = fit_result["model"].forecast(steps=steps)
    return forecast


def detect_anomalies_zscore(series: pd.Series, window: int = 30,
                             threshold: float = 2.0) -> pd.Series:
    """滚动 z-score 异常检测(对照基线)。

    Args:
        series: 时间序列
        window: 滚动窗口大小
        threshold: z-score 阈值(默认 2σ)

    Returns:
        int Series: 1 = 异常, 0 = 正常(与 series 同索引)
    """
    if len(series) < window:
        # 序列太短,用全局统计
        mu, sigma = series.mean(), series.std()
        if sigma < 1e-9:
            return pd.Series(0, index=series.index, dtype=int)
        z = (series - mu) / sigma
    else:
        rolling_mean = series.rolling(window=window, min_periods=5).mean()
        rolling_std = series.rolling(window=window, min_periods=5).std()
        z = (series - rolling_mean) / (rolling_std + 1e-9)

    flags = (z.abs() > threshold).astype(int)
    return flags


def detect_anomalies_sarima(series: pd.Series, seasonal_period: int = 52,
                             residual_threshold: float = 2.0) -> tuple[pd.Series, dict]:
    """SARIMA 残差异常检测。

    拟合 SARIMA,对残差做 z-score 阈值检测。

    Returns:
        (flags, fit_result)
        flags: int Series (1=anomaly, 0=normal)
        fit_result: dict from fit_sarima_baseline
    """
    fit = fit_sarima_baseline(series, seasonal_period=seasonal_period)
    residual = fit["residual"]
    # 残差的标准化(用残差自身的 std,不用滚动窗口因残差应已去季节)
    r_std = residual.std()
    if r_std < 1e-9:
        flags = pd.Series(0, index=series.index, dtype=int)
    else:
        z = (residual - residual.mean()) / r_std
        flags = (z.abs() > residual_threshold).astype(int)
    return flags, fit


def compare_baselines(series: pd.Series, seasonal_period: int = 52) -> dict:
    """对比 SARIMA vs z-score 两种基线的异常检测效果。

    Returns:
        {
          sarima: {n_anomalies, anomaly_rate, aic, bic},
          zscore: {n_anomalies, anomaly_rate},
          improvement: float  # zscore 异常率 - sarima 异常率(负 = sarima 更少异常)
        }
    """
    # SARIMA
    try:
        sarima_flags, sarima_fit = detect_anomalies_sarima(series, seasonal_period)
        sarima_n = int(sarima_flags.sum())
        sarima_rate = float(sarima_flags.mean())
        sarima_aic = sarima_fit["aic"]
        sarima_bic = sarima_fit["bic"]
    except ValueError:
        # 序列太短,无法 SARIMA → 只报 zscore
        sarima_n = sarima_rate = sarima_aic = sarima_bic = None

    # z-score
    zscore_flags = detect_anomalies_zscore(series, window=min(30, len(series) // 3 or 1))
    zscore_n = int(zscore_flags.sum())
    zscore_rate = float(zscore_flags.mean())

    improvement = None
    if sarima_rate is not None:
        improvement = zscore_rate - sarima_rate  # 正 = SARIMA 减少了异常标记

    return {
        "sarima": {"n_anomalies": sarima_n, "anomaly_rate": sarima_rate,
                   "aic": sarima_aic, "bic": sarima_bic},
        "zscore": {"n_anomalies": zscore_n, "anomaly_rate": zscore_rate},
        "improvement": improvement,
    }
