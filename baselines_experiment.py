# -*- coding: utf-8 -*-
"""Baseline comparison under the IDENTICAL trailing-window protocol (Doubao must-fix #2).

Methods compared (all: trailing 120d window, min 12 obs, threshold = trailing
empirical tau-quantile of the method's own score, tau = 95%):
  - adaptive Mahalanobis on robust z-scores (main, 5 features)
  - B1: univariate rolling-z on PAI_equal (classic threshold monitoring)
  - B2: Isolation Forest on the 5 raw features (classic ML anomaly)
  - B3: Mahalanobis on RAW features (no robust normalisation)

Metrics per method: pooled quiet-period flag rate; in-rule event detections and
lead times (same inclusion rule, full-series last-run search).
Output: experiments/exp5_lead_time/baselines_comparison.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import chi2

PROJ = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ))
from src.warning.lead_time import compute_pci

PAI = PROJ / "experiments" / "exp3_activity" / "pai_series"
EVENTS_YAML = PROJ / "configs" / "events.yaml"
OUT = PROJ / "experiments" / "exp5_lead_time" / "baselines_comparison.json"

COLS = ["N_total", "N_anchor", "N_berth", "delta_arrive"]
FEATURES = COLS + ["PCI"]
TRAILING_DAYS, TRAILING_MIN = 120, 12
TAU = 95.0
WINDOW = (-90, 60)
EPS = 1e-6


def trailing_idx(dates, i):
    t = dates[i]
    lo = np.searchsorted(dates, np.datetime64(pd.Timestamp(t) - pd.Timedelta(days=TRAILING_DAYS)))
    return lo


def emp_quantile_threshold(scores_hist, tau, fallback=None):
    s = np.asarray(scores_hist, dtype=float)
    s = s[~np.isnan(s)]
    if len(s) >= TRAILING_MIN:
        return float(np.quantile(s, tau / 100.0))
    return fallback  # method-specific cold-start fallback (same rule for all methods)


def last_run_start(flags, upto):
    idx = [i for i in range(upto) if flags[i] == 1]
    if not idx:
        return None
    start = idx[-1]
    while start - 1 in idx:
        start -= 1
    return start


def score_port(df, method):
    """Return flags array (0/1) aligned with df rows; NaN-position -> 0."""
    n = len(df)
    f = df[COLS].copy()
    f["PCI"] = compute_pci(df["N_total"], baseline_window=30).values
    X = f[FEATURES].values.astype(float)
    dates = df["date"].values
    score = np.full(n, np.nan)

    for i in range(n):
        lo = trailing_idx(dates, i)
        hist = X[lo:i]
        if len(hist) < TRAILING_MIN:
            continue
        if method == "main_adaptive_maha":
            med = np.median(hist, axis=0)
            mad = np.median(np.abs(hist - med), axis=0) * 1.4826 + EPS
            z = (X[i] - med) / mad
            zh = (hist - med) / mad
            cov = np.atleast_2d(np.cov(zh.T)) + np.eye(len(FEATURES)) * 1e-3
            try:
                inv = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                inv = np.linalg.pinv(cov)
            score[i] = float(z @ inv @ z)
        elif method == "b1_univariate_pai_z":
            pai = df["PAI_equal"].values if "PAI_equal" in df else df["N_total"].values
            histv = pai[lo:i]
            med, mad = np.median(histv), np.median(np.abs(histv - np.median(histv))) * 1.4826 + EPS
            score[i] = abs((pai[i] - med) / mad)
        elif method == "b2_isoforest":
            from sklearn.ensemble import IsolationForest
            iso = IsolationForest(n_estimators=100, contamination="auto", random_state=42)
            iso.fit(hist)
            score[i] = -iso.score_samples(X[i].reshape(1, -1))[0]
        elif method == "b3_raw_maha":
            mu = hist.mean(axis=0)
            cov = np.atleast_2d(np.cov(hist.T)) + np.eye(len(FEATURES)) * 1e-3
            try:
                inv = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                inv = np.linalg.pinv(cov)
            d = X[i] - mu
            score[i] = float(d @ inv @ d)
        else:
            raise ValueError(method)

    flags = np.zeros(n, dtype=int)
    fallback = {"main_adaptive_maha": float(chi2.ppf(TAU / 100.0, df=5)),
                "b1_univariate_pai_z": 1.96,            # two-sided normal 95%
                "b2_isoforest": None,                    # no parametric form; skip
                "b3_raw_maha": float(chi2.ppf(TAU / 100.0, df=5))}[method]
    for i in range(n):
        if np.isnan(score[i]):
            continue
        lo = trailing_idx(dates, i)
        thr = emp_quantile_threshold(score[lo:i], TAU, fallback)
        if thr is None:
            continue
        flags[i] = int(score[i] > thr)
    return flags


def main():
    events = yaml.safe_load(EVENTS_YAML.read_text(encoding="utf-8"))["tier1_disruptions"]
    methods = ["main_adaptive_maha", "b1_univariate_pai_z", "b2_isoforest", "b3_raw_maha"]
    report = {"protocol": {"trailing_days": TRAILING_DAYS, "min_obs": TRAILING_MIN,
                           "tau": TAU, "threshold": "trailing empirical quantile"}}
    for method in methods:
        pooled_f = pooled_n = 0
        lts = {}
        for port in ("shanghai", "los_angeles", "rotterdam"):
            df = pd.read_csv(PAI / f"{port}_pai.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
            flags = score_port(df, method)
            pe = [e for e in events if e["port"] == port]

            def in_event(ts):
                return any(pd.Timestamp(e["event_date"]) + pd.Timedelta(days=WINDOW[0]) <= ts <=
                           pd.Timestamp(e["event_date"]) + pd.Timedelta(days=WINDOW[1]) for e in pe)

            quiet = ~df["date"].apply(in_event)
            valid = quiet & (flags >= 0)  # flags all 0/1; unscored rows counted as quiet non-flags? exclude unscored
            # count only scoreable quiet rows: those with trailing history (approximate via index >= TRAILING_MIN)
            scoreable = np.array([False] * len(df))
            dates = df["date"].values
            for i in range(len(df)):
                lo = trailing_idx(dates, i)
                scoreable[i] = len(df.iloc[lo:i]) >= TRAILING_MIN
            qm = quiet.values & scoreable
            pooled_f += int(flags[qm].sum())
            pooled_n += int(qm.sum())

            dates = df["date"]
            for e in pe:
                ev = pd.Timestamp(e["event_date"])
                upto = int(np.searchsorted(dates.values, np.datetime64(ev)))
                s = last_run_start(flags, upto)
                if s is None:
                    continue
                t_sat = pd.Timestamp(dates.iloc[s])
                t_pub = pd.Timestamp(e["public_recognition_date"])
                if ev - pd.Timedelta(days=90) <= t_sat:
                    lts[e["id"].split("_")[0]] = int((t_pub - t_sat).days)
        vals = list(lts.values())
        report[method] = {
            "quiet_flags": pooled_f, "quiet_obs": pooled_n,
            "quiet_rate_pct": round(pooled_f / pooled_n * 100, 1) if pooled_n else None,
            "in_rule_detections": len(vals),
            "lts": lts,
            "median_lt": float(np.median(vals)) if vals else None,
            "mean_lt": round(float(np.mean(vals)), 1) if vals else None,
        }
        print(method, "->", report[method])

    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("saved:", OUT)


if __name__ == "__main__":
    main()
