# -*- coding: utf-8 -*-
"""Adaptive rolling-baseline scorer (method revision, round 2).

Design (fixed before evaluating event outcomes; parameters selected ONLY on
quiet-period calibration):
  - Features (5, stationary by construction): robust rolling z-scores of
    N_total, N_anchor, N_berth, delta_arrive, PCI   (delta_depart dropped:
    exactly collinear)
  - z_f(t) = (x_f(t) - med_trailing) / (1.4826 * MAD_trailing + eps)
    trailing window = past TRAILING_DAYS calendar days, min TRAILING_MIN obs
  - D2(t) = Mahalanobis on the z-vector with trailing covariance (shrunk,
    pseudo-inverse fallback)
  - Threshold = empirical tau-quantile of the trailing window's OWN D2 values
    (chi2_5 quantile reported as comparison)

Evaluation output: quiet FPR + per-event LT + inclusion-rule verdicts +
threshold sensitivity, on the FULL corpus (event + quiet windows).
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
OUT = PROJ / "experiments" / "exp5_lead_time" / "adaptive_results.json"

FEATURES = ["N_total", "N_anchor", "N_berth", "delta_arrive", "PCI"]
TRAILING_DAYS = 120
TRAILING_MIN = 12
TAUS = (95.0, 97.5, 99.0)
WINDOW = (-90, 60)  # event window for inclusion rule / quiet definition
EPS = 1e-6


def score_series(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling-baseline D2 scoring. Returns df with columns z-features, d2,
    thr_emp_<tau>, flag_<tau> for every obs with sufficient trailing history."""
    f = df[["N_total", "N_anchor", "N_berth", "delta_arrive"]].copy()
    f["PCI"] = compute_pci(df["N_total"], baseline_window=30).values
    X = f[FEATURES].values.astype(float)
    dates = df["date"].values
    n = len(df)
    d2 = np.full(n, np.nan)

    for i in range(n):
        t = dates[i]
        lo = np.searchsorted(dates, np.datetime64(pd.Timestamp(t) - pd.Timedelta(days=TRAILING_DAYS)))
        hist = X[lo:i]
        if len(hist) < TRAILING_MIN:
            continue
        med = np.median(hist, axis=0)
        mad = np.median(np.abs(hist - med), axis=0) * 1.4826 + EPS
        z = (X[i] - med) / mad
        cov = np.atleast_2d(np.cov(((hist - med) / mad).T))
        cov += np.eye(len(FEATURES)) * 1e-3
        try:
            inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(cov)
        d2[i] = float(z @ inv @ z)

    out = df.copy()
    out["d2"] = d2
    # thresholds: empirical tau-quantile of the trailing window's own D2
    for tau in TAUS:
        thr_col = np.full(n, np.nan)
        for i in range(n):
            if np.isnan(d2[i]):
                continue
            t = dates[i]
            lo = np.searchsorted(dates, np.datetime64(pd.Timestamp(t) - pd.Timedelta(days=TRAILING_DAYS)))
            hist_d2 = d2[lo:i]
            hist_d2 = hist_d2[~np.isnan(hist_d2)]
            if len(hist_d2) >= TRAILING_MIN:
                thr_col[i] = np.quantile(hist_d2, tau / 100.0)
            else:
                thr_col[i] = chi2.ppf(tau / 100.0, df=len(FEATURES))
        out[f"flag_{tau}"] = (d2 > thr_col).astype(int)
    return out


def last_run_start(flags: np.ndarray, upto_idx: int) -> int | None:
    """First index of the last contiguous True run ending at/before upto_idx."""
    idx = [i for i in range(upto_idx) if flags[i] == 1]
    if not idx:
        return None
    start = idx[-1]
    while start - 1 in idx:
        start -= 1
    return start


def main():
    events = yaml.safe_load(EVENTS_YAML.read_text(encoding="utf-8"))["tier1_disruptions"]
    report = {"config": {"trailing_days": TRAILING_DAYS, "min_obs": TRAILING_MIN,
                         "features": FEATURES, "taus": TAUS}}
    pooled = {str(t): [0, 0] for t in TAUS}
    per_event = []

    for port in ("shanghai", "los_angeles", "rotterdam"):
        df = pd.read_csv(PAI / f"{port}_pai.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
        scored = score_series(df)
        pe = [e for e in events if e["port"] == port]

        def in_event(ts):
            return any(pd.Timestamp(e["event_date"]) + pd.Timedelta(days=WINDOW[0]) <= ts <=
                       pd.Timestamp(e["event_date"]) + pd.Timedelta(days=WINDOW[1]) for e in pe)

        quiet_mask = ~scored["date"].apply(in_event) & scored["d2"].notna()
        port_rep = {"n_obs": int(len(df)), "n_scored": int(scored["d2"].notna().sum()),
                    "n_quiet_scored": int(quiet_mask.sum()), "taus": {}}
        for tau in TAUS:
            qf = int(scored.loc[quiet_mask, f"flag_{tau}"].sum())
            qn = int(quiet_mask.sum())
            port_rep["taus"][str(tau)] = {"quiet_flags": qf, "quiet_obs": qn,
                                          "quiet_rate": qf / qn if qn else None}
            pooled[str(tau)][0] += qf
            pooled[str(tau)][1] += qn
        report[port] = port_rep

        # per-event LT under each tau (full-series last-run search, then
        # inclusion rule: t_sat must lie in the event's own -90d window)
        dates = scored["date"]
        for tau in TAUS:
            flags = scored[f"flag_{tau}"].values
            for e in pe:
                ev = pd.Timestamp(e["event_date"])
                upto = int(np.searchsorted(dates.values, np.datetime64(ev)))
                s = last_run_start(flags, upto)
                if s is None:
                    per_event.append({"event": e["id"], "tau": tau, "port": port,
                                      "status": "no_detection"})
                    continue
                t_sat = pd.Timestamp(dates.iloc[s])
                t_pub = pd.Timestamp(e["public_recognition_date"])
                in_rule = ev - pd.Timedelta(days=90) <= t_sat
                per_event.append({"event": e["id"], "tau": tau, "port": port,
                                  "t_satellite": str(t_sat.date()),
                                  "lt": int((t_pub - t_sat).days),
                                  "in_window_rule": bool(in_rule)})

    report["pooled_quiet"] = {t: {"flags": f, "obs": n, "rate": f / n if n else None}
                              for t, (f, n) in pooled.items()}
    report["per_event"] = per_event
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("pooled quiet rate:", {t: round(f / n * 100, 1) for t, (f, n) in pooled.items() if n})
    for e in per_event:
        if e["tau"] == 95.0:
            print(e)


if __name__ == "__main__":
    main()
