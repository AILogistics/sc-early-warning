"""Quiet-period false-alarm analysis (revision G1/G6, Round 1 review).

Uses the EXACT same scoring path that produced the paper's LT results:
  features = [N_total, N_anchor, N_berth, delta_arrive, delta_depart, PCI]
  D^2      = src.activity.mahalanobis.mahalanobis_distance (full-series mean/cov + eps*I)
  flags    = D^2 > chi2.ppf(tau, df=6)

Reports, per port and tau in {95, 97.5, 99}:
  - total observations, total flags, empirical flag rate (calibration check)
  - flags inside vs outside the -90/+60 day event windows
  - number and max length of contiguous anomaly runs in quiet periods
Output: experiments/exp5_lead_time/quiet_period_fpr.json
"""
from __future__ import annotations
import sys, json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import chi2

PROJ_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ_ROOT))

from src.activity.mahalanobis import mahalanobis_distance
from src.warning.lead_time import compute_pci

PAI_DIR = PROJ_ROOT / "experiments" / "exp3_activity" / "pai_series"
EVENTS_YAML = PROJ_ROOT / "configs" / "events.yaml"
OUT_JSON = PROJ_ROOT / "experiments" / "exp5_lead_time" / "quiet_period_fpr.json"

ACTIVITY_COLUMNS = ["N_total", "N_anchor", "N_berth", "delta_arrive", "delta_depart"]
TAUS = (95.0, 97.5, 99.0)
WINDOW = (-90, 60)  # days around event_date, per paper section 4.1


def contiguous_runs(flags: np.ndarray) -> list[int]:
    runs, cur = [], 0
    for f in flags:
        if f:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return runs


def main():
    events = yaml.safe_load(EVENTS_YAML.read_text(encoding="utf-8"))["tier1_disruptions"]
    report = {}
    for port in ("shanghai", "los_angeles", "rotterdam"):
        df = pd.read_csv(PAI_DIR / f"{port}_pai.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)

        feat = df[ACTIVITY_COLUMNS].copy()
        feat["PCI"] = compute_pci(df["N_total"], baseline_window=30).values
        d2 = mahalanobis_distance(feat)
        df["d2"] = d2

        # event windows for this port (primary 5-event set excludes E4; but E4's
        # window is still an *event* window for FPR purposes -> include all 6)
        port_events = [e for e in events if e["port"] == port]
        ev_starts = [pd.Timestamp(e["event_date"]) + pd.Timedelta(days=WINDOW[0]) for e in port_events]
        ev_ends = [pd.Timestamp(e["event_date"]) + pd.Timedelta(days=WINDOW[1]) for e in port_events]

        def in_event(ts: pd.Timestamp) -> bool:
            return any(s <= ts <= en for s, en in zip(ev_starts, ev_ends))

        df["in_event"] = df["date"].apply(in_event)

        port_rep = {"n_obs": int(len(df)),
                    "n_event_obs": int(df["in_event"].sum()),
                    "n_quiet_obs": int((~df["in_event"]).sum()),
                    "date_min": str(df["date"].min().date()),
                    "date_max": str(df["date"].max().date()),
                    "events_at_port": [e["id"] for e in port_events],
                    "taus": {}}
        quiet = df[~df["in_event"]]
        for tau in TAUS:
            thr = float(chi2.ppf(tau / 100.0, df=feat.shape[1]))
            flags = (d2 > thr).astype(int)
            quiet_flags = flags[~df["in_event"].values]
            runs = contiguous_runs(quiet_flags)
            port_rep["taus"][str(tau)] = {
                "threshold": thr,
                "flag_rate_full_series": float(flags.mean()),
                "quiet_flag_count": int(quiet_flags.sum()),
                "quiet_obs": int(len(quiet_flags)),
                "quiet_flag_rate": float(quiet_flags.mean()) if len(quiet_flags) else None,
                "quiet_runs": len(runs),
                "quiet_max_run_len": max(runs) if runs else 0,
                "nominal_rate": (100 - tau) / 100.0,
            }
        report[port] = port_rep

        # per-event pre-event window flag structure (null-detection context)
        for e in port_events:
            ev = pd.Timestamp(e["event_date"])
            pre = df[(df["date"] < ev) & (df["date"] >= ev + pd.Timedelta(days=WINDOW[0]))]
            post = df[(df["date"] >= ev) & (df["date"] <= ev + pd.Timedelta(days=WINDOW[1]))]
            for tau in TAUS:
                thr = float(chi2.ppf(tau / 100.0, df=feat.shape[1]))
                pre_flags = (pre["d2"].values > thr).astype(int)
                post_flags = (post["d2"].values > thr).astype(int)
                p_emp = float((d2 > thr).mean())  # port-level empirical flag density
                n_pre = int(len(pre_flags))
                p_null = float(1 - (1 - p_emp) ** n_pre) if n_pre else None
                e.setdefault("taus", {})[str(tau)] = {
                    "pre_obs": n_pre,
                    "pre_flags": int(pre_flags.sum()),
                    "pre_runs": len(contiguous_runs(pre_flags)),
                    "pre_max_run": max(contiguous_runs(pre_flags), default=0),
                    "post_flags": int(post_flags.sum()),
                    "p_at_least_one_flag_null": p_null,
                }
            report[port].setdefault("per_event", {})[e["id"]] = e["taus"]

    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
