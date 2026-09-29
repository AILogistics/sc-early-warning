# -*- coding: utf-8 -*-
"""Out-of-sample quiet-period FPR under the paper's ORIGINAL (events-only) scorer.

Fit mu/Sigma (+PCI) exactly as the reported pipeline did -- on the events-only
PAI series (backed up as *_events_only_backup.csv) -- then SCORE the newly
acquired quiet-period observations (rows of the full series that fall outside
every event window) with those frozen parameters. This yields a measured
quiet-period flag rate that is coherent with the paper's headline analysis.

Output: experiments/exp5_lead_time/quiet_fpr_outofsample.json
"""
from __future__ import annotations
import sys, json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import chi2

PROJ = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ))
from src.activity.mahalanobis import mahalanobis_distance
from src.warning.lead_time import compute_pci

PAI = PROJ / "experiments" / "exp3_activity" / "pai_series"
EVENTS_YAML = PROJ / "configs" / "events.yaml"
OUT = PROJ / "experiments" / "exp5_lead_time" / "quiet_fpr_outofsample.json"

ACTIVITY_COLUMNS = ["N_total", "N_anchor", "N_berth", "delta_arrive", "delta_depart"]
TAUS = (95.0, 97.5, 99.0)
WINDOW = (-90, 60)

events = yaml.safe_load(EVENTS_YAML.read_text(encoding="utf-8"))["tier1_disruptions"]


def feats(df: pd.DataFrame) -> np.ndarray:
    f = df[ACTIVITY_COLUMNS].copy()
    f["PCI"] = compute_pci(df["N_total"], baseline_window=30).values
    return f.values.astype(float)


def main():
    report = {}
    for port in ("shanghai", "los_angeles", "rotterdam"):
        ev_df = pd.read_csv(PAI / f"{port}_pai_events_only_backup.csv", parse_dates=["date"]).sort_values("date")
        full = pd.read_csv(PAI / f"{port}_pai.csv", parse_dates=["date"]).sort_values("date")

        # frozen fit from the events-only corpus (identical to reported pipeline)
        X_ev = feats(ev_df)
        mu = X_ev.mean(axis=0)
        cov = np.cov(X_ev.T)
        cov += np.eye(cov.shape[0]) * 1e-6
        try:
            inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(cov)

        # quiet rows: observations outside every event window of this port
        pe = [e for e in events if e["port"] == port]
        def in_event(ts):
            return any(pd.Timestamp(e["event_date"]) + pd.Timedelta(days=WINDOW[0]) <= ts <=
                       pd.Timestamp(e["event_date"]) + pd.Timedelta(days=WINDOW[1]) for e in pe)
        quiet_mask = ~full["date"].apply(in_event)
        quiet = full[quiet_mask].reset_index(drop=True)
        # PCI needs the full series rolling context: compute on full, take quiet rows
        f_full = full[ACTIVITY_COLUMNS].copy()
        f_full["PCI"] = compute_pci(full["N_total"], baseline_window=30).values
        X_q = f_full[quiet_mask].values.astype(float)

        d2 = np.einsum("ij,jk,ik->i", X_q - mu, inv, X_q - mu)
        entry = {"n_quiet_obs": int(len(X_q)),
                 "window": [str(quiet["date"].min().date()), str(quiet["date"].max().date())],
                 "taus": {}}
        for tau in TAUS:
            thr = float(chi2.ppf(tau / 100.0, df=X_ev.shape[1]))
            flags = (d2 > thr).astype(int)
            runs, cur = [], 0
            for fl in flags:
                if fl:
                    cur += 1
                elif cur:
                    runs.append(cur); cur = 0
            if cur:
                runs.append(cur)
            entry["taus"][str(tau)] = {
                "threshold": thr,
                "quiet_flags": int(flags.sum()),
                "quiet_rate": float(flags.mean()),
                "nominal": (100 - tau) / 100.0,
                "runs": len(runs),
                "max_run": max(runs) if runs else 0,
            }
        report[port] = entry

    tot = {t: {"flags": sum(r["taus"][t]["quiet_flags"] for r in report.values()),
               "obs": sum(r["n_quiet_obs"] for r in report.values())} for t in ("95.0", "97.5", "99.0")}
    summary = {t: round(v["flags"] / v["obs"] * 100, 1) for t, v in tot.items()}
    OUT.write_text(json.dumps({"per_port": report, "pooled_rate_pct": summary}, indent=2), encoding="utf-8")
    print(json.dumps({"pooled_quiet_rate_pct@95/97.5/99": summary}, indent=1))
    for p, r in report.items():
        t = r["taus"]["95.0"]
        print(f"{p}: {t['quiet_flags']}/{r['n_quiet_obs']} = {t['quiet_rate']*100:.1f}% @95, max_run={t['max_run']}")


if __name__ == "__main__":
    main()
