# -*- coding: utf-8 -*-
"""Persistence-rule analysis: alert only when >= k consecutive flags."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

PROJ = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ))
import adaptive_scoring as a

events = yaml.safe_load((PROJ / "configs" / "events.yaml").read_text(encoding="utf-8"))["tier1_disruptions"]

for k in (2, 3):
    pooled_f = pooled_n = 0
    lts = {}
    for port in ("shanghai", "los_angeles", "rotterdam"):
        df = pd.read_csv(PROJ / f"experiments/exp3_activity/pai_series/{port}_pai.csv",
                         parse_dates=["date"]).sort_values("date").reset_index(drop=True)
        scored = a.score_series(df)
        flags = scored["flag_95.0"].values
        pe = [e for e in events if e["port"] == port]
        quiet = ~scored["date"].apply(lambda ts: any(
            pd.Timestamp(e["event_date"]) - pd.Timedelta(days=90) <= ts <=
            pd.Timestamp(e["event_date"]) + pd.Timedelta(days=60) for e in pe)) & scored["d2"].notna()

        # alert obs = members of any run >= k
        alerts = np.zeros(len(flags), dtype=int)
        run = 0
        starts = []
        for i, f in enumerate(flags):
            run = run + 1 if f else 0
            if run >= k:
                starts.append(i - k + 1)
                for j in range(i - k + 1, i + 1):
                    alerts[j] = 1
        pooled_f += int(alerts[quiet.values].sum())
        pooled_n += int(quiet.sum())

        dates = scored["date"]
        for e in pe:
            ev = pd.Timestamp(e["event_date"])
            upto = int(np.searchsorted(dates.values, np.datetime64(ev)))
            s = a.last_run_start(alerts, upto)
            if s is not None:
                t_sat = pd.Timestamp(dates.iloc[s])
                t_pub = pd.Timestamp(e["public_recognition_date"])
                if ev - pd.Timedelta(days=90) <= t_sat:
                    lts[e["id"].split("_")[0]] = int((t_pub - t_sat).days)
    print(f"persistence>={k}: quiet alert-obs {pooled_f}/{pooled_n} = {pooled_f/pooled_n*100:.1f}% "
          f"| detections {len(lts)}: {lts}")
