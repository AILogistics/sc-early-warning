# Sentinel Port Activity Index for Maritime Supply Chain Early Warning

Code and derived data for the study:

> **A Sentinel-1/2-Derived Port Activity Index for Early Warning of Maritime Supply Chain Disruptions**
> Xiaokang Qin, Hongsheng Xu, Zhaodi Wang (Luoyang Normal University)
> *(manuscript submitted to IEEE JSTARS; citation entry will be added upon acceptance)*

The pipeline converts free Copernicus Sentinel-1/2 imagery (via Google Earth Engine) into a
per-port **Port Activity Index (PAI)**, scores it with an **adaptive rolling-baseline anomaly
scorer**, and quantifies anticipation as a **Lead Time** metric against publicly recognised
disruption events — together with a **measured quiet-period false-alarm rate** on explicitly
event-free observation windows.

## Repository structure

```
configs/                 port ROIs (ports.yaml) and the curated Tier-1 event timeline (events.yaml)
src/                     pipeline modules
  download/              GEE acquisition (event windows + quiet windows), auth
  detection/             AMFDet-Lite loading + batch detection over GeoTIFF chips
  activity/              PAI construction (+ SARIMA/z-score baselines kept for reference)
  warning/               adaptive & static scorers, lead-time algorithm, bootstrap CIs
data/economic/           monthly port-throughput series (LA / Rotterdam / Shanghai) used for
                         the PAI-vs-throughput validation
experiments/
  exp3_activity/detections/   per-image detection outputs (counts, boxes, zone centres)
  exp3_activity/pai_series/   per-port PAI time series (event + quiet windows)
  exp5_lead_time/             lead-time results, measured quiet-FPR (static & adaptive),
                              baselines comparison, adaptive scorer results
  exp8_granger/               cross-port Granger overlap statistics (negative result)
model/                   trained AMFDet-Lite weights (state_dict.pt, 2.36 M params)
third_party/ultralytics_modules/  LAE / SPPF-SENetV2 / EUCB custom modules needed to load
                                  the weights inside the Ultralytics YOLOv8 framework
results/                 acquisition statistics (event-phase and quiet-phase downloads)
```

Root-level scripts (run in order to reproduce the study):

| Script | Purpose |
|---|---|
| `quiet_acquisition.py` | (optional) download the five quiet-window image sets from GEE |
| `make_clean_overlay.py` | regenerate the detection overlay figure from stored boxes |
| `build_pai_series.py` | rebuild PAI time series from detection JSONs |
| `adaptive_scoring.py` | adaptive 120-day rolling-baseline scorer: quiet FPR + per-event lead times |
| `quiet_fpr_outofsample.py` | static-design negative control: frozen-fit quiet FPR (19.5%) |
| `baselines_experiment.py` | univariate-z / Isolation-Forest / raw-Mahalanobis comparison |
| `persistence_analysis.py` | persistence (consecutive-flag) operating points |
| `run_lead_time_eval.py` | legacy static-design lead-time evaluation |

## Reproducing

```bash
pip install -r requirements.txt

# 1. PAI series are already in experiments/exp3_activity/pai_series/.
#    To rebuild from raw imagery you additionally need Google Earth Engine access
#    (see src/download/ee_auth.py) and ~660 GeoTIFFs (event + quiet windows);
#    the GEE collections are COPERNICUS/S1_GRD and COPERNICUS/S2_SR_HARMONIZED.

python adaptive_scoring.py          # Table: quiet FPR 7.3% pooled; per-event LTs
python baselines_experiment.py      # Table: baseline scorer comparison
python persistence_analysis.py      # k=1 -> 7.3% FPR / 5 events; k>=2 -> 0% FPR / 1 event
```

Detection re-runs require the custom YOLOv8 modules: copy the three files from
`third_party/ultralytics_modules/` into `ultralytics/nn/Addmodules/` of an Ultralytics
YOLOv8 installation and register them in its `nn/tasks.py` (module names: `LAE`,
`SPPFSENetV2`, `EUCB`), then load `model/state_dict.pt`.

## Scope notes (honesty statements from the paper)

- Imagery is aggregated to 250 m; the detector runs in **aggregate activity mode** (not
  per-vessel counting). No labelled ground truth exists at that scale; validation is at
  the count level (PAI vs monthly throughput).
- Quiet windows used for the measured false-alarm rate: Rotterdam 2021-06..10 and
  2022-06..10, Shanghai 2022-07..12 and 2023-02..06, Los Angeles 2022-01..06 (295 images).
- The static in-sample scorer is retained as a **measured negative control** (19.5% quiet
  flag rate at a nominal 5% threshold); the adaptive design measures 7.3% pooled.
- The study is retrospective; all reported lead times are upper bounds.

## License

Code: MIT (see LICENSE). Derived data (event timeline, detections, PAI series, throughput
tables): released under CC BY 4.0. Sentinel imagery: Copernicus open licence; HRSID/SSDD
benchmarks belong to their respective authors and are not redistributed here.
