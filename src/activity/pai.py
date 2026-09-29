"""Port Activity Index (PAI) composite + Congestion Pressure (CP).

三方案(响应审稿消融):
  - equal:  等权 w_i = 0.2
  - pca:    PCA 第一主成分(数据驱动,符号对齐 N_total)
  - regres: 与吞吐量回归拟合(非负最小二乘)

外加:
  - CP (Congestion Pressure) = N_anchor / N_berth  (v3 新增)
  - aggregate_port_activity(): 把检测框时间序列聚合为指标 DataFrame

论文 §2.3 Port Activity Representation 独立章节。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 五个基础活动指标列名
COLUMNS = ["N_total", "N_anchor", "N_berth", "delta_arrive", "delta_depart"]


def compute_pai_equal_weight(df: pd.DataFrame) -> np.ndarray:
    """等权 PAI:w_i = 0.2 for each of 5 metrics。
    
    最简单、最透明、可解释。作为消融基线。
    """
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}. Required: {COLUMNS}")
    return df[COLUMNS].sum(axis=1).values * 0.2


def compute_pai_pca(df: pd.DataFrame) -> np.ndarray:
    """PCA 第一主成分 PAI(数据驱动权重)。
    
    标准化后取 PC1,符号对齐使与 N_total 正相关。
    若样本 < 10 或所有值相同,降级为等权。
    """
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}. Required: {COLUMNS}")
    
    X = df[COLUMNS].values
    # 降级条件:样本太少或无方差
    if len(X) < 10 or np.allclose(X.std(axis=0), 0):
        return compute_pai_equal_weight(df)
    
    X_std = StandardScaler().fit_transform(X)
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(X_std).flatten()
    # 符号对齐:使 PAI 与 N_total 正相关(PAI 越大越繁忙)
    if np.corrcoef(pc1, df["N_total"].values)[0, 1] < 0:
        pc1 = -pc1
    return pc1


def compute_pai_regression(df: pd.DataFrame, throughput: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """与吞吐量回归拟合权重(非负最小二乘)。
    
    "金标准"——权重由数据驱动且与外部 ground truth 对齐。
    Returns:
        (pai, weights): PAI 值数组 + 5 维权重向量(非负)
    """
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}. Required: {COLUMNS}")
    if len(throughput) != len(df):
        raise ValueError(f"Length mismatch: df={len(df)}, throughput={len(throughput)}")
    
    from scipy.optimize import nnls
    X = df[COLUMNS].values
    # 非负最小二乘:保证权重可解释(每项贡献 >= 0)
    weights, residual = nnls(X, throughput)
    pai = X @ weights
    return pai, weights


def compute_congestion_pressure(n_anchor: float, n_berth: float) -> float:
    """Congestion Pressure (CP) = N_anchor / N_berth。
    
    v3 新增指标(响应审稿)。比值型,直接反映"船等泊位"压力。
    - CP 高:锚地船多但泊位少 → 拥堵
    - CP 低:泊位船多或锚地空 → 流畅
    
    防除零:n_berth=0 时,若 n_anchor>0 返回 inf,否则 0。
    """
    if n_berth == 0:
        return float("inf") if n_anchor > 0 else 0.0
    return n_anchor / n_berth


def compute_congestion_pressure_series(n_anchor: np.ndarray, n_berth: np.ndarray) -> np.ndarray:
    """CP 时间序列版本。返回 float 数组(inf 处理为 np.inf)。"""
    n_anchor = np.asarray(n_anchor, dtype=float)
    n_berth = np.asarray(n_berth, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        cp = np.where(n_berth > 0, n_anchor / n_berth,
                      np.where(n_anchor > 0, np.inf, 0.0))
    return cp


def aggregate_port_activity(detections: list[dict], anchor_polygons=None,
                             berth_polygons=None) -> pd.DataFrame:
    """把每帧检测框序列聚合为时间序列活动指标 DataFrame。
    
    Args:
        detections: [{date, boxes_xyxy, centers_lonlat, ...}, ...]
        anchor_polygons: 锚地多边形 list(shapely Polygons,可选)
        berth_polygons: 泊位多边形 list(shapely Polygons,可选)
    
    Returns:
        DataFrame[date, N_total, N_anchor, N_berth, delta_arrive, delta_depart, CP]
        按 date 升序。
    
    Note:若无 GIS 多边形,N_anchor/N_berth 用占位估计(后续 Task 3.1 接入真实 GIS)。
    """
    rows = []
    for det in detections:
        date = det.get("date") or det.get("file", "").replace(".tif", "")
        centers = det.get("centers_lonlat", [])
        n_total = len(centers) if centers else det.get("n_ships", 0)
        
        # 区域语义判定(优先用 GIS 多边形,否则占位)
        if anchor_polygons is not None and berth_polygons is not None and centers:
            from shapely.geometry import Point
            pts = [Point(c[0], c[1]) for c in centers]
            n_anchor = sum(1 for p in pts if any(poly.contains(p) for poly in anchor_polygons))
            n_berth = sum(1 for p in pts if any(poly.contains(p) for poly in berth_polygons))
        else:
            # 占位:30% 锚地,40% 泊位,30% 其他(后续 Task 3.1 替换)
            n_anchor = int(n_total * 0.3)
            n_berth = int(n_total * 0.4)
        
        rows.append({"date": date, "N_total": n_total, "N_anchor": n_anchor, "N_berth": n_berth})
    
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    # delta_arrive / delta_depart 从 N_total diff 推导
    diff = df["N_total"].diff().fillna(0)
    df["delta_arrive"] = diff.clip(lower=0)
    df["delta_depart"] = (-diff).clip(lower=0)
    # CP
    df["CP"] = compute_congestion_pressure_series(df["N_anchor"].values, df["N_berth"].values)
    return df
