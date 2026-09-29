"""港口 GIS 几何加载与对齐到 EPSG:4326。

策略(按可靠性排序):
  方案 A(主):osmnx 从 OpenStreetMap 实时拉港口/码头/锚地多边形(无需手动下载)
  方案 B(备):加载本地 WPI/MarineRegions shapefile(若用户手动下载)
  方案 C(兜底):用 ports.yaml 的 bbox 作为简单矩形 ROI
  方案 D(增强):GEE Proba-V 全球陆表 → 海域掩膜(从 bbox 进一步剔除陆地)
"""
from __future__ import annotations
import sys
from pathlib import Path
import yaml

import geopandas as gpd
import pandas as pd
from shapely.geometry import box, Polygon, MultiPolygon

PROJ_ROOT = Path(__file__).resolve().parents[2]
GIS_DIR = PROJ_ROOT / "data" / "gis"
PORTS_YAML = PROJ_ROOT / "configs" / "ports.yaml"


def load_ports_config() -> dict:
    with open(PORTS_YAML, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================================
# 方案 C(兜底):bbox 矩形 ROI —— 总是可用
# ============================================================================

def get_port_bbox_polygon(port_key: str) -> Polygon:
    """从 ports.yaml 读 bbox,返回 shapely 矩形。最可靠的兜底。"""
    cfg = load_ports_config()
    w, s, e, n = cfg["ports"][port_key]["bbox"]
    return box(w, s, e, n)


def get_port_bbox_geodataframe(port_key: str) -> gpd.GeoDataFrame:
    """bbox 作为 GeoDataFrame(EPSG:4326)。"""
    poly = get_port_bbox_polygon(port_key)
    cfg = load_ports_config()
    return gpd.GeoDataFrame(
        {"port": [port_key], "name": [cfg["ports"][port_key]["name"]],
         "type": ["bbox"]},
        geometry=[poly],
        crs="EPSG:4326",
    )


# ============================================================================
# 方案 A(主):osmnx 拉 OSM 港口几何
# ============================================================================

def get_port_osm_geometries(port_key: str, timeout_sec: int = 30) -> gpd.GeoDataFrame:
    """用 osmnx 从 OSM 拉取港口相关几何(码头/船坞/泊位等)。
    
    Windows 注:signal.alarm 不可用,用 ThreadPoolExecutor + future.result(timeout) 实现超时。
    超时或失败时返回空 GDF,由调用方走兜底方案。

    OSM tags:
      - waterfront       # 滨水区
      - waterway=dock    # 船坞
      - man_made=piers   # 码头
      - landuse=port     # 港口用地
      - man_made=harbour # 港湾
    """
    from concurrent.futures import ThreadPoolExecutor, TimeoutExpiredError

    try:
        import osmnx as ox
    except ImportError as e:
        raise RuntimeError(f"osmnx not installed: {e}")

    cfg = load_ports_config()
    w, s, e, n = cfg["ports"][port_key]["bbox"]

    tags = {
        "waterway": ["dock"],
        "man_made": ["pier", "harbour", "breakwater"],
        "landuse": ["port"],   # 不要 industrial(太宽泛,数据量爆炸)
        "waterfront": True,
    }

    def _fetch():
        return ox.features_from_bbox(bbox=(w, s, e, n), tags=tags)

    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(_fetch)
            gdf = future.result(timeout=timeout_sec)
    except TimeoutExpiredError:
        print(f"[WARN] osmnx {port_key}: timeout after {timeout_sec}s", file=sys.stderr)
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    except Exception as e:
        print(f"[WARN] osmnx {port_key}: {e}", file=sys.stderr)
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    gdf = gdf.reset_index(drop=True).to_crs(epsg=4326)
    print(f"[OK] osmnx {port_key}: {len(gdf)} OSM features", file=sys.stderr)
    return gdf


# ============================================================================
# 方案 B(备):本地 shapefile 加载(若用户手动下载)
# ============================================================================

def load_wpi_ports() -> gpd.GeoDataFrame:
    """加载 World Port Index 港口点位。需用户先下载。"""
    candidates = list((GIS_DIR / "WPI").glob("*.shp")) if (GIS_DIR / "WPI").exists() else []
    if not candidates:
        raise FileNotFoundError(
            f"WPI shapefile not found. Download from https://msi.nga.mil/ProductDownloads "
            f"and extract to {GIS_DIR / 'WPI'}"
        )
    return gpd.read_file(candidates[0]).to_crs(epsg=4326)


def load_marine_regions_ports() -> gpd.GeoDataFrame:
    """加载 MarineRegions 港口/锚地多边形。需用户先下载。"""
    candidates = list((GIS_DIR / "MarineRegions").glob("*.shp")) if (GIS_DIR / "MarineRegions").exists() else []
    if not candidates:
        raise FileNotFoundError(
            f"MarineRegions shapefile not found. Download from https://www.marineregions.org/downloads.php "
            f"and extract to {GIS_DIR / 'MarineRegions'}"
        )
    return gpd.read_file(candidates[0]).to_crs(epsg=4326)


# ============================================================================
# 统一接口:按优先级尝试三方案
# ============================================================================

def get_port_polygon(port_key: str, prefer: str = "auto") -> gpd.GeoDataFrame:
    """统一接口:获取港口 ROI 几何。
    
    Args:
        port_key: shanghai / los_angeles / rotterdam
        prefer: "auto"(按 A→B→C 尝试)/ "osm" / "shapefile" / "bbox"
    
    Returns:
        GeoDataFrame with at least one Polygon/MultiPolygon in EPSG:4326
    """
    if prefer in ("auto", "osm"):
        try:
            gdf = get_port_osm_geometries(port_key)
            if len(gdf) > 0:
                return gdf
        except Exception as e:
            print(f"[INFO] osmnx failed for {port_key}: {e}", file=sys.stderr)

    if prefer in ("auto", "shapefile"):
        try:
            mr = load_marine_regions_ports()
            name_col = "name" if "name" in mr.columns else mr.columns[0]
            keywords = {
                "shanghai": ["SHANGHAI", "NINGBO", "ZHANJIAGANG", "BEILUN"],
                "los_angeles": ["LOS ANGELES", "LONG BEACH", "SAN PEDRO"],
                "rotterdam": ["ROTTERDAM", "EUROPOORT", "MAASVLAKTE"],
            }.get(port_key, [port_key.upper()])
            mask = mr[name_col].str.contains("|".join(keywords), case=False, na=False)
            if mask.sum() > 0:
                return mr[mask]
        except FileNotFoundError as e:
            print(f"[INFO] shapefile not available: {e}", file=sys.stderr)

    # 兜底:bbox 矩形
    return get_port_bbox_geodataframe(port_key)


if __name__ == "__main__":
    # 主入口:测试三方案可用性
    cfg = load_ports_config()
    print("=" * 60, file=sys.stderr)
    for port_key in cfg["ports"]:
        print(f"\n[{port_key}]", file=sys.stderr)
        # bbox(总可用)
        bbox_gdf = get_port_bbox_geodataframe(port_key)
        print(f"  bbox: {len(bbox_gdf)} polygon", file=sys.stderr)
        # osmnx
        try:
            osm_gdf = get_port_osm_geometries(port_key)
            print(f"  osm:  {len(osm_gdf)} features", file=sys.stderr)
        except Exception as e:
            print(f"  osm:  FAILED ({e})", file=sys.stderr)
        # 统一接口
        final = get_port_polygon(port_key, prefer="auto")
        print(f"  auto: {len(final)} features (chosen)", file=sys.stderr)
    print("\n" + "=" * 60, file=sys.stderr)


# ============================================================================
# 方案 D(增强):GEE 海域掩膜 —— 从 bbox 进一步剔除陆地
# ============================================================================

def get_port_sea_mask_ee(port_key: str, water_threshold: int = 75) -> "ee.Image":
    """GEE 海域掩膜:返回 ee.Image(1=海, 0=陆),裁剪到港口 bbox。

    使用 COPERNICUS Proba-V C3 Global Land Cover 的 water-permanent-coverfraction。
    bbox 是矩形(含大量陆地),海域掩膜进一步把"船会停的地方"圈出来。

    Args:
        port_key: shanghai / los_angeles / rotterdam
        water_threshold: 0-100,水比例 >= 阈值视为水域。默认 75(保守)。
                         降到 50 可纳入更多季节性水域;提到 90 只保留稳定海洋。

    Returns:
        ee.Image: 单波段 'sea_mask',值 1(海)/0(陆),裁剪到港口 bbox
    """
    import ee  # 延迟导入:无 GEE 环境也能用其他方案
    from ..download.ee_auth import init_ee
    from ..download.sentinel import get_port_roi

    init_ee()
    roi = get_port_roi(port_key, load_ports_config())

    # 取最新一期 Proba-V C3 Global
    col = ee.ImageCollection("COPERNICUS/Landcover/100m/Proba-V-C3/Global")
    latest = col.sort("system:time_start", False).first()
    water_frac = latest.select("water-permanent-coverfraction")  # 0-100 连续值
    sea = water_frac.gte(water_threshold).rename("sea_mask")
    return sea.clip(roi)


def get_port_sea_coverage(port_key: str, water_threshold: int = 75) -> dict:
    """统计某港口 bbox 内海域占比(用于验证 bbox 大小是否合理)。

    Returns:
        {port, sea_px, land_px, sea_ratio}
    """
    import ee
    sea = get_port_sea_mask_ee(port_key, water_threshold)
    roi = get_port_roi_geography(port_key)
    stats = sea.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(),
        geometry=roi, scale=100, maxPixels=1e8,
    ).getInfo()
    hist = stats.get("sea_mask", {})
    n_sea = hist.get("1", 0)
    n_land = hist.get("0", 0)
    total = n_sea + n_land
    return {
        "port": port_key,
        "sea_px": n_sea, "land_px": n_land,
        "sea_ratio": n_sea / total if total > 0 else 0,
    }


def get_port_roi_geography(port_key: str):
    """辅助:返回港口 bbox 的 ee.Geometry。"""
    import ee
    from ..download.sentinel import get_port_roi
    return get_port_roi(port_key, load_ports_config())
