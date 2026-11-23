# -*- coding: utf-8 -*-
"""项目公共配置：路径、研究区预设、五大洋掩膜元数据与统计口径常量。"""

from pathlib import Path

# 目录结构
ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
FIG_DIR = ROOT / "figures"
ZONAL_CSV = DATA_DIR / "zonal_mean.csv"

# 全球海洋新产物路径契约
GLOBAL_RAW_DIR = DATA_DIR / "raw_global"
GLOBAL_ZONAL_CSV = DATA_DIR / "zonal_mean_global.csv"
BASINS_ZONAL_CSV = DATA_DIR / "zonal_mean_basins.csv"
BASIN_MASK_NPZ = DATA_DIR / "basin_mask.npz"
CLIMATOLOGY_NPZ = DATA_DIR / "climatology_grid.npz"
SUMMARY_JSON = DATA_DIR / "summary_metrics.json"

# 数据源：NOAA OISST v2.1 日平均，0.25° 网格
DATASET_ID = "ncdcOisst21Agg"
BASE_URL = f"https://coastwatch.pfeg.noaa.gov/erddap/griddap/{DATASET_ID}.csv"

# 向后兼容的旧版印度洋研究区常量（供既有 plot.py / run_all.py 及回归测试使用）
LON_MIN, LON_MAX = 20.0, 120.0
LAT_MIN, LAT_MAX = -40.0, 30.0
N_LON = 401          # 20.125–120.125°E，0.25° 原生格点共 401 列
N_LAT = 281          # -39.875–30.125°N，0.25° 原生格点共 281 行
EXPECTED_LINES = 112_683  # 2 行表头 + 281*401 行数据

# 区域预设
REGION_PRESETS = {
    "global": {
        "name": "global",
        "label": "全球海洋",
        "lat_min": -90.0,
        "lat_max": 90.0,
        "lon_min": 0.0,
        "lon_max": 360.0,
        "query_lat_min": -89.875,
        "query_lat_max": 89.875,
        "query_lon_min": 0.125,
        "query_lon_max": 359.875,
        "n_lat": 720,
        "n_lon": 1440,
        "expected_lines": 1_036_802,  # 2 行表头 + 720*1440 行数据
        "raw_dir": GLOBAL_RAW_DIR,
        "out_csv": GLOBAL_ZONAL_CSV,
    },
    "indian_ocean": {
        "name": "indian_ocean",
        "label": "印度洋",
        "lat_min": LAT_MIN,
        "lat_max": LAT_MAX,
        "lon_min": LON_MIN,
        "lon_max": LON_MAX,
        "query_lat_min": LAT_MIN,
        "query_lat_max": LAT_MAX,
        "query_lon_min": LON_MIN,
        "query_lon_max": LON_MAX,
        "n_lat": N_LAT,
        "n_lon": N_LON,
        "expected_lines": EXPECTED_LINES,
        "raw_dir": RAW_DIR,
        "out_csv": ZONAL_CSV,
    },
}

# 五大洋互斥掩膜 ID 与元数据（0-360° 原生网格）
BASIN_IDS = {
    "land": 0,
    "pacific": 1,
    "atlantic": 2,
    "indian": 3,
    "arctic": 4,
    "southern": 5,
}

BASINS = {
    0: {"id": 0, "key": "land", "name_zh": "陆地", "name_en": "Land"},
    1: {"id": 1, "key": "pacific", "name_zh": "太平洋", "name_en": "Pacific Ocean"},
    2: {"id": 2, "key": "atlantic", "name_zh": "大西洋", "name_en": "Atlantic Ocean"},
    3: {"id": 3, "key": "indian", "name_zh": "印度洋", "name_en": "Indian Ocean"},
    4: {"id": 4, "key": "arctic", "name_zh": "北冰洋", "name_en": "Arctic Ocean"},
    5: {"id": 5, "key": "southern", "name_zh": "南大洋", "name_en": "Southern Ocean"},
}

BASIN_ORDER = ("global", "pacific", "atlantic", "indian", "arctic", "southern")

# 时间维度：2016–2020 年，每月 16 日 12:00 UTC，一个时次/月，共 60 个时次
YEAR_START, YEAR_END = 2016, 2020
SAMPLE_DAY = 16
SAMPLE_HOUR = 12

# 曲线只绘制有效海洋格点覆盖率 >= 10% 的纬度段
COVERAGE_MIN = 0.10
