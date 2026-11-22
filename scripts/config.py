# -*- coding: utf-8 -*-
"""项目公共配置：路径、研究区、时间范围与统计口径常量。"""

from pathlib import Path

# 目录结构
ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
FIG_DIR = ROOT / "figures"
ZONAL_CSV = DATA_DIR / "zonal_mean.csv"

# 数据源：NOAA OISST v2.1 日平均，0.25° 网格
DATASET_ID = "ncdcOisst21Agg"
BASE_URL = f"https://coastwatch.pfeg.noaa.gov/erddap/griddap/{DATASET_ID}.csv"

# 研究区：印度洋 20°E–120°E、40°S–30°N
LON_MIN, LON_MAX = 20.0, 120.0
LAT_MIN, LAT_MAX = -40.0, 30.0
N_LON = 401          # 20.125–120.125°E，0.25° 原生格点共 401 列
N_LAT = 281          # -39.875–30.125°N，0.25° 原生格点共 281 行
EXPECTED_LINES = 112_683  # 2 行表头 + 281*401 行数据

# 时间维度：2016–2020 年，每月 16 日 12:00 UTC，一个时次/月，共 60 个时次
YEAR_START, YEAR_END = 2016, 2020
SAMPLE_DAY = 16
SAMPLE_HOUR = 12

# 曲线只绘制有效海洋格点覆盖率 >= 10% 的纬度段
COVERAGE_MIN = 0.10
