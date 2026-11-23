#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Wave 2 (P3)：基于 NOAA ERSST v5 (1982–2024) 计算 WMO 气候态、43 年升温趋势与 OISST 交叉校验。

输出（全部采用 LF 换行符）：
1. data/ersst_zonal.csv                 - 1982–2024 逐年逐纬度表
2. data/ersst_clim_1991_2020.csv        - WMO 1991–2020 30 年标准气候态
3. data/ersst_trend.csv                 - 1982–2024 43 年逐纬度 OLS 线性趋势与 Student-t 检验
4. data/ersst_global_series.csv         - 1982–2024 全球及分区面积加权年均时间序列
5. data/ersst_vs_oisst_2016_2020.csv    - 2016–2020 ERSST v5 vs OISST v2.1 2° 窗交叉校验
6. data/ersst_summary.json              - 汇总指标、北极放大倍数与外部交叉校验统计
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_ERSST_DIR = DATA_DIR / "raw_ersst"
OISST_GLOBAL_CSV = DATA_DIR / "zonal_mean_global.csv"

ERSST_ZONAL_CSV = DATA_DIR / "ersst_zonal.csv"
ERSST_CLIM_CSV = DATA_DIR / "ersst_clim_1991_2020.csv"
ERSST_TREND_CSV = DATA_DIR / "ersst_trend.csv"
ERSST_SERIES_CSV = DATA_DIR / "ersst_global_series.csv"
ERSST_VS_OISST_CSV = DATA_DIR / "ersst_vs_oisst_2016_2020.csv"
ERSST_SUMMARY_JSON = DATA_DIR / "ersst_summary.json"

ERSST_YEAR_START = 1982
ERSST_YEAR_END = 2024
ERSST_N_LAT = 89
ERSST_N_LON = 180
MIN_TREND_YEARS = 30


# ---------------------------------------------------------------------------
# 纯标准库数值统计函数（无 scipy 依赖）：正则化不完全 Beta 与 Student-t 检验
# ---------------------------------------------------------------------------

def _betacf(a: float, b: float, x: float) -> float:
    """Lentz 连分式计算不完全 Beta 函数核心部分（Numerical Recipes 标准算法）。"""
    max_iter = 200
    eps = 3.0e-14
    fpmin = 1.0e-300

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """正则化不完全 Beta 函数 I_x(a, b)，0 <= x <= 1。"""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def student_t_two_sided_pvalue(t_abs: float, df: int) -> float:
    """自由度 df 的 Student-t 双侧检验 p-value。"""
    if df <= 0:
        return float("nan")
    if not math.isfinite(t_abs):
        return 0.0
    if t_abs == 0.0:
        return 1.0
    x = df / (df + t_abs * t_abs)
    return max(0.0, min(1.0, betainc(df / 2.0, 0.5, x)))


def student_t_crit_95(df: int) -> float:
    """二分法求 95% 双侧置信区间临界值 t_{0.975, df}。"""
    lo, hi = 0.0, 200.0
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        if student_t_two_sided_pvalue(mid, df) > 0.05:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def ols_trend(years: np.ndarray, values: np.ndarray) -> dict[str, float | int]:
    """普通最小二乘（OLS）线性趋势、R^2、双侧 t 检验 p-value 与 95% CI 半宽。"""
    valid = np.isfinite(values)
    n = int(np.sum(valid))
    if n < 3:
        return {
            "n_years": n,
            "trend_per_decade": float("nan"),
            "r2": float("nan"),
            "p_value": float("nan"),
            "ci95_halfwidth": float("nan"),
            "sst_mean": float(np.nanmean(values)) if n > 0 else float("nan"),
        }

    x = years[valid].astype(np.float64)
    y = values[valid].astype(np.float64)
    xm = float(np.mean(x))
    ym = float(np.mean(y))
    dx = x - xm
    dy = y - ym
    ss_xx = float(np.sum(dx * dx))
    ss_yy = float(np.sum(dy * dy))
    ss_xy = float(np.sum(dx * dy))

    slope = ss_xy / ss_xx
    intercept = ym - slope * xm
    resid = y - (intercept + slope * x)
    rss = float(np.sum(resid * resid))
    df = n - 2
    r2 = 1.0 - rss / ss_yy if ss_yy > 0.0 else 0.0
    r2 = max(0.0, min(1.0, r2))

    se_slope = math.sqrt(max(rss, 0.0) / df / ss_xx)
    if se_slope == 0.0:
        p_val = 0.0 if slope != 0.0 else 1.0
    else:
        t_stat = abs(slope) / se_slope
        p_val = student_t_two_sided_pvalue(t_stat, df)

    t_crit = student_t_crit_95(df)
    return {
        "n_years": n,
        "trend_per_decade": slope * 10.0,
        "r2": r2,
        "p_value": p_val,
        "ci95_halfwidth": t_crit * se_slope * 10.0,
        "sst_mean": ym,
    }


# ---------------------------------------------------------------------------
# 数据读取与统计计算
# ---------------------------------------------------------------------------

def read_ersst_stack(raw_dir: Path, start_year: int, end_year: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[Path]]:
    """读取 1982–2024 年 43 个 ERSST v5 年度文件，返回 (years, lat, lon, sst_stack[43,12,89,180], files)。"""
    expected_years = list(range(start_year, end_year + 1))
    files: list[Path] = []
    for y in expected_years:
        p = raw_dir / f"ersst_{y}.csv.gz"
        if not p.exists():
            raise SystemExit(f"缺少 ERSST 原始文件：{p}，请先运行 scripts/fetch_ersst.py")
        files.append(p)

    lat_ref: np.ndarray | None = None
    lon_ref: np.ndarray | None = None
    expected_lat_block: np.ndarray | None = None
    expected_lon_block: np.ndarray | None = None
    sst_stack = np.empty((len(expected_years), 12, ERSST_N_LAT, ERSST_N_LON), dtype=np.float64)
    n_cells_per_month = ERSST_N_LAT * ERSST_N_LON

    for y_i, (year, path) in enumerate(zip(expected_years, files)):
        df = pd.read_csv(
            path,
            skiprows=[1],
            usecols=["time", "latitude", "longitude", "sst"],
            dtype={"time": "string", "latitude": "float64", "longitude": "float64", "sst": "float64"},
            engine="c",
        )
        if len(df) != 12 * n_cells_per_month:
            raise SystemExit(f"{path.name} 数据行数异常：{len(df)}（期望 {12 * n_cells_per_month}）")

        lat_col = df["latitude"].to_numpy()
        lon_col = df["longitude"].to_numpy()
        sst_col = df["sst"].to_numpy()

        if lat_ref is None:
            lat_ref = np.sort(np.unique(lat_col))
            lon_ref = np.sort(np.unique(lon_col))
            if len(lat_ref) != ERSST_N_LAT or len(lon_ref) != ERSST_N_LON:
                raise SystemExit(f"ERSST 网格维度异常：{len(lat_ref)}×{len(lon_ref)}")
            if not np.allclose(np.diff(lat_ref), 2.0, atol=1e-9) or not np.allclose(np.diff(lon_ref), 2.0, atol=1e-9):
                raise SystemExit("ERSST 经纬度步长不是 2.0°")
            expected_lat_block = np.tile(np.repeat(lat_ref, ERSST_N_LON), 12)
            expected_lon_block = np.tile(lon_ref, 12 * ERSST_N_LAT)

        if not (np.array_equal(lat_col, expected_lat_block) and np.array_equal(lon_col, expected_lon_block)):
            raise SystemExit(f"{path.name} 经纬度行列顺序与标准 2° 网格不一致")

        # 校验 12 个月顺序为 01..12
        months = df["time"].str.slice(5, 7).astype(int).to_numpy().reshape(12, n_cells_per_month)
        if not np.all(months == np.arange(1, 13)[:, None]):
            raise SystemExit(f"{path.name} 月份顺序不是 01..12")

        sst_stack[y_i] = sst_col.reshape(12, ERSST_N_LAT, ERSST_N_LON)

    assert lat_ref is not None and lon_ref is not None
    return np.array(expected_years, dtype=np.int64), lat_ref, lon_ref, sst_stack, files


def weighted_band_mean(lat: np.ndarray, sst: np.ndarray, valid_cnt: np.ndarray, mask: np.ndarray) -> float:
    """按 cos(lat) * valid_count 计算指定纬度带的面积加权均值。"""
    usable = mask & np.isfinite(sst) & (valid_cnt > 0)
    if not np.any(usable):
        return float("nan")
    w = valid_cnt[usable] * np.cos(np.deg2rad(lat[usable]))
    w_sum = float(np.sum(w))
    if w_sum <= 0.0:
        return float("nan")
    return float(np.sum(w * sst[usable]) / w_sum)


def compute_cross_validation_metrics(ersst_vals: np.ndarray, oisst_vals: np.ndarray) -> dict[str, float | int]:
    """计算两组纬向均值曲线的 Mean Bias, MAE, RMSE 与 Pearson r。"""
    valid = np.isfinite(ersst_vals) & np.isfinite(oisst_vals)
    n = int(np.sum(valid))
    if n < 2:
        return {"n_lat": n, "mean_bias_c": float("nan"), "mae_c": float("nan"), "rmse_c": float("nan"), "pearson_r": float("nan")}
    e = ersst_vals[valid]
    o = oisst_vals[valid]
    diff = o - e
    return {
        "n_lat": n,
        "mean_bias_c": round(float(np.mean(diff)), 4),
        "mae_c": round(float(np.mean(np.abs(diff))), 4),
        "rmse_c": round(float(math.sqrt(float(np.mean(diff * diff)))), 4),
        "pearson_r": round(float(np.corrcoef(e, o)[0, 1]), 6),
    }


def run_compute_ersst(raw_dir: Path) -> int:
    t0 = time.time()
    print(f"[ersst] 读取原始数据：{raw_dir}", flush=True)
    years, lat, lon, sst_stack, files = read_ersst_stack(raw_dir, ERSST_YEAR_START, ERSST_YEAR_END)
    n_years = len(years)
    print(f"[ersst] 已加载 {n_years} 年 × 12 月 × {len(lat)}×{len(lon)} 网格", flush=True)

    # 逐年逐月逐纬平均与有效格点数
    finite_stack = np.isfinite(sst_stack)
    counts_ym = finite_stack.sum(axis=3)  # (43, 12, 89)
    zonal_ym = np.full((n_years, 12, len(lat)), np.nan, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sums_ym = np.where(finite_stack, sst_stack, 0.0).sum(axis=3)
        nz = counts_ym > 0
        zonal_ym[nz] = sums_ym[nz] / counts_ym[nz]

        zonal_y = np.nanmean(zonal_ym, axis=1)            # (43, 89)
        valid_count_y = counts_ym.mean(axis=1)            # (43, 89)

    # 1) data/ersst_zonal.csv
    zonal_rows: list[dict] = []
    for y_i, year in enumerate(years):
        for lat_i, lat_v in enumerate(lat):
            zonal_rows.append(
                {
                    "year": int(year),
                    "latitude": round(float(lat_v), 1),
                    "sst_annual": round(float(zonal_y[y_i, lat_i]), 4) if np.isfinite(zonal_y[y_i, lat_i]) else np.nan,
                    "sst_jan": round(float(zonal_ym[y_i, 0, lat_i]), 4) if np.isfinite(zonal_ym[y_i, 0, lat_i]) else np.nan,
                    "sst_jul": round(float(zonal_ym[y_i, 6, lat_i]), 4) if np.isfinite(zonal_ym[y_i, 6, lat_i]) else np.nan,
                    "valid_count": round(float(valid_count_y[y_i, lat_i]), 1),
                }
            )
    df_zonal = pd.DataFrame(zonal_rows)
    df_zonal.to_csv(ERSST_ZONAL_CSV, index=False, encoding="utf-8", lineterminator="\n")
    print(f"已写出 {ERSST_ZONAL_CSV}（{len(df_zonal)} 行 × {len(df_zonal.columns)} 列）", flush=True)

    # 2) data/ersst_clim_1991_2020.csv (WMO 30 年气候态)
    m_91_20 = (years >= 1991) & (years <= 2020)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        clim_m_91_20 = np.nanmean(zonal_ym[m_91_20], axis=0)      # (12, 89)
        clim_ann_91_20 = np.nanmean(clim_m_91_20, axis=0)         # (89,)
        vc_91_20 = counts_ym[m_91_20].mean(axis=(0, 1))
    istd_91_20 = np.full(len(lat), np.nan, dtype=np.float64)
    n_valid_91_20 = np.sum(np.isfinite(zonal_y[m_91_20]), axis=0)
    ok_istd = n_valid_91_20 >= 2
    istd_91_20[ok_istd] = np.nanstd(zonal_y[m_91_20][:, ok_istd], axis=0, ddof=1)

    df_clim = pd.DataFrame(
        {
            "latitude": np.round(lat, 1),
            "sst_annual": np.round(clim_ann_91_20, 4),
            "sst_jan": np.round(clim_m_91_20[0], 4),
            "sst_jul": np.round(clim_m_91_20[6], 4),
            "sst_interannual_std": np.round(istd_91_20, 4),
            "valid_count": np.round(vc_91_20, 1),
        }
    )
    df_clim.to_csv(ERSST_CLIM_CSV, index=False, encoding="utf-8", lineterminator="\n")
    print(f"已写出 {ERSST_CLIM_CSV}（{len(df_clim)} 行 × {len(df_clim.columns)} 列）", flush=True)

    # 3) data/ersst_trend.csv (1982–2024 逐纬度趋势)
    trend_rows: list[dict] = []
    for lat_i, lat_v in enumerate(lat):
        res = ols_trend(years, zonal_y[:, lat_i])
        n_y = int(res["n_years"])
        if n_y >= MIN_TREND_YEARS:
            trend_rows.append(
                {
                    "latitude": round(float(lat_v), 1),
                    "trend_per_decade": round(float(res["trend_per_decade"]), 4),
                    "n_years": n_y,
                    "r2": round(float(res["r2"]), 4),
                    "p_value": float(f"{float(res['p_value']):.6g}"),
                    "ci95_halfwidth": round(float(res["ci95_halfwidth"]), 4),
                    "sst_mean_1982_2024": round(float(res["sst_mean"]), 4),
                }
            )
        else:
            trend_rows.append(
                {
                    "latitude": round(float(lat_v), 1),
                    "trend_per_decade": np.nan,
                    "n_years": n_y,
                    "r2": np.nan,
                    "p_value": np.nan,
                    "ci95_halfwidth": np.nan,
                    "sst_mean_1982_2024": round(float(res["sst_mean"]), 4) if n_y > 0 else np.nan,
                }
            )
    df_trend = pd.DataFrame(trend_rows)
    df_trend.to_csv(ERSST_TREND_CSV, index=False, encoding="utf-8", lineterminator="\n")
    print(f"已写出 {ERSST_TREND_CSV}（{len(df_trend)} 行 × {len(df_trend.columns)} 列）", flush=True)

    # 4) data/ersst_global_series.csv (1982–2024 面积加权全球及分区序列)
    m_all = np.ones(len(lat), dtype=bool)
    m_nh = lat > 0.0
    m_sh = lat < 0.0
    m_trop = (lat >= -20.0) & (lat <= 20.0)
    m_arc = lat >= 60.0
    m_sou = lat <= -60.0

    series_rows: list[dict] = []
    for y_i, year in enumerate(years):
        s_y = zonal_y[y_i]
        vc_y = valid_count_y[y_i]
        series_rows.append(
            {
                "year": int(year),
                "global_mean_sst": round(weighted_band_mean(lat, s_y, vc_y, m_all), 4),
                "nh_mean_sst": round(weighted_band_mean(lat, s_y, vc_y, m_nh), 4),
                "sh_mean_sst": round(weighted_band_mean(lat, s_y, vc_y, m_sh), 4),
                "tropics_mean_sst": round(weighted_band_mean(lat, s_y, vc_y, m_trop), 4),
                "arctic_60n_mean_sst": round(weighted_band_mean(lat, s_y, vc_y, m_arc), 4),
                "southern_60s_mean_sst": round(weighted_band_mean(lat, s_y, vc_y, m_sou), 4),
            }
        )
    df_series = pd.DataFrame(series_rows)
    df_series.to_csv(ERSST_SERIES_CSV, index=False, encoding="utf-8", lineterminator="\n")
    print(f"已写出 {ERSST_SERIES_CSV}（{len(df_series)} 行 × {len(df_series.columns)} 列）", flush=True)

    # 5) data/ersst_vs_oisst_2016_2020.csv (外部独立基准交叉校验)
    if not OISST_GLOBAL_CSV.exists():
        raise SystemExit(f"缺少 OISST 全局纬向表：{OISST_GLOBAL_CSV}")
    df_oisst = pd.read_csv(OISST_GLOBAL_CSV)
    o_lat = df_oisst["latitude"].to_numpy(dtype=np.float64)
    o_all_sst = df_oisst["sst_annual"].to_numpy(dtype=np.float64)
    o_all_vc = df_oisst["valid_count"].to_numpy(dtype=np.float64)
    o_if_sst = df_oisst["sst_annual_icefree"].to_numpy(dtype=np.float64)
    o_if_vc = df_oisst["valid_count_icefree"].to_numpy(dtype=np.float64)

    m_16_20 = (years >= 2016) & (years <= 2020)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ersst_16_20 = np.nanmean(np.nanmean(zonal_ym[m_16_20], axis=0), axis=0)

    oisst_all_2deg = np.full(len(lat), np.nan, dtype=np.float64)
    oisst_if_2deg = np.full(len(lat), np.nan, dtype=np.float64)
    for i, target_lat in enumerate(lat):
        win = (o_lat >= target_lat - 1.0) & (o_lat < target_lat + 1.0)
        oisst_all_2deg[i] = weighted_band_mean(o_lat, o_all_sst, o_all_vc, win)
        oisst_if_2deg[i] = weighted_band_mean(o_lat, o_if_sst, o_if_vc, win)

    diff_all = oisst_all_2deg - ersst_16_20
    diff_if = oisst_if_2deg - ersst_16_20

    df_cv = pd.DataFrame(
        {
            "latitude": np.round(lat, 1),
            "ersst_annual": np.round(ersst_16_20, 4),
            "oisst_all_annual": np.round(oisst_all_2deg, 4),
            "oisst_icefree_annual": np.round(oisst_if_2deg, 4),
            "diff_all_minus_ersst": np.round(diff_all, 4),
            "diff_icefree_minus_ersst": np.round(diff_if, 4),
        }
    )
    df_cv.to_csv(ERSST_VS_OISST_CSV, index=False, encoding="utf-8", lineterminator="\n")
    print(f"已写出 {ERSST_VS_OISST_CSV}（{len(df_cv)} 行 × {len(df_cv.columns)} 列）", flush=True)

    # 6) data/ersst_summary.json
    global_mean_1991_2020 = weighted_band_mean(lat, clim_ann_91_20, vc_91_20, m_all)

    def _trend_dict(col: str) -> dict:
        tr = ols_trend(years, df_series[col].to_numpy(dtype=np.float64))
        return {
            "trend_per_decade_c": round(float(tr["trend_per_decade"]), 4),
            "r2": round(float(tr["r2"]), 4),
            "p_value": float(f"{float(tr['p_value']):.6g}"),
            "ci95_halfwidth_c_per_decade": round(float(tr["ci95_halfwidth"]), 4),
            "n_years": int(tr["n_years"]),
        }

    g_trend = _trend_dict("global_mean_sst")
    arc_trend = _trend_dict("arctic_60n_mean_sst")
    trop_trend = _trend_dict("tropics_mean_sst")
    nh_trend = _trend_dict("nh_mean_sst")
    sh_trend = _trend_dict("sh_mean_sst")
    sou_trend = _trend_dict("southern_60s_mean_sst")

    amp_ratio = round(
        float(arc_trend["trend_per_decade_c"]) / float(g_trend["trend_per_decade_c"]),
        3,
    )

    g_vals = df_series["global_mean_sst"].to_numpy(dtype=np.float64)
    i_warm = int(np.argmax(g_vals))
    i_cold = int(np.argmin(g_vals))

    m_60 = (lat >= -60.0) & (lat <= 60.0)
    cv_60 = {
        "all_ocean": compute_cross_validation_metrics(ersst_16_20[m_60], oisst_all_2deg[m_60]),
        "ice_free": compute_cross_validation_metrics(ersst_16_20[m_60], oisst_if_2deg[m_60]),
    }
    cv_global = {
        "all_ocean": compute_cross_validation_metrics(ersst_16_20, oisst_all_2deg),
        "ice_free": compute_cross_validation_metrics(ersst_16_20, oisst_if_2deg),
    }

    gz_bytes = sum(p.stat().st_size for p in files)
    manifest_path = raw_dir / "manifest.json"
    csv_bytes = 0
    if manifest_path.exists():
        try:
            csv_bytes = int(json.loads(manifest_path.read_text(encoding="utf-8")).get("total_bytes_csv", 0))
        except Exception:  # noqa: BLE001
            csv_bytes = 0

    compute_sec = round(time.time() - t0, 2)
    summary = {
        "dataset": "NOAA NCEI ERSST v5 (2.0° × 2.0°, nceiErsstv5)",
        "period": f"{ERSST_YEAR_START}-{ERSST_YEAR_END}",
        "clim_period": "1991-2020",
        "weighting": "cos(lat)*valid_count",
        "global_mean_1991_2020_c": round(global_mean_1991_2020, 4),
        "global_trend_per_decade_c": g_trend,
        "arctic_60n_trend_per_decade_c": arc_trend,
        "arctic_amplification_ratio": amp_ratio,
        "tropics_trend_per_decade_c": trop_trend,
        "nh_trend_per_decade_c": nh_trend,
        "sh_trend_per_decade_c": sh_trend,
        "southern_60s_trend_per_decade_c": sou_trend,
        "warmest_year": {
            "year": int(df_series.iloc[i_warm]["year"]),
            "global_mean_sst_c": float(df_series.iloc[i_warm]["global_mean_sst"]),
        },
        "coldest_year": {
            "year": int(df_series.iloc[i_cold]["year"]),
            "global_mean_sst_c": float(df_series.iloc[i_cold]["global_mean_sst"]),
        },
        "cross_validation_2016_2020": {
            "aggregation_rule": "OISST 0.25° 经 [lat-1.0, lat+1.0) 窗按 cos(lat)*valid_count 聚合至 ERSST 2° 网格",
            "60S_60N": cv_60,
            "global": cv_global,
        },
        "pipeline": {
            "raw_files_count": len(files),
            "raw_months_count": len(files) * 12,
            "raw_gz_mb": round(gz_bytes / 1e6, 2),
            "raw_uncompressed_mb": round(csv_bytes / 1e6, 2),
            "compute_ersst_seconds": compute_sec,
        },
    }
    ERSST_SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写出 {ERSST_SUMMARY_JSON}（耗时 {compute_sec:.2f}s）", flush=True)

    print("\nERSST v5 (1982–2024) 关键结果摘要：")
    print(f"  1991–2020 全球 WMO 基准均温：{summary['global_mean_1991_2020_c']:.4f} °C")
    print(
        f"  全球升温趋势：{g_trend['trend_per_decade_c']:+.4f} ± {g_trend['ci95_halfwidth_c_per_decade']:.4f} °C/十年 "
        f"(R²={g_trend['r2']:.4f}, p={g_trend['p_value']:.3e})"
    )
    print(
        f"  北极(≥60°N)升温趋势：{arc_trend['trend_per_decade_c']:+.4f} °C/十年 "
        f"（北极放大倍数 {amp_ratio:.2f}x）"
    )
    print(f"  热带(20°S–20°N)升温趋势：{trop_trend['trend_per_decade_c']:+.4f} °C/十年")
    print(f"  最暖年份：{summary['warmest_year']['year']} ({summary['warmest_year']['global_mean_sst_c']:.4f} °C)")
    print(f"  最冷年份：{summary['coldest_year']['year']} ({summary['coldest_year']['global_mean_sst_c']:.4f} °C)")
    print(
        f"  2016–2020 交叉校验 (60°S–60°N): "
        f"all_ocean bias={cv_60['all_ocean']['mean_bias_c']:+.4f}°C, MAE={cv_60['all_ocean']['mae_c']:.4f}°C, "
        f"RMSE={cv_60['all_ocean']['rmse_c']:.4f}°C, r={cv_60['all_ocean']['pearson_r']:.6f}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="计算 ERSST v5 长期气候态、趋势与 OISST 交叉校验")
    parser.add_argument("--raw-dir", type=Path, default=RAW_ERSST_DIR, help="ERSST 原始缓存目录（默认 data/raw_ersst）")
    args = parser.parse_args()
    return run_compute_ersst(args.raw_dir)


if __name__ == "__main__":
    sys.exit(main())
