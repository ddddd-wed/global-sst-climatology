#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""读取原始 OISST 数据，计算纬向平均及全球五大洋契约产物。

支持的区域预设与产物：
- --region global（默认）：
  读取 data/raw_global/ 下的 60 个全球 0.25° (720×1440) SST 时次，并读取 data/raw_global_ice/
  下的 60 个 OISST v2.1 海冰密集度（ice）时次，构建五大洋互斥掩膜与双口径（all_ocean 兼容口径
  + ice_free 开阔无冰主口径 ice < 0.15），输出：
    1. data/zonal_mean_global.csv（原 32 列数值 100% 兼容，末尾追加 23 个 P0/P2/冰情列）
    2. data/zonal_mean_basins.csv（原 32 列数值 100% 兼容，末尾追加 23 个 P0/P2/冰情列）
    3. data/zonal_mean_global_icefree.csv（开阔无冰主口径 50 列）
    4. data/zonal_mean_basins_icefree.csv（五大洋 + global 开阔无冰主口径 50 列）
    5. data/zonal_ref.csv（720 行格点算术平均纬向参考线）
    6. data/anomaly_grid.npz（冰情次数、无冰气候态与同纬度温差异常场）
    7. data/summary_metrics.json（保留原有字段 100% 兼容，新增 basins.*.icefree 与 basins_icefree）
    8. data/basin_mask.npz 与 data/climatology_grid.npz（仅在缺失或内容变化时写入）
- --region indian_ocean：
  沿用旧版印度洋 20°E–120°E、40°S–30°N 计算口径，输出 data/zonal_mean.csv（用于回归校验）。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from basins import build_basin_mask, inland_lake_mask, verify_basin_mask
from config import (
    ANOMALY_GRID_NPZ,
    BASIN_IDS,
    BASIN_MASK_NPZ,
    BASIN_ORDER,
    BASINS,
    BASINS_ZONAL_CSV,
    BASINS_ZONAL_ICEFREE_CSV,
    CLIMATOLOGY_NPZ,
    COVERAGE_MIN,
    GLOBAL_ICE_RAW_DIR,
    GLOBAL_ZONAL_CSV,
    GLOBAL_ZONAL_ICEFREE_CSV,
    ICE_CONC_MAX,
    N_LAT,
    N_LON,
    RAW_DIR,
    REGION_PRESETS,
    ROOT,
    SUMMARY_JSON,
    ZONAL_CSV,
    ZONAL_REF_CSV,
)


def parse_name(name: str) -> tuple[int, int]:
    """从 sst_2016_01.csv.gz 解析 (year, month)。"""
    parts = name.split("_")
    year = int(parts[1])
    month = int(parts[2].split(".")[0])
    return year, month


def list_raw_files(raw_dir: Path) -> list[Path]:
    """优先使用 .csv.gz；若不存在退化为 .csv。"""
    gz = sorted(raw_dir.glob("sst_*.csv.gz"))
    if gz:
        return gz
    return sorted(raw_dir.glob("sst_*.csv"))


def check_completeness(keys: list[tuple[int, int]] | dict[tuple[int, int], object]) -> list[int]:
    key_set = set(keys)
    years = sorted({y for y, _ in key_set})
    missing = [(y, m) for y in years for m in range(1, 13) if (y, m) not in key_set]
    if missing:
        text = ", ".join(f"{y}-{m:02d}" for y, m in missing)
        raise SystemExit(f"数据不完整，缺少 {len(missing)} 个时次：{text}")
    return years


def find_isotherm(lat: np.ndarray, sst: np.ndarray, level: float = 25.0) -> list[float]:
    """求等温线纬度（线性插值），返回所有交点，按纬度升序。"""
    crossings: list[float] = []
    for i in range(len(lat) - 1):
        y1, y2 = sst[i], sst[i + 1]
        if not (np.isfinite(y1) and np.isfinite(y2)) or y1 == y2:
            continue
        if (y1 - level) * (y2 - level) <= 0:
            t = (level - y1) / (y2 - y1)
            crossings.append(float(lat[i] + t * (lat[i + 1] - lat[i])))
    deduped: list[float] = []
    for c in crossings:
        if not deduped or abs(c - deduped[-1]) > 1e-6:
            deduped.append(c)
    return deduped


def split_isotherms(lat: np.ndarray, sst: np.ndarray, level: float) -> dict:
    """求等温线全部交点，并按南北半球拆分（各取最接近极地侧或唯一交点）。"""
    crossings = [round(c, 4) for c in find_isotherm(lat, sst, level=level)]
    neg = [c for c in crossings if c < 0]
    pos = [c for c in crossings if c >= 0]
    return {
        "all": crossings,
        "north": pos[-1] if pos else None,
        "south": neg[0] if neg else None,
    }


def summarize(df: pd.DataFrame) -> dict:
    """打印旧版印度洋关键数字（README 中引用的数值均来自这里）。"""
    lat = df["latitude"].to_numpy(dtype=float)
    cov = df["coverage"].to_numpy(dtype=float)
    annual = df["sst_annual"].to_numpy(dtype=float)
    usable = np.isfinite(annual) & (cov >= COVERAGE_MIN)

    i_max = int(np.nanargmax(np.where(usable, annual, -np.inf)))
    isotherms = find_isotherm(lat[usable], annual[usable])

    band = (lat >= 20.0) & (lat <= 26.0)
    i_cov = int(np.argmin(np.where(band, cov, np.inf)))

    weights = np.cos(np.deg2rad(lat[usable]))
    basin = float(np.sum(weights * annual[usable]) / np.sum(weights))

    print("\n关键数字（与 README 一致）：")
    print(f"  年均 SST 最高值：{annual[i_max]:.2f} °C @ 纬度 {abs(lat[i_max]):.2f}°{'N' if lat[i_max] >= 0 else 'S'}")
    if isotherms:
        iso = isotherms[-1]
        print(f"  25 °C 等温线（年均，覆盖率≥10%）：{abs(iso):.2f}°{'N' if iso >= 0 else 'S'}（{iso:.2f}°）")
    else:
        print("  25 °C 等温线（年均，覆盖率≥10%）：研究区内无交点")
    print(f"  20–26°N 覆盖率最低：{cov[i_cov] * 100:.1f}%（{lat[i_cov]:.2f}°N）")
    print(f"  全海盆年均 SST（cosφ 加权，仅覆盖率≥10% 的纬度）：{basin:.2f} °C")

    return {
        "sst_max": float(annual[i_max]),
        "lat_max": float(lat[i_max]),
        "isotherms": isotherms,
        "cov_min_20_26": float(cov[i_cov]),
        "lat_cov_min_20_26": float(lat[i_cov]),
        "basin_mean": basin,
    }


def collect_indian_ocean(raw_dir: Path) -> tuple[np.ndarray, dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]]:
    """逐文件计算印度洋每个纬度的 SST 均值与有效格点数（保持与 7a4485d 一致）。"""
    files = list_raw_files(raw_dir)
    if not files:
        raise SystemExit(f"未找到原始数据：{raw_dir}，请先运行 scripts/fetch_data.py --region indian_ocean")

    lat_ref: np.ndarray | None = None
    samples: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    for i, path in enumerate(files, 1):
        year, month = parse_name(path.name)
        df = pd.read_csv(
            path,
            skiprows=[1],
            usecols=["latitude", "longitude", "sst"],
            dtype={"latitude": "float64", "longitude": "float64", "sst": "float64"},
        )
        lats = np.sort(df["latitude"].unique())
        if lat_ref is None:
            lat_ref = lats
            if len(lat_ref) != N_LAT:
                raise SystemExit(f"纬度格点数异常：{len(lat_ref)}（期望 {N_LAT}）")
            if not np.allclose(np.diff(lat_ref), 0.25, atol=1e-9):
                raise SystemExit("纬度不是 0.25° 等间距原生格点")
        elif len(lats) != len(lat_ref) or not np.allclose(lats, lat_ref):
            raise SystemExit(f"文件 {path.name} 的纬度网格与其他文件不一致")

        if df["longitude"].nunique() != N_LON:
            raise SystemExit(f"文件 {path.name} 经度格点数异常：{df['longitude'].nunique()}")

        grouped = df.groupby("latitude", sort=True)["sst"]
        means = grouped.mean().reindex(lat_ref).to_numpy(dtype=float)
        counts = grouped.count().reindex(lat_ref).to_numpy(dtype=float)
        samples[(year, month)] = (means, counts)
        print(f"  [{i:02d}/{len(files)}] {path.name}: 有效纬度圈 {int(np.isfinite(means).sum())}/{N_LAT}", flush=True)

    return lat_ref, samples


def run_indian_ocean(raw_dir: Path, out_path: Path) -> int:
    print(f"读取原始数据：{raw_dir}", flush=True)
    lat, samples = collect_indian_ocean(raw_dir)
    years = check_completeness(samples)
    print(f"时次完整：{len(samples)} 个（{years[0]}–{years[-1]}，每年 12 个月）", flush=True)

    n_lat = len(lat)
    clim = np.full((12, n_lat), np.nan, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i, month in enumerate(range(1, 13)):
            stack = np.vstack([samples[(year, month)][0] for year in years])
            clim[i] = np.nanmean(stack, axis=0)
        annual = np.nanmean(clim, axis=0)

    all_counts = np.vstack([samples[key][1] for key in sorted(samples)])
    valid_count = all_counts.mean(axis=0)
    coverage = valid_count / N_LON

    out = pd.DataFrame(
        {
            "latitude": np.round(lat, 3),
            "valid_count": np.round(valid_count, 1),
            "coverage": np.round(coverage, 4),
            "sst_annual": np.round(annual, 3),
            "sst_jan": np.round(clim[0], 3),
            "sst_jul": np.round(clim[6], 3),
        }
    )
    for m in range(12):
        out[f"sst_{m + 1:02d}"] = np.round(clim[m], 3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8", lineterminator="\n")
    print(f"已写出 {out_path}（{len(out)} 行 × {len(out.columns)} 列）", flush=True)

    summarize(out)
    return 0


def read_global_grid_stack(
    raw_dir: Path,
    preset: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], list[Path]]:
    """高速向量化读取全球 0.25° 月度 CSV，返回 (lat_ref, lon_ref, sst_5_12_lat_lon, years, files)。"""
    files = list_raw_files(raw_dir)
    if not files:
        raise SystemExit(f"未找到全球原始数据：{raw_dir}，请先运行 scripts/fetch_data.py --region global")

    file_map: dict[tuple[int, int], Path] = {}
    for path in files:
        file_map[parse_name(path.name)] = path

    years = check_completeness(list(file_map.keys()))
    n_years = len(years)
    n_lat = preset["n_lat"]
    n_lon = preset["n_lon"]
    n_cells = n_lat * n_lon

    lat_ref: np.ndarray | None = None
    lon_ref: np.ndarray | None = None
    expected_lat_col: np.ndarray | None = None
    expected_lon_col: np.ndarray | None = None

    sst_stack = np.empty((n_years, 12, n_lat, n_lon), dtype=np.float64)

    ordered_keys = [(y, m) for y in years for m in range(1, 13)]
    for idx, (year, month) in enumerate(ordered_keys, 1):
        path = file_map[(year, month)]
        df = pd.read_csv(
            path,
            skiprows=[1],
            usecols=["latitude", "longitude", "sst"],
            dtype={"latitude": "float64", "longitude": "float64", "sst": "float64"},
            engine="c",
        )
        if len(df) != n_cells:
            raise SystemExit(f"文件 {path.name} 行数异常：{len(df)}（期望 {n_cells}）")

        lat_col = df["latitude"].to_numpy()
        lon_col = df["longitude"].to_numpy()
        sst_col = df["sst"].to_numpy()

        if lat_ref is None:
            lat_ref = np.unique(lat_col)
            lon_ref = np.unique(lon_col)
            if len(lat_ref) != n_lat or len(lon_ref) != n_lon:
                raise SystemExit(f"网格维度异常：{len(lat_ref)}×{len(lon_ref)}（期望 {n_lat}×{n_lon}）")
            if not np.allclose(np.diff(lat_ref), 0.25, atol=1e-9) or not np.allclose(np.diff(lon_ref), 0.25, atol=1e-9):
                raise SystemExit("经纬度步长不是 0.25° 原生网格")
            expected_lat_col = np.repeat(lat_ref, n_lon)
            expected_lon_col = np.tile(lon_ref, n_lat)

        if not (np.array_equal(lat_col, expected_lat_col) and np.array_equal(lon_col, expected_lon_col)):
            raise SystemExit(f"文件 {path.name} 的经纬度行列顺序与标准网格不一致")

        y_i = years.index(year)
        m_i = month - 1
        sst_stack[y_i, m_i] = sst_col.reshape(n_lat, n_lon)

        if idx % 10 == 0 or idx == len(ordered_keys):
            print(f"  [{idx:02d}/{len(ordered_keys)}] 已读取 {path.name}", flush=True)

    assert lat_ref is not None and lon_ref is not None
    return lat_ref, lon_ref, sst_stack, years, files


def list_ice_files(ice_dir: Path) -> list[Path]:
    """优先使用 .csv.gz；若不存在退化为 .csv。"""
    gz = sorted(ice_dir.glob("ice_*.csv.gz"))
    if gz:
        return gz
    return sorted(ice_dir.glob("ice_*.csv"))


def read_global_ice_stack(
    ice_dir: Path,
    preset: dict,
    expected_years: list[int],
    lat_ref: np.ndarray,
    lon_ref: np.ndarray,
) -> tuple[np.ndarray, list[Path]]:
    """读取 60 个全球 0.25° 海冰密集度（ice）月度 CSV，并归一化到 0.0–1.0。"""
    files = list_ice_files(ice_dir)
    if not files:
        raise SystemExit(
            f"未找到全球海冰原始数据：{ice_dir}，请先运行 scripts/fetch_data.py --region global --var ice"
        )

    file_map: dict[tuple[int, int], Path] = {}
    for path in files:
        file_map[parse_name(path.name)] = path

    years = check_completeness(list(file_map.keys()))
    if years != expected_years:
        raise SystemExit(f"海冰数据年份 {years} 与 SST 年份 {expected_years} 不一致")

    n_years = len(years)
    n_lat = preset["n_lat"]
    n_lon = preset["n_lon"]
    n_cells = n_lat * n_lon
    expected_lat_col = np.repeat(lat_ref, n_lon)
    expected_lon_col = np.tile(lon_ref, n_lat)

    ice_stack = np.empty((n_years, 12, n_lat, n_lon), dtype=np.float64)
    max_observed = 0.0
    ordered_keys = [(y, m) for y in years for m in range(1, 13)]
    for idx, (year, month) in enumerate(ordered_keys, 1):
        path = file_map[(year, month)]
        df = pd.read_csv(
            path,
            skiprows=[1],
            usecols=["latitude", "longitude", "ice"],
            dtype={"latitude": "float64", "longitude": "float64", "ice": "float64"},
            engine="c",
        )
        if len(df) != n_cells:
            raise SystemExit(f"海冰文件 {path.name} 行数异常：{len(df)}（期望 {n_cells}）")

        lat_col = df["latitude"].to_numpy()
        lon_col = df["longitude"].to_numpy()
        ice_col = df["ice"].to_numpy()
        if not (np.array_equal(lat_col, expected_lat_col) and np.array_equal(lon_col, expected_lon_col)):
            raise SystemExit(f"海冰文件 {path.name} 的经纬度行列顺序与标准网格不一致")

        finite_vals = ice_col[np.isfinite(ice_col)]
        if finite_vals.size > 0:
            max_observed = max(max_observed, float(np.max(finite_vals)))

        y_i = years.index(year)
        m_i = month - 1
        ice_stack[y_i, m_i] = ice_col.reshape(n_lat, n_lon)

        if idx % 10 == 0 or idx == len(ordered_keys):
            print(f"  [{idx:02d}/{len(ordered_keys)}] 已读取海冰 {path.name}", flush=True)

    if max_observed > 1.5:
        print(f"检测到海冰密集度最大值为 {max_observed:.2f}（百分数），自动归一化至 0.0–1.0", flush=True)
        ice_stack = ice_stack / 100.0

    return ice_stack, files


def compute_zonal_core(
    bmask: np.ndarray,
    sst_stack: np.ndarray,
    finite_stack: np.ndarray,
) -> dict[str, np.ndarray]:
    """按给定海盆掩膜计算月度/年均纬向平均及有效格点统计。"""
    n_years, n_months, n_lat, _ = sst_stack.shape
    ocean_grid_count = bmask.sum(axis=1).astype(np.int64)

    counts_ym = np.empty((n_years, n_months, n_lat), dtype=np.int64)
    zonal_ym = np.full((n_years, n_months, n_lat), np.nan, dtype=np.float64)

    for y_i in range(n_years):
        for m_i in range(n_months):
            valid_ij = finite_stack[y_i, m_i] & bmask
            c = valid_ij.sum(axis=1)
            counts_ym[y_i, m_i] = c
            s = np.where(valid_ij, sst_stack[y_i, m_i], 0.0).sum(axis=1)
            nonzero = c > 0
            zonal_ym[y_i, m_i, nonzero] = s[nonzero] / c[nonzero]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        clim_m = np.nanmean(zonal_ym, axis=0)   # (12, n_lat)
        annual = np.nanmean(clim_m, axis=0)     # (n_lat,)
        zonal_y = np.nanmean(zonal_ym, axis=1)  # (n_years, n_lat)

    valid_count = counts_ym.mean(axis=(0, 1))
    month_valid_count = counts_ym.mean(axis=0)  # (12, n_lat)

    denom = np.where(ocean_grid_count > 0, ocean_grid_count, 1)
    coverage = np.where(ocean_grid_count > 0, valid_count / denom, 0.0)
    cov_m = np.where(ocean_grid_count[None, :] > 0, month_valid_count / denom[None, :], 0.0)

    return {
        "ocean_grid_count": ocean_grid_count,
        "valid_count": valid_count,
        "coverage": coverage,
        "annual": annual,
        "clim_m": clim_m,
        "cov_m": cov_m,
        "zonal_ym": zonal_ym,
        "zonal_y": zonal_y,
    }


def _row_spatial_stats(
    grid_field: np.ndarray,
    bmask: np.ndarray,
    compute_std: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """计算多年平均格点场在海盆每条纬度上的 P10、P90（以及可选空间标准差）。"""
    n_lat = bmask.shape[0]
    p10 = np.full(n_lat, np.nan, dtype=np.float64)
    p90 = np.full(n_lat, np.nan, dtype=np.float64)
    std = np.full(n_lat, np.nan, dtype=np.float64) if compute_std else None

    for i in range(n_lat):
        row_mask = bmask[i]
        if not np.any(row_mask):
            continue
        vals = grid_field[i, row_mask]
        valid = vals[np.isfinite(vals)]
        if valid.size == 0:
            continue
        p10[i] = float(np.percentile(valid, 10))
        p90[i] = float(np.percentile(valid, 90))
        if std is not None:
            std[i] = float(np.std(valid, ddof=0))
    return p10, p90, std


def _interannual_std_1d(values_yn: np.ndarray) -> np.ndarray:
    """对 (n_years, n_lat) 计算 ddof=1 年际标准差，有效年份 < 2 时置 NaN。"""
    n_valid = np.sum(np.isfinite(values_yn), axis=0)
    out = np.full(values_yn.shape[1], np.nan, dtype=np.float64)
    mask = n_valid >= 2
    if np.any(mask):
        out[mask] = np.nanstd(values_yn[:, mask], axis=0, ddof=1)
    return out


def compute_icefree_extras(
    bmask: np.ndarray,
    icefree_core: dict[str, np.ndarray],
    ice_masked_stack: np.ndarray,
    grid_annual_icefree: np.ndarray,
    grid_jan_icefree: np.ndarray,
    grid_jul_icefree: np.ndarray,
) -> dict[str, object]:
    """计算海冰遮蔽比例、P2 空间离散、5 年年际离散及真实最暖/最冷月包络。"""
    n_years, n_months, n_lat, _ = ice_masked_stack.shape
    ocean_grid_count = icefree_core["ocean_grid_count"]
    denom = np.where(ocean_grid_count > 0, ocean_grid_count, 1)

    ice_counts_ym = np.empty((n_years, n_months, n_lat), dtype=np.int64)
    for y_i in range(n_years):
        for m_i in range(n_months):
            ice_counts_ym[y_i, m_i] = (ice_masked_stack[y_i, m_i] & bmask).sum(axis=1)
    ice_frac = np.where(ocean_grid_count > 0, ice_counts_ym.mean(axis=(0, 1)) / denom, 0.0)

    # 1) 空间离散（基于 ice-free 多年平均格点场）
    p10_ann, p90_ann, sst_std = _row_spatial_stats(grid_annual_icefree, bmask, compute_std=True)
    p10_jan, p90_jan, _ = _row_spatial_stats(grid_jan_icefree, bmask, compute_std=False)
    p10_jul, p90_jul, _ = _row_spatial_stats(grid_jul_icefree, bmask, compute_std=False)
    assert sst_std is not None

    # 2) 5 年年际离散（基于 ice-free 每年纬向均值）
    zonal_y = icefree_core["zonal_y"]         # (5, n_lat)
    zonal_ym = icefree_core["zonal_ym"]       # (5, 12, n_lat)
    sst_interannual_std = _interannual_std_1d(zonal_y)
    sst_jan_interannual_std = _interannual_std_1d(zonal_ym[:, 0, :])
    sst_jul_interannual_std = _interannual_std_1d(zonal_ym[:, 6, :])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        interannual_min = np.nanmin(zonal_y, axis=0)
        interannual_max = np.nanmax(zonal_y, axis=0)

    # 3) 最暖/最冷月真实季节包络（基于 ice-free 12 个月气候态）
    clim_m = icefree_core["clim_m"]  # (12, n_lat)
    has_any = np.any(np.isfinite(clim_m), axis=0)
    sst_warmest = np.full(n_lat, np.nan, dtype=np.float64)
    sst_coldest = np.full(n_lat, np.nan, dtype=np.float64)
    warm_months: list[int | object] = [pd.NA] * n_lat
    cold_months: list[int | object] = [pd.NA] * n_lat
    for i in range(n_lat):
        if not has_any[i]:
            continue
        col = clim_m[:, i]
        sst_warmest[i] = float(np.nanmax(col))
        sst_coldest[i] = float(np.nanmin(col))
        warm_months[i] = int(np.nanargmax(col)) + 1
        cold_months[i] = int(np.nanargmin(col)) + 1
    seasonal_amp = sst_warmest - sst_coldest

    return {
        "ice_frac": np.round(ice_frac, 4),
        "sst_std": np.round(sst_std, 3),
        "sst_p10": np.round(p10_ann, 3),
        "sst_p90": np.round(p90_ann, 3),
        "sst_jan_p10": np.round(p10_jan, 3),
        "sst_jan_p90": np.round(p90_jan, 3),
        "sst_jul_p10": np.round(p10_jul, 3),
        "sst_jul_p90": np.round(p90_jul, 3),
        "sst_interannual_std": np.round(sst_interannual_std, 3),
        "sst_jan_interannual_std": np.round(sst_jan_interannual_std, 3),
        "sst_jul_interannual_std": np.round(sst_jul_interannual_std, 3),
        "interannual_min": np.round(interannual_min, 3),
        "interannual_max": np.round(interannual_max, 3),
        "sst_warmest": np.round(sst_warmest, 3),
        "warmest_month": pd.array(warm_months, dtype="Int64"),
        "sst_coldest": np.round(sst_coldest, 3),
        "coldest_month": pd.array(cold_months, dtype="Int64"),
        "seasonal_amp": np.round(seasonal_amp, 3),
    }


def build_core_df(basin_key: str, lat: np.ndarray, core: dict[str, np.ndarray]) -> pd.DataFrame:
    """将核心纬向统计组装为兼容旧版 32 列的 DataFrame（不含 zonal_ocean_frac）。"""
    clim_m = core["clim_m"]
    cov_m = core["cov_m"]
    data: dict[str, object] = {
        "basin": basin_key,
        "latitude": np.round(lat, 3),
        "ocean_grid_count": core["ocean_grid_count"],
        "valid_count": np.round(core["valid_count"], 1),
        "coverage": np.round(core["coverage"], 4),
        "sst_annual": np.round(core["annual"], 3),
        "sst_jan": np.round(clim_m[0], 3),
        "sst_jul": np.round(clim_m[6], 3),
    }
    for m in range(12):
        data[f"sst_{m + 1:02d}"] = np.round(clim_m[m], 3)
    for m in range(12):
        data[f"cov_{m + 1:02d}"] = np.round(cov_m[m], 4)
    return pd.DataFrame(data)


def append_v2_columns(
    df_all: pd.DataFrame,
    df_icefree: pd.DataFrame,
    extras: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """在兼容表末尾追加双口径+P2列，并在 icefree 表末尾追加 ice_frac+P2列。"""
    df_all_out = df_all.copy()
    df_all_out["valid_count_icefree"] = df_icefree["valid_count"]
    df_all_out["coverage_icefree"] = df_icefree["coverage"]
    df_all_out["sst_annual_icefree"] = df_icefree["sst_annual"]
    df_all_out["sst_jan_icefree"] = df_icefree["sst_jan"]
    df_all_out["sst_jul_icefree"] = df_icefree["sst_jul"]

    df_icefree_out = df_icefree.copy()
    for col, values in extras.items():
        df_all_out[col] = values
        df_icefree_out[col] = values
    return df_all_out, df_icefree_out


def build_basin_summary(
    basin_id: int | None,
    df_basin: pd.DataFrame,
    bmask: np.ndarray,
    lat: np.ndarray,
    grid_sst_annual: np.ndarray,
) -> dict:
    """计算单个分区的面积加权均温、极值、25°C/0°C 等温线与有效纬度范围。"""
    lat_arr = df_basin["latitude"].to_numpy(dtype=float)
    cov = df_basin["coverage"].to_numpy(dtype=float)
    valid_cnt = df_basin["valid_count"].to_numpy(dtype=float)
    annual = df_basin["sst_annual"].to_numpy(dtype=float)
    usable = (cov >= COVERAGE_MIN) & np.isfinite(annual)

    # 1) 纬向有效格点 × cos(lat) 加权全海盆年均 SST
    w_zonal = valid_cnt[usable] * np.cos(np.deg2rad(lat_arr[usable]))
    area_weighted_sst = float(np.sum(w_zonal * annual[usable]) / np.sum(w_zonal))

    # 2) 逐格点直接 cos(lat) 加权全海盆年均 SST（用于交叉验证）
    cos_2d = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, bmask.shape[1]), dtype=np.float64)
    grid_usable = bmask & np.isfinite(grid_sst_annual)
    w_grid = cos_2d[grid_usable]
    grid_area_weighted_sst = float(np.sum(w_grid * grid_sst_annual[grid_usable]) / np.sum(w_grid))

    i_max = int(np.nanargmax(np.where(usable, annual, -np.inf)))
    i_min = int(np.nanargmin(np.where(usable, annual, np.inf)))

    usable_lats = lat_arr[usable]
    usable_sst = annual[usable]

    meta = (
        {"id": None, "key": "global", "name_zh": "全球海洋", "name_en": "Global Ocean"}
        if basin_id is None
        else BASINS[basin_id]
    )

    return {
        "basin_id": meta["id"],
        "name_zh": meta["name_zh"],
        "name_en": meta["name_en"],
        "weighting": "cos(lat)*valid_count",
        "coverage_definition": "valid_count / ocean_grid_count",
        "mean_sst_cos_weighted": round(area_weighted_sst, 4),
        "area_weighted_annual_sst": round(area_weighted_sst, 4),
        "grid_area_weighted_annual_sst": round(grid_area_weighted_sst, 4),
        "max_sst_lat": round(float(lat_arr[i_max]), 3),
        "max_sst_c": round(float(annual[i_max]), 3),
        "min_sst_lat": round(float(lat_arr[i_min]), 3),
        "min_sst_c": round(float(annual[i_min]), 3),
        "isotherms_25c": split_isotherms(usable_lats, usable_sst, 25.0),
        "isotherms_0c": split_isotherms(usable_lats, usable_sst, 0.0),
        "valid_lat_min": round(float(usable_lats[0]), 3),
        "valid_lat_max": round(float(usable_lats[-1]), 3),
        "ocean_grid_total": int(np.sum(bmask)),
        "lat_row_count": int(len(df_basin)),
    }


def save_npz_if_changed(path: Path, arrays: dict[str, np.ndarray]) -> tuple[float, bool]:
    """仅当文件不存在或数组内容发生变化时覆写 npz，避免无意义修改既有产物时间戳。"""
    if path.exists():
        try:
            with np.load(path) as existing:
                if set(existing.files) == set(arrays):
                    same = True
                    for k, arr in arrays.items():
                        old = existing[k]
                        if old.shape != arr.shape or old.dtype != arr.dtype:
                            same = False
                            break
                        eq_nan = bool(np.issubdtype(arr.dtype, np.floating))
                        if not np.array_equal(old, arr, equal_nan=eq_nan):
                            same = False
                            break
                    if same:
                        return path.stat().st_size / 1e6, False
        except Exception:  # noqa: BLE001
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path.stat().st_size / 1e6, True


def check_indian_regression_if_available() -> bool | None:
    """若本地存在印度洋原始缓存（含主工作区 D:/PythonProject1/data/raw），执行逐字节回归比对。"""
    candidates = [RAW_DIR, Path("D:/PythonProject1/data/raw")]
    indian_raw = next((p for p in candidates if p.exists() and any(p.glob("sst_*.csv*"))), None)
    if indian_raw is None or not ZONAL_CSV.exists():
        return None
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_csv = Path(tmpdir) / "indian_zonal_mean.csv"
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--region",
            "indian_ocean",
            "--raw-dir",
            str(indian_raw),
            "--out",
            str(tmp_csv),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            return False
        ref_bytes = ZONAL_CSV.read_bytes().replace(b"\r\n", b"\n")
        new_bytes = tmp_csv.read_bytes().replace(b"\r\n", b"\n")
        return ref_bytes == new_bytes


def extract_anomaly_checkpoints(
    lat: np.ndarray,
    lon: np.ndarray,
    anom_annual: np.ndarray,
) -> dict[str, dict]:
    """抽取五大典型洋流区（黑潮/湾流/秘鲁/东太冷舌/本格拉）的同纬度异常值。"""
    specs = [
        ("kuroshio", "黑潮区", 35.125, 141.625),
        ("gulf_stream", "湾流区", 38.0, 295.0),
        ("peru_current", "秘鲁寒流区", -15.0, 282.0),
        ("east_pacific_cold_tongue", "东太平洋冷舌", -1.0, 250.0),
        ("benguela_current", "本格拉寒流区", -20.0, 12.0),
    ]
    out: dict[str, dict] = {}
    for key, name_zh, target_lat, target_lon in specs:
        i0 = int(np.argmin(np.abs(lat - target_lat)))
        j0 = int(np.argmin(np.abs(lon - (target_lon % 360.0))))
        val = float(anom_annual[i0, j0])
        if not np.isfinite(val):
            # 若最近点恰落岸上，则在 1° 邻域内搜索最近有效海面点
            best_d2 = np.inf
            best_ij: tuple[int, int] | None = None
            for di in range(-4, 5):
                for dj in range(-4, 5):
                    ii = min(max(i0 + di, 0), len(lat) - 1)
                    jj = (j0 + dj) % len(lon)
                    if np.isfinite(anom_annual[ii, jj]):
                        d2 = (float(lat[ii]) - target_lat) ** 2 + (float(lon[jj]) - target_lon) ** 2
                        if d2 < best_d2:
                            best_d2 = d2
                            best_ij = (ii, jj)
            if best_ij is None:
                raise AssertionError(f"异常场抽检点 {name_zh} 附近无有效海面格点")
            i0, j0 = best_ij
            val = float(anom_annual[i0, j0])

        out[key] = {
            "name_zh": name_zh,
            "target_lat": target_lat,
            "target_lon": target_lon,
            "grid_lat": round(float(lat[i0]), 3),
            "grid_lon": round(float(lon[j0]), 3),
            "sst_annual_anom_c": round(val, 3),
        }
    return out


def extract_dispersion_highlights(df_global_icefree: pd.DataFrame) -> dict:
    """提取空间东西岸温差最大纬度、5年年际波动最大纬度及40°N/40°S季节滞后特征。"""
    cov = df_global_icefree["coverage"].to_numpy(dtype=float)
    usable = (cov >= COVERAGE_MIN) & np.isfinite(df_global_icefree["sst_annual"].to_numpy(dtype=float))
    sub = df_global_icefree[usable].copy()

    spread = sub["sst_p90"].to_numpy(dtype=float) - sub["sst_p10"].to_numpy(dtype=float)
    i_spread = int(np.nanargmax(spread))
    row_spread = sub.iloc[i_spread]

    istd = sub["sst_interannual_std"].to_numpy(dtype=float)
    i_istd = int(np.nanargmax(istd))
    row_istd = sub.iloc[i_istd]

    def lag_at(target_lat: float) -> dict:
        idx = int(np.argmin(np.abs(sub["latitude"].to_numpy(dtype=float) - target_lat)))
        r = sub.iloc[idx]
        return {
            "lat": round(float(r["latitude"]), 3),
            "warmest_month": int(r["warmest_month"]),
            "coldest_month": int(r["coldest_month"]),
            "sst_warmest": round(float(r["sst_warmest"]), 3),
            "sst_coldest": round(float(r["sst_coldest"]), 3),
            "seasonal_amp": round(float(r["seasonal_amp"]), 3),
        }

    return {
        "max_zonal_p90_p10": {
            "lat": round(float(row_spread["latitude"]), 3),
            "spread_c": round(float(spread[i_spread]), 3),
            "sst_p10": round(float(row_spread["sst_p10"]), 3),
            "sst_p90": round(float(row_spread["sst_p90"]), 3),
        },
        "max_interannual_std": {
            "lat": round(float(row_istd["latitude"]), 3),
            "std_c": round(float(istd[i_istd]), 3),
            "interannual_min": round(float(row_istd["interannual_min"]), 3),
            "interannual_max": round(float(row_istd["interannual_max"]), 3),
        },
        "midlat_40n_seasonal_lag": lag_at(40.0),
        "midlat_40s_seasonal_lag": lag_at(-40.0),
    }


def run_global(raw_dir: Path, ice_dir: Path, global_csv_out: Path) -> int:
    t0 = time.time()
    preset = REGION_PRESETS["global"]
    print(f"读取全球 SST 原始数据：{raw_dir}", flush=True)
    lat, lon, sst_stack, years, files = read_global_grid_stack(raw_dir, preset)
    print(f"SST 时次完整：{len(files)} 个（{years[0]}–{years[-1]}，每年 12 个月）", flush=True)

    print(f"读取全球海冰原始数据：{ice_dir}", flush=True)
    ice_stack, ice_files = read_global_ice_stack(ice_dir, preset, years, lat, lon)
    print(f"海冰时次完整：{len(ice_files)} 个（阈值 ice >= {ICE_CONC_MAX}）", flush=True)

    finite_stack = np.isfinite(sst_stack)
    valid_steps = finite_stack.sum(axis=(0, 1)).astype(np.uint8)
    ocean_valid_mask = valid_steps > 0

    inland_mask, inland_breakdown = inland_lake_mask(lat, lon, ocean_valid_mask)
    excluded_inland_cells = int(np.sum(inland_mask))
    basin_mask = build_basin_mask(lat, lon, ocean_valid_mask)
    checkpoints = verify_basin_mask(lat, lon, basin_mask, ocean_valid_mask)
    print(
        f"五大洋掩膜构建与 {len(checkpoints)} 个关键地理点抽检通过"
        f"（剔除封闭内陆湖泊 {excluded_inland_cells} 格点：{inland_breakdown}）",
        flush=True,
    )

    # 双口径格点场：all_ocean 保持兼容，ice_free 先剔除 ice >= 0.15 并屏蔽陆地/内陆湖
    ice_masked_stack = np.isfinite(ice_stack) & (ice_stack >= ICE_CONC_MAX)
    sst_stack_icefree = np.where(ice_masked_stack, np.nan, sst_stack)
    finite_stack_icefree = np.isfinite(sst_stack_icefree)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        grid_clim = np.nanmean(sst_stack, axis=0)                    # (12, 720, 1440)
        grid_annual = np.nanmean(grid_clim, axis=0)                  # (720, 1440)
        grid_clim_icefree = np.nanmean(sst_stack_icefree, axis=0)    # (12, 720, 1440)
        grid_annual_icefree_raw = np.nanmean(grid_clim_icefree, axis=0)

    grid_jan = grid_clim[0]
    grid_jul = grid_clim[6]

    ocean_2d = basin_mask > 0
    grid_annual_icefree = np.where(ocean_2d, grid_annual_icefree_raw, np.nan)
    grid_jan_icefree = np.where(ocean_2d, grid_clim_icefree[0], np.nan)
    grid_jul_icefree = np.where(ocean_2d, grid_clim_icefree[6], np.nan)

    ice_steps = (ice_masked_stack & ocean_2d[None, None, :, :]).sum(axis=(0, 1)).astype(np.uint8)
    ice_freq = (ice_steps.astype(np.float32) / np.float32(60.0)).astype(np.float32)
    valid_steps_icefree = (finite_stack_icefree & ocean_2d[None, None, :, :]).sum(axis=(0, 1)).astype(np.uint8)

    # 保存轻量级 basin_mask.npz（仅当缺失或内容变化时写入）
    basin_names = np.array([BASINS[i]["key"] for i in range(6)], dtype=str)
    basin_id_grid = np.where(basin_mask == 0, np.int8(-1), basin_mask.astype(np.int8))
    mask_arrays = {
        "lat": lat.astype(np.float32),
        "lon": lon.astype(np.float32),
        "basin_mask": basin_mask.astype(np.uint8),
        "basin_id": basin_id_grid,
        "basin_names": basin_names,
    }
    mask_npz_mb, mask_written = save_npz_if_changed(BASIN_MASK_NPZ, mask_arrays)
    print(f"{'已写出' if mask_written else '保留既有'} {BASIN_MASK_NPZ}（{mask_npz_mb:.3f} MB）", flush=True)

    # 保留 climatology_grid.npz 原有结构不被破坏（仅当缺失或与当前数据不一致时写入）
    base_arrays = {
        "lat": lat.astype(np.float32),
        "lon": lon.astype(np.float32),
        "latitude": lat.astype(np.float32),
        "longitude": lon.astype(np.float32),
        "basin_mask": basin_mask.astype(np.uint8),
        "sst_annual": grid_annual.astype(np.float32),
        "sst_jan": grid_jan.astype(np.float32),
        "sst_jul": grid_jul.astype(np.float32),
        "valid_steps": valid_steps.astype(np.uint8),
    }
    if CLIMATOLOGY_NPZ.exists():
        npz_mb, clim_written = save_npz_if_changed(CLIMATOLOGY_NPZ, base_arrays)
        includes_monthly_grids = False
    else:
        monthly_arrays = {f"sst_{m + 1:02d}": grid_clim[m].astype(np.float32) for m in range(12)}
        CLIMATOLOGY_NPZ.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CLIMATOLOGY_NPZ, **base_arrays, **monthly_arrays)
        npz_mb = CLIMATOLOGY_NPZ.stat().st_size / 1e6
        includes_monthly_grids = npz_mb <= 20.0
        if not includes_monthly_grids:
            np.savez_compressed(CLIMATOLOGY_NPZ, **base_arrays)
            npz_mb = CLIMATOLOGY_NPZ.stat().st_size / 1e6
        clim_written = True
    print(
        f"{'已写出' if clim_written else '保留既有'} {CLIMATOLOGY_NPZ}"
        f"（{npz_mb:.2f} MB，含逐月网格={includes_monthly_grids}）",
        flush=True,
    )

    # P1：参考纬向均值表 data/zonal_ref.csv 与同纬度异常场 data/anomaly_grid.npz
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ref_annual = np.round(np.nanmean(grid_annual_icefree, axis=1), 6)
        ref_jan = np.round(np.nanmean(grid_jan_icefree, axis=1), 6)
        ref_jul = np.round(np.nanmean(grid_jul_icefree, axis=1), 6)

    df_zonal_ref = pd.DataFrame(
        {
            "latitude": np.round(lat, 3),
            "ref_annual": ref_annual,
            "ref_jan": ref_jan,
            "ref_jul": ref_jul,
        }
    )
    ZONAL_REF_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_zonal_ref.to_csv(ZONAL_REF_CSV, index=False, encoding="utf-8", lineterminator="\n")
    print(f"已写出 {ZONAL_REF_CSV}（{len(df_zonal_ref)} 行 × {len(df_zonal_ref.columns)} 列）", flush=True)

    sst_annual_anom = (grid_annual_icefree - ref_annual[:, None]).astype(np.float32)
    sst_jan_anom = (grid_jan_icefree - ref_jan[:, None]).astype(np.float32)
    sst_jul_anom = (grid_jul_icefree - ref_jul[:, None]).astype(np.float32)

    anom_arrays = {
        "lat": lat.astype(np.float32),
        "latitude": lat.astype(np.float32),
        "lon": lon.astype(np.float32),
        "longitude": lon.astype(np.float32),
        "basin_mask": basin_mask.astype(np.uint8),
        "ice_steps": ice_steps,
        "ice_freq": ice_freq,
        "valid_steps_icefree": valid_steps_icefree,
        "sst_annual_icefree": grid_annual_icefree.astype(np.float32),
        "sst_jan_icefree": grid_jan_icefree.astype(np.float32),
        "sst_jul_icefree": grid_jul_icefree.astype(np.float32),
        "sst_annual_anom": sst_annual_anom,
        "sst_jan_anom": sst_jan_anom,
        "sst_jul_anom": sst_jul_anom,
    }
    ANOMALY_GRID_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ANOMALY_GRID_NPZ, **anom_arrays)
    anomaly_npz_mb = ANOMALY_GRID_NPZ.stat().st_size / 1e6
    print(f"已写出 {ANOMALY_GRID_NPZ}（{anomaly_npz_mb:.2f} MB）", flush=True)

    anomaly_checkpoints = extract_anomaly_checkpoints(lat, lon, sst_annual_anom)

    # 逐海盆双口径计算纬向平均 CSV 与汇总指标
    basin_frames_all: list[pd.DataFrame] = []
    basin_frames_icefree: list[pd.DataFrame] = []
    summaries_all: dict[str, dict] = {}
    summaries_icefree: dict[str, dict] = {}
    df_global_icefree_for_highlights: pd.DataFrame | None = None

    for basin_key in BASIN_ORDER:
        if basin_key == "global":
            bid = None
            bmask = ocean_2d
        else:
            bid = BASIN_IDS[basin_key]
            bmask = basin_mask == bid

        core_all = compute_zonal_core(bmask, sst_stack, finite_stack)
        core_icefree = compute_zonal_core(bmask, sst_stack_icefree, finite_stack_icefree)
        extras = compute_icefree_extras(
            bmask,
            core_icefree,
            ice_masked_stack,
            grid_annual_icefree,
            grid_jan_icefree,
            grid_jul_icefree,
        )

        df_all_base = build_core_df(basin_key, lat, core_all)
        df_icefree_base = build_core_df(basin_key, lat, core_icefree)
        df_all_full, df_icefree_full = append_v2_columns(df_all_base, df_icefree_base, extras)

        if basin_key == "global":
            df_g_all = df_all_full.drop(columns=["basin"]).copy()
            cov_pos = df_g_all.columns.get_loc("coverage") + 1
            df_g_all.insert(
                cov_pos,
                "zonal_ocean_frac",
                np.round(df_g_all["valid_count"].to_numpy(dtype=float) / preset["n_lon"], 4),
            )
            global_csv_out.parent.mkdir(parents=True, exist_ok=True)
            df_g_all.to_csv(global_csv_out, index=False, encoding="utf-8", lineterminator="\n")
            print(f"已写出 {global_csv_out}（{len(df_g_all)} 行 × {len(df_g_all.columns)} 列）", flush=True)

            df_g_if = df_icefree_full.drop(columns=["basin"]).copy()
            cov_pos_if = df_g_if.columns.get_loc("coverage") + 1
            df_g_if.insert(
                cov_pos_if,
                "zonal_ocean_frac",
                np.round(df_g_if["valid_count"].to_numpy(dtype=float) / preset["n_lon"], 4),
            )
            GLOBAL_ZONAL_ICEFREE_CSV.parent.mkdir(parents=True, exist_ok=True)
            df_g_if.to_csv(GLOBAL_ZONAL_ICEFREE_CSV, index=False, encoding="utf-8", lineterminator="\n")
            print(
                f"已写出 {GLOBAL_ZONAL_ICEFREE_CSV}（{len(df_g_if)} 行 × {len(df_g_if.columns)} 列）",
                flush=True,
            )
            df_global_icefree_for_highlights = df_g_if

        df_all_nz = df_all_full[df_all_full["ocean_grid_count"] > 0].copy()
        df_if_nz = df_icefree_full[df_icefree_full["ocean_grid_count"] > 0].copy()
        basin_frames_all.append(df_all_nz)
        basin_frames_icefree.append(df_if_nz)

        s_all = build_basin_summary(bid, df_all_nz, bmask, lat, grid_annual)
        s_if = build_basin_summary(bid, df_if_nz, bmask, lat, grid_annual_icefree)

        # 海冰遮蔽全海盆面积加权占比：cos(lat) * ocean_grid_count 加权
        lat_nz = df_if_nz["latitude"].to_numpy(dtype=float)
        ogc_nz = df_if_nz["ocean_grid_count"].to_numpy(dtype=float)
        ifrac_nz = df_if_nz["ice_frac"].to_numpy(dtype=float)
        w_area = ogc_nz * np.cos(np.deg2rad(lat_nz))
        mean_ice_frac = float(np.sum(w_area * ifrac_nz) / np.sum(w_area))

        s_if["delta_vs_all_ocean_c"] = round(
            s_if["area_weighted_annual_sst"] - s_all["area_weighted_annual_sst"], 4
        )
        s_if["mean_ice_frac"] = round(mean_ice_frac, 4)

        summaries_icefree[basin_key] = dict(s_if)
        s_all_with_if = dict(s_all)
        s_all_with_if["icefree"] = dict(s_if)
        summaries_all[basin_key] = s_all_with_if

    df_basins_out = pd.concat(basin_frames_all, ignore_index=True)
    BASINS_ZONAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_basins_out.to_csv(BASINS_ZONAL_CSV, index=False, encoding="utf-8", lineterminator="\n")
    print(f"已写出 {BASINS_ZONAL_CSV}（{len(df_basins_out)} 行 × {len(df_basins_out.columns)} 列）", flush=True)

    df_basins_if_out = pd.concat(basin_frames_icefree, ignore_index=True)
    BASINS_ZONAL_ICEFREE_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_basins_if_out.to_csv(BASINS_ZONAL_ICEFREE_CSV, index=False, encoding="utf-8", lineterminator="\n")
    print(
        f"已写出 {BASINS_ZONAL_ICEFREE_CSV}（{len(df_basins_if_out)} 行 × {len(df_basins_if_out.columns)} 列）",
        flush=True,
    )

    assert df_global_icefree_for_highlights is not None
    dispersion_highlights = extract_dispersion_highlights(df_global_icefree_for_highlights)

    # 统计 raw_global 与 raw_global_ice 体积
    def _manifest_csv_bytes(d: Path) -> int:
        mp = d / "manifest.json"
        if not mp.exists():
            return 0
        try:
            return int(json.loads(mp.read_text(encoding="utf-8")).get("total_bytes_csv", 0))
        except Exception:  # noqa: BLE001
            return 0

    raw_gz_bytes = sum(p.stat().st_size for p in files)
    raw_csv_bytes = _manifest_csv_bytes(raw_dir)
    ice_raw_gz_bytes = sum(p.stat().st_size for p in ice_files)
    ice_raw_csv_bytes = _manifest_csv_bytes(ice_dir)

    compute_seconds = round(time.time() - t0, 2)
    reg_match = check_indian_regression_if_available()

    summary_payload = {
        "icefree_definition": "ice < 0.15 (OISST ice variable)",
        "ice_threshold": ICE_CONC_MAX,
        "weighting": "cos(lat)*valid_count",
        "coverage_definition": "valid_count / ocean_grid_count",
        "basins": summaries_all,
        "basins_icefree": summaries_icefree,
        "dispersion_highlights": dispersion_highlights,
        "anomaly_checkpoints": anomaly_checkpoints,
        "pipeline": {
            "raw_files_count": len(files),
            "raw_gz_mb": round(raw_gz_bytes / 1e6, 2),
            "raw_uncompressed_mb": round(raw_csv_bytes / 1e6, 2),
            "ice_raw_files_count": len(ice_files),
            "ice_raw_gz_mb": round(ice_raw_gz_bytes / 1e6, 2),
            "ice_raw_uncompressed_mb": round(ice_raw_csv_bytes / 1e6, 2),
            "compute_seconds": compute_seconds,
            "compute_v2_seconds": compute_seconds,
            "basin_mask_npz_mb": round(mask_npz_mb, 3),
            "npz_mb": round(npz_mb, 2),
            "anomaly_grid_npz_mb": round(anomaly_npz_mb, 2),
            "npz_includes_monthly_grids": includes_monthly_grids,
            "excluded_inland_cells": excluded_inland_cells,
            "excluded_inland_breakdown": inland_breakdown,
            "indian_regression_exact_match": reg_match,
        },
        "checkpoints": checkpoints,
    }
    SUMMARY_JSON.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写出 {SUMMARY_JSON}（计算耗时 {compute_seconds:.2f}s）", flush=True)

    print("\n全球与五大洋年均 SST 双口径摘要（all_ocean vs ice_free < 0.15）：")
    for k in BASIN_ORDER:
        s = summaries_all[k]
        sif = s["icefree"]
        print(
            f"  {k:8s} ({s['name_zh']}): "
            f"全海面={s['area_weighted_annual_sst']:.3f}°C | "
            f"无冰={sif['area_weighted_annual_sst']:.3f}°C "
            f"(Δ={sif['delta_vs_all_ocean_c']:+.3f}°C, 冰占比={sif['mean_ice_frac']*100:.2f}%)"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="计算 NOAA OISST v2.1 纬向平均海表温度")
    parser.add_argument(
        "--region",
        choices=tuple(REGION_PRESETS),
        default="global",
        help="区域预设：global（默认）或 indian_ocean",
    )
    parser.add_argument("--raw-dir", type=Path, default=None, help="覆盖 SST 原始缓存目录")
    parser.add_argument("--ice-dir", type=Path, default=None, help="覆盖海冰原始缓存目录（默认 data/raw_global_ice）")
    parser.add_argument("--out", type=Path, default=None, help="覆盖主输出 CSV 路径")
    args = parser.parse_args()

    preset = REGION_PRESETS[args.region]
    raw_dir = args.raw_dir if args.raw_dir is not None else preset["raw_dir"]
    ice_dir = args.ice_dir if args.ice_dir is not None else preset.get("ice_raw_dir", GLOBAL_ICE_RAW_DIR)
    out_path = args.out if args.out is not None else preset["out_csv"]

    if args.region == "indian_ocean":
        return run_indian_ocean(raw_dir, out_path)
    return run_global(raw_dir, ice_dir, out_path)


if __name__ == "__main__":
    sys.exit(main())
