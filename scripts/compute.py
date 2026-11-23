#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""读取原始 OISST 数据，计算纬向平均及全球五大洋契约产物。

支持两个区域预设：
- --region global（默认）：
  读取 data/raw_global/ 下的 60 个全球 0.25° (720×1440) 时次，构建五大洋互斥掩膜，
  输出：
    1. data/zonal_mean_global.csv
    2. data/zonal_mean_basins.csv
    3. data/climatology_grid.npz
    4. data/summary_metrics.json
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
    BASIN_IDS,
    BASIN_MASK_NPZ,
    BASIN_ORDER,
    BASINS,
    BASINS_ZONAL_CSV,
    CLIMATOLOGY_NPZ,
    COVERAGE_MIN,
    GLOBAL_ZONAL_CSV,
    N_LAT,
    N_LON,
    RAW_DIR,
    REGION_PRESETS,
    ROOT,
    SUMMARY_JSON,
    ZONAL_CSV,
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


def compute_basin_zonal_df(
    basin_key: str,
    bmask: np.ndarray,
    lat: np.ndarray,
    sst_stack: np.ndarray,
    finite_stack: np.ndarray,
) -> pd.DataFrame:
    """按给定海盆布尔掩膜向量化计算逐纬度月度/年均纬向平均与分月覆盖率。"""
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

    valid_count = counts_ym.mean(axis=(0, 1))
    month_valid_count = counts_ym.mean(axis=0)  # (12, n_lat)

    denom = np.where(ocean_grid_count > 0, ocean_grid_count, 1)
    coverage = np.where(ocean_grid_count > 0, valid_count / denom, 0.0)
    cov_m = np.where(ocean_grid_count[None, :] > 0, month_valid_count / denom[None, :], 0.0)

    data: dict[str, object] = {
        "basin": basin_key,
        "latitude": np.round(lat, 3),
        "ocean_grid_count": ocean_grid_count,
        "valid_count": np.round(valid_count, 1),
        "coverage": np.round(coverage, 4),
        "sst_annual": np.round(annual, 3),
        "sst_jan": np.round(clim_m[0], 3),
        "sst_jul": np.round(clim_m[6], 3),
    }
    for m in range(12):
        data[f"sst_{m + 1:02d}"] = np.round(clim_m[m], 3)
    for m in range(12):
        data[f"cov_{m + 1:02d}"] = np.round(cov_m[m], 4)

    return pd.DataFrame(data)


def build_basin_summary(
    basin_key: str,
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


def run_global(raw_dir: Path, global_csv_out: Path) -> int:
    t0 = time.time()
    preset = REGION_PRESETS["global"]
    print(f"读取全球原始数据：{raw_dir}", flush=True)
    lat, lon, sst_stack, years, files = read_global_grid_stack(raw_dir, preset)
    print(f"时次完整：{len(files)} 个（{years[0]}–{years[-1]}，每年 12 个月）", flush=True)

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

    # 网格级月气候态与年均气候态 (720, 1440)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        grid_clim = np.nanmean(sst_stack, axis=0)   # (12, 720, 1440)
        grid_annual = np.nanmean(grid_clim, axis=0)  # (720, 1440)
    grid_jan = grid_clim[0]
    grid_jul = grid_clim[6]

    # 保存轻量级 basin_mask.npz（供绘图海盆边界复用，同时提供 uint8 basin_mask 与 int8 basin_id）
    basin_names = np.array([BASINS[i]["key"] for i in range(6)], dtype=str)
    basin_id_grid = np.where(basin_mask == 0, np.int8(-1), basin_mask.astype(np.int8))
    BASIN_MASK_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        BASIN_MASK_NPZ,
        lat=lat.astype(np.float32),
        lon=lon.astype(np.float32),
        basin_mask=basin_mask.astype(np.uint8),
        basin_id=basin_id_grid,
        basin_names=basin_names,
    )
    mask_npz_mb = BASIN_MASK_NPZ.stat().st_size / 1e6
    print(f"已写出 {BASIN_MASK_NPZ}（{mask_npz_mb:.3f} MB）", flush=True)

    # 保存 climatology_grid.npz（兼容 lat/lon 与 latitude/longitude；若含 12 月气候态 <= 20 MB 则保留）
    CLIMATOLOGY_NPZ.parent.mkdir(parents=True, exist_ok=True)
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
    monthly_arrays = {f"sst_{m + 1:02d}": grid_clim[m].astype(np.float32) for m in range(12)}
    np.savez_compressed(CLIMATOLOGY_NPZ, **base_arrays, **monthly_arrays)
    npz_mb = CLIMATOLOGY_NPZ.stat().st_size / 1e6
    includes_monthly_grids = npz_mb <= 20.0
    if not includes_monthly_grids:
        print(f"climatology_grid.npz 含 12 月网格时体积 {npz_mb:.2f} MB > 20 MB，回退为仅保留 annual/jan/jul", flush=True)
        np.savez_compressed(CLIMATOLOGY_NPZ, **base_arrays)
        npz_mb = CLIMATOLOGY_NPZ.stat().st_size / 1e6
    print(f"已写出 {CLIMATOLOGY_NPZ}（{npz_mb:.2f} MB，含逐月网格={includes_monthly_grids}）", flush=True)

    # 逐海盆计算纬向平均 CSV 与汇总指标
    basin_frames: list[pd.DataFrame] = []
    summaries: dict[str, dict] = {}

    for basin_key in BASIN_ORDER:
        if basin_key == "global":
            bid = None
            bmask = basin_mask > 0
        else:
            bid = BASIN_IDS[basin_key]
            bmask = basin_mask == bid

        df_full = compute_basin_zonal_df(basin_key, bmask, lat, sst_stack, finite_stack)

        if basin_key == "global":
            df_global_out = df_full.drop(columns=["basin"]).copy()
            cov_pos = df_global_out.columns.get_loc("coverage") + 1
            df_global_out.insert(
                cov_pos,
                "zonal_ocean_frac",
                np.round(df_global_out["valid_count"].to_numpy(dtype=float) / preset["n_lon"], 4),
            )
            global_csv_out.parent.mkdir(parents=True, exist_ok=True)
            df_global_out.to_csv(global_csv_out, index=False, encoding="utf-8", lineterminator="\n")
            print(f"已写出 {global_csv_out}（{len(df_global_out)} 行 × {len(df_global_out.columns)} 列）", flush=True)

        df_nonzero = df_full[df_full["ocean_grid_count"] > 0].copy()
        basin_frames.append(df_nonzero)
        summaries[basin_key] = build_basin_summary(basin_key, bid, df_nonzero, bmask, lat, grid_annual)

    df_basins_out = pd.concat(basin_frames, ignore_index=True)
    BASINS_ZONAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_basins_out.to_csv(BASINS_ZONAL_CSV, index=False, encoding="utf-8", lineterminator="\n")
    print(f"已写出 {BASINS_ZONAL_CSV}（{len(df_basins_out)} 行 × {len(df_basins_out.columns)} 列）", flush=True)

    # 统计 raw_global 体积
    raw_gz_bytes = sum(p.stat().st_size for p in files)
    manifest_path = raw_dir / "manifest.json"
    raw_csv_bytes = 0
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            raw_csv_bytes = int(manifest.get("total_bytes_csv", 0))
        except Exception:  # noqa: BLE001
            raw_csv_bytes = 0

    compute_seconds = round(time.time() - t0, 2)
    reg_match = check_indian_regression_if_available()

    summary_payload = {
        "weighting": "cos(lat)*valid_count",
        "coverage_definition": "valid_count / ocean_grid_count",
        "basins": summaries,
        "pipeline": {
            "raw_files_count": len(files),
            "raw_gz_mb": round(raw_gz_bytes / 1e6, 2),
            "raw_uncompressed_mb": round(raw_csv_bytes / 1e6, 2),
            "compute_seconds": compute_seconds,
            "basin_mask_npz_mb": round(mask_npz_mb, 3),
            "npz_mb": round(npz_mb, 2),
            "npz_includes_monthly_grids": includes_monthly_grids,
            "excluded_inland_cells": excluded_inland_cells,
            "excluded_inland_breakdown": inland_breakdown,
            "indian_regression_exact_match": reg_match,
        },
        "checkpoints": checkpoints,
    }
    SUMMARY_JSON.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写出 {SUMMARY_JSON}（计算耗时 {compute_seconds:.2f}s）", flush=True)

    print("\n全球与五大洋年均 SST 摘要：")
    for k in BASIN_ORDER:
        s = summaries[k]
        print(
            f"  {k:8s} ({s['name_zh']}): 面积加权={s['area_weighted_annual_sst']:.2f}°C "
            f"(网格={s['grid_area_weighted_annual_sst']:.2f}°C), "
            f"最高={s['max_sst_c']:.2f}°C@{s['max_sst_lat']:.2f}°, "
            f"最低={s['min_sst_c']:.2f}°C@{s['min_sst_lat']:.2f}°, "
            f"25°C={s['isotherms_25c']}, 0°C={s['isotherms_0c']}"
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
    parser.add_argument("--raw-dir", type=Path, default=None, help="覆盖原始缓存目录")
    parser.add_argument("--out", type=Path, default=None, help="覆盖主输出 CSV 路径")
    args = parser.parse_args()

    preset = REGION_PRESETS[args.region]
    raw_dir = args.raw_dir if args.raw_dir is not None else preset["raw_dir"]
    out_path = args.out if args.out is not None else preset["out_csv"]

    if args.region == "indian_ocean":
        return run_indian_ocean(raw_dir, out_path)
    return run_global(raw_dir, out_path)


if __name__ == "__main__":
    sys.exit(main())
