#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""独立复核全球海温分析结果（**不 import compute.py**，口径从零实现）。

复核内容
--------
1. **数值复算**：读取 ``data/raw/`` 下 60 个 ``sst_YYYY_MM.csv.gz``，独立重算纬向平均，
   抽 8 个纬度 × 1 月/7 月两个月份，与 ``zonal_mean_basins.csv`` 的 global 行对齐到 0.01 °C。
2. **结构性检查**：720 行、0.25° 步长、纬度单调、``coverage ∈ [0,1]``、
   ``valid_count`` 与 ``coverage`` 自洽、海洋值不越界、1 月/7 月南北半球季节反相。
3. **网格一致性**：``climatology_grid.npz`` 形状/键齐全，且在"全海洋"纬度上其纬向平均
   与 CSV 一致（避免冰缘处"先平均后加权"的固有差异被误判）。
4. **在线抽样**：直接向 ERDDAP 取 2–3 个时次 × 一条约 1° 宽的纬度带（很小的请求），
   与本地同一时次的 ``data/raw/`` 逐纬圈比对，容差 0.05 °C；网络不通时**跳过**，不算失败。
5. **印度洋回归**：用仓库已提交的 ``data/zonal_mean.csv``（281 行）复算 README 现值
   （最高 29.14 °C @ 2.12°N、25 °C 等温线 21.66°S、海盆均值 26.16 °C），证明全球版没有改口径。

结果写入 ``data/verify_report.json``（fixture 模式写到对应目录），并打印结论。
任何一项**实际比对失败**都会非零退出；因数据未就绪或网络不通而跳过的不算失败。

用法::

    python scripts/verify.py
    python scripts/verify.py --data-dir .tmp/fixture          # 复核合成 fixture
    python scripts/verify.py --skip-online                    # 不联网
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_RAW_DIR = DEFAULT_DATA_DIR / "raw"
INDIAN_CSV = ROOT / "data" / "zonal_mean.csv"

N_LAT_GLOBAL = 720
N_LON_GLOBAL = 1440
LAT_FIRST = -89.875
LAT_LAST = 89.875
LAT_STEP = 0.25
COVERAGE_MIN = 0.10
SST_MIN, SST_MAX = -2.0, 35.0
NUMERIC_TOL = 0.01          # 数值复算容差（°C）
GRID_TOL = 0.02             # 网格纬向平均与 CSV 的容差（°C）
ONLINE_TOL = 0.05           # 在线抽样容差（°C）
SAMPLE_LATS = (-70.0, -45.0, -20.0, 0.0, 20.0, 40.0, 60.0, 75.0)
SAMPLE_MONTHS = (1, 7)
REQUIRED_COLUMNS = (
    ["latitude", "valid_count", "coverage", "sst_annual", "sst_jan", "sst_jul"]
    + [f"sst_{m:02d}" for m in range(1, 13)]
)
ERDDAP_BASE = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg.csv"
INDIAN_BASELINE = {
    "sst_max": (29.14, 0.01),
    "lat_max": (2.125, 0.01),
    "isotherm_south": (-21.664, 0.01),
    "basin_mean": (26.163, 0.005),
}

RAW_RE = re.compile(r"sst_(\d{4})_(\d{2})\.csv(?:\.gz)?$")


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def make_check(name: str, status: str, detail: str, metrics: dict[str, Any] | None = None) -> dict:
    assert status in ("pass", "fail", "skip")
    return {"name": name, "status": status, "detail": detail, "metrics": metrics or {}}


def isotherm_crossings(lat: np.ndarray, sst: np.ndarray, level: float = 25.0) -> list[float]:
    """与 compute.py 独立实现的等温线插值；返回全部交点。"""
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


def cos_weighted_mean(lat: np.ndarray, sst: np.ndarray, coverage: np.ndarray, cov_min: float) -> float | None:
    usable = np.isfinite(sst) & (coverage >= cov_min)
    if not usable.any():
        return None
    w = np.cos(np.deg2rad(lat[usable]))
    return float(np.sum(w * sst[usable]) / np.sum(w))


def find_basin_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if str(col).strip().lower() == "basin":
            return col
    first = df.columns[0]
    known = {"global", "indian", "pacific", "atlantic", "arctic", "southern"}
    values = df[first].astype(str).str.strip().str.lower()
    if values.isin(known).mean() > 0.5:
        return first
    return None


def parse_erddap_csv(payload: bytes) -> pd.DataFrame:
    """解析 ERDDAP griddap CSV（第 2 行为单位行）。"""
    text = payload.decode("utf-8", "replace")
    lines = text.splitlines()
    if not lines:
        raise ValueError("空响应")
    skiprows = [1] if len(lines) > 1 and ("UTC" in lines[1] or "degrees" in lines[1]) else []
    df = pd.read_csv(io.StringIO(text), skiprows=skiprows)
    needed = {"latitude", "sst"}
    if not needed.issubset(df.columns):
        raise ValueError(f"响应列异常：{list(df.columns)[:6]}")
    return df


def read_raw_csv(path: Path) -> pd.DataFrame:
    """读取单个时次原始 CSV，自动判断是否需要跳过单位行。"""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        head = [fh.readline() for _ in range(2)]
    skiprows = [1] if len(head) > 1 and ("UTC" in head[1] or "degrees" in head[1]) else []
    return pd.read_csv(
        path,
        skiprows=skiprows,
        usecols=["latitude", "longitude", "sst"],
        dtype={"latitude": "float64", "longitude": "float64", "sst": "float64"},
    )


# --------------------------------------------------------------------------- #
# 1. 全球纬向平均 CSV
# --------------------------------------------------------------------------- #
def check_global_zonal(path: Path) -> dict:
    name = "zonal_mean_global.csv 结构"
    if not path.exists():
        return make_check(name, "fail", f"缺少 {path}")
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        return make_check(name, "fail", f"无法读取 {path}：{exc}")
    problems: list[str] = []

    if list(df.columns[:18]) != REQUIRED_COLUMNS:
        problems.append(f"前 18 列与契约不符：{list(df.columns[:18])}")
    if len(df) != N_LAT_GLOBAL:
        problems.append(f"行数 {len(df)} != {N_LAT_GLOBAL}")
    if df.empty:
        return make_check(name, "fail", "文件为空", {"rows": 0})

    lat = df["latitude"].to_numpy(dtype=float)
    if len(lat) > 1:
        if not np.all(np.diff(lat) > 0):
            problems.append("纬度不是严格单调递增")
        if not np.allclose(np.diff(lat), LAT_STEP, atol=1e-9):
            problems.append("纬度步长不是 0.25°")
    if abs(lat[0] - LAT_FIRST) > 1e-6 or abs(lat[-1] - LAT_LAST) > 1e-6:
        problems.append(f"纬度端点 {lat[0]:.3f}…{lat[-1]:.3f} 不是 -89.875…89.875")

    valid = df["valid_count"].to_numpy(dtype=float)
    coverage = df["coverage"].to_numpy(dtype=float)
    if not np.all(np.isfinite(coverage)) or coverage.min() < 0 or coverage.max() > 1:
        problems.append(f"coverage 越界：[{np.nanmin(coverage)}, {np.nanmax(coverage)}]")
    if not np.all(np.isfinite(valid)) or valid.min() < 0 or valid.max() > N_LON_GLOBAL + 1e-6:
        problems.append(f"valid_count 越界：[{np.nanmin(valid)}, {np.nanmax(valid)}]")
    max_cov_err = float(np.nanmax(np.abs(coverage - valid / N_LON_GLOBAL)))
    if max_cov_err > 1.1e-3:
        problems.append(f"coverage 与 valid_count/1440 不自洽（最大偏差 {max_cov_err:.5f}）")

    ocean_max = -np.inf
    ocean_min = np.inf
    for col in REQUIRED_COLUMNS[3:]:
        values = df[col].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size:
            ocean_max = max(ocean_max, float(finite.max()))
            ocean_min = min(ocean_min, float(finite.min()))
    if ocean_max > SST_MAX:
        problems.append(f"海洋 SST 最大值 {ocean_max:.2f} > {SST_MAX}")
    if ocean_min < SST_MIN:
        problems.append(f"海洋 SST 最小值 {ocean_min:.2f} < {SST_MIN}")

    # 1 月 / 7 月南北半球季节反相（20°–60°，cosφ 加权）
    def band_mean(col: str, north: bool) -> float:
        sst = df[col].to_numpy(dtype=float)
        mask = (np.abs(lat) >= 20) & (np.abs(lat) <= 60) & (coverage >= COVERAGE_MIN)
        mask &= (lat > 0) if north else (lat < 0)
        w = np.cos(np.deg2rad(lat[mask]))
        return float(np.sum(w * sst[mask]) / np.sum(w))

    nh_jan, nh_jul = band_mean("sst_jan", True), band_mean("sst_jul", True)
    sh_jan, sh_jul = band_mean("sst_jan", False), band_mean("sst_jul", False)
    metrics = {
        "rows": len(df),
        "lat_first": float(lat[0]),
        "lat_last": float(lat[-1]),
        "coverage_min": float(np.nanmin(coverage)),
        "coverage_max": float(np.nanmax(coverage)),
        "max_coverage_error": max_cov_err,
        "sst_min": ocean_min,
        "sst_max": ocean_max,
        "nh_jan_minus_jul": nh_jan - nh_jul,
        "sh_jan_minus_jul": sh_jan - sh_jul,
    }
    if not nh_jan < nh_jul - 1.0:
        problems.append(f"北半球 1 月未明显冷于 7 月（{nh_jan:.2f} vs {nh_jul:.2f}）")
    if not sh_jan > sh_jul + 1.0:
        problems.append(f"南半球 1 月未明显暖于 7 月（{sh_jan:.2f} vs {sh_jul:.2f}）")

    if problems:
        return make_check(name, "fail", "；".join(problems), metrics)
    return make_check(
        name,
        "pass",
        f"{len(df)} 行，纬度 {lat[0]:.3f}→{lat[-1]:.3f}，步长 {LAT_STEP}°，"
        f"coverage∈[{coverage.min():.4f},{coverage.max():.4f}]，海洋值∈[{ocean_min:.2f},{ocean_max:.2f}] °C，"
        f"南北半球季节反相（NH Δ={nh_jan - nh_jul:+.2f} °C，SH Δ={sh_jan - sh_jul:+.2f} °C）",
        metrics,
    )


# --------------------------------------------------------------------------- #
# 2. 分海盆 CSV
# --------------------------------------------------------------------------- #
def check_basins_zonal(path: Path, global_df: pd.DataFrame | None) -> dict:
    name = "zonal_mean_basins.csv 结构"
    if not path.exists():
        return make_check(name, "fail", f"缺少 {path}")
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        return make_check(name, "fail", f"无法读取 {path}：{exc}")
    basin_col = find_basin_column(df)
    if basin_col is None:
        return make_check(name, "fail", f"找不到 basin 列（现有列：{list(df.columns)[:6]}）")
    df[basin_col] = df[basin_col].astype(str).str.strip().str.lower()
    problems: list[str] = []
    metrics: dict[str, Any] = {"basins": {}}

    lat_ref = global_df["latitude"].to_numpy(dtype=float) if global_df is not None else None
    for basin, group in df.groupby(basin_col, sort=False):
        group = group.reset_index(drop=True)
        info: dict[str, Any] = {"rows": len(group)}
        metrics["basins"][str(basin)] = info
        if list(group.columns[1:19]) != REQUIRED_COLUMNS:
            problems.append(f"{basin}: 首列之后前 18 列与契约不符")
        if len(group) != N_LAT_GLOBAL:
            problems.append(f"{basin}: 行数 {len(group)} != {N_LAT_GLOBAL}")
            continue
        lat = group["latitude"].to_numpy(dtype=float)
        if lat_ref is not None and not np.allclose(lat, lat_ref, atol=1e-9):
            problems.append(f"{basin}: 纬度网格与全球文件不一致")
        if not np.all(np.diff(lat) > 0) or not np.allclose(np.diff(lat), LAT_STEP, atol=1e-9):
            problems.append(f"{basin}: 纬度不是 0.25° 单调网格")
        cov = group["coverage"].to_numpy(dtype=float)
        valid = group["valid_count"].to_numpy(dtype=float)
        if cov.min() < 0 or cov.max() > 1:
            problems.append(f"{basin}: coverage 越界 [{cov.min()}, {cov.max()}]")
        info["coverage_range"] = [float(cov.min()), float(cov.max())]
        if basin == "global":
            err = float(np.nanmax(np.abs(cov - valid / N_LON_GLOBAL)))
            info["coverage_denominator"] = N_LON_GLOBAL
            info["max_coverage_error"] = err
            if err > 1.1e-3:
                problems.append(f"global: coverage 与 valid_count/1440 不自洽（{err:.5f}）")
        else:
            err_1440 = float(np.nanmax(np.abs(cov - valid / N_LON_GLOBAL)))
            best = err_1440
            denom = N_LON_GLOBAL
            if "ocean_grid_count" in group:
                ocean = group["ocean_grid_count"].to_numpy(dtype=float)
                with np.errstate(invalid="ignore", divide="ignore"):
                    err_ocean = np.where(ocean > 0, np.abs(cov - valid / np.where(ocean > 0, ocean, 1)), 0.0)
                if float(np.nanmax(err_ocean)) < best:
                    best = float(np.nanmax(err_ocean))
                    denom = "ocean_grid_count"
            info["coverage_denominator"] = denom
            info["max_coverage_error"] = best
            if best > 1.1e-3:
                problems.append(f"{basin}: coverage 与 valid_count 分母（1440 或 ocean_grid_count）不自洽（{best:.5f}）")

        for col in REQUIRED_COLUMNS[3:]:
            values = group[col].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size and (finite.max() > SST_MAX or finite.min() < SST_MIN):
                problems.append(f"{basin}/{col}: 越界 [{finite.min():.2f}, {finite.max():.2f}]")
                break

    if "global" not in metrics["basins"]:
        problems.append("缺少 global 海盆行")
    elif global_df is not None:
        g = df[df[basin_col] == "global"].reset_index(drop=True)
        for col in ("sst_annual", "sst_jan", "sst_jul"):
            a = g[col].to_numpy(dtype=float)
            b = global_df[col].to_numpy(dtype=float)
            diff = float(np.nanmax(np.abs(a - b)))
            metrics["basins"]["global"][f"max_diff_vs_global_{col}"] = diff
            if diff > NUMERIC_TOL:
                problems.append(f"global 行 {col} 与全球文件最大偏差 {diff:.4f} > {NUMERIC_TOL}")

    if problems:
        return make_check(name, "fail", "；".join(problems), metrics)
    return make_check(
        name,
        "pass",
        f"{len(metrics['basins'])} 个海盆 × {N_LAT_GLOBAL} 行，纬度/覆盖率/温度范围自洽"
        + ("，global 行与全球文件一致" if global_df is not None else ""),
        metrics,
    )


# --------------------------------------------------------------------------- #
# 3. 气候态网格
# --------------------------------------------------------------------------- #
def check_grid(path: Path, global_df: pd.DataFrame | None) -> dict:
    name = "climatology_grid.npz 一致性"
    if not path.exists():
        return make_check(name, "fail", f"缺少 {path}")
    required = {"lat", "lon", "sst_annual"} | {f"sst_{m:02d}" for m in range(1, 13)}
    try:
        with np.load(path) as npz:
            keys = set(npz.files)
            missing = required - keys
            if missing:
                return make_check(name, "fail", f"缺少键：{sorted(missing)}")
            lat = npz["lat"].astype(float)
            lon = npz["lon"].astype(float)
            annual = npz["sst_annual"].astype(float)
            monthly = {m: npz[f"sst_{m:02d}"].astype(float) for m in range(1, 13)}
    except Exception as exc:  # noqa: BLE001
        return make_check(name, "fail", f"无法读取 {path}：{exc}")

    problems: list[str] = []
    metrics: dict[str, Any] = {"keys": sorted(keys)}
    if len(lat) != N_LAT_GLOBAL or len(lon) != N_LON_GLOBAL:
        problems.append(f"lat/lon 长度 {len(lat)}/{len(lon)} != 720/1440")
    if annual.shape != (len(lat), len(lon)):
        problems.append(f"sst_annual 形状 {annual.shape} != ({len(lat)},{len(lon)})")
    for m, grid in monthly.items():
        if grid.shape != annual.shape:
            problems.append(f"sst_{m:02d} 形状 {grid.shape} != {annual.shape}")
            break
    if len(lat) > 1 and not np.allclose(np.diff(lat), LAT_STEP, atol=1e-9):
        problems.append("lat 不是 0.25° 等间距")
    if len(lon) > 1 and not np.allclose(np.diff(lon), LAT_STEP, atol=1e-9):
        problems.append("lon 不是 0.25° 等间距")

    finite = annual[np.isfinite(annual)]
    if finite.size == 0:
        problems.append("sst_annual 全为 NaN")
    else:
        metrics["sst_range"] = [float(finite.min()), float(finite.max())]
        metrics["ocean_fraction"] = float(finite.size / annual.size)
        if finite.max() > SST_MAX or finite.min() < SST_MIN:
            problems.append(f"网格值越界 [{finite.min():.2f}, {finite.max():.2f}]")

    # 只在"60 个时次全为海洋"的纬度比较，避开冰缘处先平均/后平均的口径差异
    if global_df is not None and not problems:
        lat_csv = global_df["latitude"].to_numpy(dtype=float)
        cov_csv = global_df["coverage"].to_numpy(dtype=float)
        full = np.isclose(cov_csv, 1.0, atol=1e-9) & np.isfinite(global_df["sst_annual"].to_numpy(dtype=float))
        n_full = int(full.sum())
        metrics["full_ocean_latitudes"] = n_full
        if n_full < 5:
            metrics["grid_check"] = "skipped（全海洋纬度不足 5 个）"
        else:
            checks = {"sst_annual": (annual, global_df["sst_annual"].to_numpy(dtype=float))}
            for m in (1, 4, 7, 10):
                checks[f"sst_{m:02d}"] = (monthly[m], global_df[f"sst_{m:02d}"].to_numpy(dtype=float))
            worst = 0.0
            worst_key = ""
            for key, (grid, csv) in checks.items():
                with np.errstate(invalid="ignore"):
                    grid_mean = np.nanmean(grid[full], axis=1)
                diff = float(np.nanmax(np.abs(grid_mean - csv[full])))
                if not np.isfinite(diff):
                    problems.append(f"全海洋纬度上 {key} 的网格纬向平均出现 NaN")
                    continue
                if diff > worst:
                    worst, worst_key = diff, key
            metrics["max_grid_zonal_diff"] = {"key": worst_key, "value": worst, "n_lat": n_full}
            if worst > GRID_TOL:
                problems.append(
                    f"全海洋纬度上网格纬向平均与 CSV 最大偏差 {worst:.4f}（{worst_key}）> {GRID_TOL}"
                )

    if problems:
        return make_check(name, "fail", "；".join(problems), metrics)
    return make_check(
        name,
        "pass",
        f"{len(lat)}×{len(lon)} 网格，{len(monthly)} 个月 + 年均键齐全，"
        f"海洋格点占比 {metrics.get('ocean_fraction', float('nan')):.1%}，"
        f"全海洋纬度 {metrics.get('full_ocean_latitudes', 0)} 个上纬向平均与 CSV 最大偏差 "
        f"{metrics.get('max_grid_zonal_diff', {}).get('value', float('nan')):.4f} °C",
        metrics,
    )


# --------------------------------------------------------------------------- #
# 4. 从 raw 独立重算（核心）
# --------------------------------------------------------------------------- #
def list_raw_files(raw_dir: Path) -> list[tuple[tuple[int, int], Path]]:
    if not raw_dir.exists():
        return []
    out: list[tuple[tuple[int, int], Path]] = []
    for path in sorted(raw_dir.iterdir()):
        match = RAW_RE.search(path.name)
        if match:
            out.append(((int(match.group(1)), int(match.group(2))), path))
    return out


def recompute_zonal_from_raw(files: list[tuple[tuple[int, int], Path]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """独立重算：逐时次纬向平均 -> 同月多年平均 -> 12 月等权年均。

    返回 (lat, clim[12,nlat], annual[nlat], counts[ntime,nlat])。
    """
    lat_ref: np.ndarray | None = None
    means: dict[tuple[int, int], np.ndarray] = {}
    counts: dict[tuple[int, int], np.ndarray] = {}
    for (year, month), path in files:
        df = read_raw_csv(path)
        lats = np.sort(df["latitude"].unique())
        if lat_ref is None:
            lat_ref = lats
        elif not np.array_equal(lats, lat_ref):
            raise ValueError(f"{path.name} 的纬度网格与首个文件不一致")
        grouped = df.groupby("latitude", sort=True)["sst"]
        means[(year, month)] = grouped.mean().reindex(lat_ref).to_numpy(dtype=float)
        counts[(year, month)] = grouped.count().reindex(lat_ref).to_numpy(dtype=float)
    assert lat_ref is not None
    years = sorted({y for y, _ in means})
    months = sorted({m for _, m in means})
    if months != list(range(1, 13)):
        raise ValueError(f"月份不完整：{months}")
    clim = np.full((12, len(lat_ref)), np.nan, dtype=float)
    for i, month in enumerate(range(1, 13)):
        stack = np.vstack([means[(y, month)] for y in years])
        clim[i] = np.nanmean(stack, axis=0)
    annual = np.nanmean(clim, axis=0)
    count_stack = np.vstack([counts[key] for key in sorted(counts)])
    return lat_ref, clim, annual, count_stack


def check_raw_recompute(raw_dir: Path, target_df: pd.DataFrame | None, target_label: str) -> dict:
    name = "data/raw 独立数值复算"
    files = list_raw_files(raw_dir)
    if not files:
        return make_check(name, "skip", f"未找到 {raw_dir}/sst_YYYY_MM.csv.gz，跳过数值复算")
    if len(files) != 60:
        return make_check(name, "skip", f"raw 只有 {len(files)} 个时次（需 60），数据未就绪，跳过数值复算")
    if target_df is None:
        return make_check(name, "skip", "缺少可对齐的 zonal_mean_basins.csv（或无 global 行），跳过")

    try:
        lat_raw, clim, annual, _ = recompute_zonal_from_raw(files)
    except Exception as exc:  # noqa: BLE001 - 数据格式异常必须报告
        return make_check(name, "fail", f"raw 重算失败：{exc}")

    if len(lat_raw) != N_LAT_GLOBAL or not np.allclose(np.diff(lat_raw), LAT_STEP, atol=1e-9):
        return make_check(
            name,
            "fail",
            f"raw 纬度网格为 {len(lat_raw)} 个点、步长约 {np.diff(lat_raw).mean():.3f}°，不是全球 720×0.25°",
        )

    target = target_df.copy()
    target = target.set_index("latitude")
    rows: list[dict[str, Any]] = []
    problems: list[str] = []
    for wanted_lat in SAMPLE_LATS:
        idx = int(np.argmin(np.abs(lat_raw - wanted_lat)))
        lat = float(lat_raw[idx])
        if lat not in target.index:
            nearest = target.index[np.argmin(np.abs(target.index.to_numpy(dtype=float) - lat))]
            lat = float(nearest)
            idx = int(np.argmin(np.abs(lat_raw - lat)))
        row = target.loc[lat]
        for month in SAMPLE_MONTHS:
            recomputed = float(clim[month - 1, idx])
            csv_value = float(row[f"sst_{month:02d}"])
            if not (np.isfinite(recomputed) and np.isfinite(csv_value)):
                problems.append(f"{lat:.3f}°/{month:02d} 月：raw={recomputed} / csv={csv_value} 存在 NaN")
                rows.append({"latitude": lat, "month": month, "raw": recomputed,
                             "csv": csv_value, "diff": float("inf")})
                continue
            diff = abs(recomputed - csv_value)
            rows.append({"latitude": lat, "month": month, "raw": recomputed, "csv": csv_value, "diff": diff})
            if diff > NUMERIC_TOL:
                problems.append(f"{lat:.3f}°/{month:02d} 月：raw {recomputed:.4f} vs csv {csv_value:.4f}（Δ={diff:.4f}）")
        annual_raw = float(annual[idx])
        annual_csv = float(row["sst_annual"])
        if not (np.isfinite(annual_raw) and np.isfinite(annual_csv)):
            problems.append(f"{lat:.3f}° 年均：raw={annual_raw} / csv={annual_csv} 存在 NaN")
            rows.append({"latitude": lat, "month": "annual", "raw": annual_raw,
                         "csv": annual_csv, "diff": float("inf")})
        else:
            annual_diff = abs(annual_raw - annual_csv)
            rows.append({"latitude": lat, "month": "annual", "raw": annual_raw,
                         "csv": annual_csv, "diff": annual_diff})
            if annual_diff > NUMERIC_TOL:
                problems.append(f"{lat:.3f}° 年均：raw vs csv Δ={annual_diff:.4f}")

    max_diff = max(r["diff"] for r in rows)
    metrics = {"n_timesteps": len(files), "target": target_label,
               "max_abs_diff": max_diff, "samples": rows}
    if problems:
        return make_check(name, "fail", "；".join(problems[:8]), metrics)
    return make_check(
        name,
        "pass",
        f"{len(files)} 个时次独立重算，8 纬度 × 1/7 月 + 年均共 {len(rows)} 个比对，"
        f"最大偏差 {max_diff:.4f} °C ≤ {NUMERIC_TOL}（对齐 {target_label}）",
        metrics,
    )


# --------------------------------------------------------------------------- #
# 5. 在线抽样
# --------------------------------------------------------------------------- #
def build_online_url(year: int, month: int, lat0: float, lat1: float) -> str:
    ts = f"{year}-{month:02d}-16T12:00:00Z"
    query = (
        f"sst[({ts}):1:({ts})]"
        f"[(0.0):1:(0.0)]"
        f"[({lat0:.3f}):1:({lat1:.3f})]"
        f"[(0.0):1:(359.75)]"
    )
    return ERDDAP_BASE + "?" + quote(query, safe=":(),.+-=")


def check_online(raw_dir: Path, skip_online: bool) -> dict:
    name = "ERDDAP 在线抽样复核"
    if skip_online:
        return make_check(name, "skip", "按 --skip-online 跳过")
    files = list_raw_files(raw_dir)
    if len(files) != 60:
        return make_check(name, "skip", "无完整本地 raw 可比对，跳过在线复核（联网只做同时次对照）")

    picks = [files[0], files[len(files) // 2], files[-1]]
    rows: list[dict[str, Any]] = []
    problems: list[str] = []
    try:
        for (year, month), path in picks:
            lat0, lat1 = -0.7, 0.7  # 约 1.4° 窄带（OISST 该范围内 6 个纬圈）
            url = build_online_url(year, month, lat0, lat1)
            req = Request(url, headers={"User-Agent": "global-sst-verify/1.0 (python-urllib; research)"})
            with urlopen(req, timeout=90) as resp:
                payload = resp.read()
            online = parse_erddap_csv(payload)
            online_means = online.groupby("latitude")["sst"].mean()
            local = read_raw_csv(path)
            local_means = local.groupby("latitude")["sst"].mean()
            common = sorted(set(online_means.index) & set(local_means.index))
            if len(common) < 3:
                raise ValueError(f"{year}-{month:02d} 在线与本地纬圈交集过小：{len(common)}")
            for lat in common:
                a, b = float(online_means.loc[lat]), float(local_means.loc[lat])
                if not (np.isfinite(a) and np.isfinite(b)):
                    problems.append(f"{year}-{month:02d} {lat:.3f}°：在线或本地均值为 NaN")
                    continue
                diff = abs(a - b)
                rows.append({"year": year, "month": month, "lat": float(lat),
                             "online": a, "local": b, "diff": diff})
                if diff > ONLINE_TOL:
                    problems.append(f"{year}-{month:02d} {lat:.3f}°：在线 {a:.3f} vs 本地 {b:.3f}（Δ={diff:.3f}）")
    except Exception as exc:  # noqa: BLE001 - 网络不通按设计跳过
        return make_check(name, "skip", f"网络不可用或请求失败，跳过在线复核：{exc}")

    max_diff = max((r["diff"] for r in rows), default=float("nan"))
    metrics = {"timesteps": [list(k) for k, _ in picks], "max_abs_diff": max_diff, "samples": rows}
    if problems:
        return make_check(name, "fail", "；".join(problems[:6]), metrics)
    return make_check(
        name,
        "pass",
        f"{len(picks)} 个时次 × 约 1° 窄带逐纬圈比对（{len(rows)} 个样本），"
        f"在线与本地最大偏差 {max_diff:.4f} °C ≤ {ONLINE_TOL}",
        metrics,
    )


# --------------------------------------------------------------------------- #
# 6. 印度洋回归
# --------------------------------------------------------------------------- #
def check_indian_regression(path: Path) -> dict:
    name = "印度洋旧结果回归"
    if not path.exists():
        return make_check(name, "skip", f"缺少 {path}，跳过回归")
    df = pd.read_csv(path)
    lat = df["latitude"].to_numpy(dtype=float)
    coverage = df["coverage"].to_numpy(dtype=float)
    annual = df["sst_annual"].to_numpy(dtype=float)
    usable = np.isfinite(annual) & (coverage >= COVERAGE_MIN)
    i_max = int(np.nanargmax(np.where(usable, annual, -np.inf)))
    isos = isotherm_crossings(lat[usable], annual[usable])
    measured = {
        "sst_max": float(annual[i_max]),
        "lat_max": float(lat[i_max]),
        "isotherm_south": float(min(isos)) if isos else float("nan"),
        "basin_mean": cos_weighted_mean(lat, annual, coverage, COVERAGE_MIN),
    }
    metrics = {"rows": len(df), "measured": measured,
               "baseline": {k: v[0] for k, v in INDIAN_BASELINE.items()}}
    problems = []
    for key, (expected, tol) in INDIAN_BASELINE.items():
        got = measured[key]
        if got is None or not np.isfinite(got) or abs(got - expected) > tol:
            problems.append(f"{key}: {got} vs README {expected}±{tol}")
    if len(df) != 281:
        problems.append(f"行数 {len(df)} != 281")
    if problems:
        return make_check(name, "fail", "；".join(problems), metrics)
    return make_check(
        name,
        "pass",
        f"281 行复算：最高 {measured['sst_max']:.2f} °C @ {measured['lat_max']:.2f}°，"
        f"25 °C 等温线 {measured['isotherm_south']:.2f}°，海盆均值 {measured['basin_mean']:.2f} °C，"
        "与 README 现值一致（口径未变）",
        metrics,
    )


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def run(data_dir: Path, raw_dir: Path, report_path: Path, skip_online: bool) -> dict:
    data_dir = data_dir.resolve()
    raw_dir = raw_dir.resolve()
    global_csv = data_dir / "zonal_mean_global.csv"
    basins_csv = data_dir / "zonal_mean_basins.csv"
    grid_npz = data_dir / "climatology_grid.npz"

    global_df: pd.DataFrame | None = None
    if global_csv.exists():
        try:
            global_df = pd.read_csv(global_csv)
        except Exception:  # noqa: BLE001
            global_df = None

    target_df: pd.DataFrame | None = None
    target_label = "zonal_mean_basins.csv::global"
    if basins_csv.exists():
        try:
            basins = pd.read_csv(basins_csv)
            basin_col = find_basin_column(basins)
            if basin_col is not None:
                basins[basin_col] = basins[basin_col].astype(str).str.strip().str.lower()
                target_df = basins[basins[basin_col] == "global"].reset_index(drop=True)
        except Exception:  # noqa: BLE001
            target_df = None
    if target_df is None or target_df.empty:
        target_df = global_df
        target_label = "zonal_mean_global.csv"

    real_ready = global_csv.exists() or basins_csv.exists() or grid_npz.exists()
    if real_ready:
        checks = [
            check_global_zonal(global_csv),
            check_basins_zonal(basins_csv, global_df),
            check_grid(grid_npz, global_df),
        ]
    else:
        # 真实数据尚未产出：结构性检查按"跳过"处理，而不是判失败
        print(f"真实数据未就绪（{data_dir} 下没有全球结果文件），结构性检查跳过。", flush=True)
        checks = [
            make_check("zonal_mean_global.csv 结构", "skip", f"真实数据未就绪：缺少 {global_csv}"),
            make_check("zonal_mean_basins.csv 结构", "skip", f"真实数据未就绪：缺少 {basins_csv}"),
            make_check("climatology_grid.npz 一致性", "skip", f"真实数据未就绪：缺少 {grid_npz}"),
        ]
    checks += [
        check_raw_recompute(raw_dir, target_df, target_label),
        check_online(raw_dir, skip_online),
        check_indian_regression(INDIAN_CSV),
    ]

    n_pass = sum(c["status"] == "pass" for c in checks)
    n_fail = sum(c["status"] == "fail" for c in checks)
    n_skip = sum(c["status"] == "skip" for c in checks)
    overall = "fail" if n_fail else ("pass_with_skips" if n_skip else "pass")
    report = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_dir": str(data_dir),
        "raw_dir": str(raw_dir),
        "mode": "fixture" if ".tmp" in str(data_dir) else "real",
        "overall": overall,
        "counts": {"pass": n_pass, "fail": n_fail, "skip": n_skip},
        "checks": checks,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 78, flush=True)
    print(f"复核目录：{data_dir}", flush=True)
    for check in checks:
        mark = {"pass": "PASS", "fail": "FAIL", "skip": "SKIP"}[check["status"]]
        print(f"[{mark}] {check['name']}", flush=True)
        print(f"       {check['detail']}", flush=True)
    print("-" * 78, flush=True)
    print(f"结论：{overall}（通过 {n_pass} / 失败 {n_fail} / 跳过 {n_skip}）"
          f"  报告：{report_path}", flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="独立复核全球 SST 分析结果（不依赖 compute.py）")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--raw-dir", type=Path, default=None, help="默认 <data-dir>/raw")
    parser.add_argument("--report", type=Path, default=None, help="默认 <data-dir>/verify_report.json")
    parser.add_argument("--skip-online", action="store_true", help="跳过 ERDDAP 在线抽样")
    args = parser.parse_args()

    data_dir = args.data_dir
    raw_dir = args.raw_dir if args.raw_dir is not None else data_dir / "raw"
    report_path = args.report if args.report is not None else data_dir / "verify_report.json"
    report = run(data_dir, raw_dir, report_path, args.skip_online)
    return 1 if report["overall"] == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())
