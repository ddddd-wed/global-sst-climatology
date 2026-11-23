#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""独立复核全球海温分析结果（**不 import compute.py**，口径从零实现）。

复核内容
--------
1. **数值复算**：读取 ``data/raw_global/``（或 ``--raw-dir``）下 60 个 ``sst_YYYY_MM.csv.gz``，独立重算纬向平均，
   抽 8 个纬度 × 1 月/7 月两个月份，与 ``zonal_mean_basins.csv`` 的 global 行**按纬度**对齐到 0.01 °C；
   统计域用 ``basin_mask.npz::basin_mask`` 剔除内陆湖泊（SST 均值仍独立计算）。
2. **结构性检查**：720 行、0.25° 步长、纬度单调，``coverage = valid_count/ocean_grid_count``、
   ``zonal_ocean_frac = valid_count/1440``、``cov_MM`` 与 ``sst_MM`` 的 NaN 模式（均容差 0.002）、
   海洋值不越界、1 月/7 月南北半球季节反相；新列/月度列缺失时自动退回旧口径或跳过。
   分海盆文件只输出 ``ocean_grid_count > 0`` 的纬度行，校验按 ``latitude`` 交集比对，并正向断言
   缺失行在 global CSV 中必须 ``ocean_grid_count == 0``（反之亦然）。
3. **网格一致性**：``climatology_grid.npz`` 至少含 lat/lon/sst_annual，月度键按需；在"有效格点全覆盖"
   的纬度上比较网格纬向平均与 CSV（比对前按 basin_mask 剔除内陆湖泊）。
4. **无冰复算**：读取 ``raw_global`` + ``raw_global_ice``（ice 缺失/NaN 或 <0.15 视为无冰），
   抽 6 个纬度 × 1/7 月 + 无冰年均，与 ``sst_*_icefree`` 对齐到 0.01 °C；海冰 raw 缺失时跳过。
5. **异常场自洽**：``sst_annual_anom ≈ sst_annual_icefree − zonal_ref.ref_annual``，残差 < 1e-3 °C。
6. **统计列健全性**：``p10 ≤ p90``、``interannual_std ≥ 0``、月份 1–12、``warmest ≥ coldest``、
   ``seasonal_amp ≈ warmest − coldest``。
7. **在线抽样**：直接向 ERDDAP 取 2–3 个时次 × 一条约 1° 宽的纬度带（很小的请求），
   与本地同一时次的 ``data/raw/`` 逐纬圈比对，容差 0.05 °C；网络不通时**跳过**，不算失败。
8. **印度洋回归**：用仓库已提交的 ``data/zonal_mean.csv``（281 行）复算 README 现值
   （最高 29.14 °C @ 2.12°N、25 °C 等温线 21.66°S、海盆均值 26.16 °C），证明全球版没有改口径。

结果写入 ``data/verify_report.json``（或 ``--report`` 指定路径），并打印结论。
任何一项**实际比对失败**都会非零退出；因数据未就绪或网络不通而跳过的不算失败。

用法::

    python scripts/verify.py
    python scripts/verify.py --data-dir .tmp/fixture --skip-online   # 复核合成 fixture
    python scripts/verify.py --data-dir <数据目录> --raw-dir <raw目录> --ice-raw-dir <ice目录> --report .tmp/x.json --skip-online
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_RAW_DIR = DEFAULT_DATA_DIR / "raw_global"   # 全球 raw；旧 data/raw 只服务印度洋回归
DEFAULT_ICE_RAW_DIR = DEFAULT_DATA_DIR / "raw_global_ice"
ANOMALY_NPZ = DEFAULT_DATA_DIR / "anomaly_grid.npz"
ZONAL_REF_CSV = DEFAULT_DATA_DIR / "zonal_ref.csv"
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
CORE_COLUMNS = ["latitude", "valid_count", "coverage", "sst_annual", "sst_jan", "sst_jul"]
SST_COLUMNS = ["sst_annual", "sst_jan", "sst_jul"] + [f"sst_{m:02d}" for m in range(1, 13)]
MONTHLY_COLUMNS = [f"sst_{m:02d}" for m in range(1, 13)]
ERDDAP_BASE = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg.csv"
INDIAN_BASELINE = {
    "sst_max": (29.14, 0.01),
    "lat_max": (2.125, 0.01),
    "isotherm_south": (-21.664, 0.01),
    "basin_mean": (26.163, 0.005),
}

RAW_RE = re.compile(r"sst_(\d{4})_(\d{2})\.csv(?:\.gz)?$")
ICE_RE = re.compile(r"ice_(\d{4})_(\d{2})\.csv(?:\.gz)?$")


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


def read_ice_csv(path: Path) -> pd.DataFrame:
    """读取单个时次海冰 CSV（ice 为 0–1；无冰格点缺失/NaN）。"""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        head = [fh.readline() for _ in range(2)]
    skiprows = [1] if len(head) > 1 and ("UTC" in head[1] or "degrees" in head[1]) else []
    return pd.read_csv(
        path,
        skiprows=skiprows,
        usecols=["latitude", "longitude", "ice"],
        dtype={"latitude": "float64", "longitude": "float64", "ice": "float64"},
    )


def load_ocean_mask(data_dir: Path) -> tuple[np.ndarray | None, str]:
    """读取数据侧的海盆/海洋掩膜（basin_mask>0），用于把内陆湖泊从纬向平均域中剔除。

    这是**定义域**，不是 SST 数值；SST 的纬向平均仍由 verify.py 从 raw 独立计算。
    缺失时返回 (None, "none")，校验会退化为使用全部有效格点。
    """
    for name in ("basin_mask.npz", "climatology_grid.npz"):
        path = data_dir / name
        if not path.exists():
            continue
        try:
            with np.load(path) as npz:
                if "basin_mask" in npz.files:
                    mask = npz["basin_mask"].astype(np.uint8) > 0
                    if mask.shape == (N_LAT_GLOBAL, N_LON_GLOBAL):
                        return mask, f"{name}::basin_mask"
                if "basin_id" in npz.files:
                    mask = npz["basin_id"].astype(np.int16) >= 0
                    if mask.shape == (N_LAT_GLOBAL, N_LON_GLOBAL):
                        return mask, f"{name}::basin_id"
        except Exception:  # noqa: BLE001
            continue
    return None, "none"


# --------------------------------------------------------------------------- #
# 1. 全球纬向平均 CSV
# --------------------------------------------------------------------------- #
def coverage_consistency(frame: pd.DataFrame, label: str) -> tuple[list[str], dict[str, Any]]:
    """按最终契约检查 coverage / zonal_ocean_frac / cov_MM 的自洽性（存在则用，缺失则降级）。"""
    problems: list[str] = []
    info: dict[str, Any] = {}
    cov = frame["coverage"].to_numpy(dtype=float)
    valid = frame["valid_count"].to_numpy(dtype=float)
    finite_cov = cov[np.isfinite(cov)]
    if finite_cov.size == 0:
        problems.append(f"{label}: coverage 全为 NaN")
    else:
        if finite_cov.min() < 0 or finite_cov.max() > 1:
            problems.append(f"{label}: coverage 越界 [{finite_cov.min():.4f}, {finite_cov.max():.4f}]")
        info["coverage_range"] = [float(finite_cov.min()), float(finite_cov.max())]
    if not np.all(np.isfinite(valid)) or valid.min() < 0 or valid.max() > N_LON_GLOBAL + 1e-6:
        problems.append(f"{label}: valid_count 越界或含 NaN")

    if "ocean_grid_count" in frame:
        ocean = frame["ocean_grid_count"].to_numpy(dtype=float)
        if not np.all(np.isfinite(ocean)) or ocean.min() < 0 or ocean.max() > N_LON_GLOBAL + 1e-6:
            problems.append(f"{label}: ocean_grid_count 越界或含 NaN")
        mask = (ocean > 0) & np.isfinite(valid) & np.isfinite(cov)
        cov_err = float(np.max(np.abs(cov[mask] - valid[mask] / ocean[mask]))) if mask.any() else float("nan")
        if not mask.any():
            problems.append(f"{label}: 没有可用于核对 coverage 的海洋纬度")
        if np.any((ocean == 0) & np.isfinite(cov) & (cov > 1e-9)):
            problems.append(f"{label}: ocean_grid_count=0 的纬度 coverage 应为 NaN 或 0")
        annual_finite = (
            np.isfinite(frame["sst_annual"].to_numpy(dtype=float))
            if "sst_annual" in frame
            else np.ones(len(frame), dtype=bool)
        )
        if np.any((ocean > 0) & annual_finite & ~np.isfinite(cov)):
            problems.append(f"{label}: 年均 SST 有效的海洋纬度出现 coverage=NaN")
        info["coverage_denominator"] = "ocean_grid_count"
    else:
        mask = np.isfinite(cov) & np.isfinite(valid)
        cov_err = float(np.max(np.abs(cov[mask] - valid[mask] / N_LON_GLOBAL))) if mask.any() else float("nan")
        info["coverage_denominator"] = f"{N_LON_GLOBAL}(legacy)"
    info["max_coverage_error"] = cov_err
    if not np.isfinite(cov_err) or cov_err > 0.002:
        problems.append(f"{label}: coverage 与 valid_count/分母不自洽（最大偏差 {cov_err:.5f} > 0.002）")

    if "zonal_ocean_frac" in frame:
        zof = frame["zonal_ocean_frac"].to_numpy(dtype=float)
        finite_zof = zof[np.isfinite(zof)]
        if finite_zof.size and (finite_zof.min() < 0 or finite_zof.max() > 1):
            problems.append(f"{label}: zonal_ocean_frac 越界 [{finite_zof.min():.4f}, {finite_zof.max():.4f}]")
        zmask = np.isfinite(valid) & np.isfinite(zof)
        zof_err = float(np.max(np.abs(zof[zmask] - valid[zmask] / N_LON_GLOBAL))) if zmask.any() else float("nan")
        info["max_zonal_ocean_frac_error"] = zof_err
        if np.any(np.isfinite(valid) & (valid > 0) & ~np.isfinite(zof)):
            problems.append(f"{label}: 有效格点数 >0 的纬度出现 zonal_ocean_frac=NaN")
        if not np.isfinite(zof_err) or zof_err > 0.002:
            problems.append(f"{label}: zonal_ocean_frac 与 valid_count/1440 不自洽（最大偏差 {zof_err:.5f} > 0.002）")

    checked = 0
    mismatched: list[str] = []
    # 无海洋格点（ocean_grid_count=0）的行允许 cov=0 而 sst=NaN；有海洋格点的行必须同 NaN 模式
    row_mask = (
        frame["ocean_grid_count"].to_numpy(dtype=float) > 0
        if "ocean_grid_count" in frame
        else np.ones(len(frame), dtype=bool)
    )
    for m in range(1, 13):
        cov_col = f"cov_{m:02d}"
        sst_col = f"sst_{m:02d}"
        if cov_col not in frame or sst_col not in frame:
            continue
        checked += 1
        cov_nan = np.isnan(frame[cov_col].to_numpy(dtype=float))
        sst_nan = np.isnan(frame[sst_col].to_numpy(dtype=float))
        if not np.array_equal(cov_nan[row_mask], sst_nan[row_mask]):
            mismatched.append(f"cov_{m:02d}/sst_{m:02d}")
    if 0 < checked < 12:
        problems.append(f"{label}: cov_MM 列不完整（只有 {checked}/12 个月）")
    if mismatched:
        problems.append(f"{label}: cov_MM 与 sst_MM 的 NaN 模式不一致（{', '.join(mismatched[:4])}）")
    info["cov_sst_nan_pattern_months"] = checked
    return problems, info


def check_global_zonal(path: Path) -> dict:
    name = "zonal_mean_global.csv 结构"
    if not path.exists():
        return make_check(name, "fail", f"缺少 {path}")
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        return make_check(name, "fail", f"无法读取 {path}：{exc}")
    if df.empty:
        return make_check(name, "fail", "文件为空", {"rows": 0})
    missing = [c for c in CORE_COLUMNS if c not in df.columns]
    if missing:
        return make_check(name, "fail", f"缺少必需列：{missing}")
    problems: list[str] = []
    monthly_present = [c for c in MONTHLY_COLUMNS if c in df.columns]

    if len(df) != N_LAT_GLOBAL:
        problems.append(f"行数 {len(df)} != {N_LAT_GLOBAL}")

    lat = df["latitude"].to_numpy(dtype=float)
    if len(lat) > 1:
        if not np.all(np.diff(lat) > 0):
            problems.append("纬度不是严格单调递增")
        if not np.allclose(np.diff(lat), LAT_STEP, atol=1e-9):
            problems.append("纬度步长不是 0.25°")
    if abs(lat[0] - LAT_FIRST) > 1e-6 or abs(lat[-1] - LAT_LAST) > 1e-6:
        problems.append(f"纬度端点 {lat[0]:.3f}…{lat[-1]:.3f} 不是 -89.875…89.875")

    # coverage / zonal_ocean_frac / cov_MM 自洽性
    cov_problems, cov_info = coverage_consistency(df, "global")
    problems += cov_problems

    coverage = df["coverage"].to_numpy(dtype=float)
    valid = df["valid_count"].to_numpy(dtype=float)
    ocean_frac = (
        df["zonal_ocean_frac"].to_numpy(dtype=float)
        if "zonal_ocean_frac" in df
        else valid / N_LON_GLOBAL
    )

    ocean_max = -np.inf
    ocean_min = np.inf
    for col in SST_COLUMNS:
        if col not in df:
            continue
        values = df[col].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size:
            ocean_max = max(ocean_max, float(finite.max()))
            ocean_min = min(ocean_min, float(finite.min()))
    if ocean_max > SST_MAX:
        problems.append(f"海洋 SST 最大值 {ocean_max:.2f} > {SST_MAX}")
    if ocean_min < SST_MIN:
        problems.append(f"海洋 SST 最小值 {ocean_min:.2f} < {SST_MIN}")

    # 1 月 / 7 月南北半球季节反相（20°–60°，cosφ 加权；门槛用海洋占比）
    def band_mean(col: str, north: bool) -> float:
        sst = df[col].to_numpy(dtype=float)
        mask = (np.abs(lat) >= 20) & (np.abs(lat) <= 60)
        mask &= np.isfinite(ocean_frac) & (ocean_frac >= COVERAGE_MIN)
        mask &= (lat > 0) if north else (lat < 0)
        w = np.cos(np.deg2rad(lat[mask]))
        return float(np.sum(w * sst[mask]) / np.sum(w)) if mask.any() else float("nan")

    nh_jan, nh_jul = band_mean("sst_jan", True), band_mean("sst_jul", True)
    sh_jan, sh_jul = band_mean("sst_jan", False), band_mean("sst_jul", False)
    metrics = {
        "rows": len(df),
        "lat_first": float(lat[0]),
        "lat_last": float(lat[-1]),
        "sst_min": ocean_min,
        "sst_max": ocean_max,
        "monthly_columns_present": len(monthly_present),
        "nh_jan_minus_jul": nh_jan - nh_jul,
        "sh_jan_minus_jul": sh_jan - sh_jul,
        **cov_info,
    }
    if not nh_jan < nh_jul - 1.0:
        problems.append(f"北半球 1 月未明显冷于 7 月（{nh_jan:.2f} vs {nh_jul:.2f}）")
    if not sh_jan > sh_jul + 1.0:
        problems.append(f"南半球 1 月未明显暖于 7 月（{sh_jan:.2f} vs {sh_jul:.2f}）")

    if problems:
        return make_check(name, "fail", "；".join(problems), metrics)
    cov_range = metrics.get("coverage_range", [float("nan"), float("nan")])
    return make_check(
        name,
        "pass",
        f"{len(df)} 行，纬度 {lat[0]:.3f}→{lat[-1]:.3f}，步长 {LAT_STEP}°；"
        f"coverage∈[{cov_range[0]:.4f},{cov_range[1]:.4f}]（分母 {metrics.get('coverage_denominator')}），"
        f"zonal_ocean_frac 误差≤{metrics.get('max_zonal_ocean_frac_error', float('nan')):.5f}，"
        f"cov_MM 与 sst_MM NaN 模式一致（{metrics.get('cov_sst_nan_pattern_months', 0)} 个月），"
        f"海洋值∈[{ocean_min:.2f},{ocean_max:.2f}] °C，南北半球季节反相",
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

    expected_order = ["global", "pacific", "atlantic", "indian", "arctic", "southern"]
    seen = list(dict.fromkeys(df[basin_col].tolist()))
    metrics["basin_order"] = seen
    if set(expected_order) - set(seen):
        problems.append(f"缺少海盆：{sorted(set(expected_order) - set(seen))}")

    global_lats = (
        np.round(global_df["latitude"].to_numpy(dtype=float), 3) if global_df is not None else None
    )
    global_ocean = (
        dict(zip(global_lats, global_df["ocean_grid_count"].to_numpy(dtype=float)))
        if global_df is not None and "ocean_grid_count" in global_df
        else None
    )

    for basin, group in df.groupby(basin_col, sort=False):
        group = group.reset_index(drop=True)
        info: dict[str, Any] = {"rows": len(group)}
        metrics["basins"][str(basin)] = info
        missing = [c for c in CORE_COLUMNS if c not in group.columns]
        if missing:
            problems.append(f"{basin}: 缺少必需列 {missing}")
            continue
        if group.empty:
            problems.append(f"{basin}: 没有数据行")
            continue
        lat = group["latitude"].to_numpy(dtype=float)
        info["lat_min"] = float(lat.min())
        info["lat_max"] = float(lat.max())
        if not np.all(np.diff(lat) > 0):
            problems.append(f"{basin}: 纬度不是严格递增")
        elif not np.allclose(np.diff(lat) / LAT_STEP, np.round(np.diff(lat) / LAT_STEP), atol=1e-9):
            problems.append(f"{basin}: 纬度不是 0.25° 的整数倍（允许缺行）")
        if global_lats is not None:
            outside = np.setdiff1d(np.round(lat, 3), global_lats)
            if outside.size:
                problems.append(f"{basin}: {outside.size} 个纬度不在全球网格上（如 {outside[:3]}）")

        cov_problems, cov_info = coverage_consistency(group, str(basin))
        problems += cov_problems
        info.update(cov_info)

        for col in SST_COLUMNS:
            if col not in group:
                continue
            values = group[col].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size and (finite.max() > SST_MAX or finite.min() < SST_MIN):
                problems.append(f"{basin}/{col}: 越界 [{finite.min():.2f}, {finite.max():.2f}]")
                break

    if "global" not in metrics["basins"]:
        problems.append("缺少 global 海盆行")
    elif global_df is not None:
        g = df[df[basin_col] == "global"].reset_index(drop=True)
        g_lat = np.round(g["latitude"].to_numpy(dtype=float), 3)
        common, ia, ib = np.intersect1d(global_lats, g_lat, return_indices=True)
        metrics["basins"]["global"]["common_latitudes"] = int(common.size)
        if common.size == 0:
            problems.append("global 行与全球 CSV 没有公共纬度")
        compare_cols = [
            c
            for c in g.columns
            if c not in (basin_col, "latitude")
            and c in global_df.columns
            and pd.api.types.is_numeric_dtype(g[c])
            and pd.api.types.is_numeric_dtype(global_df[c])
        ]
        worst: dict[str, float] = {}
        for col in compare_cols:
            a = global_df[col].to_numpy(dtype=float)[ia]
            b = g[col].to_numpy(dtype=float)[ib]
            if np.any(np.isnan(a) != np.isnan(b)):
                problems.append(f"global 行 {col} 与全球 CSV 在公共纬度上 NaN 模式不一致")
                continue
            finite = np.isfinite(a) & np.isfinite(b)
            if not finite.any():
                continue
            diff = float(np.max(np.abs(a[finite] - b[finite])))
            worst[col] = diff
            # 计数类容差放宽；温度/比例列按 0.002（原口径）
            tol = 0.05 if col == "valid_count" else (1e-9 if col == "ocean_grid_count" else 0.002)
            if diff > tol:
                problems.append(f"global 行 {col} 与全球 CSV 最大偏差 {diff:.5f} > {tol}")
        metrics["basins"]["global"]["max_diff"] = worst

        # 正向断言：global CSV 有、basin=global 无的纬度，必须是 ocean_grid_count == 0
        missing_lats = np.setdiff1d(global_lats, g_lat)
        metrics["basins"]["global"]["missing_in_basins"] = int(missing_lats.size)
        if missing_lats.size:
            if global_ocean is not None:
                bad = [float(x) for x in missing_lats if global_ocean.get(float(x), np.nan) != 0]
                if bad:
                    problems.append(
                        f"global CSV 有 {len(missing_lats)} 个纬度不在 basin=global 中，"
                        f"其中 {len(bad)} 个 ocean_grid_count != 0（如 {bad[:3]}）"
                    )
            else:
                metrics["basins"]["global"]["missing_assertion"] = "skipped（全球 CSV 无 ocean_grid_count 列）"
        # 反之亦然：basin=global 有、global CSV 无的纬度，必须是 ocean_grid_count == 0
        extra_lats = np.setdiff1d(g_lat, global_lats)
        metrics["basins"]["global"]["extra_in_basins"] = int(extra_lats.size)
        if extra_lats.size:
            if "ocean_grid_count" in g:
                ocean_map = dict(zip(g_lat, g["ocean_grid_count"].to_numpy(dtype=float)))
                bad = [float(x) for x in extra_lats if ocean_map.get(float(x), np.nan) != 0]
                if bad:
                    problems.append(
                        f"basin=global 有 {len(extra_lats)} 个纬度不在全球 CSV 中，"
                        f"其中 {len(bad)} 个 ocean_grid_count != 0（如 {bad[:3]}）"
                    )
            else:
                metrics["basins"]["global"]["extra_assertion"] = "skipped（basin 无 ocean_grid_count 列）"

    if problems:
        return make_check(name, "fail", "；".join(problems[:10]), metrics)
    return make_check(
        name,
        "pass",
        f"{len(metrics['basins'])} 个海盆（行数 "
        + "/".join(f"{b}:{info['rows']}" for b, info in metrics["basins"].items())
        + "），按纬度与全球 CSV 对齐；coverage 分母 "
        f"{metrics['basins'].get('global', {}).get('coverage_denominator', '?')}，"
        "coverage/zonal_ocean_frac/cov_MM 自洽，缺失纬度均为 ocean_grid_count=0",
        metrics,
    )


# --------------------------------------------------------------------------- #
# 3. 气候态网格
# --------------------------------------------------------------------------- #
def check_grid(
    path: Path,
    global_df: pd.DataFrame | None,
    ocean_mask: np.ndarray | None = None,
) -> dict:
    name = "climatology_grid.npz 一致性"
    if not path.exists():
        return make_check(name, "fail", f"缺少 {path}")
    mask_source = "external" if ocean_mask is not None else "none"
    try:
        with np.load(path) as npz:
            keys = set(npz.files)
            if "sst_annual" not in keys:
                return make_check(name, "fail", "缺少键：sst_annual")
            if "lat" in keys:
                lat = npz["lat"].astype(float)
            elif "latitude" in keys:
                lat = npz["latitude"].astype(float)
            else:
                return make_check(name, "fail", "缺少键：lat/latitude")
            if "lon" in keys:
                lon = npz["lon"].astype(float)
            elif "longitude" in keys:
                lon = npz["longitude"].astype(float)
            else:
                return make_check(name, "fail", "缺少键：lon/longitude")
            annual = npz["sst_annual"].astype(float)
            # 月度键按需检测：没有月键也能复核，热图会用 CSV 的月度列
            monthly = {m: npz[f"sst_{m:02d}"].astype(float) for m in range(1, 13) if f"sst_{m:02d}" in keys}
            extra_fields = {
                key: npz[key].astype(float) for key in ("sst_jan", "sst_jul") if key in keys
            }
            if ocean_mask is None and "basin_mask" in keys:
                candidate = npz["basin_mask"].astype(np.uint8) > 0
                if candidate.shape == annual.shape:
                    ocean_mask = candidate
                    mask_source = "npz::basin_mask"
    except Exception as exc:  # noqa: BLE001
        return make_check(name, "fail", f"无法读取 {path}：{exc}")

    problems: list[str] = []
    metrics: dict[str, Any] = {
        "keys": sorted(keys),
        "monthly_keys_present": len(monthly),
        "mask_source": mask_source,
    }
    if ocean_mask is not None and ocean_mask.shape != annual.shape:
        problems.append(f"ocean_mask 形状 {ocean_mask.shape} != sst_annual {annual.shape}")
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

    # 只在"该纬圈所有海洋格点在 60 个时次都有数据"（coverage==1）的纬度比较，
    # 避开冰缘处先平均/后平均的口径差异；coverage 新定义下这些纬度允许仍有陆地。
    if global_df is not None and not problems:
        cov_csv = global_df["coverage"].to_numpy(dtype=float)
        full = np.isclose(cov_csv, 1.0, atol=1e-9) & np.isfinite(global_df["sst_annual"].to_numpy(dtype=float))
        n_full = int(full.sum())
        metrics["full_ocean_latitudes"] = n_full
        if n_full < 5:
            metrics["grid_check"] = "skipped（有效格点全覆盖纬度不足 5 个）"
        else:
            checks = {"sst_annual": (annual, global_df["sst_annual"].to_numpy(dtype=float))}
            for key in ("sst_jan", "sst_jul"):
                if key in extra_fields and key in global_df:
                    checks[key] = (extra_fields[key], global_df[key].to_numpy(dtype=float))
            for m, grid in monthly.items():
                key = f"sst_{m:02d}"
                if key in global_df:
                    checks[key] = (grid, global_df[key].to_numpy(dtype=float))
            worst = 0.0
            worst_key = ""
            for key, (grid, csv) in checks.items():
                grid_use = np.where(ocean_mask, grid, np.nan) if ocean_mask is not None else grid
                with np.errstate(invalid="ignore"):
                    grid_mean = np.nanmean(grid_use[full], axis=1)
                diff = float(np.nanmax(np.abs(grid_mean - csv[full])))
                if not np.isfinite(diff):
                    problems.append(f"有效格点全覆盖纬度上 {key} 的网格纬向平均出现 NaN")
                    continue
                if diff > worst:
                    worst, worst_key = diff, key
            metrics["max_grid_zonal_diff"] = {"key": worst_key, "value": worst, "n_lat": n_full}
            if worst > GRID_TOL:
                problems.append(
                    f"有效格点全覆盖纬度上网格纬向平均与 CSV 最大偏差 {worst:.4f}（{worst_key}）> {GRID_TOL}"
                )

    if problems:
        return make_check(name, "fail", "；".join(problems), metrics)
    return make_check(
        name,
        "pass",
        f"{len(lat)}×{len(lon)} 网格，sst_annual + {len(monthly)} 个月度键（按需），"
        f"海洋格点占比 {metrics.get('ocean_fraction', float('nan')):.1%}，"
        f"有效格点全覆盖纬度 {metrics.get('full_ocean_latitudes', 0)} 个上纬向平均与 CSV 最大偏差 "
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


def list_ice_files(ice_dir: Path) -> list[tuple[tuple[int, int], Path]]:
    if not ice_dir.exists():
        return []
    out: list[tuple[tuple[int, int], Path]] = []
    for path in sorted(ice_dir.iterdir()):
        match = ICE_RE.search(path.name)
        if match:
            out.append(((int(match.group(1)), int(match.group(2))), path))
    return out


def recompute_zonal_from_raw(
    files: list[tuple[tuple[int, int], Path]],
    ocean_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """独立重算：逐时次纬向平均 -> 同月多年平均 -> 12 月等权年均。

    ``ocean_mask`` 只用于定义统计域（剔除内陆湖泊/陆地），SST 均值仍在本函数内独立计算。
    返回 (lat, clim[12,nlat], annual[nlat], counts[ntime,nlat])。
    """
    lat_ref: np.ndarray | None = None
    lon_ref: np.ndarray | None = None
    n_lat = n_lon = 0
    means: dict[tuple[int, int], np.ndarray] = {}
    counts: dict[tuple[int, int], np.ndarray] = {}
    for i, ((year, month), path) in enumerate(files, start=1):
        df = read_raw_csv(path)
        lat_col = df["latitude"].to_numpy(dtype=float)
        lon_col = df["longitude"].to_numpy(dtype=float)
        sst_col = df["sst"].to_numpy(dtype=float)
        if lat_ref is None:
            lat_ref = np.unique(lat_col)
            lon_ref = np.unique(lon_col)
            n_lat, n_lon = len(lat_ref), len(lon_ref)
            if n_lat * n_lon != len(df):
                raise ValueError(f"{path.name} 行数 {len(df)} != {n_lat}×{n_lon}")
            if ocean_mask is not None and ocean_mask.shape != (n_lat, n_lon):
                raise ValueError(
                    f"ocean_mask 形状 {ocean_mask.shape} 与 raw 网格 {(n_lat, n_lon)} 不一致"
                )
            expected_lat = np.repeat(lat_ref, n_lon)
            expected_lon = np.tile(lon_ref, n_lat)
        if len(df) != n_lat * n_lon:
            raise ValueError(f"{path.name} 行数 {len(df)} != {n_lat}×{n_lon}")
        if not (np.array_equal(lat_col, expected_lat) and np.array_equal(lon_col, expected_lon)):
            raise ValueError(f"{path.name} 的经纬度行列顺序与标准网格不一致")

        grid = sst_col.reshape(n_lat, n_lon)
        valid = np.isfinite(grid)
        if ocean_mask is not None:
            valid &= ocean_mask
        c = valid.sum(axis=1).astype(float)
        s = np.where(valid, grid, 0.0).sum(axis=1)
        zonal = np.full(n_lat, np.nan, dtype=float)
        nonzero = c > 0
        zonal[nonzero] = s[nonzero] / c[nonzero]
        means[(year, month)] = zonal
        counts[(year, month)] = c
        if i % 10 == 0 or i == len(files):
            print(f"  raw 重算 [{i:02d}/{len(files)}] {path.name}", flush=True)

    assert lat_ref is not None
    years = sorted({y for y, _ in means})
    months = sorted({m for _, m in means})
    if months != list(range(1, 13)):
        raise ValueError(f"月份不完整：{months}")
    clim = np.full((12, len(lat_ref)), np.nan, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # 无海洋格点的纬度
        for i, month in enumerate(range(1, 13)):
            stack = np.vstack([means[(y, month)] for y in years])
            clim[i] = np.nanmean(stack, axis=0)
        annual = np.nanmean(clim, axis=0)
    count_stack = np.vstack([counts[key] for key in sorted(counts)])
    return lat_ref, clim, annual, count_stack


def recompute_icefree_zonal(
    raw_dir: Path,
    ice_dir: Path,
    ocean_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """独立复算无冰纬向平均：ice 缺失/NaN 或 ice<0.15 视为无冰，SST 取 raw_global。"""
    sst_files = dict(list_raw_files(raw_dir))
    ice_files = dict(list_ice_files(ice_dir))
    keys = sorted(set(sst_files) & set(ice_files))
    if len(keys) != 60:
        raise ValueError(f"raw_global/raw_global_ice 配对时次为 {len(keys)}，需 60")

    lat_ref: np.ndarray | None = None
    n_lat = n_lon = 0
    means: dict[tuple[int, int], np.ndarray] = {}
    counts: dict[tuple[int, int], np.ndarray] = {}
    for i, key in enumerate(keys, start=1):
        sst_df = read_raw_csv(sst_files[key])
        ice_df = read_ice_csv(ice_files[key])
        lat_s = sst_df["latitude"].to_numpy(dtype=float)
        lon_s = sst_df["longitude"].to_numpy(dtype=float)
        lat_i = ice_df["latitude"].to_numpy(dtype=float)
        lon_i = ice_df["longitude"].to_numpy(dtype=float)
        if not (np.array_equal(lat_s, lat_i) and np.array_equal(lon_s, lon_i)):
            raise ValueError(f"{sst_files[key].name} 与 {ice_files[key].name} 经纬度顺序不一致")
        if lat_ref is None:
            lat_ref = np.unique(lat_s)
            n_lat, n_lon = len(lat_ref), len(np.unique(lon_s))
            if n_lat * n_lon != len(sst_df):
                raise ValueError(f"{sst_files[key].name} 行数 {len(sst_df)} != {n_lat}×{n_lon}")
            if ocean_mask is not None and ocean_mask.shape != (n_lat, n_lon):
                raise ValueError(f"ocean_mask 形状 {ocean_mask.shape} != {(n_lat, n_lon)}")
        if len(sst_df) != n_lat * n_lon or len(ice_df) != n_lat * n_lon:
            raise ValueError(f"{key} 行数与网格 {n_lat}×{n_lon} 不一致")
        sst = sst_df["sst"].to_numpy(dtype=float).reshape(n_lat, n_lon)
        ice = ice_df["ice"].to_numpy(dtype=float).reshape(n_lat, n_lon)
        # 海冰文件只含 ice>=0.15 的格点；无冰/无海冰格点为 NaN。NaN 视为无冰。
        ice_free = (~np.isfinite(ice)) | (ice < 0.15)
        valid = np.isfinite(sst) & ice_free
        if ocean_mask is not None:
            valid &= ocean_mask
        c = valid.sum(axis=1).astype(float)
        s = np.where(valid, sst, 0.0).sum(axis=1)
        zonal = np.full(n_lat, np.nan, dtype=float)
        nz = c > 0
        zonal[nz] = s[nz] / c[nz]
        means[key] = zonal
        counts[key] = c
        if i % 10 == 0 or i == len(keys):
            print(f"  无冰 raw 重算 [{i:02d}/{len(keys)}] {sst_files[key].name}", flush=True)

    assert lat_ref is not None
    years = sorted({y for y, _ in means})
    clim = np.full((12, len(lat_ref)), np.nan, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i, month in enumerate(range(1, 13)):
            stack = np.vstack([means[(y, month)] for y in years])
            clim[i] = np.nanmean(stack, axis=0)
        annual = np.nanmean(clim, axis=0)
    return lat_ref, clim, annual



    """快速读取首个 raw 文件，返回 (纬度格点数, 经度格点数)。"""
    df = read_raw_csv(path)
    return int(df["latitude"].nunique()), int(df["longitude"].nunique())


def probe_raw_grid(path: Path) -> tuple[int, int]:
    """快速读取首个 raw 文件，返回 (纬度格点数, 经度格点数)。"""
    df = read_raw_csv(path)
    return int(df["latitude"].nunique()), int(df["longitude"].nunique())


def check_raw_recompute(
    raw_dir: Path,
    target_df: pd.DataFrame | None,
    target_label: str,
    ocean_mask: np.ndarray | None = None,
    mask_source: str = "none",
) -> dict:
    name = "data/raw 独立数值复算"
    files = list_raw_files(raw_dir)
    if not files:
        return make_check(name, "skip", f"未找到 {raw_dir}/sst_YYYY_MM.csv.gz，跳过数值复算")
    try:
        grid_shape = probe_raw_grid(files[0][1])
    except Exception as exc:  # noqa: BLE001
        return make_check(name, "fail", f"raw 读取失败：{exc}")
    if grid_shape != (N_LAT_GLOBAL, N_LON_GLOBAL):
        hint = "，疑似旧印度洋缓存" if grid_shape == (281, 401) else ""
        return make_check(
            name,
            "skip",
            f"raw 不是全球网格（{grid_shape[0]}×{grid_shape[1]}{hint}），"
            "请用 --raw-dir 指向全球缓存",
        )
    if len(files) != 60:
        return make_check(name, "skip", f"raw 只有 {len(files)} 个时次（需 60），数据未就绪，跳过数值复算")
    if target_df is None:
        return make_check(name, "skip", "缺少可对齐的 zonal_mean_basins.csv（或无 global 行），跳过")

    try:
        lat_raw, clim, annual, _ = recompute_zonal_from_raw(files, ocean_mask=ocean_mask)
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
    sample_months = [m for m in SAMPLE_MONTHS if f"sst_{m:02d}" in target.columns]
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
        for month in sample_months:
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
    metrics = {"n_timesteps": len(files), "target": target_label, "mask_source": mask_source,
               "months": sample_months, "max_abs_diff": max_diff, "samples": rows}
    if problems:
        return make_check(name, "fail", "；".join(problems[:8]), metrics)
    month_text = "/".join(f"{m:02d}" for m in sample_months) if sample_months else "无月度列"
    return make_check(
        name,
        "pass",
        f"{len(files)} 个时次独立重算，8 纬度 × {month_text} 月 + 年均共 {len(rows)} 个比对，"
        f"最大偏差 {max_diff:.4f} °C ≤ {NUMERIC_TOL}（对齐 {target_label}；统计域 {mask_source}）",
        metrics,
    )


def check_icefree_recompute(
    raw_dir: Path,
    ice_dir: Path,
    target_df: pd.DataFrame | None,
    ocean_mask: np.ndarray | None,
    mask_source: str,
) -> dict:
    name = "raw_global_ice 无冰纬向平均独立复算"
    if not ice_dir.exists():
        return make_check(name, "skip", f"未找到 {ice_dir}，跳过无冰复算（不判失败）")
    files = list_ice_files(ice_dir)
    if len(files) != 60:
        return make_check(name, "skip", f"{ice_dir} 只有 {len(files)} 个时次（需 60），跳过")
    if target_df is None:
        return make_check(name, "skip", "缺少可对齐的全球纬向平均 CSV，跳过")
    cols = {"annual": "sst_annual_icefree", "jan": "sst_jan_icefree", "jul": "sst_jul_icefree"}
    if any(c not in target_df.columns for c in cols.values()):
        return make_check(name, "skip", "sst_*_icefree 列未就绪，跳过")
    try:
        lat_raw, clim, annual = recompute_icefree_zonal(raw_dir, ice_dir, ocean_mask=ocean_mask)
    except Exception as exc:  # noqa: BLE001
        return make_check(name, "fail", f"无冰 raw 重算失败：{exc}")

    target = target_df.set_index("latitude")
    sample_lats = (-60.0, -40.0, -20.0, 0.0, 20.0, 40.0)
    rows: list[dict[str, Any]] = []
    problems: list[str] = []
    for wanted in sample_lats:
        idx = int(np.argmin(np.abs(lat_raw - wanted)))
        lat = float(lat_raw[idx])
        if lat not in target.index:
            nearest = target.index[np.argmin(np.abs(target.index.to_numpy(dtype=float) - lat))]
            lat = float(nearest)
            idx = int(np.argmin(np.abs(lat_raw - lat)))
        row = target.loc[lat]
        for month, key in ((1, "jan"), (7, "jul")):
            raw_v = float(clim[month - 1, idx])
            csv_v = float(row[cols[key]])
            diff = abs(raw_v - csv_v) if np.isfinite(raw_v) and np.isfinite(csv_v) else float("inf")
            rows.append({"latitude": lat, "month": month, "raw": raw_v, "csv": csv_v, "diff": diff})
            if diff > NUMERIC_TOL:
                problems.append(f"{lat:.2f}°/{month:02d} 月：raw {raw_v:.4f} vs csv {csv_v:.4f}（Δ={diff:.4f}）")
        raw_a = float(annual[idx])
        csv_a = float(row[cols["annual"]])
        diff_a = abs(raw_a - csv_a) if np.isfinite(raw_a) and np.isfinite(csv_a) else float("inf")
        rows.append({"latitude": lat, "month": "annual", "raw": raw_a, "csv": csv_a, "diff": diff_a})
        if diff_a > NUMERIC_TOL:
            problems.append(f"{lat:.2f}° 无冰年均：raw vs csv Δ={diff_a:.4f}")
    max_diff = max(r["diff"] for r in rows)
    metrics = {"n_timesteps": len(files), "ice_raw_dir": str(ice_dir), "mask_source": mask_source,
               "sample_lats": sample_lats, "max_abs_diff": max_diff, "samples": rows}
    if problems:
        return make_check(name, "fail", "；".join(problems[:8]), metrics)
    return make_check(
        name, "pass",
        f"60 时次无冰复算，6 纬度 × 1/7 月 + 无冰年均共 {len(rows)} 个比对，"
        f"最大偏差 {max_diff:.4f} °C ≤ {NUMERIC_TOL}（统计域 {mask_source}）", metrics,
    )


def check_anomaly_grid(data_dir: Path) -> dict:
    name = "anomaly_grid 同纬度残差"
    anom_path = data_dir / "anomaly_grid.npz"
    ref_path = data_dir / "zonal_ref.csv"
    if not anom_path.exists() or not ref_path.exists():
        return make_check(name, "skip", "anomaly_grid.npz 或 zonal_ref.csv 未就绪，跳过")
    try:
        with np.load(anom_path) as npz:
            needed = {"sst_annual_anom", "sst_annual_icefree"}
            if not needed.issubset(npz.files):
                return make_check(name, "skip", f"anomaly_grid.npz 缺少 {sorted(needed - set(npz.files))}")
            anom = npz["sst_annual_anom"].astype(float)
            icefree = npz["sst_annual_icefree"].astype(float)
        ref = pd.read_csv(ref_path)
        expected = icefree - ref["ref_annual"].to_numpy(dtype=float)[:, None]
        ok = np.isfinite(expected) & np.isfinite(anom)
        if not ok.any():
            return make_check(name, "skip", "anomaly/ref 没有共同有效格点")
        residual = np.abs(expected[ok] - anom[ok])
        max_res = float(residual.max())
        metrics = {"n_valid": int(ok.sum()), "max_abs_residual": max_res,
                   "p99_residual": float(np.percentile(residual, 99))}
        if max_res > 1e-3:
            return make_check(name, "fail",
                              f"max|anom-(icefree-ref)|={max_res:.6f} °C > 1e-3（{ok.sum()} 格点）", metrics)
        return make_check(name, "pass",
                          f"{ok.sum()} 个有效格点，max|anom-(icefree-ref)|={max_res:.2e} °C ≤ 1e-3",
                          metrics)
    except Exception as exc:  # noqa: BLE001
        return make_check(name, "fail", f"异常场校验失败：{exc}")


def check_stats_columns(data_dir: Path) -> dict:
    name = "统计列健全性"
    path = data_dir / "zonal_mean_global.csv"
    if not path.exists():
        return make_check(name, "skip", "zonal_mean_global.csv 未就绪，跳过")
    df = pd.read_csv(path)
    needed = [
        "sst_p10", "sst_p90", "sst_interannual_std", "sst_warmest", "sst_coldest",
        "seasonal_amp", "warmest_month", "coldest_month",
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        return make_check(name, "skip", f"统计列未就绪：缺少 {', '.join(missing)}")
    issues: list[str] = []
    counts: dict[str, int] = {}
    for col in needed:
        counts[col] = int(np.isfinite(df[col].to_numpy(dtype=float)).sum())
    p10 = df["sst_p10"].to_numpy(float)
    p90 = df["sst_p90"].to_numpy(float)
    ok = np.isfinite(p10) & np.isfinite(p90)
    bad = ok & (p10 > p90 + 1e-9)
    if bad.any():
        issues.append(f"p10>p90 在 {int(bad.sum())} 个纬度")
    istd = df["sst_interannual_std"].to_numpy(float)
    bad = np.isfinite(istd) & (istd < -1e-9)
    if bad.any():
        issues.append(f"interannual_std<0 在 {int(bad.sum())} 个纬度")
    for col in ("warmest_month", "coldest_month"):
        vals = df[col].to_numpy(float)
        okm = np.isfinite(vals)
        if okm.any():
            bad = okm & ((vals < 1 - 1e-9) | (vals > 12 + 1e-9) | (np.abs(vals - np.round(vals)) > 1e-9))
            if bad.any():
                issues.append(f"{col} 越界/非整数在 {int(bad.sum())} 个纬度")
        counts[col] = int(okm.sum())
    warm = df["sst_warmest"].to_numpy(float)
    cold = df["sst_coldest"].to_numpy(float)
    ok = np.isfinite(warm) & np.isfinite(cold)
    bad = ok & (warm < cold - 1e-9)
    if bad.any():
        issues.append(f"sst_warmest<sst_coldest 在 {int(bad.sum())} 个纬度")
    amp = df["seasonal_amp"].to_numpy(float)
    okamp = ok & np.isfinite(amp)
    if okamp.any():
        amp_diff = float(np.max(np.abs((warm - cold)[okamp] - amp[okamp])))
        if amp_diff > 0.01:
            issues.append(f"seasonal_amp 与 warmest-coldest 最大差 {amp_diff:.4f} °C > 0.01")
    else:
        amp_diff = float("nan")
    metrics = {"counts": counts, "seasonal_amp_max_diff": amp_diff}
    if issues:
        return make_check(name, "fail", "；".join(issues), metrics)
    return make_check(name, "pass",
                      f"{len(needed)} 个统计列通过：p10≤p90、std≥0、月份 1–12、warmest≥coldest、"
                      f"seasonal_amp≈warmest−coldest（max diff {amp_diff:.4f} °C）", metrics)



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
    coverage = df["coverage"].to_numpy(dtype=float)  # 旧文件 coverage=valid/401，即海洋占比
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
def run(data_dir: Path, raw_dir: Path, report_path: Path, skip_online: bool,
        ice_raw_dir: Path | None = None) -> dict:
    data_dir = data_dir.resolve()
    raw_dir = raw_dir.resolve()
    if ice_raw_dir is None:
        ice_raw_dir = data_dir / "raw_global_ice"
    ice_raw_dir = ice_raw_dir.resolve()
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
    ocean_mask, mask_source = load_ocean_mask(data_dir) if real_ready else (None, "none")
    if real_ready:
        if mask_source != "none":
            print(f"统计域掩膜：{mask_source}（剔除内陆湖泊/陆地）", flush=True)
        else:
            print("未找到 basin_mask，raw 重算与网格比对将使用全部有效格点（可能含内陆湖泊）。", flush=True)
        checks = [
            check_global_zonal(global_csv),
            check_basins_zonal(basins_csv, global_df),
            check_grid(grid_npz, global_df, ocean_mask),
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
        check_raw_recompute(raw_dir, target_df, target_label, ocean_mask, mask_source),
        check_icefree_recompute(raw_dir, ice_raw_dir, target_df, ocean_mask, mask_source),
        check_anomaly_grid(data_dir),
        check_stats_columns(data_dir),
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
        "ice_raw_dir": str(ice_raw_dir),
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
    parser.add_argument("--raw-dir", type=Path, default=None, help="默认 <data-dir>/raw_global")
    parser.add_argument("--ice-raw-dir", type=Path, default=None,
                        help="海冰 raw，默认 <data-dir>/raw_global_ice；不存在则 SKIP")
    parser.add_argument("--report", type=Path, default=None, help="默认 <data-dir>/verify_report.json")
    parser.add_argument("--skip-online", action="store_true", help="跳过 ERDDAP 在线抽样")
    args = parser.parse_args()

    data_dir = args.data_dir
    if args.raw_dir is not None:
        raw_dir = args.raw_dir
    elif data_dir.resolve() == DEFAULT_DATA_DIR.resolve():
        raw_dir = DEFAULT_RAW_DIR
    else:
        raw_dir = data_dir / "raw_global"
    report_path = args.report if args.report is not None else data_dir / "verify_report.json"
    if args.ice_raw_dir is not None:
        ice_raw_dir = args.ice_raw_dir
    elif data_dir.resolve() == DEFAULT_DATA_DIR.resolve():
        ice_raw_dir = DEFAULT_ICE_RAW_DIR
    else:
        ice_raw_dir = data_dir / "raw_global_ice"
    report = run(data_dir, raw_dir, report_path, args.skip_online, ice_raw_dir=ice_raw_dir)
    return 1 if report["overall"] == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())
