#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""读取 data/raw/ 下的原始 OISST 数据，计算纬向平均，输出 data/zonal_mean.csv。

统计口径：
- 研究区 20°E–120°E、40°S–30°N，仅使用 0.25° 原生格点（401 列 × 281 行），不降采样、不重采样、不平滑。
- 纬向平均：同一纬度圈内对有效的非 NaN 海洋格点取算术平均（不乘 cosφ；同纬度圈格点面积相同）。
- 时间维度：2016–2020 年、每年每月 16 日 12:00 UTC 一个时次，共 60 个时次。
  先逐时次求纬向平均 -> 对同一月份 5 个年份取平均得到 12 个月气候态 -> 12 个月等权平均得到年均。
- 覆盖率：该纬度 60 个时次有效海洋格点数的平均值 / 401。
- 陆地与海冰为 NaN，直接跳过，并记录有效格点数与覆盖率。

输出列：latitude, valid_count, coverage, sst_annual, sst_jan, sst_jul,
        以及 sst_01 ... sst_12（12 个月气候态，供热图使用）。
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from config import COVERAGE_MIN, N_LAT, N_LON, RAW_DIR, YEAR_END, YEAR_START, ZONAL_CSV


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


def collect(raw_dir: Path) -> tuple[np.ndarray, dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]]:
    """逐文件计算每个纬度的 SST 均值与有效格点数。"""
    files = list_raw_files(raw_dir)
    if not files:
        raise SystemExit(f"未找到原始数据：{raw_dir}，请先运行 scripts/fetch_data.py")

    lat_ref: np.ndarray | None = None
    samples: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    for i, path in enumerate(files, 1):
        year, month = parse_name(path.name)
        df = pd.read_csv(
            path,
            skiprows=[1],  # 第 2 行是单位行（UTC,m,degrees_north,...）
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


def check_completeness(samples: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]) -> list[int]:
    years = sorted({y for y, _ in samples})
    missing = [(y, m) for y in years for m in range(1, 13) if (y, m) not in samples]
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


def summarize(df: pd.DataFrame) -> dict:
    """打印关键数字（README 中引用的数值均来自这里）。"""
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
        iso = isotherms[-1]  # 自北向南最后一个交点，即南半球 25°C 等温线
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


def main() -> int:
    parser = argparse.ArgumentParser(description="计算印度洋纬向平均海表温度")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out", type=Path, default=ZONAL_CSV)
    args = parser.parse_args()

    print(f"读取原始数据：{args.raw_dir}", flush=True)
    lat, samples = collect(args.raw_dir)
    years = check_completeness(samples)
    print(f"时次完整：{len(samples)} 个（{years[0]}–{years[-1]}，每年 12 个月）", flush=True)

    n_lat = len(lat)
    clim = np.full((12, n_lat), np.nan, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # 全 NaN 纬度圈（无海洋格点）
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

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8", lineterminator="\n")
    print(f"已写出 {args.out}（{len(out)} 行 × {len(out.columns)} 列）", flush=True)

    summarize(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
