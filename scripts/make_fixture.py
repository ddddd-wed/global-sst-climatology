#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成**合成**全球海表温度 fixture，用于在真实数据未就绪时开发与自测绘图/复核链路。

输出目录默认为 ``.tmp/fixture/``（已被 .gitignore 忽略），内容严格对齐数据契约：

- ``zonal_mean_global.csv``：720 行 × 18 列（-89.875…89.875，0.25°），``coverage = valid_count / 1440``
- ``zonal_mean_basins.csv``：global / indian / pacific / atlantic / arctic / southern 纵向堆叠，
  额外带 ``ocean_grid_count`` 与 ``coverage_01..coverage_12``
- ``climatology_grid.npz``：``lat(720)``、``lon(1440)``、``sst_annual``、``sst_01..sst_12``
- ``basin_mask.npz``：``basin_id(720×1440)``（-1=陆地）、``basin_names``、``lat``、``lon``
- ``summary_metrics.json``：各海盆 cosφ 加权均温等（字段名与真实数据侧保持"能读就读"的宽松约定）

**这些数字全部是解析公式生成的合成值，不是观测结果，也绝不会写入 ``data/``。**
物理上刻意保留：赤道暖、两极冷、南北半球季节反相、陆地 NaN、极地海冰随季节进退。

用法::

    python scripts/make_fixture.py
    python scripts/make_fixture.py --out .tmp/fixture --seed 7
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / ".tmp" / "fixture"

N_LAT, N_LON = 720, 1440
LAT = np.round(-89.875 + 0.25 * np.arange(N_LAT), 3)
LON = np.round(0.125 + 0.25 * np.arange(N_LON), 3)
BASIN_NAMES = ("indian", "pacific", "atlantic", "arctic", "southern")
BASIN_COLUMNS = (
    ["latitude", "valid_count", "coverage", "sst_annual", "sst_jan", "sst_jul"]
    + [f"sst_{m:02d}" for m in range(1, 13)]
)


def _box(lon0: float, lon1: float, lat0: float, lat1: float) -> np.ndarray:
    lon2 = LON[None, :]
    lat2 = LAT[:, None]
    return (lon2 >= lon0) & (lon2 <= lon1) & (lat2 >= lat0) & (lat2 <= lat1)


def make_land_mask() -> np.ndarray:
    """用粗糙矩形拼出大陆轮廓；只为让覆盖率/海盆边界看起来合理，不追求地理精度。"""
    land = np.zeros((N_LAT, N_LON), dtype=bool)
    land |= _box(190, 310, 15, 72)      # 北美
    land |= _box(280, 325, -55, 12)     # 南美
    land |= _box(340, 360, -35, 35)     # 非洲西部
    land |= _box(0, 40, -35, 35)        # 非洲东部
    land |= _box(35, 60, 12, 32)        # 阿拉伯半岛
    land |= _box(0, 180, 35, 75)        # 欧亚大陆
    land |= _box(68, 90, 8, 30)         # 印度半岛
    land |= _box(95, 125, 0, 25)        # 中南半岛/印尼
    land |= _box(110, 155, -45, -10)    # 澳大利亚
    land |= _box(300, 340, 60, 84)      # 格陵兰
    land |= _box(0, 360, -90, -66)      # 南极洲
    return land


def make_basin_id(ocean: np.ndarray) -> np.ndarray:
    """海盆编号：0 indian / 1 pacific / 2 atlantic / 3 arctic / 4 southern，-1 陆地。"""
    lat2 = LAT[:, None]
    lon2 = LON[None, :]
    bid = np.full((N_LAT, N_LON), -1, dtype=np.int16)
    bid[ocean] = 2  # 默认大西洋，再按经度覆盖
    bid[ocean & (lon2 >= 20) & (lon2 < 120)] = 0
    bid[ocean & (lon2 >= 120) & (lon2 < 280)] = 1
    bid[ocean & (lat2 <= -60)] = 4
    bid[ocean & (lat2 >= 66)] = 3
    return bid


def build_monthly_grids(seed: int = 2024) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (monthly[12,720,1440] float32, annual[720,1440] float32, basin_id)。"""
    rng = np.random.default_rng(seed)
    land = make_land_mask()
    ocean = ~land
    basin_id = make_basin_id(ocean)

    lat2 = LAT[:, None]
    lon2 = LON[None, :]

    # 年均背景场：赤道 ~29 °C，极地 ~-1.8 °C；加一点经向暖池结构。
    base = -1.8 + 30.8 * np.cos(np.deg2rad(lat2)) ** 2
    base = base + 0.6 * np.cos(np.deg2rad(lon2 - 150.0)) * np.exp(-(lat2 / 35.0) ** 2)
    base = base + 0.5 * np.exp(-((lon2 - 70.0) / 30.0) ** 2 - ((lat2 - 8.0) / 14.0) ** 2)
    base = np.broadcast_to(base, (N_LAT, N_LON)).copy()

    # 季节振幅：赤道很小，中高纬增大；用 cos(lat)^0.5 压制极点夏季增温。
    amp = (0.5 + 7.2 * np.sin(np.deg2rad(np.abs(lat2))) ** 1.2) * np.cos(np.deg2rad(lat2)) ** 0.5
    phase = np.where(lat2 >= 0, 8.0, 2.0)  # 北半球 8 月最暖，南半球 2 月最暖

    monthly = np.full((12, N_LAT, N_LON), np.nan, dtype=np.float32)
    for i, month in enumerate(range(1, 13), start=1):
        season = amp * np.cos(2.0 * np.pi * (month - phase) / 12.0)
        sst = base + season + rng.normal(0.0, 0.05, size=(N_LAT, N_LON))
        sst = np.where(ocean, sst, np.nan)
        sst = np.where(np.isfinite(sst) & (sst < -1.75), np.nan, sst)  # 海冰
        monthly[i - 1] = sst.astype(np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        annual = np.nanmean(monthly, axis=0).astype(np.float32)
    return monthly, annual, basin_id


def zonal_means_counts(grids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """逐时次纬向平均与有效格点数，返回 (means[...,720], counts[...,720])。"""
    counts = np.isfinite(grids).sum(axis=-1).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        means = np.nanmean(grids, axis=-1)
    return np.asarray(means, dtype=float), counts


def build_global_frame(monthly: np.ndarray) -> pd.DataFrame:
    means, counts = zonal_means_counts(monthly)  # (12,720)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        annual = np.nanmean(means, axis=0)
    valid_count = counts.mean(axis=0)
    frame = pd.DataFrame(
        {
            "latitude": LAT,
            "valid_count": np.round(valid_count, 1),
            "coverage": np.round(valid_count / N_LON, 4),
            "sst_annual": np.round(annual, 3),
            "sst_jan": np.round(means[0], 3),
            "sst_jul": np.round(means[6], 3),
        }
    )
    for m in range(12):
        frame[f"sst_{m + 1:02d}"] = np.round(means[m], 3)
    return frame[BASIN_COLUMNS]


def build_basin_frames(monthly: np.ndarray, basin_id: np.ndarray, global_frame: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for idx, name in enumerate(BASIN_NAMES):
        mask = basin_id == idx
        if not mask.any():
            continue
        masked = np.where(mask[None, :, :], monthly, np.nan)
        means, counts = zonal_means_counts(masked)  # (12,720)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            annual = np.nanmean(means, axis=0)
        ocean_count = mask.sum(axis=1).astype(float)
        valid_count = counts.mean(axis=0)
        coverage = np.divide(valid_count, ocean_count, out=np.zeros_like(valid_count), where=ocean_count > 0)
        frame = pd.DataFrame(
            {
                "basin": name,
                "latitude": LAT,
                "valid_count": np.round(valid_count, 1),
                "coverage": np.round(coverage, 4),
                "sst_annual": np.round(annual, 3),
                "sst_jan": np.round(means[0], 3),
                "sst_jul": np.round(means[6], 3),
                "ocean_grid_count": ocean_count.astype(int),
            }
        )
        for m in range(12):
            frame[f"sst_{m + 1:02d}"] = np.round(means[m], 3)
        for m in range(12):
            cov_m = np.divide(counts[m], ocean_count, out=np.zeros(N_LAT), where=ocean_count > 0)
            frame[f"coverage_{m + 1:02d}"] = np.round(cov_m, 4)
        frames.append(frame)

    # global 行沿用全球文件的 18 列口径（coverage = valid_count / 1440）
    g = global_frame.copy()
    extra = pd.DataFrame(
        {
            "basin": "global",
            "ocean_grid_count": np.full(len(g), N_LON, dtype=int),
            **{f"coverage_{m:02d}": g["coverage"].to_numpy() for m in range(1, 13)},
        }
    )
    g = pd.concat([g, extra], axis=1)
    ordered = ["basin"] + BASIN_COLUMNS + ["ocean_grid_count"] + [f"coverage_{m:02d}" for m in range(1, 13)]
    frames.insert(0, g[ordered])
    return pd.concat(frames, ignore_index=True)


def basin_mean_cos(frame: pd.DataFrame, coverage_min: float = 0.10) -> dict:
    lat = frame["latitude"].to_numpy(dtype=float)
    sst = frame["sst_annual"].to_numpy(dtype=float)
    cov = frame["coverage"].to_numpy(dtype=float)
    usable = np.isfinite(sst) & (cov >= coverage_min)
    if not usable.any():
        return {"mean_sst_cos_weighted": None, "lat_min": None, "lat_max": None}
    w = np.cos(np.deg2rad(lat[usable]))
    return {
        "mean_sst_cos_weighted": round(float(np.sum(w * sst[usable]) / np.sum(w)), 3),
        "lat_min": round(float(lat[usable].min()), 3),
        "lat_max": round(float(lat[usable].max()), 3),
        "coverage_min_threshold": coverage_min,
    }


def build_fixture(out_dir: Path, seed: int = 2024) -> dict:
    out_dir = Path(out_dir).resolve()
    data_dir = (ROOT / "data").resolve()
    if out_dir == data_dir or data_dir in out_dir.parents:
        raise SystemExit(f"拒绝写入 data/：{out_dir}（fixture 只能放 .tmp/）")

    monthly, annual, basin_id = build_monthly_grids(seed=seed)
    if not (np.isfinite(annual).any() and np.nanmax(annual) < 35 and np.nanmin(annual) > -2.5):
        raise SystemExit("合成温度场超出合理范围，检查公式")

    global_frame = build_global_frame(monthly)
    basin_frame = build_basin_frames(monthly, basin_id, global_frame)

    out_dir.mkdir(parents=True, exist_ok=True)
    global_frame.to_csv(out_dir / "zonal_mean_global.csv", index=False, encoding="utf-8", lineterminator="\n")
    basin_frame.to_csv(out_dir / "zonal_mean_basins.csv", index=False, encoding="utf-8", lineterminator="\n")

    grid_kwargs = {"lat": LAT, "lon": LON, "sst_annual": annual}
    for m in range(12):
        grid_kwargs[f"sst_{m + 1:02d}"] = monthly[m]
    np.savez_compressed(out_dir / "climatology_grid.npz", **grid_kwargs)

    np.savez_compressed(
        out_dir / "basin_mask.npz",
        basin_id=basin_id,
        basin_names=np.array(BASIN_NAMES, dtype="U16"),
        lat=LAT,
        lon=LON,
    )

    summary: dict = {
        "synthetic": True,
        "generated_by": Path(__file__).name,
        "note": "合成数据，仅用于开发自测；不是 OISST 观测结果。",
        "dataset": "synthetic fixture (not observations)",
        "grid": {"resolution_deg": 0.25, "n_latitude": N_LAT, "n_longitude": N_LON},
        "coverage_rule": "valid_count / 1440（global）；分海盆为 valid_count / ocean_grid_count",
        "basins": {},
    }
    for name, group in basin_frame.groupby("basin", sort=False):
        summary["basins"][name] = basin_mean_cos(group)
    summary["basins"]["global"] = basin_mean_cos(global_frame)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        summary["global_annual_sst_range"] = [
            round(float(np.nanmin(global_frame["sst_annual"])), 3),
            round(float(np.nanmax(global_frame["sst_annual"])), 3),
        ]
    (out_dir / "summary_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "README_fixture.txt").write_text(
        "本目录全部为 make_fixture.py 生成的合成数据，仅供 scripts/plot.py 与 scripts/verify.py 开发自测。\n"
        "不是 NOAA OISST 观测结果，禁止复制到 data/ 冒充真实结果。\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="生成合成全球海温 fixture（不写入 data/）")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="输出目录（默认 .tmp/fixture）")
    parser.add_argument("--seed", type=int, default=2024, help="随机种子")
    args = parser.parse_args()

    print(f"生成合成 fixture -> {args.out}", flush=True)
    summary = build_fixture(args.out, seed=args.seed)
    print(f"  global 年均 SST 范围：{summary['global_annual_sst_range']} °C", flush=True)
    print("  海盆 cosφ 加权均温：", flush=True)
    for name, info in summary["basins"].items():
        print(f"    {name:<9s} {info['mean_sst_cos_weighted']} °C", flush=True)
    print("完成（合成数据，未写入 data/）。", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
