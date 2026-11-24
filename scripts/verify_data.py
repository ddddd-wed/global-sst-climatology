#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""数据与计算侧独立验收脚本（Wave 1：双口径、P1 异常场、P2 离散包络与印度洋逐字节回归）。

用法：
    python scripts/verify_data.py
    python scripts/verify_data.py --raw-dir D:/PythonProject1/data/raw_global --indian-raw-dir D:/PythonProject1/data/raw
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from basins import verify_basin_mask
from config import (
    ANOMALY_GRID_NPZ,
    BASIN_MASK_NPZ,
    BASIN_ORDER,
    BASINS,
    BASINS_ZONAL_CSV,
    BASINS_ZONAL_ICEFREE_CSV,
    CLIMATOLOGY_NPZ,
    GLOBAL_ICE_RAW_DIR,
    GLOBAL_RAW_DIR,
    GLOBAL_ZONAL_CSV,
    GLOBAL_ZONAL_ICEFREE_CSV,
    ICE_CONC_MAX,
    RAW_DIR,
    ROOT,
    SCRIPTS_DIR,
    SUMMARY_JSON,
    ZONAL_CSV,
    ZONAL_REF_CSV,
)

OLD_GLOBAL_COLS = (
    ["latitude", "ocean_grid_count", "valid_count", "coverage", "zonal_ocean_frac", "sst_annual", "sst_jan", "sst_jul"]
    + [f"sst_{m:02d}" for m in range(1, 13)]
    + [f"cov_{m:02d}" for m in range(1, 13)]
)
OLD_BASINS_COLS = (
    ["basin", "latitude", "ocean_grid_count", "valid_count", "coverage", "sst_annual", "sst_jan", "sst_jul"]
    + [f"sst_{m:02d}" for m in range(1, 13)]
    + [f"cov_{m:02d}" for m in range(1, 13)]
)

DUAL_ICE_COLS = [
    "valid_count_icefree",
    "coverage_icefree",
    "sst_annual_icefree",
    "sst_jan_icefree",
    "sst_jul_icefree",
]
P2_AND_ICE_COLS = [
    "ice_frac",
    "sst_std",
    "sst_p10",
    "sst_p90",
    "sst_jan_p10",
    "sst_jan_p90",
    "sst_jul_p10",
    "sst_jul_p90",
    "sst_interannual_std",
    "sst_jan_interannual_std",
    "sst_jul_interannual_std",
    "interannual_min",
    "interannual_max",
    "sst_warmest",
    "warmest_month",
    "sst_coldest",
    "coldest_month",
    "seasonal_amp",
]

EXPECTED_GLOBAL_V2_COLS = OLD_GLOBAL_COLS + DUAL_ICE_COLS + P2_AND_ICE_COLS
EXPECTED_BASINS_V2_COLS = OLD_BASINS_COLS + DUAL_ICE_COLS + P2_AND_ICE_COLS
EXPECTED_GLOBAL_ICEFREE_COLS = OLD_GLOBAL_COLS + P2_AND_ICE_COLS
EXPECTED_BASINS_ICEFREE_COLS = OLD_BASINS_COLS + P2_AND_ICE_COLS


def _resolve_raw_dir(raw_dir: Path | None, glob_pat: str, candidates: list[Path | None]) -> Path:
    for c in [raw_dir, *candidates]:
        if c is not None and c.exists() and len(list(c.glob(glob_pat))) == 60:
            return c
    raise SystemExit(f"未找到完整 60 个 {glob_pat} 文件目录（已检查 {[str(c) for c in [raw_dir, *candidates] if c]}）")


def check_raw_sst_and_ice(sst_raw_dir: Path | None, ice_raw_dir: Path | None) -> tuple[Path, Path]:
    sst_dir = _resolve_raw_dir(
        sst_raw_dir,
        "sst_*.csv.gz",
        [
            GLOBAL_RAW_DIR,
            Path("D:/PythonProject1/data/raw_global"),
            Path("C:/Users/Lenovo/orca/workspaces/PythonProject1/global-ocean-sst-audit/data/raw_global"),
        ],
    )
    ice_dir = _resolve_raw_dir(ice_raw_dir, "ice_*.csv.gz", [GLOBAL_ICE_RAW_DIR])

    sst_files = sorted(sst_dir.glob("sst_*.csv.gz"))
    sst_manifest = json.loads((sst_dir / "manifest.json").read_text(encoding="utf-8"))
    assert len(sst_manifest.get("files", [])) == 60, "SST manifest.json files 数 != 60"
    for p in sst_files:
        with gzip.open(p, "rb") as fh:
            payload = fh.read()
        lines = payload.splitlines()
        assert len(lines) == 1_036_802, f"{p.name} 行数 {len(lines)} != 1036802"
        assert lines[0] == b"time,zlev,latitude,longitude,sst", f"{p.name} 表头异常"
        assert lines[1] == b"UTC,m,degrees_north,degrees_east,degree_C", f"{p.name} 单位行异常"

    ice_files = sorted(ice_dir.glob("ice_*.csv.gz"))
    ice_manifest_path = ice_dir / "manifest.json"
    assert ice_manifest_path.exists(), f"缺少 {ice_manifest_path}"
    ice_manifest = json.loads(ice_manifest_path.read_text(encoding="utf-8"))
    assert len(ice_manifest.get("files", [])) == 60, "ICE manifest.json files 数 != 60"
    ice_min, ice_max = np.inf, -np.inf
    for p in ice_files:
        with gzip.open(p, "rb") as fh:
            payload = fh.read()
        lines = payload.splitlines()
        assert len(lines) == 1_036_802, f"{p.name} 行数 {len(lines)} != 1036802"
        assert lines[0] == b"time,zlev,latitude,longitude,ice", f"{p.name} 表头异常"
        assert lines[1] == b"UTC,m,degrees_north,degrees_east,1", f"{p.name} 单位行异常"
        if p.name in ("ice_2018_01.csv.gz", "ice_2018_07.csv.gz"):
            arr = pd.read_csv(io.BytesIO(payload), skiprows=[1], usecols=["ice"], dtype={"ice": "float64"})["ice"].to_numpy()
            fin = arr[np.isfinite(arr)]
            ice_min = min(ice_min, float(np.min(fin)))
            ice_max = max(ice_max, float(np.max(fin)))
            assert 0.0 <= ice_min and ice_max <= 1.0, f"{p.name} ice 值域越界: [{ice_min}, {ice_max}]"

    sst_gz_mb = sum(p.stat().st_size for p in sst_files) / 1e6
    ice_gz_mb = sum(p.stat().st_size for p in ice_files) / 1e6
    print(
        f"[1] 原始缓存通过：SST={len(sst_files)} 文件 ({sst_gz_mb:.2f} MB @ {sst_dir}), "
        f"ICE={len(ice_files)} 文件 ({ice_gz_mb:.2f} MB @ {ice_dir}, 值域 [{ice_min:.2f}, {ice_max:.2f}])"
    )
    return sst_dir, ice_dir


def check_csv_contracts() -> None:
    df_g = pd.read_csv(GLOBAL_ZONAL_CSV)
    assert len(df_g) == 720, f"zonal_mean_global.csv 行数 {len(df_g)} != 720"
    assert list(df_g.columns) == EXPECTED_GLOBAL_V2_COLS, f"zonal_mean_global.csv 列不匹配：{list(df_g.columns)}"

    df_b = pd.read_csv(BASINS_ZONAL_CSV)
    assert list(df_b.columns) == EXPECTED_BASINS_V2_COLS, f"zonal_mean_basins.csv 列不匹配：{list(df_b.columns)}"

    df_g_if = pd.read_csv(GLOBAL_ZONAL_ICEFREE_CSV)
    assert len(df_g_if) == 720, f"zonal_mean_global_icefree.csv 行数 {len(df_g_if)} != 720"
    assert list(df_g_if.columns) == EXPECTED_GLOBAL_ICEFREE_COLS, (
        f"zonal_mean_global_icefree.csv 列不匹配：{list(df_g_if.columns)}"
    )

    df_b_if = pd.read_csv(BASINS_ZONAL_ICEFREE_CSV)
    assert list(df_b_if.columns) == EXPECTED_BASINS_ICEFREE_COLS, (
        f"zonal_mean_basins_icefree.csv 列不匹配：{list(df_b_if.columns)}"
    )

    # 1) 强回归：zonal_mean_global.csv 与 zonal_mean_basins.csv 前 32 列与 fe3a68b 100% 数值一致
    ref_g_bytes = subprocess.run(
        ["git", "show", "fe3a68b:data/zonal_mean_global.csv"], cwd=ROOT, capture_output=True, check=True
    ).stdout
    ref_g_32 = pd.read_csv(io.BytesIO(ref_g_bytes))
    pd.testing.assert_frame_equal(df_g[OLD_GLOBAL_COLS], ref_g_32[OLD_GLOBAL_COLS], check_exact=False, atol=1e-12)

    ref_b_bytes = subprocess.run(
        ["git", "show", "fe3a68b:data/zonal_mean_basins.csv"], cwd=ROOT, capture_output=True, check=True
    ).stdout
    ref_b_32 = pd.read_csv(io.BytesIO(ref_b_bytes))
    pd.testing.assert_frame_equal(df_b[OLD_BASINS_COLS], ref_b_32[OLD_BASINS_COLS], check_exact=False, atol=1e-12)

    # 2) basins 表 global 行与 global 表公共列完全一致（含 all_ocean 与 icefree 两套）
    common_all = [c for c in EXPECTED_BASINS_V2_COLS if c not in ("basin",)]
    df_g_nz = df_g[df_g["ocean_grid_count"] > 0][common_all].reset_index(drop=True)
    df_b_global = df_b[df_b["basin"] == "global"][common_all].reset_index(drop=True)
    pd.testing.assert_frame_equal(df_b_global, df_g_nz, check_exact=False, check_dtype=False, atol=1e-12)

    common_if = [c for c in EXPECTED_BASINS_ICEFREE_COLS if c not in ("basin",)]
    df_g_if_nz = df_g_if[df_g_if["ocean_grid_count"] > 0][common_if].reset_index(drop=True)
    df_b_if_global = df_b_if[df_b_if["basin"] == "global"][common_if].reset_index(drop=True)
    pd.testing.assert_frame_equal(df_b_if_global, df_g_if_nz, check_exact=False, check_dtype=False, atol=1e-12)

    # 3) zonal_mean_global.csv 中的 icefree 列与 zonal_mean_global_icefree.csv 完全一致
    pd.testing.assert_series_equal(df_g["valid_count_icefree"], df_g_if["valid_count"], check_names=False)
    pd.testing.assert_series_equal(df_g["coverage_icefree"], df_g_if["coverage"], check_names=False)
    pd.testing.assert_series_equal(df_g["sst_annual_icefree"], df_g_if["sst_annual"], check_names=False)
    pd.testing.assert_series_equal(df_g["sst_jan_icefree"], df_g_if["sst_jan"], check_names=False)
    pd.testing.assert_series_equal(df_g["sst_jul_icefree"], df_g_if["sst_jul"], check_names=False)
    pd.testing.assert_frame_equal(df_g[P2_AND_ICE_COLS], df_g_if[P2_AND_ICE_COLS], check_dtype=False)

    # 4) 双口径物理逻辑自洽检查
    for name, d in [("zonal_mean_basins", df_b), ("zonal_mean_basins_icefree", df_b_if)]:
        assert tuple(dict.fromkeys(d["basin"].tolist())) == BASIN_ORDER, f"{name} 海盆顺序异常"
        assert (d["ocean_grid_count"] > 0).all(), f"{name} 存在 ocean_grid_count <= 0"

    cov_if = df_b["coverage_icefree"].to_numpy(dtype=float)
    ice_f = df_b["ice_frac"].to_numpy(dtype=float)
    assert np.allclose(cov_if + ice_f, 1.0, atol=3e-4), "coverage_icefree + ice_frac 与 1.0 不自洽"

    valid_p = np.isfinite(df_b["sst_p10"]) & np.isfinite(df_b["sst_p90"])
    assert (df_b.loc[valid_p, "sst_p10"] <= df_b.loc[valid_p, "sst_p90"] + 1e-6).all(), "sst_p10 > sst_p90"
    valid_w = np.isfinite(df_b["sst_warmest"]) & np.isfinite(df_b["sst_coldest"])
    assert (df_b.loc[valid_w, "sst_coldest"] <= df_b.loc[valid_w, "sst_warmest"] + 1e-6).all(), "coldest > warmest"
    assert np.allclose(
        df_b.loc[valid_w, "seasonal_amp"],
        df_b.loc[valid_w, "sst_warmest"] - df_b.loc[valid_w, "sst_coldest"],
        atol=2e-3,
    ), "seasonal_amp != sst_warmest - sst_coldest"

    print(
        f"[2] 四张 CSV 契约与 fe3a68b 前 32 列回归全通过："
        f"global({df_g.shape[0]}×{df_g.shape[1]}), basins({df_b.shape[0]}×{df_b.shape[1]}), "
        f"global_icefree({df_g_if.shape[0]}×{df_g_if.shape[1]}), basins_icefree({df_b_if.shape[0]}×{df_b_if.shape[1]})"
    )


def check_npz_and_zonal_ref() -> None:
    assert BASIN_MASK_NPZ.exists(), f"缺少 {BASIN_MASK_NPZ}"
    mask_mb = BASIN_MASK_NPZ.stat().st_size / 1e6
    with np.load(BASIN_MASK_NPZ) as mz:
        assert set(mz.files) == {"lat", "lon", "basin_mask", "basin_id", "basin_names"}

    assert CLIMATOLOGY_NPZ.exists(), f"缺少 {CLIMATOLOGY_NPZ}"
    clim_mb = CLIMATOLOGY_NPZ.stat().st_size / 1e6

    assert ZONAL_REF_CSV.exists(), f"缺少 {ZONAL_REF_CSV}"
    df_ref = pd.read_csv(ZONAL_REF_CSV)
    assert len(df_ref) == 720, f"zonal_ref.csv 行数 {len(df_ref)} != 720"
    assert list(df_ref.columns) == ["latitude", "ref_annual", "ref_jan", "ref_jul"], (
        f"zonal_ref.csv 列不匹配：{list(df_ref.columns)}"
    )

    assert ANOMALY_GRID_NPZ.exists(), f"缺少 {ANOMALY_GRID_NPZ}"
    anom_mb = ANOMALY_GRID_NPZ.stat().st_size / 1e6
    assert 1.0 < anom_mb <= 15.0, f"anomaly_grid.npz 体积异常：{anom_mb:.2f} MB"

    expected_anom = {
        "lat": ((720,), np.float32),
        "latitude": ((720,), np.float32),
        "lon": ((1440,), np.float32),
        "longitude": ((1440,), np.float32),
        "basin_mask": ((720, 1440), np.uint8),
        "ice_steps": ((720, 1440), np.uint8),
        "ice_freq": ((720, 1440), np.float32),
        "valid_steps_icefree": ((720, 1440), np.uint8),
        "sst_annual_icefree": ((720, 1440), np.float32),
        "sst_jan_icefree": ((720, 1440), np.float32),
        "sst_jul_icefree": ((720, 1440), np.float32),
        "sst_annual_anom": ((720, 1440), np.float32),
        "sst_jan_anom": ((720, 1440), np.float32),
        "sst_jul_anom": ((720, 1440), np.float32),
    }

    with np.load(ANOMALY_GRID_NPZ) as az:
        assert set(az.files) == set(expected_anom), f"anomaly_grid.npz 键不匹配：{az.files}"
        for k, (shape, dt) in expected_anom.items():
            assert az[k].shape == shape and az[k].dtype == dt, f"anomaly_grid[{k}] 维度/类型异常"

        bm = az["basin_mask"]
        ocean_2d = bm > 0
        ice_steps = az["ice_steps"]
        vs_if = az["valid_steps_icefree"]
        assert np.all(ice_steps[~ocean_2d] == 0) and np.all(vs_if[~ocean_2d] == 0), "陆地/内陆湖计数应为 0"
        assert np.all(ice_steps[ocean_2d].astype(int) + vs_if[ocean_2d].astype(int) == 60), "ice_steps + valid_steps_icefree != 60"
        assert np.allclose(az["ice_freq"], ice_steps.astype(np.float32) / 60.0, atol=1e-6)

        for field_key, ref_col, anom_key in [
            ("sst_annual_icefree", "ref_annual", "sst_annual_anom"),
            ("sst_jan_icefree", "ref_jan", "sst_jan_anom"),
            ("sst_jul_icefree", "ref_jul", "sst_jul_anom"),
        ]:
            sst_2d = az[field_key].astype(np.float64)
            anom_2d = az[anom_key].astype(np.float64)
            assert np.all(~np.isfinite(sst_2d[~ocean_2d])), f"{field_key} 陆地/内陆湖应为 NaN"
            assert np.all(~np.isfinite(anom_2d[~ocean_2d])), f"{anom_key} 陆地/内陆湖应为 NaN"

            ref_vals = df_ref[ref_col].to_numpy(dtype=np.float64)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                row_mean = np.nanmean(sst_2d, axis=1)
                anom_mean = np.nanmean(anom_2d, axis=1)

            both_fin = np.isfinite(row_mean) & np.isfinite(ref_vals)
            max_ref_err = float(np.max(np.abs(row_mean[both_fin] - ref_vals[both_fin])))
            max_anom_mean = float(np.max(np.abs(anom_mean[np.isfinite(anom_mean)])))
            assert max_ref_err < 1e-4, f"{ref_col} 与 {field_key} 逐纬 nanmean 残差过大：{max_ref_err}"
            assert max_anom_mean < 1e-4, f"{anom_key} 逐纬均值偏离 0 过大：{max_anom_mean}"

            diff = anom_2d - (sst_2d - ref_vals[:, None])
            fin_diff = diff[np.isfinite(diff)]
            assert float(np.max(np.abs(fin_diff))) < 1e-4, f"{anom_key} != {field_key} - {ref_col}"

        lat = az["lat"]
        lon = az["lon"]
        anom_ann = az["sst_annual_anom"]
        for label, tlat, tlon, op, thresh in [
            ("黑潮", 35.125, 141.625, ">", 2.0),
            ("湾流", 38.0, 295.0, ">", 2.0),
            ("秘鲁寒流", -15.0, 282.0, "<", -1.5),
            ("东太冷舌", -1.0, 250.0, "<", -1.5),
            ("本格拉寒流", -20.0, 12.0, "<", -1.5),
        ]:
            i = int(np.argmin(np.abs(lat - tlat)))
            j = int(np.argmin(np.abs(lon - tlon)))
            v = float(anom_ann[i, j])
            assert np.isfinite(v), f"{label} ({lat[i]:.3f}, {lon[j]:.3f}) 为 NaN"
            if op == ">":
                assert v > thresh, f"{label} 异常值 {v:.3f} <= {thresh}"
            else:
                assert v < thresh, f"{label} 异常值 {v:.3f} >= {thresh}"

        with np.load(CLIMATOLOGY_NPZ) as cz:
            checkpoints = verify_basin_mask(cz["latitude"], cz["longitude"], cz["basin_mask"], cz["valid_steps"] > 0)

    print(
        f"[3] anomaly_grid.npz ({anom_mb:.2f} MB) 与 zonal_ref.csv 自洽（逐纬残差 < 1e-4°C），"
        f"五大洋掩膜({mask_mb:.3f} MB)/climatology({clim_mb:.2f} MB)/5 大洋流异常物理符号全通过（抽检 {len(checkpoints)} 点）"
    )


def check_summary_json() -> dict:
    assert SUMMARY_JSON.exists(), f"缺少 {SUMMARY_JSON}"
    data = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    assert data.get("icefree_definition") == "ice < 0.15 (OISST ice variable)"
    assert data.get("ice_threshold") == ICE_CONC_MAX
    assert data.get("weighting") == "cos(lat)*valid_count"
    assert "basins" in data and "basins_icefree" in data and "pipeline" in data
    assert "dispersion_highlights" in data and "anomaly_checkpoints" in data

    req_fields = (
        "weighting",
        "coverage_definition",
        "mean_sst_cos_weighted",
        "area_weighted_annual_sst",
        "grid_area_weighted_annual_sst",
        "max_sst_lat",
        "max_sst_c",
        "min_sst_lat",
        "min_sst_c",
        "isotherms_25c",
        "isotherms_0c",
        "valid_lat_min",
        "valid_lat_max",
        "ocean_grid_total",
        "lat_row_count",
    )
    for k in BASIN_ORDER:
        b = data["basins"][k]
        for f in req_fields:
            assert f in b, f"basins[{k}] 缺少 {f}"
        assert "icefree" in b, f"basins[{k}] 缺少 icefree 子对象"
        bif = data["basins_icefree"][k]
        assert b["icefree"] == bif, f"basins[{k}].icefree 与 basins_icefree[{k}] 不一致"
        for f in req_fields + ("delta_vs_all_ocean_c", "mean_ice_frac"):
            assert f in bif, f"basins_icefree[{k}] 缺少 {f}"

    pipe = data["pipeline"]
    assert pipe.get("ice_raw_files_count") == 60
    assert "anomaly_grid_npz_mb" in pipe and "npz_mb" in pipe
    print(f"[4] summary_metrics.json 校验通过（{len(data['basins'])} 分区含 icefree 子对象与双口径对照）")
    return data


def check_indian_regression(indian_raw_dir: Path | None) -> bool:
    candidates = [indian_raw_dir, RAW_DIR, Path("D:/PythonProject1/data/raw")]
    src = next((p for p in candidates if p is not None and p.exists() and any(p.glob("sst_*.csv*"))), None)
    if src is None:
        raise SystemExit("未找到印度洋原始缓存目录（data/raw 或 D:/PythonProject1/data/raw），请通过 --indian-raw-dir 指定")

    git_show = subprocess.run(
        ["git", "show", "7a4485d:data/zonal_mean.csv"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    )
    ref_bytes = git_show.stdout.replace(b"\r\n", b"\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_tmp = Path(tmpdir) / "zonal_mean_indian_test.csv"
        cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "compute.py"),
            "--region",
            "indian_ocean",
            "--raw-dir",
            str(src),
            "--out",
            str(out_tmp),
        ]
        res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            raise SystemExit(f"印度洋回归重算失败：\n{res.stdout}\n{res.stderr}")
        new_bytes = out_tmp.read_bytes().replace(b"\r\n", b"\n")

    assert new_bytes == ref_bytes, "印度洋回归重算结果与 git 7a4485d:data/zonal_mean.csv 不一致！"
    assert ZONAL_CSV.read_bytes().replace(b"\r\n", b"\n") == ref_bytes, "工作区 data/zonal_mean.csv 与 7a4485d 不一致"
    print(f"[5] 印度洋逐字节回归校验通过（数据源：{src}，与 7a4485d:data/zonal_mean.csv 100% 匹配）")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="数据与计算侧独立验收与回归校验")
    parser.add_argument("--raw-dir", type=Path, default=None, help="指定全球 SST 原始目录")
    parser.add_argument("--ice-dir", type=Path, default=None, help="指定全球 ICE 原始目录")
    parser.add_argument("--indian-raw-dir", type=Path, default=None, help="指定印度洋原始数据目录用于回归校验")
    args = parser.parse_args()

    check_raw_sst_and_ice(args.raw_dir, args.ice_dir)
    check_csv_contracts()
    check_npz_and_zonal_ref()
    summary = check_summary_json()
    reg_ok = check_indian_regression(args.indian_raw_dir)

    if summary["pipeline"].get("indian_regression_exact_match") is not reg_ok:
        summary["pipeline"]["indian_regression_exact_match"] = reg_ok
        SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n数据侧 Wave 1 全部验收通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
