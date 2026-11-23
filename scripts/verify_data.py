#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""数据与计算侧独立验收脚本（含全球契约产物验收与印度洋逐字节回归校验）。

用法：
    python scripts/verify_data.py
    python scripts/verify_data.py --indian-raw-dir D:/PythonProject1/data/raw
"""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from basins import verify_basin_mask
from config import (
    BASIN_MASK_NPZ,
    BASIN_ORDER,
    BASINS,
    BASINS_ZONAL_CSV,
    CLIMATOLOGY_NPZ,
    GLOBAL_RAW_DIR,
    GLOBAL_ZONAL_CSV,
    RAW_DIR,
    ROOT,
    SCRIPTS_DIR,
    SUMMARY_JSON,
    ZONAL_CSV,
)


EXPECTED_BASE_COLS = (
    ["latitude", "ocean_grid_count", "valid_count", "coverage", "sst_annual", "sst_jan", "sst_jul"]
    + [f"sst_{m:02d}" for m in range(1, 13)]
    + [f"cov_{m:02d}" for m in range(1, 13)]
)
EXPECTED_GLOBAL_COLS = (
    ["latitude", "ocean_grid_count", "valid_count", "coverage", "zonal_ocean_frac", "sst_annual", "sst_jan", "sst_jul"]
    + [f"sst_{m:02d}" for m in range(1, 13)]
    + [f"cov_{m:02d}" for m in range(1, 13)]
)
EXPECTED_BASINS_COLS = ["basin"] + EXPECTED_BASE_COLS


def check_raw_global() -> None:
    raw_files = sorted(GLOBAL_RAW_DIR.glob("sst_*.csv.gz"))
    assert len(raw_files) == 60, f"data/raw_global 文件数 {len(raw_files)} != 60"
    manifest_path = GLOBAL_RAW_DIR / "manifest.json"
    assert manifest_path.exists(), f"缺少 {manifest_path}"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest.get("files", [])) == 60, "manifest.json files 数 != 60"

    # 校验全部 60 个原始 .csv.gz 完整保留 ERDDAP 两行表头（列名行 + 单位行）与 720×1440 格点行数
    expected_h1 = b"time,zlev,latitude,longitude,sst"
    expected_h2 = b"UTC,m,degrees_north,degrees_east,degree_C"
    for p in raw_files:
        with gzip.open(p, "rb") as fh:
            payload = fh.read()
        lines = payload.splitlines()
        assert len(lines) == 1_036_802, f"{p.name} 行数 {len(lines)} != 1036802"
        assert lines[0] == expected_h1, f"{p.name} 第 1 行表头异常：{lines[0]!r}"
        assert lines[1] == expected_h2, f"{p.name} 第 2 行单位表头异常：{lines[1]!r}"

    gz_mb = sum(p.stat().st_size for p in raw_files) / 1e6
    csv_mb = manifest.get("total_bytes_csv", 0) / 1e6
    print(f"[1] data/raw_global: {len(raw_files)} 个文件（含 ERDDAP 双行表头与 720×1440 格点），gzip={gz_mb:.2f} MB，解压={csv_mb:.2f} MB")


def check_csv_contracts() -> None:
    df_g = pd.read_csv(GLOBAL_ZONAL_CSV)
    assert len(df_g) == 720, f"zonal_mean_global.csv 行数 {len(df_g)} != 720"
    assert list(df_g.columns) == EXPECTED_GLOBAL_COLS, f"zonal_mean_global.csv 列不匹配：{list(df_g.columns)}"
    lats = df_g["latitude"].to_numpy(dtype=float)
    assert np.isclose(lats[0], -89.875) and np.isclose(lats[-1], 89.875), "全球纬度起止范围异常"
    assert np.allclose(np.diff(lats), 0.25, atol=1e-9), "全球纬度步长不是 0.25°"
    assert (df_g["coverage"].dropna() >= 0).all() and (df_g["coverage"].dropna() <= 1.0 + 1e-6).all(), "覆盖率越界"

    df_b = pd.read_csv(BASINS_ZONAL_CSV)
    assert list(df_b.columns) == EXPECTED_BASINS_COLS, f"zonal_mean_basins.csv 列不匹配：{list(df_b.columns)}"
    assert (df_b["ocean_grid_count"] > 0).all(), "zonal_mean_basins.csv 存在 ocean_grid_count <= 0 的行"
    present_basins = tuple(dict.fromkeys(df_b["basin"].tolist()))
    assert present_basins == BASIN_ORDER, f"海盆顺序或种类异常：{present_basins} != {BASIN_ORDER}"
    assert (df_b["coverage"] >= 0).all() and (df_b["coverage"] <= 1.0 + 1e-6).all(), "分海盆覆盖率越界"

    # 强断言：zonal_mean_basins.csv 中 basin == "global" 与 zonal_mean_global.csv 公共列逐行完全一致
    df_g_nz = df_g[df_g["ocean_grid_count"] > 0][EXPECTED_BASE_COLS].reset_index(drop=True)
    df_b_global = df_b[df_b["basin"] == "global"][EXPECTED_BASE_COLS].reset_index(drop=True)
    pd.testing.assert_frame_equal(df_b_global, df_g_nz, check_exact=True)

    print(
        f"[2] CSV 契约通过：zonal_mean_global.csv ({len(df_g)}×{len(df_g.columns)}), "
        f"zonal_mean_basins.csv ({len(df_b)}×{len(df_b.columns)})（global 行与全局表 100% 匹配）"
    )


def check_npz_and_mask() -> None:
    assert BASIN_MASK_NPZ.exists(), f"缺少 {BASIN_MASK_NPZ}"
    mask_mb = BASIN_MASK_NPZ.stat().st_size / 1e6
    assert mask_mb < 1.0, f"basin_mask.npz 体积过大：{mask_mb:.3f} MB"
    with np.load(BASIN_MASK_NPZ) as mz:
        assert set(mz.files) == {"lat", "lon", "basin_mask", "basin_id", "basin_names"}, (
            f"basin_mask.npz 键不匹配：{mz.files}"
        )
        assert mz["lat"].shape == (720,) and mz["lat"].dtype == np.float32
        assert mz["lon"].shape == (1440,) and mz["lon"].dtype == np.float32
        assert mz["basin_mask"].shape == (720, 1440) and mz["basin_mask"].dtype == np.uint8
        assert mz["basin_id"].shape == (720, 1440) and mz["basin_id"].dtype == np.int8
        assert np.array_equal(
            mz["basin_id"],
            np.where(mz["basin_mask"] == 0, np.int8(-1), mz["basin_mask"].astype(np.int8)),
        )
        assert list(mz["basin_names"]) == [BASINS[i]["key"] for i in range(6)]

    assert CLIMATOLOGY_NPZ.exists(), f"缺少 {CLIMATOLOGY_NPZ}"
    npz_mb = CLIMATOLOGY_NPZ.stat().st_size / 1e6
    assert 0.5 < npz_mb <= 20.0, f"climatology_grid.npz 体积异常：{npz_mb:.2f} MB"
    with np.load(CLIMATOLOGY_NPZ) as z:
        expected = {
            "lat": ((720,), np.float32),
            "lon": ((1440,), np.float32),
            "latitude": ((720,), np.float32),
            "longitude": ((1440,), np.float32),
            "basin_mask": ((720, 1440), np.uint8),
            "sst_annual": ((720, 1440), np.float32),
            "sst_jan": ((720, 1440), np.float32),
            "sst_jul": ((720, 1440), np.float32),
            "valid_steps": ((720, 1440), np.uint8),
        }
        monthly_keys = {f"sst_{m:02d}": ((720, 1440), np.float32) for m in range(1, 13)}
        has_monthly = "sst_01" in z.files
        if has_monthly:
            expected.update(monthly_keys)
        assert set(z.files) == set(expected), f"npz 键不匹配：{z.files}"
        for k, (shape, dt) in expected.items():
            assert z[k].shape == shape, f"npz[{k}] shape {z[k].shape} != {shape}"
            assert z[k].dtype == dt, f"npz[{k}] dtype {z[k].dtype} != {dt}"
        assert np.array_equal(z["lat"], z["latitude"]) and np.array_equal(z["lon"], z["longitude"])

        lat = z["latitude"]
        lon = z["longitude"]
        basin_mask = z["basin_mask"]
        valid_steps = z["valid_steps"]
        ocean_valid = valid_steps > 0
        checkpoints = verify_basin_mask(lat, lon, basin_mask, ocean_valid)

    print(
        f"[3] basin_mask.npz ({mask_mb:.3f} MB) 与 climatology_grid.npz ({npz_mb:.2f} MB, "
        f"含逐月={has_monthly}) 通过，关键地理点掩膜抽检 {len(checkpoints)}/{len(checkpoints)} 全通过："
    )
    for cp in checkpoints:
        print(f"    - {cp['name']} ({cp['grid_lat']:.3f}, {cp['grid_lon']:.3f}) -> {cp['actual']}")


def check_summary_json() -> dict:
    assert SUMMARY_JSON.exists(), f"缺少 {SUMMARY_JSON}"
    data = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    assert data.get("weighting") == "cos(lat)*valid_count", "summary_metrics.json 顶层缺少 weighting"
    assert "basins" in data and "pipeline" in data, "summary_metrics.json 缺少 basins 或 pipeline"
    assert "excluded_inland_cells" in data["pipeline"], "pipeline 缺少 excluded_inland_cells"
    for k in BASIN_ORDER:
        assert k in data["basins"], f"summary_metrics.json 缺少海盆 {k}"
        b = data["basins"][k]
        assert b.get("weighting") == "cos(lat)*valid_count", f"summary_metrics.json[{k}] 缺少 weighting"
        for field in (
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
        ):
            assert field in b, f"summary_metrics.json[{k}] 缺少字段 {field}"
    print(f"[4] summary_metrics.json 结构校验通过（{len(data['basins'])} 个分区）")
    return data


def check_indian_regression(indian_raw_dir: Path | None) -> bool:
    candidates = [indian_raw_dir, RAW_DIR, Path("D:/PythonProject1/data/raw")]
    src = next((p for p in candidates if p is not None and p.exists() and any(p.glob("sst_*.csv*"))), None)
    if src is None:
        raise SystemExit("未找到印度洋原始缓存目录（data/raw 或 D:/PythonProject1/data/raw），请通过 --indian-raw-dir 指定")

    # 从 git 7a4485d 提取基准 CSV，避免受工作区当前文件影响
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
    # 顺便确保工作区 data/zonal_mean.csv 未被破坏
    assert ZONAL_CSV.read_bytes().replace(b"\r\n", b"\n") == ref_bytes, "工作区 data/zonal_mean.csv 与 7a4485d 不一致"
    print(f"[5] 印度洋逐字节回归校验通过（数据源：{src}，与 7a4485d:data/zonal_mean.csv 100% 匹配）")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="数据与计算侧独立验收与回归校验")
    parser.add_argument("--indian-raw-dir", type=Path, default=None, help="指定印度洋原始数据目录用于回归校验")
    args = parser.parse_args()

    check_raw_global()
    check_csv_contracts()
    check_npz_and_mask()
    summary = check_summary_json()
    reg_ok = check_indian_regression(args.indian_raw_dir)

    if summary["pipeline"].get("indian_regression_exact_match") is not reg_ok:
        summary["pipeline"]["indian_regression_exact_match"] = reg_ok
        SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n数据侧全部验收通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
