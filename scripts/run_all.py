#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""全球版一键复现：下载 → 计算 → 绘图 → 独立复核 → 验收。

用法::

    python scripts/run_all.py                       # 使用 data/raw_global 缓存（缺少则下载）
    python scripts/run_all.py --force-download      # 忽略缓存重新下载
    python scripts/run_all.py --skip-download       # 直接用现有缓存
    python scripts/run_all.py --raw-dir <缓存目录> --skip-download
    python scripts/run_all.py --jobs 8

任一步骤非零退出立即停止。验收口径为全球版：720 行 × 32 列、6 个海盆、720×1440 网格、
5 张正式图、summary 字段与 verify.py 6 项无失败（ERDDAP 在线抽样不可用时可跳过并打印说明）；
并打印旧印度洋回归三项数值。
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from config import (  # noqa: E402
    BASINS_ZONAL_CSV,
    CLIMATOLOGY_NPZ,
    FIG_DIR,
    GLOBAL_RAW_DIR,
    GLOBAL_ZONAL_CSV,
    SCRIPTS_DIR,
    SUMMARY_JSON,
)

# 数据侧最终契约的列顺序（前 7 列）
FIRST7_COLUMNS = [
    "latitude",
    "ocean_grid_count",
    "valid_count",
    "coverage",
    "zonal_ocean_frac",
    "sst_annual",
    "sst_jan",
]
BASIN_NAMES = {"global", "pacific", "atlantic", "indian", "arctic", "southern"}
FIGURES = [
    "global_sst_vs_latitude.png",
    "basins_sst_vs_latitude.png",
    "global_sst_map.png",
    "lat_month_heatmap.png",
    "indian_vs_global.png",
]
VERIFY_REPORT = ROOT / ".tmp" / "verify_report_run_all.json"

INDIAN_BASELINE = {
    "sst_max": (29.14, 0.01),
    "lat_max": (2.125, 0.01),
    "isotherm_south": (-21.664, 0.01),
    "basin_mean": (26.163, 0.005),
}


def run_step(title: str, script: str, extra: list[str] | None = None) -> float:
    print("\n" + "=" * 78, flush=True)
    print(f"▶ {title}  ({script})", flush=True)
    print("=" * 78, flush=True)
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(ROOT / ".mplconfig")
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("PYTHONUTF8", "1")
    cmd = [sys.executable, str(SCRIPTS_DIR / script)] + (extra or [])
    t0 = time.time()
    result = subprocess.run(cmd, cwd=ROOT, env=env)
    elapsed = time.time() - t0
    if result.returncode != 0:
        raise SystemExit(f"步骤失败（退出码 {result.returncode}）：{script}")
    print(f"✔ 完成，用时 {elapsed:.1f}s", flush=True)
    return elapsed


def png_size(path: Path) -> tuple[int, int]:
    with open(path, "rb") as fh:
        head = fh.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        raise SystemExit(f"{path} 不是合法 PNG")
    return struct.unpack(">II", head[16:24])


def verify_outputs(report_path: Path) -> None:
    """按全球口径验收数据产物、图片、summary 与 verify 报告。"""
    print("\n" + "=" * 78, flush=True)
    print("验收检查（全球口径）", flush=True)
    print("=" * 78, flush=True)

    # [1] 全球纬向平均 CSV
    df = pd.read_csv(GLOBAL_ZONAL_CSV)
    print(f"[1] {GLOBAL_ZONAL_CSV.name}：{df.shape[0]} 行 × {df.shape[1]} 列")
    assert df.shape == (720, 32), f"形状 {df.shape} != (720, 32)"
    assert list(df.columns[:7]) == FIRST7_COLUMNS, f"前 7 列异常：{list(df.columns[:7])}"
    print(f"    前 7 列：{', '.join(FIRST7_COLUMNS)}")

    # [2] 分海盆 CSV
    basins = pd.read_csv(BASINS_ZONAL_CSV)
    names = set(basins["basin"].astype(str).str.strip().str.lower())
    print(f"[2] {BASINS_ZONAL_CSV.name}：{len(basins)} 行，海盆 {sorted(names)}")
    assert names == BASIN_NAMES, f"海盆集合异常：{names} != {BASIN_NAMES}"

    # [3] 气候态网格
    with np.load(CLIMATOLOGY_NPZ) as npz:
        lat = npz["lat"] if "lat" in npz.files else npz["latitude"]
        lon = npz["lon"] if "lon" in npz.files else npz["longitude"]
        annual = npz["sst_annual"]
    npz_mb = CLIMATOLOGY_NPZ.stat().st_size / 1e6
    print(f"[3] {CLIMATOLOGY_NPZ.name}：{npz_mb:.2f} MB，sst_annual {annual.shape}")
    assert 3.0 < npz_mb < 20.0, f"npz 体积 {npz_mb:.2f} MB 不接近 5 MB"
    assert (len(lat), len(lon)) == (720, 1440), f"lat/lon 长度 {len(lat)}/{len(lon)} != 720/1440"
    assert annual.shape == (720, 1440), f"sst_annual 形状 {annual.shape} != (720, 1440)"

    # [4] 正式图
    print("[4] 正式图：")
    for name in FIGURES:
        path = FIG_DIR / name
        assert path.exists(), f"缺少 {path}"
        width, height = png_size(path)
        size_kb = path.stat().st_size / 1024
        print(f"    figures/{name}：{width}×{height} px，{size_kb:.0f} KB")
        assert size_kb > 100, f"{name} 只有 {size_kb:.1f} KB（要求 > 100 KB）"

    # [5] summary_metrics
    summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    value = summary["basins"]["global"]["mean_sst_cos_weighted"]
    print(f"[5] {SUMMARY_JSON.name}：basins.global.mean_sst_cos_weighted = {value}")
    assert isinstance(value, (int, float)) and np.isfinite(value), "summary 均温字段异常"

    # [6] verify 报告：6 项无失败（ERDDAP 在线抽样不可用时允许 skip，但必须打印说明）
    report = json.loads(report_path.read_text(encoding="utf-8"))
    counts = report["counts"]
    print(f"[6] verify.py：pass={counts['pass']} fail={counts['fail']} skip={counts['skip']}")
    assert counts["fail"] == 0, f"verify 存在失败项：{counts}"
    assert counts["pass"] + counts["skip"] == 6, f"verify 检查项数异常：{counts}"
    if counts["skip"]:
        skipped = "、".join(c["name"] for c in report["checks"] if c["status"] == "skip")
        print(f"    注意：{counts['skip']} 项被跳过（{skipped}），无失败项，视为通过。")

    # [7] 旧印度洋回归证据
    indian = next((c for c in report["checks"] if c["name"] == "印度洋旧结果回归"), None)
    assert indian is not None and indian["status"] == "pass", "缺少印度洋回归 PASS"
    measured = indian["metrics"]["measured"]
    print("[7] 旧印度洋回归（口径未被破坏）：")
    print(f"    最高 {measured['sst_max']:.2f} °C @ {abs(measured['lat_max']):.2f}°"
          f"{'N' if measured['lat_max'] >= 0 else 'S'}")
    print(f"    25 °C 等温线 {measured['isotherm_south']:.2f}°")
    print(f"    海盆均值 {measured['basin_mean']:.2f} °C")
    for key, (expected, tol) in INDIAN_BASELINE.items():
        assert abs(measured[key] - expected) <= tol, f"{key} 回归失败：{measured[key]} vs {expected}"

    print("\n验收通过。")


def main() -> int:
    parser = argparse.ArgumentParser(description="全球版一键复现：下载 → 计算 → 绘图 → 独立复核")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--force-download", action="store_true", help="忽略已有缓存重新下载")
    group.add_argument("--skip-download", action="store_true", help="跳过下载，直接用现有 raw 缓存")
    parser.add_argument("--raw-dir", type=Path, default=GLOBAL_RAW_DIR, help="raw 缓存目录（默认 data/raw_global）")
    parser.add_argument("--jobs", type=int, default=4, help="下载并发数（默认 4）")
    args = parser.parse_args()

    raw_dir = args.raw_dir.resolve()
    default_raw = GLOBAL_RAW_DIR.resolve()
    if raw_dir != default_raw and not args.skip_download:
        raise SystemExit(
            "fetch_data.py 只写入默认缓存 data/raw_global；使用 --raw-dir 自定义目录时请同时加 --skip-download。"
        )

    print(f"Python {sys.version.split()[0]}  {sys.executable}")
    print(f"项目根目录：{ROOT}")
    print(f"raw 缓存：{raw_dir}")

    timings: dict[str, float] = {}
    total0 = time.time()

    if args.skip_download:
        print("\n▶ 已按 --skip-download 跳过下载步骤", flush=True)
        timings["下载（跳过）"] = 0.0
    else:
        fetch_args = ["--region", "global", "--jobs", str(args.jobs)]
        if args.force_download:
            fetch_args.append("--force")
        timings["下载"] = run_step("下载全球原始数据（已缓存则跳过）", "fetch_data.py", fetch_args)

    timings["计算"] = run_step(
        "计算全球纬向平均、五大洋与网格产物",
        "compute.py",
        ["--region", "global", "--raw-dir", str(raw_dir)],
    )
    timings["绘图"] = run_step("绘制 5 张正式图", "plot.py")
    timings["复核"] = run_step(
        "独立复核（60 时次 raw + ERDDAP 在线抽样）",
        "verify.py",
        ["--raw-dir", str(raw_dir), "--report", str(VERIFY_REPORT)],
    )

    verify_outputs(VERIFY_REPORT)

    total = time.time() - total0
    print("\n" + "=" * 78)
    print("各步骤耗时：")
    for name, elapsed in timings.items():
        print(f"  {name:<10s} {elapsed:8.1f} s")
    print(f"  {'总计':<10s} {total:8.1f} s")
    print("\n输出：")
    print(f"  {GLOBAL_ZONAL_CSV}")
    print(f"  {BASINS_ZONAL_CSV}")
    print(f"  {CLIMATOLOGY_NPZ}")
    print(f"  {SUMMARY_JSON}")
    for name in FIGURES:
        print(f"  figures/{name}")
    print(f"  {VERIFY_REPORT}")
    print("\n全部完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
