#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""一键复现：下载 -> 计算纬向平均 -> 绘图 -> 验收检查。

用法：
    .venv\\Scripts\\python.exe scripts\\run_all.py
    .venv\\Scripts\\python.exe scripts\\run_all.py --force-download   # 忽略缓存重新下载
"""

from __future__ import annotations

import argparse
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

from compute import summarize  # noqa: E402
from config import (  # noqa: E402
    COVERAGE_MIN,
    FIG_DIR,
    N_LAT,
    RAW_DIR,
    SCRIPTS_DIR,
    ZONAL_CSV,
)

STEPS = [
    ("下载原始数据（已缓存则跳过）", "fetch_data.py"),
    ("计算纬向平均并输出 CSV", "compute.py"),
    ("绘制曲线与热图", "plot.py"),
]


def run_step(title: str, script: str, extra: list[str] | None = None) -> None:
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
    if result.returncode != 0:
        raise SystemExit(f"步骤失败（退出码 {result.returncode}）：{script}")
    print(f"✔ 完成，用时 {time.time() - t0:.1f}s", flush=True)


def png_size(path: Path) -> tuple[int, int]:
    with open(path, "rb") as fh:
        head = fh.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        raise SystemExit(f"{path} 不是合法 PNG")
    return struct.unpack(">II", head[16:24])


def verify() -> None:
    print("\n" + "=" * 78)
    print("验收检查")
    print("=" * 78)

    df = pd.read_csv(ZONAL_CSV)
    lat = df["latitude"].to_numpy(dtype=float)
    print(f"[1] data/zonal_mean.csv：{len(df)} 行，纬度 {lat[0]:.3f} → {lat[-1]:.3f}，"
          f"步长 {np.diff(lat).mean():.3f}°")
    assert len(df) == N_LAT, f"行数 {len(df)} != {N_LAT}"
    assert np.allclose(np.diff(lat), 0.25, atol=1e-9), "纬度步长不是 0.25°"
    required = ["latitude", "valid_count", "coverage", "sst_annual", "sst_jan", "sst_jul"]
    assert list(df.columns[:6]) == required, f"列顺序异常：{list(df.columns[:6])}"
    print(f"    列：{', '.join(df.columns[:6])} + 12 个月气候态列")

    print("[2] 图片：")
    for name in ("sst_vs_latitude.png", "lat_month_heatmap.png"):
        path = FIG_DIR / name
        assert path.exists(), f"缺少 {path}"
        size_kb = path.stat().st_size / 1024
        width, height = png_size(path)
        print(f"    figures/{name}：{width}×{height} px，{size_kb:.0f} KB")
        assert size_kb > 100, f"{name} 只有 {size_kb:.1f} KB（要求 > 100 KB）"

    raw_files = sorted(RAW_DIR.glob("sst_*.csv.gz")) + sorted(RAW_DIR.glob("sst_*.csv"))
    raw_mb = sum(p.stat().st_size for p in raw_files) / 1e6
    print(f"[3] data/raw：{len(raw_files)} 个时次文件，磁盘占用 {raw_mb:.1f} MB（gzip，上限 300 MB）")
    assert len(raw_files) == 60, f"原始文件数 {len(raw_files)} != 60"
    assert raw_mb < 300, "原始数据超过 300 MB"

    print(f"[4] 关键数字（覆盖率阈值 {COVERAGE_MIN:.0%}）：")
    summarize(df)
    print("\n验收通过。")


def main() -> int:
    parser = argparse.ArgumentParser(description="一键复现印度洋 SST-纬度曲线")
    parser.add_argument("--force-download", action="store_true", help="忽略已有缓存重新下载")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()

    t0 = time.time()
    print(f"Python {sys.version.split()[0]}  {sys.executable}")
    print(f"项目根目录：{ROOT}")
    run_step(*STEPS[0], extra=(["--force"] if args.force_download else []) + ["--jobs", str(args.jobs)])
    run_step(*STEPS[1])
    run_step(*STEPS[2])
    verify()
    print(f"\n全部完成，总用时 {time.time() - t0:.1f}s")
    print("输出：data/zonal_mean.csv、figures/sst_vs_latitude.png、figures/lat_month_heatmap.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
