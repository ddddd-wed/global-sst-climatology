#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""按月下载印度洋区域的 NOAA OISST v2.1 原始数据。

- 每个时次对应的原始 CSV 以 gzip 无损压缩保存到 data/raw/（约 5 MB/月，压缩后约 2.5 MB）。
  60 个时次的未压缩 CSV 合计约 300 MB，压缩后约 150 MB，满足 "原始 CSV 总量 <= 300 MB" 的约束，
  且数据本身未做任何降采样、重采样或平滑。
- 已存在的文件默认跳过，--force 强制重新下载。
- 下载成功后校验表头、行数与时间戳，避免把 ERDDAP 的错误页当成数据缓存下来。

用法：
    python scripts/fetch_data.py            # 断点续传，已存在则跳过
    python scripts/fetch_data.py --force    # 全部重新下载
    python scripts/fetch_data.py --jobs 1   # 串行下载（默认 4 并发）
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

from config import (
    BASE_URL,
    DATASET_ID,
    EXPECTED_LINES,
    LAT_MAX,
    LAT_MIN,
    LON_MAX,
    LON_MIN,
    N_LAT,
    N_LON,
    RAW_DIR,
    SAMPLE_DAY,
    SAMPLE_HOUR,
    YEAR_END,
    YEAR_START,
)


def build_url(year: int, month: int) -> str:
    """构造 ERDDAP griddap CSV 请求（原生 0.25° 网格，不设 stride）。"""
    ts = f"{year}-{month:02d}-{SAMPLE_DAY:02d}T{SAMPLE_HOUR:02d}:00:00Z"
    query = (
        f"sst[({ts}):1:({ts})]"
        f"[(0.0):1:(0.0)]"
        f"[({LAT_MIN:.1f}):1:({LAT_MAX:.1f})]"
        f"[({LON_MIN:.1f}):1:({LON_MAX:.1f})]"
    )
    # 只保留 :(),.+- 不编码，方括号会被编码为 %5B/%5D
    return BASE_URL + "?" + quote(query, safe=":(),.+-=")


def validate(payload: bytes, year: int, month: int) -> None:
    """校验下载内容确实是完整的 281x401 原始 CSV。"""
    if not payload.startswith(b"time,zlev,latitude,longitude,sst"):
        head = payload[:300].decode("utf-8", "replace")
        raise ValueError(f"响应表头异常: {head!r}")
    n_lines = payload.count(b"\n")
    if n_lines not in (EXPECTED_LINES - 1, EXPECTED_LINES):
        raise ValueError(f"数据行数异常: {n_lines}（期望 {EXPECTED_LINES}）")
    stamp = f"{year}-{month:02d}-{SAMPLE_DAY:02d}T{SAMPLE_HOUR:02d}:00:00Z".encode()
    if stamp not in payload[:500]:
        raise ValueError(f"响应时间戳与请求不符: 期望 {stamp!r}")


def fetch_one(year: int, month: int, force: bool = False) -> tuple[str, Path, int, int]:
    """下载单个时次，返回 (状态, 路径, 压缩后字节数, 原始字节数)。"""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / f"sst_{year}_{month:02d}.csv.gz"
    if dest.exists() and not force and dest.stat().st_size > 100_000:
        return "skip", dest, dest.stat().st_size, 0

    url = build_url(year, month)
    last_err: Exception | None = None
    for attempt in range(1, 4):
        try:
            req = Request(
                url,
                headers={"User-Agent": "indian-ocean-sst/1.0 (python-urllib; research)"},
            )
            with urlopen(req, timeout=180) as resp:
                payload = resp.read()
            validate(payload, year, month)
            packed = gzip.compress(payload, compresslevel=6, mtime=0)
            tmp = dest.with_name(dest.name + ".tmp")
            tmp.write_bytes(packed)
            tmp.replace(dest)
            return "ok", dest, len(packed), len(payload)
        except Exception as exc:  # noqa: BLE001 - 下载失败需要重试
            last_err = exc
            wait = 3 * attempt
            print(f"  [重试 {attempt}/3] {year}-{month:02d}: {exc}（{wait}s 后重试）", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"{year}-{month:02d} 下载失败: {last_err}")


def write_manifest(files_info: list[dict]) -> None:
    manifest = {
        "dataset": "NOAA OISST v2.1 日平均海表温度（ncdcOisst21Agg）",
        "endpoint": BASE_URL,
        "region": {"lon_min": LON_MIN, "lon_max": LON_MAX, "lat_min": LAT_MIN, "lat_max": LAT_MAX},
        "native_grid": {
            "resolution_deg": 0.25,
            "n_latitude": N_LAT,
            "n_longitude": N_LON,
            "latitude_centers": "-39.875 ~ 30.125（覆盖 -40 ~ 30）",
            "longitude_convention": "0-360（本区 20.125-120.125°E）",
        },
        "timestamps": f"{YEAR_START}-01-{SAMPLE_DAY:02d}T{SAMPLE_HOUR:02d}:00:00Z ~ "
                      f"{YEAR_END}-12-{SAMPLE_DAY:02d}T{SAMPLE_HOUR:02d}:00:00Z，每月 1 个时次，共 60 个",
        "storage": "gzip 无损压缩 CSV（.csv.gz），未做任何降采样或重采样",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": sorted(files_info, key=lambda d: (d["year"], d["month"])),
    }
    path = RAW_DIR / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def csv_size(path: Path) -> int:
    """原始（解压后）CSV 字节数，写进 manifest 便于核对体积。"""
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as fh:
            return len(fh.read())
    return path.stat().st_size


def main() -> int:
    parser = argparse.ArgumentParser(description="下载印度洋 OISST v2.1 月度原始数据")
    parser.add_argument("--force", action="store_true", help="忽略已有缓存，强制重新下载")
    parser.add_argument("--jobs", type=int, default=4, help="并发下载数（默认 4）")
    parser.add_argument("--start-year", type=int, default=YEAR_START)
    parser.add_argument("--end-year", type=int, default=YEAR_END)
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tasks = [(y, m) for y in range(args.start_year, args.end_year + 1) for m in range(1, 13)]
    total_packed = 0
    total_raw = 0
    done = 0
    failed: list[tuple[int, int, str]] = []
    files_info: list[dict] = []
    t0 = time.time()

    print(f"开始下载 {len(tasks)} 个时次 -> {RAW_DIR}（并发 {args.jobs}）", flush=True)
    if args.jobs <= 1:
        for y, m in tasks:
            try:
                status, path, packed, raw = fetch_one(y, m, args.force)
            except Exception as exc:  # noqa: BLE001
                failed.append((y, m, str(exc)))
                print(f"  [失败] {y}-{m:02d}: {exc}", flush=True)
                continue
            done += 1
            if status == "ok":
                total_packed += packed
                total_raw += raw
                print(f"  [{done:02d}/{len(tasks)}] 下载 {path.name}  {packed/1e6:.2f} MB", flush=True)
            else:
                total_packed += packed
                print(f"  [{done:02d}/{len(tasks)}] 跳过 {path.name}（已存在）", flush=True)
            files_info.append({"file": path.name, "year": y, "month": m, "bytes_gz": packed})
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(fetch_one, y, m, args.force): (y, m) for y, m in tasks}
            for fut in as_completed(futures):
                y, m = futures[fut]
                try:
                    status, path, packed, raw = fut.result()
                except Exception as exc:  # noqa: BLE001
                    failed.append((y, m, str(exc)))
                    print(f"  [失败] {y}-{m:02d}: {exc}", flush=True)
                    continue
                done += 1
                total_packed += packed
                if status == "ok":
                    total_raw += raw
                    print(f"  [{done:02d}/{len(tasks)}] 下载 {path.name}  {packed/1e6:.2f} MB", flush=True)
                else:
                    print(f"  [{done:02d}/{len(tasks)}] 跳过 {path.name}（已存在）", flush=True)
                files_info.append({"file": path.name, "year": y, "month": m, "bytes_gz": packed})

    # 清单只保留磁盘上实际存在的文件
    existing = sorted(RAW_DIR.glob("sst_*.csv.gz"))
    if len(existing) == len(tasks):
        files_info = []
        for path in existing:
            _, year_s, month_s = path.name.split(".")[0].split("_")
            files_info.append(
                {
                    "file": path.name,
                    "year": int(year_s),
                    "month": int(month_s),
                    "bytes_gz": path.stat().st_size,
                    "bytes_csv": csv_size(path),
                }
            )
        write_manifest(files_info)
    else:
        print(f"注意：仅 {len(existing)}/{len(tasks)} 个文件就绪，未写入 manifest。", flush=True)

    elapsed = time.time() - t0
    total_disk = sum(p.stat().st_size for p in existing)
    total_csv = sum(csv_size(p) for p in existing) if existing else 0
    print(
        f"\n完成：{done} 个时次，失败 {len(failed)} 个，用时 {elapsed:.1f}s；"
        f"本次新下载原始 CSV {total_raw/1e6:.1f} MB（未压缩），"
        f"data/raw 共 {len(existing)} 个文件 / {total_disk/1e6:.1f} MB（gzip，原始 CSV {total_csv/1e6:.1f} MB）。",
        flush=True,
    )
    if failed:
        print("失败清单：" + ", ".join(f"{y}-{m:02d}" for y, m, _ in failed), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
