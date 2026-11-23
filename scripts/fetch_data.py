#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""按月下载 NOAA OISST v2.1 原始数据（支持 global 与 indian_ocean 预设）。

- 请求头携带 Accept-Encoding: gzip 压低网络传输体积，流式解压后校验完整性再以 gzip 无损落盘。
- 支持断点续传（已存在且通过基础校验的文件自动跳过）、--force 强制重下、指数退避重试。
- 下载完成后写入 manifest.json，记录各文件时次、原始 CSV 字节数、gzip 落盘字节数与耗时。

用法：
    python scripts/fetch_data.py                        # 默认下载 global 60 时次
    python scripts/fetch_data.py --region indian_ocean  # 下载旧版印度洋区域
    python scripts/fetch_data.py --force --jobs 4       # 强制重下，4 并发
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
    REGION_PRESETS,
    SAMPLE_DAY,
    SAMPLE_HOUR,
    YEAR_END,
    YEAR_START,
)


def build_url(year: int, month: int, preset: dict) -> str:
    """构造 ERDDAP griddap CSV 请求（原生 0.25° 网格，不设 stride）。"""
    ts = f"{year}-{month:02d}-{SAMPLE_DAY:02d}T{SAMPLE_HOUR:02d}:00:00Z"
    query = (
        f"sst[({ts}):1:({ts})]"
        f"[(0.0):1:(0.0)]"
        f"[({preset['query_lat_min']}):1:({preset['query_lat_max']})]"
        f"[({preset['query_lon_min']}):1:({preset['query_lon_max']})]"
    )
    return BASE_URL + "?" + quote(query, safe=":(),.+-=")


def validate(payload: bytes, year: int, month: int, preset: dict) -> None:
    """校验下载内容表头、行数、时间戳及起止坐标点。"""
    if not payload.startswith(b"time,zlev,latitude,longitude,sst"):
        head = payload[:300].decode("utf-8", "replace")
        raise ValueError(f"响应表头异常: {head!r}")

    expected_lines = preset["expected_lines"]
    n_lines = payload.count(b"\n")
    if n_lines not in (expected_lines - 1, expected_lines):
        raise ValueError(f"数据行数异常: {n_lines}（期望 {expected_lines}）")

    stamp = f"{year}-{month:02d}-{SAMPLE_DAY:02d}T{SAMPLE_HOUR:02d}:00:00Z".encode()
    if stamp not in payload[:500]:
        raise ValueError(f"响应时间戳与请求不符: 期望 {stamp!r}")

    tail = payload[-200:].strip()
    last_line = tail.splitlines()[-1] if tail else b""
    if not last_line.startswith(stamp):
        raise ValueError(f"末尾行时间戳异常: {last_line!r}")


def is_valid_cached_file(path: Path, year: int, month: int, preset: dict) -> tuple[bool, int]:
    """检查本地 gzip 缓存是否完整有效，返回 (是否有效, 解压后字节数)。"""
    if not path.exists() or path.stat().st_size < 100_000:
        return False, 0
    try:
        with gzip.open(path, "rb") as fh:
            payload = fh.read()
        validate(payload, year, month, preset)
        return True, len(payload)
    except Exception:  # noqa: BLE001 - 缓存损坏时视为无效并触发重下
        return False, 0


def fetch_one(year: int, month: int, preset: dict, force: bool = False) -> tuple[str, Path, int, int, float]:
    """下载单个时次，返回 (状态, 路径, 压缩字节数, 解压字节数, 耗时秒)。"""
    raw_dir: Path = preset["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / f"sst_{year}_{month:02d}.csv.gz"

    t0 = time.time()
    if not force:
        ok, raw_len = is_valid_cached_file(dest, year, month, preset)
        if ok:
            return "skip", dest, dest.stat().st_size, raw_len, time.time() - t0

    url = build_url(year, month, preset)
    last_err: Exception | None = None
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            req = Request(
                url,
                headers={
                    "Accept-Encoding": "gzip",
                    "User-Agent": "global-ocean-sst/1.0 (python-urllib; research)",
                },
            )
            with urlopen(req, timeout=180) as resp:
                chunks: list[bytes] = []
                while True:
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    chunks.append(chunk)
                wire = b"".join(chunks)
                enc = (resp.headers.get("Content-Encoding") or "").lower()
            payload = gzip.decompress(wire) if "gzip" in enc else wire
            validate(payload, year, month, preset)
            packed = gzip.compress(payload, compresslevel=6, mtime=0)
            tmp = dest.with_name(dest.name + ".tmp")
            tmp.write_bytes(packed)
            tmp.replace(dest)
            return "ok", dest, len(packed), len(payload), time.time() - t0
        except Exception as exc:  # noqa: BLE001 - 下载失败需指数退避重试
            last_err = exc
            if attempt < max_attempts:
                wait = min(3 * (2 ** (attempt - 1)), 30)
                print(f"  [重试 {attempt}/{max_attempts}] {year}-{month:02d}: {exc}（{wait}s 后重试）", flush=True)
                time.sleep(wait)
    raise RuntimeError(f"{year}-{month:02d} 下载失败: {last_err}")


def write_manifest(
    preset: dict,
    files_info: list[dict],
    start_year: int,
    end_year: int,
    total_elapsed: float,
) -> None:
    raw_dir: Path = preset["raw_dir"]
    manifest = {
        "dataset": f"NOAA OISST v2.1 日平均海表温度（{DATASET_ID}）",
        "region_preset": preset["name"],
        "endpoint": BASE_URL,
        "region": {
            "lon_min": preset["lon_min"],
            "lon_max": preset["lon_max"],
            "lat_min": preset["lat_min"],
            "lat_max": preset["lat_max"],
        },
        "native_grid": {
            "resolution_deg": 0.25,
            "n_latitude": preset["n_lat"],
            "n_longitude": preset["n_lon"],
            "expected_lines": preset["expected_lines"],
            "longitude_convention": "0-360",
        },
        "timestamps": (
            f"{start_year}-01-{SAMPLE_DAY:02d}T{SAMPLE_HOUR:02d}:00:00Z ~ "
            f"{end_year}-12-{SAMPLE_DAY:02d}T{SAMPLE_HOUR:02d}:00:00Z，每月 1 个时次，共 {len(files_info)} 个"
        ),
        "storage": "gzip 无损压缩 CSV（.csv.gz），未做任何降采样或重采样",
        "total_bytes_gz": sum(d["bytes_gz"] for d in files_info),
        "total_bytes_csv": sum(d["bytes_csv"] for d in files_info),
        "total_elapsed_sec": round(total_elapsed, 2),
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": sorted(files_info, key=lambda d: (d["year"], d["month"])),
    }
    path = raw_dir / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="下载 NOAA OISST v2.1 月度原始数据")
    parser.add_argument(
        "--region",
        choices=tuple(REGION_PRESETS),
        default="global",
        help="区域预设：global（默认）或 indian_ocean",
    )
    parser.add_argument("--force", action="store_true", help="忽略已有缓存，强制重新下载")
    parser.add_argument("--jobs", type=int, default=4, help="并发下载数（默认 4）")
    parser.add_argument("--start-year", type=int, default=YEAR_START)
    parser.add_argument("--end-year", type=int, default=YEAR_END)
    args = parser.parse_args()

    preset = REGION_PRESETS[args.region]
    raw_dir: Path = preset["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(y, m) for y in range(args.start_year, args.end_year + 1) for m in range(1, 13)]

    done = 0
    new_raw_bytes = 0
    failed: list[tuple[int, int, str]] = []
    files_info: list[dict] = []
    t0 = time.time()

    print(
        f"[{preset['name']}] 开始下载 {len(tasks)} 个时次 -> {raw_dir}（并发 {args.jobs}）",
        flush=True,
    )

    def handle_result(y: int, m: int, status: str, path: Path, packed: int, raw: int, sec: float) -> None:
        nonlocal done, new_raw_bytes
        done += 1
        if status == "ok":
            new_raw_bytes += raw
            print(f"  [{done:02d}/{len(tasks)}] 下载 {path.name}  {packed/1e6:.2f} MB (csv {raw/1e6:.2f} MB, {sec:.1f}s)", flush=True)
        else:
            print(f"  [{done:02d}/{len(tasks)}] 跳过 {path.name}（已校验缓存，{packed/1e6:.2f} MB）", flush=True)
        files_info.append(
            {
                "file": path.name,
                "year": y,
                "month": m,
                "bytes_gz": packed,
                "bytes_csv": raw,
                "elapsed_sec": round(sec, 2),
                "status": status,
            }
        )

    if args.jobs <= 1:
        for y, m in tasks:
            try:
                res = fetch_one(y, m, preset, args.force)
                handle_result(y, m, *res)
            except Exception as exc:  # noqa: BLE001
                failed.append((y, m, str(exc)))
                print(f"  [失败] {y}-{m:02d}: {exc}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(fetch_one, y, m, preset, args.force): (y, m) for y, m in tasks}
            for fut in as_completed(futures):
                y, m = futures[fut]
                try:
                    res = fut.result()
                    handle_result(y, m, *res)
                except Exception as exc:  # noqa: BLE001
                    failed.append((y, m, str(exc)))
                    print(f"  [失败] {y}-{m:02d}: {exc}", flush=True)

    elapsed = time.time() - t0
    if not failed and len(files_info) == len(tasks):
        write_manifest(preset, files_info, args.start_year, args.end_year, elapsed)
    else:
        print(f"注意：仅 {len(files_info)}/{len(tasks)} 个文件就绪，未写入 manifest。", flush=True)

    total_gz = sum(d["bytes_gz"] for d in files_info)
    total_csv = sum(d["bytes_csv"] for d in files_info)
    print(
        f"\n完成：{done}/{len(tasks)} 个时次，失败 {len(failed)} 个，用时 {elapsed:.1f}s；"
        f"本次新下载原始 CSV {new_raw_bytes/1e6:.1f} MB；"
        f"{raw_dir.name} 共 {len(files_info)} 个文件 / {total_gz/1e6:.1f} MB（gzip，解压 {total_csv/1e6:.1f} MB）。",
        flush=True,
    )
    if failed:
        print("失败清单：" + ", ".join(f"{y}-{m:02d}" for y, m, _ in failed), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
