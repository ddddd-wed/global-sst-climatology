#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""按年下载 NOAA NCEI ERSST v5（2° × 2° 月平均 SST，1982–2024 共 43 年 = 516 个月）。

特性：
- 自动探测并按年对齐 ERDDAP nceiErsstv5 精确时间戳（注：实测早期年份时间戳为每月 1 日
  YYYY-MM-01T00:00:00Z，后期为每月 15 日 YYYY-MM-15T00:00:00Z），确保每年精确包含 12 个月
  （192,242 行 = 2 行表头 + 12 × 89 × 180 格点）。
- 优先使用 coastwatch.pfeg.noaa.gov，若遇 503/超时自动回退至 upwell / oceanwatch 镜像。
- 携带 Accept-Encoding: gzip 流式传输，校验完整性后按年压缩落盘至 data/raw_ersst/ersst_YYYY.csv.gz，
  支持断点续传、指数退避重试与 manifest.json 审计记录。

用法：
    python scripts/fetch_ersst.py                # 默认下载 1982–2024（4 并发）
    python scripts/fetch_ersst.py --jobs 6       # 6 并发
    python scripts/fetch_ersst.py --force        # 忽略缓存强制重下
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
RAW_ERSST_DIR = ROOT / "data" / "raw_ersst"

DATASET_ID = "nceiErsstv5"
ENDPOINTS = (
    f"https://coastwatch.pfeg.noaa.gov/erddap/griddap/{DATASET_ID}.csv",
    f"https://upwell.pfeg.noaa.gov/erddap/griddap/{DATASET_ID}.csv",
    f"https://oceanwatch.pifsc.noaa.gov/erddap/griddap/{DATASET_ID}.csv",
)

ERSST_YEAR_START = 1982
ERSST_YEAR_END = 2024
ERSST_N_LAT = 89     # -88.0 ~ 88.0，步长 2.0°
ERSST_N_LON = 180    # 0.0 ~ 358.0，步长 2.0°
ERSST_LAT_MIN, ERSST_LAT_MAX = -88.0, 88.0
ERSST_LON_MIN, ERSST_LON_MAX = 0.0, 358.0
ERSST_MONTHS_PER_YEAR = 12
ERSST_EXPECTED_LINES = 2 + ERSST_MONTHS_PER_YEAR * ERSST_N_LAT * ERSST_N_LON  # 192,242

HEADER1 = b"time,depth,latitude,longitude,sst"
HEADER2 = b"UTC,m,degrees_north,degrees_east,degree_C"


def _http_get_bytes(url: str, timeout: int = 120) -> bytes:
    req = Request(
        url,
        headers={
            "Accept-Encoding": "gzip",
            "User-Agent": "global-ocean-sst-ersst/2.0 (python-urllib; research)",
        },
    )
    with urlopen(req, timeout=timeout) as resp:
        chunks: list[bytes] = []
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            chunks.append(chunk)
        wire = b"".join(chunks)
        enc = (resp.headers.get("Content-Encoding") or "").lower()
    return gzip.decompress(wire) if "gzip" in enc else wire


def fetch_time_axis(start_year: int, end_year: int) -> tuple[dict[int, list[str]], str]:
    """向 ERDDAP 查询精确时间轴，返回 ({year: [12 个精确时间戳]}, 命中的端点)。"""
    q = f"time[({start_year - 1}-12-01T00:00:00Z):({end_year + 1}-03-01T00:00:00Z)]"
    last_err: Exception | None = None
    for base in ENDPOINTS:
        url = base + "?" + quote(q, safe=":(),.+-=")
        for attempt in range(1, 4):
            try:
                raw = _http_get_bytes(url, timeout=45)
                lines = [ln.decode("utf-8").strip() for ln in raw.splitlines()[2:] if ln.strip()]
                by_year: dict[int, list[str]] = defaultdict(list)
                for ts in lines:
                    y = int(ts[:4])
                    if start_year <= y <= end_year:
                        by_year[y].append(ts)
                for y in range(start_year, end_year + 1):
                    if len(by_year[y]) != ERSST_MONTHS_PER_YEAR:
                        raise ValueError(f"{y} 年时间点数异常：{len(by_year[y])} != 12")
                return dict(by_year), base
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(2 * attempt)
    raise RuntimeError(f"无法获取 ERSST v5 时间轴: {last_err}")


def build_year_url(base_url: str, ts_start: str, ts_end: str) -> str:
    query = (
        f"sst[({ts_start}):1:({ts_end})]"
        f"[(0.0):1:(0.0)]"
        f"[({ERSST_LAT_MIN}):1:({ERSST_LAT_MAX})]"
        f"[({ERSST_LON_MIN}):1:({ERSST_LON_MAX})]"
    )
    return base_url + "?" + quote(query, safe=":(),.+-=")


def validate_year_payload(
    payload: bytes,
    year: int,
    expected_ts: list[str] | None = None,
) -> None:
    """校验单年 ERSST CSV 表头、单位行、行数及起止时间戳。"""
    head_lines = payload[:300].splitlines()
    if len(head_lines) < 2 or head_lines[0] != HEADER1 or head_lines[1] != HEADER2:
        head = payload[:300].decode("utf-8", "replace")
        raise ValueError(f"ERSST {year} 表头异常: {head!r}")

    n_lines = payload.count(b"\n")
    if n_lines not in (ERSST_EXPECTED_LINES - 1, ERSST_EXPECTED_LINES):
        raise ValueError(f"ERSST {year} 行数异常: {n_lines}（期望 {ERSST_EXPECTED_LINES}）")

    first_row = head_lines[2] if len(head_lines) >= 3 else b""
    tail = payload[-200:].strip()
    last_row = tail.splitlines()[-1] if tail else b""

    if expected_ts:
        if not first_row.startswith(expected_ts[0].encode()):
            raise ValueError(f"ERSST {year} 首行时间戳不匹配: {first_row[:30]!r} vs {expected_ts[0]}")
        if not last_row.startswith(expected_ts[-1].encode()):
            raise ValueError(f"ERSST {year} 末行时间戳不匹配: {last_row[:30]!r} vs {expected_ts[-1]}")
    else:
        if not first_row.startswith(f"{year}-01-".encode()):
            raise ValueError(f"ERSST {year} 首行非 01 月: {first_row[:30]!r}")
        if not last_row.startswith(f"{year}-12-".encode()):
            raise ValueError(f"ERSST {year} 末行非 12 月: {last_row[:30]!r}")


def is_valid_cached_year(path: Path, year: int, expected_ts: list[str] | None = None) -> tuple[bool, int]:
    if not path.exists() or path.stat().st_size < 10_000:
        return False, 0
    try:
        with gzip.open(path, "rb") as fh:
            payload = fh.read()
        validate_year_payload(payload, year, expected_ts)
        return True, len(payload)
    except Exception:  # noqa: BLE001
        return False, 0


def fetch_one_year(
    year: int,
    expected_ts: list[str],
    raw_dir: Path,
    preferred_base: str,
    force: bool = False,
) -> tuple[str, Path, int, int, float, str]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / f"ersst_{year}.csv.gz"

    t0 = time.time()
    if not force:
        ok, raw_len = is_valid_cached_year(dest, year, expected_ts)
        if ok:
            return "skip", dest, dest.stat().st_size, raw_len, time.time() - t0, "cache"

    ordered_endpoints = [preferred_base] + [ep for ep in ENDPOINTS if ep != preferred_base]
    last_err: Exception | None = None
    max_attempts = 6
    for attempt in range(1, max_attempts + 1):
        base = ordered_endpoints[(attempt - 1) % len(ordered_endpoints)]
        url = build_year_url(base, expected_ts[0], expected_ts[-1])
        try:
            payload = _http_get_bytes(url, timeout=180)
            validate_year_payload(payload, year, expected_ts)
            packed = gzip.compress(payload, compresslevel=6, mtime=0)
            tmp = dest.with_name(dest.name + ".tmp")
            tmp.write_bytes(packed)
            tmp.replace(dest)
            return "ok", dest, len(packed), len(payload), time.time() - t0, base
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt < max_attempts:
                wait = min(3 * (2 ** (attempt - 1)), 30)
                print(f"  [重试 {attempt}/{max_attempts}] ERSST {year} ({base}): {exc}（{wait}s 后切端点/重试）", flush=True)
                time.sleep(wait)
    raise RuntimeError(f"ERSST {year} 下载失败: {last_err}")


def write_manifest(
    raw_dir: Path,
    files_info: list[dict],
    start_year: int,
    end_year: int,
    primary_endpoint: str,
    total_elapsed: float,
) -> None:
    manifest = {
        "dataset": f"NOAA NCEI ERSST v5 月平均海表温度（{DATASET_ID}）",
        "primary_endpoint": primary_endpoint,
        "fallback_endpoints": list(ENDPOINTS),
        "native_grid": {
            "resolution_deg": 2.0,
            "n_latitude": ERSST_N_LAT,
            "n_longitude": ERSST_N_LON,
            "lat_range": [ERSST_LAT_MIN, ERSST_LAT_MAX],
            "lon_range": [ERSST_LON_MIN, ERSST_LON_MAX],
            "months_per_year": ERSST_MONTHS_PER_YEAR,
            "expected_lines_per_year": ERSST_EXPECTED_LINES,
            "longitude_convention": "0-360",
        },
        "period": f"{start_year}-01 ~ {end_year}-12（共 {len(files_info)} 年 / {len(files_info) * 12} 个月）",
        "storage": "按年 gzip 无损压缩 CSV（.csv.gz），未做任何降采样",
        "total_bytes_gz": sum(d["bytes_gz"] for d in files_info),
        "total_bytes_csv": sum(d["bytes_csv"] for d in files_info),
        "total_elapsed_sec": round(total_elapsed, 2),
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": sorted(files_info, key=lambda d: d["year"]),
    }
    (raw_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="下载 NOAA NCEI ERSST v5 (1982–2024) 年度原始数据")
    parser.add_argument("--raw-dir", type=Path, default=RAW_ERSST_DIR, help="输出缓存目录（默认 data/raw_ersst）")
    parser.add_argument("--start-year", type=int, default=ERSST_YEAR_START)
    parser.add_argument("--end-year", type=int, default=ERSST_YEAR_END)
    parser.add_argument("--jobs", type=int, default=4, help="并发下载数（默认 4）")
    parser.add_argument("--force", action="store_true", help="忽略已有缓存，强制重新下载")
    args = parser.parse_args()

    t0 = time.time()
    print(f"[ersst] 探测 {DATASET_ID} 时间轴 ({args.start_year}–{args.end_year})...", flush=True)
    ts_by_year, primary_ep = fetch_time_axis(args.start_year, args.end_year)
    years = list(range(args.start_year, args.end_year + 1))
    print(f"[ersst] 时间轴确认完整：{len(years)} 年 / {len(years) * 12} 个月（主端点 {primary_ep}）", flush=True)

    done = 0
    new_csv_bytes = 0
    failed: list[tuple[int, str]] = []
    files_info: list[dict] = []

    def handle_res(y: int, status: str, path: Path, packed: int, raw: int, sec: float, src: str) -> None:
        nonlocal done, new_csv_bytes
        done += 1
        if status == "ok":
            new_csv_bytes += raw
            print(f"  [{done:02d}/{len(years)}] 下载 {path.name}  {packed/1e6:.2f} MB (csv {raw/1e6:.2f} MB, {sec:.1f}s)", flush=True)
        else:
            print(f"  [{done:02d}/{len(years)}] 跳过 {path.name}（已校验缓存，{packed/1e6:.2f} MB）", flush=True)
        files_info.append(
            {
                "year": y,
                "file": path.name,
                "bytes_gz": packed,
                "bytes_csv": raw,
                "elapsed_sec": round(sec, 2),
                "status": status,
                "source": src,
                "timestamps": [ts_by_year[y][0], ts_by_year[y][-1]],
            }
        )

    if args.jobs <= 1:
        for y in years:
            try:
                res = fetch_one_year(y, ts_by_year[y], args.raw_dir, primary_ep, args.force)
                handle_res(y, *res)
            except Exception as exc:  # noqa: BLE001
                failed.append((y, str(exc)))
                print(f"  [失败] {y}: {exc}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futs = {
                pool.submit(fetch_one_year, y, ts_by_year[y], args.raw_dir, primary_ep, args.force): y
                for y in years
            }
            for fut in as_completed(futs):
                y = futs[fut]
                try:
                    res = fut.result()
                    handle_res(y, *res)
                except Exception as exc:  # noqa: BLE001
                    failed.append((y, str(exc)))
                    print(f"  [失败] {y}: {exc}", flush=True)

    elapsed = time.time() - t0
    if not failed and len(files_info) == len(years):
        write_manifest(args.raw_dir, files_info, args.start_year, args.end_year, primary_ep, elapsed)
    else:
        print(f"注意：仅 {len(files_info)}/{len(years)} 个年度文件就绪，未写入 manifest。", flush=True)

    total_gz = sum(d["bytes_gz"] for d in files_info)
    total_csv = sum(d["bytes_csv"] for d in files_info)
    print(
        f"\n完成 [ersst]：{done}/{len(years)} 年，失败 {len(failed)} 年，用时 {elapsed:.1f}s；"
        f"本次新下载原始 CSV {new_csv_bytes/1e6:.1f} MB；"
        f"{args.raw_dir.name} 共 {len(files_info)} 个文件 / {total_gz/1e6:.1f} MB（gzip，解压 {total_csv/1e6:.1f} MB）。",
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
