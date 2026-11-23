#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成自包含的全球 SST 交互地图（离线 HTML，零外部依赖、零网络请求）。

把 ``climatology_grid.npz`` 的 ``sst_annual``（720×1440，float32）转成
**float16 → base64** 内嵌，浏览器在 ``<canvas>`` 上按当前视口逐像素重绘，
因此任意缩放级别都保持 0.25° 原生精度。陆地/内陆湖（``basin_mask == 0``）显示为灰色；
25 °C 等值线由 contourpy 预计算后内嵌。

用法::

    python scripts/make_map_viewer.py
    python scripts/make_map_viewer.py --out figures/global_sst_map.html
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
from contourpy import contour_generator  # noqa: E402
from matplotlib import colormaps  # noqa: E402

DEFAULT_GRID = ROOT / "data" / "climatology_grid.npz"
DEFAULT_MASK = ROOT / "data" / "basin_mask.npz"
DEFAULT_OUT = ROOT / "figures" / "global_sst_map.html"
VMIN, VMAX = -2.0, 32.0
ISOTHERM = 25.0
MAX_BYTES = 12 * 1024 * 1024


def load_annual(grid_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(grid_path) as npz:
        lat = (npz["lat"] if "lat" in npz.files else npz["latitude"]).astype(np.float64)
        lon = (npz["lon"] if "lon" in npz.files else npz["longitude"]).astype(np.float64)
        annual = npz["sst_annual"].astype(np.float32)
    if annual.shape != (len(lat), len(lon)):
        raise SystemExit(f"sst_annual 形状 {annual.shape} 与 lat/lon 不符")
    mask = None
    if mask_path.exists():
        with np.load(mask_path) as npz:
            if "basin_mask" in npz.files:
                mask = npz["basin_mask"].astype(np.uint8) > 0
            elif "basin_id" in npz.files:
                mask = npz["basin_id"].astype(np.int16) >= 0
    if mask is None:
        mask = np.isfinite(annual)
    if mask.shape != annual.shape:
        raise SystemExit(f"basin_mask 形状 {mask.shape} 与 sst_annual 不符")
    # 陆地与内陆湖统一为 NaN，浏览器里直接画灰色
    annual = np.where(mask, annual, np.nan).astype(np.float32)
    return lat, lon, annual, mask


def build_contours(lon: np.ndarray, lat: np.ndarray, annual: np.ndarray) -> list[list[list[float]]]:
    cg = contour_generator(x=lon, y=lat, z=np.ma.masked_invalid(annual), name="serial")
    lines = cg.lines(ISOTHERM)
    segments: list[list[list[float]]] = []
    for line in lines:
        arr = np.asarray(line, dtype=float)
        if len(arr) >= 2:
            segments.append(np.round(arr, 3).tolist())
    return segments


def make_html(
    lat: np.ndarray,
    lon: np.ndarray,
    annual: np.ndarray,
    contours: list[list[list[float]]],
) -> str:
    h, w = annual.shape
    sst16 = annual.astype(np.float16)
    sst_b64 = base64.b64encode(sst16.tobytes(order="C")).decode("ascii")
    cmap = colormaps["RdYlBu_r"]
    lut = (cmap(np.linspace(0.0, 1.0, 256))[:, :3] * 255.0).round().astype(np.uint8)
    lut_b64 = base64.b64encode(lut.tobytes(order="C")).decode("ascii")
    meta = {
        "w": int(w),
        "h": int(h),
        "lat0": float(lat[0]),
        "lon0": float(lon[0]),
        "dlat": float(lat[1] - lat[0]),
        "dlon": float(lon[1] - lon[0]),
        "vmin": VMIN,
        "vmax": VMAX,
        "isotherm": ISOTHERM,
    }
    template = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>全球 SST 交互地图（0.25°）</title>
<style>
  html, body { height: 100%; margin: 0; background: #0b1220; overflow: hidden; }
  #map { display: block; width: 100vw; height: 100vh; cursor: grab; background: #d9d9d9; }
  #map.dragging { cursor: grabbing; }
  #hud {
    position: fixed; left: 14px; top: 14px; z-index: 10; max-width: 390px;
    padding: 12px 14px; border-radius: 10px; color: #eef4ff;
    background: rgba(9, 18, 34, 0.86); border: 1px solid rgba(255,255,255,0.18);
    font: 13px/1.55 "Microsoft YaHei", "PingFang SC", sans-serif;
    box-shadow: 0 8px 30px rgba(0,0,0,0.35); backdrop-filter: blur(4px);
  }
  #hud h1 { margin: 0 0 6px; font-size: 16px; font-weight: 600; }
  #hud .hint { color: #a9b8d6; font-size: 12px; }
  #readout { margin: 8px 0 6px; padding: 6px 8px; border-radius: 6px;
             background: rgba(255,255,255,0.08); font-variant-numeric: tabular-nums; }
  #legend { display: flex; align-items: center; gap: 6px; margin-top: 7px; font-size: 11px; color: #c8d6f2; }
  #bar { height: 10px; flex: 1; border-radius: 3px; border: 1px solid rgba(255,255,255,0.25); }
  button {
    margin-top: 8px; padding: 5px 12px; border-radius: 6px; border: 1px solid #6ea8ff;
    background: #1d4ed8; color: #fff; cursor: pointer; font: inherit;
  }
  button:hover { background: #2563eb; }
  #badge { position: fixed; right: 14px; bottom: 12px; z-index: 10; color: #d8e4ff;
           font: 12px/1.4 "Microsoft YaHei", sans-serif; text-align: right;
           text-shadow: 0 1px 4px rgba(0,0,0,0.8); }
</style>
</head>
<body>
<canvas id="map"></canvas>
<div id="hud">
  <h1>全球年均 SST · 0.25° 交互地图</h1>
  <div class="hint">滚轮缩放（以鼠标为中心） · 拖动平移 · 双击放大 · 按钮重置</div>
  <div id="readout">移动鼠标查看经纬度与 SST</div>
  <div id="legend"><span>−2 °C</span><div id="bar"></div><span>32 °C</span></div>
  <button id="reset">重置视图</button>
</div>
<div id="badge">OISST v2.1 · 2016–2020 气候态<br>0.25°≈28 km，为数据分辨率上限</div>
<script>
const META = __META__;
const SST_B64 = "__SST__";
const LUT_B64 = "__LUT__";
const CONTOURS = __CONTOURS__;

function b64ToU8(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}
function halfToFloat(h) {
  const s = (h & 0x8000) ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const f = h & 0x3ff;
  if (e === 0) return s * Math.pow(2, -14) * (f / 1024);
  if (e === 31) return f ? NaN : s * Infinity;
  return s * Math.pow(2, e - 15) * (1 + f / 1024);
}

const SST_BYTES = b64ToU8(SST_B64);
const sstDV = new DataView(SST_BYTES.buffer);
const SST = new Float32Array(META.w * META.h);
for (let i = 0; i < SST.length; i++) {
  SST[i] = halfToFloat(sstDV.getUint16(i * 2, true));
}
const LUT = b64ToU8(LUT_B64);
const LAND = [217, 217, 217];

const canvas = document.getElementById('map');
const ctx = canvas.getContext('2d', { alpha: false });
const readout = document.getElementById('readout');
const bar = document.getElementById('bar');
let view = { lonMin: 0, lonMax: 360, latMin: -90, latMax: 90 };
let dragging = null;
let scheduled = false;

function setLegend() {
  const stops = [];
  for (let i = 0; i <= 12; i++) {
    const k = Math.round(i / 12 * 255) * 3;
    stops.push(`rgb(${LUT[k]},${LUT[k + 1]},${LUT[k + 2]}) ${(i / 12 * 100).toFixed(1)}%`);
  }
  bar.style.background = `linear-gradient(to right, ${stops.join(',')})`;
}

function sizeCanvas() {
  const dpr = 1;  // 逐像素重绘，保持性能；0.25° 原生数据本身已足够细
  canvas.width = Math.max(320, Math.floor(window.innerWidth * dpr));
  canvas.height = Math.max(240, Math.floor(window.innerHeight * dpr));
  scheduleDraw();
}

function lonToX(lon) {
  return (lon - view.lonMin) / (view.lonMax - view.lonMin) * canvas.width;
}
function latToY(lat) {
  return (view.latMax - lat) / (view.latMax - view.latMin) * canvas.height;
}

function clampView(v) {
  v.lonMin = Math.max(0, Math.min(360 - 0.05, v.lonMin));
  v.lonMax = Math.min(360, Math.max(v.lonMin + 0.05, v.lonMax));
  v.latMin = Math.max(-90, Math.min(90 - 0.05, v.latMin));
  v.latMax = Math.min(90, Math.max(v.latMin + 0.05, v.latMax));
  return v;
}

function draw() {
  const w = canvas.width, h = canvas.height;
  const spanLon = view.lonMax - view.lonMin;
  const spanLat = view.latMax - view.latMin;
  const cols = new Int32Array(w);
  for (let px = 0; px < w; px++) {
    const lon = view.lonMin + (px + 0.5) / w * spanLon;
    cols[px] = Math.floor((lon - META.lon0) / META.dlon + 1e-9);
  }
  const rows = new Int32Array(h);
  for (let py = 0; py < h; py++) {
    const lat = view.latMax - (py + 0.5) / h * spanLat;
    rows[py] = Math.floor((lat - META.lat0) / META.dlat + 1e-9);
  }
  const img = ctx.createImageData(w, h);
  const d = img.data;
  const scale = 255 / (META.vmax - META.vmin);
  for (let py = 0; py < h; py++) {
    const iy = rows[py];
    let o = py * w * 4;
    if (iy < 0 || iy >= META.h) {
      for (let px = 0; px < w; px++) {
        d[o] = LAND[0]; d[o + 1] = LAND[1]; d[o + 2] = LAND[2]; d[o + 3] = 255; o += 4;
      }
      continue;
    }
    const rowBase = iy * META.w;
    for (let px = 0; px < w; px++) {
      const ix = cols[px];
      if (ix < 0 || ix >= META.w) {
        d[o] = LAND[0]; d[o + 1] = LAND[1]; d[o + 2] = LAND[2];
      } else {
        const v = SST[rowBase + ix];
        if (Number.isNaN(v)) {
          d[o] = LAND[0]; d[o + 1] = LAND[1]; d[o + 2] = LAND[2];
        } else {
          let t = (v - META.vmin) * scale;
          t = Math.max(0, Math.min(255, Math.round(t)));
          const k = t * 3;
          d[o] = LUT[k]; d[o + 1] = LUT[k + 1]; d[o + 2] = LUT[k + 2];
        }
      }
      d[o + 3] = 255;
      o += 4;
    }
  }
  ctx.putImageData(img, 0, 0);

  // 30° 经纬网
  ctx.save();
  ctx.strokeStyle = 'rgba(20,35,55,0.16)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let lon = 0; lon <= 360; lon += 30) {
    if (lon < view.lonMin || lon > view.lonMax) continue;
    const x = lonToX(lon);
    ctx.moveTo(x, latToY(view.latMax));
    ctx.lineTo(x, latToY(view.latMin));
  }
  for (let lat = -90; lat <= 90; lat += 30) {
    if (lat < view.latMin || lat > view.latMax) continue;
    const y = latToY(lat);
    ctx.moveTo(lonToX(view.lonMin), y);
    ctx.lineTo(lonToX(view.lonMax), y);
  }
  ctx.stroke();
  ctx.restore();

  // 25 °C 等值线
  ctx.save();
  ctx.strokeStyle = 'rgba(0,0,0,0.88)';
  ctx.lineWidth = 1.15;
  ctx.beginPath();
  for (const seg of CONTOURS) {
    let started = false;
    for (const p of seg) {
      const lon = p[0], lat = p[1];
      if (lon < view.lonMin - 1 || lon > view.lonMax + 1 || lat < view.latMin - 1 || lat > view.latMax + 1) {
        started = false;
        continue;
      }
      const x = lonToX(lon), y = latToY(lat);
      if (!started) { ctx.moveTo(x, y); started = true; } else { ctx.lineTo(x, y); }
    }
  }
  ctx.stroke();
  ctx.restore();
}

function scheduleDraw() {
  if (scheduled) return;
  scheduled = true;
  requestAnimationFrame(() => { scheduled = false; draw(); });
}

function zoomAt(px, py, factor) {
  const w = canvas.width, h = canvas.height;
  const lon = view.lonMin + px / w * (view.lonMax - view.lonMin);
  const lat = view.latMax - py / h * (view.latMax - view.latMin);
  let lonSpan = (view.lonMax - view.lonMin) / factor;
  let latSpan = (view.latMax - view.latMin) / factor;
  lonSpan = Math.max(0.5, Math.min(360, lonSpan));
  latSpan = Math.max(0.5, Math.min(180, latSpan));
  view = clampView({
    lonMin: lon - px / w * lonSpan, lonMax: lon - px / w * lonSpan + lonSpan,
    latMin: lat - (1 - py / h) * latSpan, latMax: lat - (1 - py / h) * latSpan + latSpan
  });
  scheduleDraw();
}

function panByPixels(dx, dy) {
  const dLon = -dx / canvas.width * (view.lonMax - view.lonMin);
  const dLat = dy / canvas.height * (view.latMax - view.latMin);
  view = clampView({
    lonMin: view.lonMin + dLon, lonMax: view.lonMax + dLon,
    latMin: view.latMin + dLat, latMax: view.latMax + dLat
  });
  scheduleDraw();
}

function mouseLonLat(ev) {
  const r = canvas.getBoundingClientRect();
  const px = Math.max(0, Math.min(canvas.width - 1, (ev.clientX - r.left) / r.width * canvas.width));
  const py = Math.max(0, Math.min(canvas.height - 1, (ev.clientY - r.top) / r.height * canvas.height));
  return {
    px, py,
    lon: view.lonMin + px / canvas.width * (view.lonMax - view.lonMin),
    lat: view.latMax - py / canvas.height * (view.latMax - view.latMin)
  };
}

function fmtLon(lon) {
  const x = ((lon % 360) + 360) % 360;
  return x <= 180 ? `${x.toFixed(2)}°E` : `${(360 - x).toFixed(2)}°W`;
}
function fmtLat(lat) {
  return `${Math.abs(lat).toFixed(2)}°${lat >= 0 ? 'N' : 'S'}`;
}
function updateReadout(m) {
  const lon = Math.max(META.lon0, Math.min(META.lon0 + (META.w - 1) * META.dlon, m.lon));
  const lat = Math.max(META.lat0, Math.min(META.lat0 + (META.h - 1) * META.dlat, m.lat));
  const ix = Math.floor((lon - META.lon0) / META.dlon + 1e-9);
  const iy = Math.floor((lat - META.lat0) / META.dlat + 1e-9);
  let text = `${fmtLon(lon)} · ${fmtLat(lat)}`;
  if (ix >= 0 && ix < META.w && iy >= 0 && iy < META.h) {
    const v = SST[iy * META.w + ix];
    text += Number.isNaN(v) ? ' · 陆地/无数据' : ` · SST ${v.toFixed(2)} °C`;
  }
  readout.textContent = text;
}

canvas.addEventListener('mousedown', (ev) => {
  dragging = { x: ev.clientX, y: ev.clientY };
  canvas.classList.add('dragging');
});
window.addEventListener('mouseup', () => {
  dragging = null;
  canvas.classList.remove('dragging');
});
window.addEventListener('mousemove', (ev) => {
  if (dragging) {
    panByPixels(ev.clientX - dragging.x, ev.clientY - dragging.y);
    dragging = { x: ev.clientX, y: ev.clientY };
  } else {
    updateReadout(mouseLonLat(ev));
  }
});
canvas.addEventListener('wheel', (ev) => {
  ev.preventDefault();
  const m = mouseLonLat(ev);
  zoomAt(m.px, m.py, Math.exp(-ev.deltaY * 0.0015));
}, { passive: false });
canvas.addEventListener('dblclick', (ev) => {
  const m = mouseLonLat(ev);
  zoomAt(m.px, m.py, 1.8);
});
document.getElementById('reset').addEventListener('click', () => {
  view = { lonMin: 0, lonMax: 360, latMin: -90, latMax: 90 };
  scheduleDraw();
});
window.addEventListener('resize', sizeCanvas);

setLegend();
sizeCanvas();
</script>
</body>
</html>
"""
    return (
        template.replace("__META__", json.dumps(meta, ensure_ascii=False))
        .replace("__SST__", sst_b64)
        .replace("__LUT__", lut_b64)
        .replace("__CONTOURS__", json.dumps(contours, separators=(",", ":")))
    )


def validate_embedded(sst_b64: str, lat: np.ndarray, lon: np.ndarray, annual: np.ndarray, mask: np.ndarray) -> None:
    decoded = np.frombuffer(base64.b64decode(sst_b64), dtype="<f2").astype(np.float32).reshape(annual.shape)
    rng = np.random.default_rng(2024)
    candidates = np.flatnonzero(mask & np.isfinite(annual))
    picks = rng.choice(candidates, size=5, replace=False)
    print("  内嵌 float16 随机抽 5 个海洋格点与原始 npz 对比：")
    max_diff = 0.0
    for k in picks:
        iy, ix = np.unravel_index(int(k), annual.shape)
        a = float(annual[iy, ix])
        b = float(decoded[iy, ix])
        diff = abs(a - b)
        max_diff = max(max_diff, diff)
        print(
            f"    {fmt_lon(lon[ix])} {fmt_lat(lat[iy])}: npz={a:.3f} °C, embedded={b:.3f} °C, diff={diff:.4f}"
        )
    if max_diff > 0.02:
        raise SystemExit(f"内嵌 float16 与 npz 最大差异 {max_diff:.4f} °C > 0.02 °C")
    print(f"  最大差异 {max_diff:.4f} °C（float16 量化容差 0.02 °C）")


def fmt_lon(lon: float) -> str:
    x = ((lon % 360) + 360) % 360
    return f"{x:.2f}°E" if x <= 180 else f"{360 - x:.2f}°W"


def fmt_lat(lat: float) -> str:
    return f"{abs(lat):.2f}°{'N' if lat >= 0 else 'S'}"


def main() -> int:
    parser = argparse.ArgumentParser(description="生成自包含全球 SST 交互地图 HTML")
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID)
    parser.add_argument("--mask", type=Path, default=DEFAULT_MASK)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    lat, lon, annual, mask = load_annual(args.grid, args.mask)
    print(f"读取 {args.grid}：{annual.shape[1]}×{annual.shape[0]} 网格，海洋格点 {int(mask.sum())}")
    contours = build_contours(lon, lat, annual)
    print(f"25 °C 等值线：{len(contours)} 段，{sum(len(s) for s in contours)} 个点")
    html = make_html(lat, lon, annual, contours)

    # 重新截取 base64 做校验（与写入 HTML 的数据一致）
    sst_b64 = base64.b64encode(annual.astype(np.float16).tobytes(order="C")).decode("ascii")
    validate_embedded(sst_b64, lat, lon, annual, mask)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    size = args.out.stat().st_size
    print(f"已生成 {args.out}：{size / 1e6:.2f} MB")
    print(f"  float16 base64 长度：{len(sst_b64):,} 字符（原始 {annual.size * 2:,} 字节）")
    if "__SST__" in html or "__META__" in html or "__CONTOURS__" in html:
        raise SystemExit("HTML 模板占位符未替换")
    if size >= MAX_BYTES:
        raise SystemExit(f"HTML 体积 {size / 1e6:.1f} MB ≥ 12 MB")
    print(f"  校验通过：文件 < 12 MB，交互为原生 canvas，无外部请求。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
