#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成自包含的全球 SST 交互地图（离线 HTML，零外部依赖、零网络请求）。

内嵌 ``sst_annual`` / ``sst_jan`` / ``sst_jul`` 的 **int16 量化（round(SST×100)，
-32768 = NaN，解码 /100）**，按行差分后 **gzip + base64** 压缩（页面用
``DecompressionStream('gzip')`` 异步解压；不支持时给出提示并回退静态图）。浏览器在
``<canvas>`` 上按当前视口逐像素重绘，支持年均 / 1 月 / 7 月 / 7 月−1 月 / 同纬度异常五图层，
Shift 框选统计、URL hash 深链，叠加 25 °C 等值线、灰色陆地与中文地名/洋流标注。

用法::

    python scripts/make_map_viewer.py
    python scripts/make_map_viewer.py --out figures/global_sst_map.html
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
from contourpy import contour_generator  # noqa: E402
from matplotlib import colormaps  # noqa: E402

from map_labels import CURRENT_LABELS, OCEAN_LABELS, SEA_LABELS_T2, SEA_LABELS_T3  # noqa: E402

DEFAULT_GRID = ROOT / "data" / "climatology_grid.npz"
DEFAULT_MASK = ROOT / "data" / "basin_mask.npz"
DEFAULT_ZONAL = ROOT / "data" / "zonal_mean_global.csv"
DEFAULT_OUT = ROOT / "figures" / "global_sst_map.html"
VMIN, VMAX = -2.0, 32.0
DIFF_LIMIT = 12.0
ANOM_LIMIT = 8.0
ISOTHERM = 25.0
MAX_BYTES = 12 * 1024 * 1024
TARGET_BYTES = 3 * 1024 * 1024
BASIN_NAMES = ["陆地", "太平洋", "大西洋", "印度洋", "北冰洋", "南大洋"]


def load_layers(grid_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    with np.load(grid_path) as npz:
        lat = (npz["lat"] if "lat" in npz.files else npz["latitude"]).astype(np.float64)
        lon = (npz["lon"] if "lon" in npz.files else npz["longitude"]).astype(np.float64)
        fields = {
            "annual": npz["sst_annual"].astype(np.float32),
            "jan": npz["sst_jan"].astype(np.float32) if "sst_jan" in npz.files else None,
            "jul": npz["sst_jul"].astype(np.float32) if "sst_jul" in npz.files else None,
        }
    if fields["jan"] is None or fields["jul"] is None:
        raise SystemExit("climatology_grid.npz 缺少 sst_jan/sst_jul，无法提供四图层")
    shape = fields["annual"].shape
    if shape != (len(lat), len(lon)):
        raise SystemExit(f"sst 网格形状 {shape} 与 lat/lon 不符")

    mask = None
    basin = np.zeros(shape, dtype=np.uint8)
    if mask_path.exists():
        with np.load(mask_path) as npz:
            if "basin_mask" in npz.files:
                bm = npz["basin_mask"].astype(np.uint8)
                mask = bm > 0
                if bm.shape == shape:
                    basin = bm
            elif "basin_id" in npz.files:
                bid = npz["basin_id"].astype(np.int16)
                mask = bid >= 0
                if bid.shape == shape:
                    basin = np.where(bid >= 0, bid + 1, 0).astype(np.uint8)
    if mask is None:
        mask = np.isfinite(fields["annual"])
        basin = np.where(mask, 1, 0).astype(np.uint8)
    if mask.shape != shape:
        raise SystemExit(f"basin_mask 形状 {mask.shape} 与 sst 不符")

    quantized: dict[str, np.ndarray] = {}
    originals: dict[str, np.ndarray] = {}
    for key, arr in fields.items():
        masked_arr = np.where(mask, arr, np.nan).astype(np.float32)
        originals[key] = masked_arr
        q = np.full(shape, -32768, dtype=np.int16)
        valid = mask & np.isfinite(arr)
        q[valid] = np.clip(np.rint(arr[valid] * 100.0), -32767, 32767).astype(np.int16)
        quantized[key] = q
    return lat, lon, quantized, mask, basin, originals


def _delta_u16(arr: np.ndarray) -> np.ndarray:
    """按行做模 65536 差分，使平滑 SST 的 gzip 体积接近减半；首列从 0 起算。"""
    q = arr.astype(np.int32)
    prev = np.empty_like(q)
    prev[:, 0] = 0
    prev[:, 1:] = q[:, :-1]
    return ((q - prev) & 0xFFFF).astype("<u2")


def b64_int16(arr: np.ndarray) -> str:
    raw = gzip.compress(_delta_u16(arr).tobytes(order="C"), compresslevel=9, mtime=0)
    return base64.b64encode(raw).decode("ascii")


def _undelta_u16(raw: bytes, shape: tuple[int, int]) -> np.ndarray:
    d = np.frombuffer(raw, dtype="<u2").astype(np.int64).reshape(shape)
    q = np.empty(shape, dtype=np.int16)
    prev = np.zeros(shape[0], dtype=np.int64)
    for j in range(shape[1]):
        v = (prev + d[:, j]) & 0xFFFF
        q[:, j] = np.where(v >= 0x8000, v - 0x10000, v).astype(np.int16)
        prev = v
    return q


def b64_u8_gzip(arr: np.ndarray) -> str:
    raw = gzip.compress(arr.astype(np.uint8, copy=False).tobytes(order="C"), compresslevel=9, mtime=0)
    return base64.b64encode(raw).decode("ascii")


def build_contours(lon: np.ndarray, lat: np.ndarray, q: np.ndarray) -> list[list[list[float]]]:
    z = q.astype(np.float32) / 100.0
    z[q == -32768] = np.nan
    cg = contour_generator(x=lon, y=lat, z=np.ma.masked_invalid(z), name="serial")
    segments: list[list[list[float]]] = []
    for line in cg.lines(ISOTHERM):
        arr = np.asarray(line, dtype=float)
        if len(arr) >= 2:
            segments.append(np.round(arr, 3).tolist())
    return segments


def all_labels() -> list[dict]:
    out: list[dict] = []
    for name, lon, lat, tier, kind in OCEAN_LABELS:
        out.append({"name": name, "lon": lon, "lat": lat, "tier": tier, "kind": kind})
    for name, lon, lat, kind in SEA_LABELS_T2:
        out.append({"name": name, "lon": lon, "lat": lat, "tier": 2, "kind": kind})
    for name, lon, lat in SEA_LABELS_T3:
        out.append({"name": name, "lon": lon, "lat": lat, "tier": 3, "kind": "sea"})
    for name, lon, lat, kind in CURRENT_LABELS:
        out.append({"name": name, "lon": lon, "lat": lat, "tier": 2, "kind": kind})
    return out


def make_html(
    lat: np.ndarray,
    lon: np.ndarray,
    quantized: dict[str, np.ndarray],
    basin: np.ndarray,
    contours: dict[str, list[list[list[float]]]],
    ref: np.ndarray,
) -> str:
    h, w = quantized["annual"].shape
    cmap = colormaps["RdYlBu_r"]
    cmap_diff = colormaps["RdBu_r"]
    lut = (cmap(np.linspace(0.0, 1.0, 256))[:, :3] * 255.0).round().astype(np.uint8)
    lut_diff = (cmap_diff(np.linspace(0.0, 1.0, 256))[:, :3] * 255.0).round().astype(np.uint8)
    meta = {
        "w": int(w),
        "h": int(h),
        "lat0": float(lat[0]),
        "lon0": float(lon[0]),
        "dlat": float(lat[1] - lat[0]),
        "dlon": float(lon[1] - lon[0]),
        "vmin": VMIN,
        "vmax": VMAX,
        "diffLimit": DIFF_LIMIT,
        "anomLimit": ANOM_LIMIT,
        "isotherm": ISOTHERM,
        "basinNames": BASIN_NAMES,
    }
    template = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>全球 SST 交互地图（0.25°）</title>
<style>
  html, body { height: 100%; margin: 0; background: #0b1220; overflow: hidden; }
  #map { display: block; width: 100vw; height: 100vh; cursor: crosshair; background: #d9d9d9; }
  #map.dragging { cursor: grabbing; }
  #hud {
    position: fixed; left: 14px; top: 14px; z-index: 10; max-width: 430px;
    padding: 12px 14px; border-radius: 10px; color: #eef4ff;
    background: rgba(9, 18, 34, 0.88); border: 1px solid rgba(255,255,255,0.18);
    font: 13px/1.55 "Microsoft YaHei", "PingFang SC", sans-serif;
    box-shadow: 0 8px 30px rgba(0,0,0,0.35); backdrop-filter: blur(4px);
  }
  #hud h1 { margin: 0 0 6px; font-size: 16px; font-weight: 600; }
  #hud .hint { color: #a9b8d6; font-size: 12px; }
  .row { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 8px; }
  button {
    padding: 4px 9px; border-radius: 6px; border: 1px solid #4f6a9a;
    background: rgba(30, 52, 88, 0.9); color: #dbe7ff; cursor: pointer; font: inherit;
    font-size: 12px;
  }
  button:hover { background: #28508a; }
  button.active { background: #1d4ed8; border-color: #8db7ff; color: #fff; }
  #layerInfo { margin-top: 8px; color: #cfe0ff; font-size: 12px; }
  #legend { display: flex; align-items: center; gap: 6px; margin-top: 7px; font-size: 11px; color: #c8d6f2; }
  #bar { height: 10px; flex: 1; border-radius: 3px; border: 1px solid rgba(255,255,255,0.25); }
  #inspect {
    position: fixed; right: 14px; bottom: 52px; z-index: 10; width: 268px;
    padding: 10px 12px; border-radius: 9px; color: #eef4ff;
    background: rgba(9, 18, 34, 0.88); border: 1px solid rgba(255,255,255,0.18);
    font: 12px/1.55 "Microsoft YaHei", "PingFang SC", sans-serif;
    box-shadow: 0 8px 30px rgba(0,0,0,0.35); font-variant-numeric: tabular-nums;
  }
  #inspect b { color: #9cc4ff; font-weight: 600; }
  #inspect .cross { font-size: 18px; color: #ffd166; margin-right: 4px; vertical-align: -1px; }
  #badge { position: fixed; right: 14px; bottom: 12px; z-index: 10; color: #d8e4ff;
           font: 11px/1.4 "Microsoft YaHei", sans-serif; text-align: right;
           text-shadow: 0 1px 4px rgba(0,0,0,0.8); }
  #loading {
    position: fixed; inset: 0; z-index: 50; display: flex; align-items: center;
    justify-content: center; color: #eef4ff; background: #0b1220;
    font: 16px/1.6 "Microsoft YaHei", sans-serif; text-align: center; padding: 20px;
  }
  #selection {
    position: fixed; left: 14px; bottom: 14px; z-index: 10; width: 300px;
    padding: 10px 12px; border-radius: 9px; color: #eef4ff;
    background: rgba(9, 18, 34, 0.9); border: 1px solid rgba(255,255,255,0.18);
    font: 12px/1.55 "Microsoft YaHei", sans-serif; display: none;
    font-variant-numeric: tabular-nums;
  }
  #selection b { color: #9cc4ff; }
  .sep { color: #5f7599; margin: 0 4px; }
</style>
</head>
<body>
<canvas id="map"></canvas>
<div id="hud">
  <h1>全球 SST 交互地图 · 0.25°</h1>
  <div class="hint">滚轮缩放（以鼠标为中心） · 拖动平移 · 双击放大 · Shift+拖动框选（Esc 取消）</div>
  <div class="row" id="layers">
    <button data-layer="annual" class="active">年均</button>
    <button data-layer="jan">1 月</button>
    <button data-layer="jul">7 月</button>
    <button data-layer="diff">7 月−1 月</button>
    <button data-layer="anomaly">同纬度异常</button>
  </div>
  <div class="row" id="jumps">
    <button data-jump="114,14,22,14">南海</button>
    <button data-jump="125,29,18,12">东海</button>
    <button data-jump="18,36,45,20">地中海</button>
    <button data-jump="270,25,35,20">墨西哥湾</button>
    <button id="reset">重置视图</button>
  </div>
  <div id="layerInfo">当前图层：年均 SST</div>
  <div id="legend"><span id="vmin">−2 °C</span><div id="bar"></div><span id="vmax">32 °C</span></div>
</div>
<div id="inspect">
  <div><span class="cross">┼</span><b>鼠标读数</b></div>
  <div id="ins-pos">经度 — · 纬度 —</div>
  <div id="ins-basin">海盆 —</div>
  <div id="ins-sst">年均 — · 1 月 — · 7 月 — · 温差 —</div>
</div>
<div id="badge">OISST v2.1 · 2016–2020 气候态<br>0.25°≈28 km，为数据分辨率上限</div>
<div id="selection">
  <div><b>框选统计</b> <span id="sel-close" style="float:right;cursor:pointer">×</span></div>
  <div id="sel-range">按住 Shift 拖动框选海域</div>
  <div id="sel-stats"></div>
</div>
<div id="loading">正在解压内嵌数据…<br><span style="color:#9cc4ff;font-size:12px">（gzip + DecompressionStream）</span></div>
<script>
const META = __META__;
const ANNUAL_B64 = "__ANNUAL__";
const JAN_B64 = "__JAN__";
const JUL_B64 = "__JUL__";
const BASIN_B64 = "__BASIN__";
const LUT_B64 = "__LUT__";
const LUT_DIFF_B64 = "__LUT_DIFF__";
const CONTOURS = __CONTOURS__;
const LABELS = __LABELS__;
const REF = __REF__;

function b64ToU8(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}
async function gunzipB64(b64) {
  if (typeof DecompressionStream === "undefined") {
    throw new Error("当前浏览器不支持 DecompressionStream（Safari 需 16.4+；Chrome/Edge/Firefox 新版本可用）");
  }
  const bytes = b64ToU8(b64);
  const ds = new DecompressionStream("gzip");
  const stream = new Blob([bytes]).stream().pipeThrough(ds);
  return new Uint8Array(await new Response(stream).arrayBuffer());
}
function bytesToI16(u8) {
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const out = new Int16Array(u8.length / 2);
  let prev = 0;
  for (let i = 0; i < out.length; i++) {
    if (i % META.w === 0) prev = 0;
    const v = (prev + dv.getUint16(i * 2, true)) & 0xffff;
    out[i] = v >= 0x8000 ? v - 0x10000 : v;
    prev = v;
  }
  return out;
}
async function loadI16(b64) {
  return bytesToI16(await gunzipB64(b64));
}

const DATA = { annual: null, jan: null, jul: null };
let BASIN = null;
let ready = false;
const LUT = b64ToU8(LUT_B64);
const LUT_DIFF = b64ToU8(LUT_DIFF_B64);
const LAND = [217, 217, 217];
const LAYERS = {
  annual: { label: "年均 SST", key: "annual", vmin: META.vmin, vmax: META.vmax, lut: LUT, contour: "annual" },
  jan: { label: "1 月 SST", key: "jan", vmin: META.vmin, vmax: META.vmax, lut: LUT, contour: "jan" },
  jul: { label: "7 月 SST", key: "jul", vmin: META.vmin, vmax: META.vmax, lut: LUT, contour: "jul" },
  diff: { label: "7 月−1 月温差", key: null, vmin: -META.diffLimit, vmax: META.diffLimit, lut: LUT_DIFF, contour: null },
  anomaly: { label: "同纬度异常（SST−同纬度平均）", key: null, vmin: -META.anomLimit, vmax: META.anomLimit, lut: LUT_DIFF, contour: null },
};
let layer = "annual";

const canvas = document.getElementById('map');
const ctx = canvas.getContext('2d', { alpha: false });
const bar = document.getElementById('bar');
const vminEl = document.getElementById('vmin');
const vmaxEl = document.getElementById('vmax');
const layerInfo = document.getElementById('layerInfo');
const loadingEl = document.getElementById('loading');
const selectionEl = document.getElementById('selection');
let view = { lonMin: 0, lonMax: 360, latMin: -90, latMax: 90 };
let dragging = null;
let cursor = null;
let boxStart = null;
let selection = null;
let scheduled = false;

function valueAt(idx) {
  if (layer === "diff") {
    const a = DATA.jan[idx], b = DATA.jul[idx];
    if (a === -32768 || b === -32768) return NaN;
    return (b - a) / 100;
  }
  if (layer === "anomaly") {
    const q = DATA.annual[idx];
    if (q === -32768) return NaN;
    const r = REF[Math.floor(idx / META.w)];
    if (!Number.isFinite(r)) return NaN;
    return q / 100 - r;
  }
  const q = DATA[LAYERS[layer].key][idx];
  return q === -32768 ? NaN : q / 100;
}
function valueOf(key, idx) {
  const q = DATA[key][idx];
  return q === -32768 ? NaN : q / 100;
}
function basinName(idx) {
  return META.basinNames[BASIN[idx]] || "—";
}

function setLegend() {
  const cfg = LAYERS[layer];
  const stops = [];
  for (let i = 0; i <= 12; i++) {
    const k = Math.round(i / 12 * 255) * 3;
    stops.push(`rgb(${cfg.lut[k]},${cfg.lut[k + 1]},${cfg.lut[k + 2]}) ${(i / 12 * 100).toFixed(1)}%`);
  }
  bar.style.background = `linear-gradient(to right, ${stops.join(',')})`;
  vminEl.textContent = `${cfg.vmin > 0 ? '+' : ''}${cfg.vmin.toFixed(0)} °C`;
  vmaxEl.textContent = `${cfg.vmax > 0 ? '+' : ''}${cfg.vmax.toFixed(0)} °C`;
  layerInfo.textContent = `当前图层：${cfg.label}`;
}

function sizeCanvas() {
  canvas.width = Math.max(320, Math.floor(window.innerWidth));
  canvas.height = Math.max(240, Math.floor(window.innerHeight));
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
  if (!ready) return;
  const cfg = LAYERS[layer];
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
  const scale = 255 / (cfg.vmax - cfg.vmin);
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
        const idx = rowBase + ix;
        const v = valueAt(idx);
        if (Number.isNaN(v) || DATA.annual[idx] === -32768) {
          d[o] = LAND[0]; d[o + 1] = LAND[1]; d[o + 2] = LAND[2];
        } else {
          let t = (v - cfg.vmin) * scale;
          t = Math.max(0, Math.min(255, Math.round(t)));
          const k = t * 3;
          d[o] = cfg.lut[k]; d[o + 1] = cfg.lut[k + 1]; d[o + 2] = cfg.lut[k + 2];
        }
      }
      d[o + 3] = 255;
      o += 4;
    }
  }
  ctx.putImageData(img, 0, 0);

  // 15°/30° 经纬网
  ctx.save();
  ctx.beginPath();
  for (let lon = 0; lon <= 360; lon += 15) {
    if (lon < view.lonMin || lon > view.lonMax) continue;
    ctx.strokeStyle = (lon % 30 === 0) ? "rgba(20,35,55,0.20)" : "rgba(20,35,55,0.10)";
    ctx.lineWidth = (lon % 30 === 0) ? 1.0 : 0.6;
    const x = lonToX(lon);
    ctx.beginPath(); ctx.moveTo(x, latToY(view.latMax)); ctx.lineTo(x, latToY(view.latMin)); ctx.stroke();
  }
  for (let lat = -90; lat <= 90; lat += 15) {
    if (lat < view.latMin || lat > view.latMax) continue;
    ctx.strokeStyle = (lat % 30 === 0) ? "rgba(20,35,55,0.20)" : "rgba(20,35,55,0.10)";
    ctx.lineWidth = (lat % 30 === 0) ? 1.0 : 0.6;
    const y = latToY(lat);
    ctx.beginPath(); ctx.moveTo(lonToX(view.lonMin), y); ctx.lineTo(lonToX(view.lonMax), y); ctx.stroke();
  }
  ctx.restore();

  // 当前图层的 25 °C 等值线（温差图层无 25 °C 线）
  if (cfg.contour && CONTOURS[cfg.contour]) {
    ctx.save();
    ctx.strokeStyle = "rgba(0,0,0,0.88)";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    for (const seg of CONTOURS[cfg.contour]) {
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

  drawLabels();

  // 十字定位线
  if (cursor) {
    ctx.save();
    ctx.strokeStyle = "rgba(255,255,255,0.85)";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(cursor.px, 0); ctx.lineTo(cursor.px, canvas.height);
    ctx.moveTo(0, cursor.py); ctx.lineTo(canvas.width, cursor.py);
    ctx.stroke();
    ctx.restore();
  }

  const rect = (boxStart && boxStart.active) ? boxStart : selection;
  if (rect && rect.lonMin !== undefined) {
    const x0 = lonToX(rect.lonMin), x1 = lonToX(rect.lonMax);
    const y0 = latToY(rect.latMax), y1 = latToY(rect.latMin);
    ctx.save();
    ctx.fillStyle = "rgba(255,209,102,0.12)";
    ctx.strokeStyle = "rgba(255,209,102,0.95)";
    ctx.lineWidth = 1.4;
    ctx.setLineDash([5, 3]);
    ctx.fillRect(Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0));
    ctx.strokeRect(Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0));
    ctx.restore();
  }
}

function drawLabels() {
  const spanLon = view.lonMax - view.lonMin;
  ctx.save();
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  for (const lab of LABELS) {
    if (lab.tier === 3 && spanLon >= 40) continue;
    if (lab.lon < view.lonMin || lab.lon > view.lonMax || lab.lat < view.latMin || lab.lat > view.latMax) continue;
    const x = lonToX(lab.lon), y = latToY(lab.lat);
    let size = 12, weight = "normal", style = "normal", color = "#17202a";
    if (lab.kind === "ocean") { size = 18; weight = "bold"; }
    else if (lab.kind === "warm") { color = "#8B1A1A"; style = "italic"; }
    else if (lab.kind === "cold") { color = "#123B6D"; style = "italic"; }
    else if (lab.tier === 3) { size = 11; }
    ctx.font = `${style} ${weight} ${size}px "Microsoft YaHei", "PingFang SC", sans-serif`;
    ctx.lineWidth = 3.2;
    ctx.strokeStyle = "rgba(255,255,255,0.92)";
    ctx.strokeText(lab.name, x, y);
    ctx.fillStyle = color;
    ctx.fillText(lab.name, x, y);
  }
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
  updateHash();
  scheduleDraw();
}

function panByPixels(dx, dy) {
  const dLon = -dx / canvas.width * (view.lonMax - view.lonMin);
  const dLat = dy / canvas.height * (view.latMax - view.latMin);
  view = clampView({
    lonMin: view.lonMin + dLon, lonMax: view.lonMax + dLon,
    latMin: view.latMin + dLat, latMax: view.latMax + dLat
  });
  updateHash();
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
function fmtVal(v) {
  return Number.isNaN(v) ? "—" : `${v.toFixed(2)} °C`;
}

function cellIndex(lon, lat) {
  const x = Math.max(META.lon0, Math.min(META.lon0 + (META.w - 1) * META.dlon, lon));
  const y = Math.max(META.lat0, Math.min(META.lat0 + (META.h - 1) * META.dlat, lat));
  const ix = Math.floor((x - META.lon0) / META.dlon + 1e-9);
  const iy = Math.floor((y - META.lat0) / META.dlat + 1e-9);
  if (ix < 0 || ix >= META.w || iy < 0 || iy >= META.h) return -1;
  return iy * META.w + ix;
}

function updateInspect(m) {
  if (!ready) return;
  const idx = cellIndex(m.lon, m.lat);
  document.getElementById('ins-pos').textContent = `${fmtLon(m.lon)} · ${fmtLat(m.lat)}`;
  if (idx < 0) {
    document.getElementById('ins-basin').textContent = "海盆 —";
    document.getElementById('ins-sst').textContent = "年均 — · 1 月 — · 7 月 — · 温差 — · 异常 —";
    return;
  }
  const a = valueOf("annual", idx), j = valueOf("jan", idx), u = valueOf("jul", idx);
  let diff = "—";
  if (!Number.isNaN(j) && !Number.isNaN(u)) diff = `${(u - j).toFixed(2)} °C`;
  const ref = REF[Math.floor(idx / META.w)];
  const anom = (!Number.isNaN(a) && Number.isFinite(ref)) ? `${(a - ref).toFixed(2)} °C` : "—";
  document.getElementById('ins-basin').textContent = `海盆 ${basinName(idx)}`;
  document.getElementById('ins-sst').textContent =
    `年均 ${fmtVal(a)} · 1 月 ${fmtVal(j)} · 7 月 ${fmtVal(u)} · 温差 ${diff} · 异常 ${anom}`;
}

function updateHash() {
  const lonSpan = view.lonMax - view.lonMin;
  const zoom = 360 / lonSpan;
  const lat = (view.latMin + view.latMax) / 2;
  const lon = (view.lonMin + view.lonMax) / 2;
  try {
    history.replaceState(null, "", `#zoom=${zoom.toFixed(3)}&lat=${lat.toFixed(4)}&lon=${lon.toFixed(4)}`);
  } catch (e) {
    try { location.hash = `zoom=${zoom.toFixed(3)}&lat=${lat.toFixed(4)}&lon=${lon.toFixed(4)}`; } catch (e2) { /* ignore */ }
  }
}

function restoreViewFromHash() {
  const raw = location.hash.replace(/^#/, "");
  if (!raw) return;
  const p = new URLSearchParams(raw);
  const zoom = parseFloat(p.get("zoom"));
  const lat = parseFloat(p.get("lat"));
  const lon = parseFloat(p.get("lon"));
  if (!Number.isFinite(zoom) || zoom <= 0 || !Number.isFinite(lat) || !Number.isFinite(lon)) return;
  const lonSpan = Math.max(0.5, Math.min(360, 360 / zoom));
  const latSpan = Math.max(0.5, Math.min(180, 180 / zoom));
  view = clampView({
    lonMin: lon - lonSpan / 2, lonMax: lon + lonSpan / 2,
    latMin: lat - latSpan / 2, latMax: lat + latSpan / 2,
  });
}

function computeSelection(sel) {
  const ix0 = Math.max(0, Math.floor((sel.lonMin - META.lon0) / META.dlon + 1e-9));
  const ix1 = Math.min(META.w - 1, Math.floor((sel.lonMax - META.lon0) / META.dlon + 1e-9));
  const iy0 = Math.max(0, Math.floor((sel.latMin - META.lat0) / META.dlat + 1e-9));
  const iy1 = Math.min(META.h - 1, Math.floor((sel.latMax - META.lat0) / META.dlat + 1e-9));
  let sumW = 0, sumWV = 0, minV = Infinity, maxV = -Infinity, valid = 0;
  const total = Math.max(0, (ix1 - ix0 + 1) * (iy1 - iy0 + 1));
  for (let iy = iy0; iy <= iy1; iy++) {
    const w = Math.cos((META.lat0 + iy * META.dlat) * Math.PI / 180);
    const row = iy * META.w;
    for (let ix = ix0; ix <= ix1; ix++) {
      const v = valueAt(row + ix);
      if (Number.isNaN(v)) continue;
      sumW += w; sumWV += w * v; valid++;
      if (v < minV) minV = v;
      if (v > maxV) maxV = v;
    }
  }
  const mean = sumW > 0 ? sumWV / sumW : NaN;
  const frac = total > 0 ? valid / total * 100 : 0;
  document.getElementById('sel-range').textContent =
    `经度 ${fmtLon(sel.lonMin)}–${fmtLon(sel.lonMax)} · 纬度 ${fmtLat(sel.latMin)}–${fmtLat(sel.latMax)}`;
  document.getElementById('sel-stats').innerHTML = Number.isFinite(mean)
    ? `图层：${LAYERS[layer].label}<br>cos(lat) 加权均温：<b>${mean.toFixed(2)} °C</b><br>` +
      `最低 ${minV.toFixed(2)} °C · 最高 ${maxV.toFixed(2)} °C<br>` +
      `有效格点 ${valid.toLocaleString()} / ${total.toLocaleString()}（${frac.toFixed(1)}%）`
    : "该范围内没有有效海洋格点";
  selectionEl.style.display = "block";
}

function setSelectionFrom(start, current) {
  return {
    lonMin: Math.min(start.lon, current.lon), lonMax: Math.max(start.lon, current.lon),
    latMin: Math.min(start.lat, current.lat), latMax: Math.max(start.lat, current.lat),
  };
}

canvas.addEventListener('mousedown', (ev) => {
  if (ev.shiftKey) {
    const m = mouseLonLat(ev);
    boxStart = { active: true, lon: m.lon, lat: m.lat, lonMin: m.lon, lonMax: m.lon, latMin: m.lat, latMax: m.lat };
    selection = null;
    selectionEl.style.display = "none";
    scheduleDraw();
    return;
  }
  dragging = { x: ev.clientX, y: ev.clientY };
  canvas.classList.add('dragging');
});
window.addEventListener('mouseup', (ev) => {
  if (boxStart && boxStart.active) {
    const sel = { lonMin: boxStart.lonMin, lonMax: boxStart.lonMax, latMin: boxStart.latMin, latMax: boxStart.latMax };
    boxStart.active = false;
    boxStart = null;
    if (sel.lonMax - sel.lonMin > 0.1 && sel.latMax - sel.latMin > 0.1) {
      selection = sel;
      computeSelection(sel);
    }
    scheduleDraw();
    return;
  }
  dragging = null;
  canvas.classList.remove('dragging');
});
window.addEventListener('mousemove', (ev) => {
  if (boxStart && boxStart.active) {
    const m = mouseLonLat(ev);
    const sel = setSelectionFrom(boxStart, m);
    boxStart.lonMin = sel.lonMin; boxStart.lonMax = sel.lonMax;
    boxStart.latMin = sel.latMin; boxStart.latMax = sel.latMax;
  } else if (dragging) {
    panByPixels(ev.clientX - dragging.x, ev.clientY - dragging.y);
    dragging = { x: ev.clientX, y: ev.clientY };
  }
  const m = mouseLonLat(ev);
  cursor = m;
  updateInspect(m);
  scheduleDraw();
});
window.addEventListener('keydown', (ev) => {
  if (ev.key === "Escape") {
    selection = null;
    boxStart = null;
    selectionEl.style.display = "none";
    scheduleDraw();
  }
});
canvas.addEventListener('mouseleave', () => { cursor = null; scheduleDraw(); });
canvas.addEventListener('wheel', (ev) => {
  ev.preventDefault();
  const m = mouseLonLat(ev);
  zoomAt(m.px, m.py, Math.exp(-ev.deltaY * 0.0015));
}, { passive: false });
canvas.addEventListener('dblclick', (ev) => {
  const m = mouseLonLat(ev);
  zoomAt(m.px, m.py, 1.8);
});
document.querySelectorAll('#layers button').forEach((btn) => {
  btn.addEventListener('click', () => {
    if (!ready) return;
    layer = btn.dataset.layer;
    document.querySelectorAll('#layers button').forEach((b) => b.classList.toggle('active', b === btn));
    setLegend();
    if (selection) computeSelection(selection);
    scheduleDraw();
  });
});
document.querySelectorAll('#jumps button').forEach((btn) => {
  btn.addEventListener('click', () => {
    if (!ready) return;
    const [lon, lat, lonSpan, latSpan] = btn.dataset.jump.split(',').map(Number);
    view = clampView({
      lonMin: lon - lonSpan / 2, lonMax: lon + lonSpan / 2,
      latMin: lat - latSpan / 2, latMax: lat + latSpan / 2,
    });
    updateHash();
    scheduleDraw();
  });
});
document.getElementById('reset').addEventListener('click', () => {
  view = { lonMin: 0, lonMax: 360, latMin: -90, latMax: 90 };
  selection = null;
  selectionEl.style.display = "none";
  updateHash();
  scheduleDraw();
});
document.getElementById('sel-close').addEventListener('click', () => {
  selection = null;
  selectionEl.style.display = "none";
  scheduleDraw();
});
window.addEventListener('resize', sizeCanvas);

async function init() {
  try {
    const [a, j, u, b] = await Promise.all([
      loadI16(ANNUAL_B64), loadI16(JAN_B64), loadI16(JUL_B64),
      gunzipB64(BASIN_B64),
    ]);
    DATA.annual = a; DATA.jan = j; DATA.jul = u;
    BASIN = b;
    ready = true;
    loadingEl.style.display = "none";
    restoreViewFromHash();
    setLegend();
    updateHash();
    sizeCanvas();
  } catch (err) {
    loadingEl.innerHTML = `<div><b>交互地图加载失败</b><br>${String(err.message || err)}<br><br>` +
      `请使用支持 DecompressionStream 的浏览器（Chrome/Edge/Firefox/Safari 16.4+），` +
      `或查看静态图 <a href="global_sst_map.png" target="_blank" style="color:#9cc4ff">global_sst_map.png</a>。</div>`;
    console.error(err);
  }
}
init();
</script>
</body>
</html>
"""
    return (
        template.replace("__META__", json.dumps(meta, ensure_ascii=False))
        .replace("__ANNUAL__", b64_int16(quantized["annual"]))
        .replace("__JAN__", b64_int16(quantized["jan"]))
        .replace("__JUL__", b64_int16(quantized["jul"]))
        .replace("__BASIN__", b64_u8_gzip(basin))
        .replace("__LUT__", base64.b64encode(lut.tobytes(order="C")).decode("ascii"))
        .replace("__LUT_DIFF__", base64.b64encode(lut_diff.tobytes(order="C")).decode("ascii"))
        .replace("__CONTOURS__", json.dumps(contours, separators=(",", ":")))
        .replace("__LABELS__", json.dumps(all_labels(), ensure_ascii=False, separators=(",", ":")))
        .replace("__REF__", json.dumps(np.round(ref.astype(float), 3).tolist(), separators=(",", ":")))
    )


def _decode_embedded(html: str, token: str, shape: tuple[int, int]) -> np.ndarray:
    m = re.search(rf'{token} = "([A-Za-z0-9+/=]+)"', html)
    if m is None:
        raise SystemExit(f"HTML 中找不到 {token}")
    raw = gzip.decompress(base64.b64decode(m.group(1)))
    q = _undelta_u16(raw, shape)
    return q.astype(np.float32) / 100.0


def validate(html: str, originals: dict[str, np.ndarray], basin: np.ndarray,
             lat: np.ndarray, lon: np.ndarray) -> float:
    shape = originals["annual"].shape
    embedded = {key: _decode_embedded(html, f"{key.upper()}_B64", shape) for key in ("annual", "jan", "jul")}
    rng = np.random.default_rng(2024)
    ocean = np.flatnonzero(
        np.isfinite(originals["annual"]) & np.isfinite(originals["jan"]) & np.isfinite(originals["jul"])
    )
    picks = rng.choice(ocean, size=5, replace=False)
    print("  内嵌 int16 随机抽 5 个海洋格点（embedded vs npz，容差 0.01 °C）：")
    worst = 0.0
    for k in picks:
        iy, ix = np.unravel_index(int(k), shape)
        vals = {key: float(embedded[key][iy, ix]) for key in ("annual", "jan", "jul")}
        orig = {key: float(originals[key][iy, ix]) for key in ("annual", "jan", "jul")}
        diffs = {key: abs(vals[key] - orig[key]) for key in vals}
        worst = max(worst, max(diffs.values()))
        print(
            f"    {fmt_lon(lon[ix])} {fmt_lat(lat[iy])} ({BASIN_NAMES[int(basin[iy, ix])]}): "
            f"annual {vals['annual']:.2f}/{orig['annual']:.2f}, "
            f"jan {vals['jan']:.2f}/{orig['jan']:.2f}, "
            f"jul {vals['jul']:.2f}/{orig['jul']:.2f}, "
            f"maxdiff={max(diffs.values()):.4f}"
        )
    if worst > 0.01:
        raise SystemExit(f"内嵌 int16 与 npz 原始值最大差异 {worst:.4f} °C > 0.01 °C")
    print(f"  最大偏差 {worst:.4f} °C（int16×100 量化上限 0.005 °C）")
    return worst


def fmt_lon(lon: float) -> str:
    x = ((lon % 360) + 360) % 360
    return f"{x:.2f}°E" if x <= 180 else f"{360 - x:.2f}°W"


def fmt_lat(lat: float) -> str:
    return f"{abs(lat):.2f}°{'N' if lat >= 0 else 'S'}"


def main() -> int:
    parser = argparse.ArgumentParser(description="生成自包含全球 SST 交互地图 HTML")
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID)
    parser.add_argument("--mask", type=Path, default=DEFAULT_MASK)
    parser.add_argument("--zonal", type=Path, default=DEFAULT_ZONAL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    lat, lon, quantized, mask, basin, originals = load_layers(args.grid, args.mask)
    ref_df = pd.read_csv(args.zonal)
    ref_lat = ref_df["latitude"].to_numpy(dtype=float)
    ref = ref_df["sst_annual"].to_numpy(dtype=float)
    if len(ref_lat) != len(lat) or not np.allclose(ref_lat, lat, atol=1e-6):
        ref = np.interp(lat, ref_lat, ref)
    print(f"读取 {args.grid}：{quantized['annual'].shape[1]}×{quantized['annual'].shape[0]} 网格，"
          f"海洋格点 {int(mask.sum())}")
    contours = {key: build_contours(lon, lat, q) for key, q in quantized.items()}
    print("25 °C 等值线：" + "，".join(f"{key}={len(seg)} 段" for key, seg in contours.items()))
    html = make_html(lat, lon, quantized, basin, contours, ref)
    worst = validate(html, originals, basin, lat, lon)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    size = args.out.stat().st_size
    print(f"已生成 {args.out}：{size / 1e6:.2f} MB")
    for key in ("annual", "jan", "jul"):
        b64_len = len(b64_int16(quantized[key]))
        print(f"  {key} int16差分+gzip base64 长度：{b64_len:,} 字符（原始 {quantized[key].size * 2:,} 字节）")
    if "__ANNUAL__" in html or "__META__" in html or "__CONTOURS__" in html:
        raise SystemExit("HTML 模板占位符未替换")
    if size >= MAX_BYTES:
        raise SystemExit(f"HTML 体积 {size / 1e6:.1f} MB ≥ 12 MB")
    if size >= TARGET_BYTES:
        print(f"  注意：体积 {size / 1e6:.2f} MB 略高于 3 MB 目标")
    print(f"  校验通过：文件 < 3 MB 目标检查完成（{size / 1e6:.2f} MB），无外部请求，量化最大偏差 {worst:.4f} °C。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
