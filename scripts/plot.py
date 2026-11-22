#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""根据 data/zonal_mean.csv 绘制两张图：

1. figures/sst_vs_latitude.png  —— 年均 / 1 月 / 7 月 SST 随纬度变化 + 覆盖率面板
2. figures/lat_month_heatmap.png —— 12 个月 × 纬度的纬向平均 SST 热图

只绘制有效海洋格点覆盖率 >= 10% 的纬度段（北端约 26°N 以北不绘制）。
图像标签全部为中文，使用 Microsoft YaHei / SimHei 字体；保存后检查是否有字形缺失（豆腐块）。
"""

from __future__ import annotations

import os
import struct
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from config import COVERAGE_MIN, FIG_DIR, LAT_MAX, LAT_MIN, ZONAL_CSV

ROOT = Path(__file__).resolve().parents[1]
# 沙箱/只读 HOME 环境下 matplotlib 无法写入 ~/.matplotlib，缓存改放仓库内
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager, text as mtext  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from compute import find_isotherm  # noqa: E402

DPI = 200
FONT_NAME = "Microsoft YaHei"


def setup_chinese_font() -> str:
    global FONT_NAME
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = None
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC"):
        if name in available:
            chosen = name
            break
    if chosen is None:
        raise SystemExit("未找到可用的中文字体（Microsoft YaHei / SimHei），无法保证中文标签正确渲染")
    FONT_NAME = chosen
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [chosen, "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 11
    print(f"中文字体：{chosen}", flush=True)
    return chosen


def verify_glyphs(fig: plt.Figure) -> None:
    """逐个检查图中实际用到的字符在所选字体中是否有字形，防止出现豆腐块（□）。"""
    font_path = font_manager.findfont(font_manager.FontProperties(family=FONT_NAME))
    face = font_manager.get_font(font_path)
    chars = sorted(
        {ch for text in fig.findobj(mtext.Text) for ch in text.get_text() if ord(ch) > 127}
    )
    missing = [ch for ch in chars if face.get_char_index(ord(ch)) == 0]
    if missing:
        raise RuntimeError(f"字体 {FONT_NAME} 缺少字形：{''.join(missing)}")
    print(f"  字形检查通过：图中 {len(chars)} 个非 ASCII 字符在 {FONT_NAME} 中均有字形", flush=True)


def lat_formatter(value: float, _pos=None) -> str:
    if abs(value) < 1e-9:
        return "0°"
    return f"{abs(value):.0f}°{'N' if value > 0 else 'S'}"


def bad_runs(lat: np.ndarray, bad: np.ndarray) -> list[tuple[float, float]]:
    """把覆盖率不足的纬度段合并成连续区间。"""
    runs: list[tuple[float, float]] = []
    start = None
    for i, flag in enumerate(bad):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((float(lat[start]), float(lat[i - 1])))
            start = None
    if start is not None:
        runs.append((float(lat[start]), float(lat[-1])))
    return runs


def save_png(fig: plt.Figure, path: Path) -> tuple[int, int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    verify_glyphs(fig)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    missing = [str(w.message) for w in caught if "missing from font" in str(w.message)]
    plt.close(fig)
    if missing:
        raise RuntimeError("PNG 中存在缺失字形（豆腐块）：\n" + "\n".join(missing))
    with open(path, "rb") as fh:
        head = fh.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError(f"{path} 不是合法 PNG")
    width, height = struct.unpack(">II", head[16:24])
    print(f"已生成 {path.relative_to(ROOT)}：{width}×{height} px，{path.stat().st_size / 1024:.0f} KB", flush=True)
    return width, height, path.stat().st_size


def plot_profiles(df: pd.DataFrame) -> Path:
    lat = df["latitude"].to_numpy(dtype=float)
    coverage = df["coverage"].to_numpy(dtype=float)
    usable = (coverage >= COVERAGE_MIN) & np.isfinite(df["sst_annual"].to_numpy(dtype=float))
    iso_list = find_isotherm(lat[usable], df["sst_annual"].to_numpy(dtype=float)[usable])

    fig = plt.figure(figsize=(11.6, 7.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[4.3, 1.05], wspace=0.06)
    ax = fig.add_subplot(gs[0, 0])
    axc = fig.add_subplot(gs[0, 1], sharey=ax)

    # 覆盖率 <10% 的纬度段（主要是北端）用浅灰底色标出，曲线不绘制
    shaded = False
    for y0, y1 in bad_runs(lat, ~usable):
        ax.axhspan(y0 - 0.125, y1 + 0.125, color="0.90", zorder=0,
                   label=None if shaded else f"覆盖率 < {COVERAGE_MIN:.0%}（不绘制）")
        axc.axhspan(y0 - 0.125, y1 + 0.125, color="0.90", zorder=0)
        shaded = True

    series = [
        ("sst_annual", "年平均", "#111111", 2.8, "-"),
        ("sst_jan", "1 月", "#2E6DB4", 1.8, "-"),
        ("sst_jul", "7 月", "#C0392B", 1.8, "-"),
    ]
    for col, label, color, lw, ls in series:
        values = np.where(usable, df[col].to_numpy(dtype=float), np.nan)
        ax.plot(values, lat, color=color, lw=lw, ls=ls, label=label, zorder=3)

    valid_values = np.concatenate(
        [np.where(usable, df[col].to_numpy(dtype=float), np.nan) for col, *_ in series]
    )
    ax.set_xlim(np.floor(np.nanmin(valid_values)) - 0.6, np.ceil(np.nanmax(valid_values)) + 0.8)
    ax.set_ylim(LAT_MIN - 0.8, LAT_MAX + 0.8)

    ax.set_xlabel("海表温度 SST（°C）")
    ax.set_ylabel("纬度")
    ax.yaxis.set_major_formatter(FuncFormatter(lat_formatter))
    ax.set_yticks(np.arange(-40, 31, 10))
    ax.grid(True, ls=":", lw=0.8, alpha=0.55)
    ax.set_axisbelow(True)
    ax.axvline(25.0, color="0.35", ls="--", lw=1.0, alpha=0.85, zorder=2)
    ax.text(25.15, LAT_MIN + 0.5, "25 °C", color="0.30", fontsize=9, rotation=90, va="bottom")
    ax.legend(loc="lower right", framealpha=0.92, fontsize=10)
    ax.set_title("印度洋纬向平均海表温度随纬度变化\n（2016–2020 气候态，20°E–120°E）", fontsize=13)

    if iso_list:
        iso = iso_list[-1]
        ax.plot(25.0, iso, "o", ms=6.5, color="#6A3D9A", zorder=5)
        ax.annotate(
            f"年均 25 °C 等温线\n{abs(iso):.2f}°{'N' if iso >= 0 else 'S'}",
            xy=(25.0, iso),
            xytext=(25.6, 5.0),
            fontsize=9,
            color="#4A2A6A",
            arrowprops={"arrowstyle": "->", "color": "#6A3D9A", "lw": 1.0},
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#6A3D9A", "alpha": 0.9},
            zorder=6,
        )

    # 覆盖率面板
    axc.fill_betweenx(lat, 0, coverage * 100, color="#7FA6C9", alpha=0.55, lw=0)
    axc.plot(coverage * 100, lat, color="#2E6DB4", lw=1.3, zorder=3)
    axc.axvline(COVERAGE_MIN * 100, color="#C0392B", ls="--", lw=1.1)
    axc.set_xlim(0, 100)
    axc.set_xticks([0, 50, 100])
    axc.tick_params(axis="x", labelsize=9)
    axc.set_xlabel("覆盖率（%）", fontsize=9)
    axc.set_title("有效海洋格点\n覆盖率", fontsize=10, pad=8)
    axc.grid(True, axis="x", ls=":", lw=0.8, alpha=0.55)
    axc.set_axisbelow(True)
    plt.setp(axc.get_yticklabels(), visible=False)

    band_cov = coverage[(lat >= 20) & (lat <= 26)]
    band_lat = lat[(lat >= 20) & (lat <= 26)]
    idx = int(np.argmin(band_cov))
    axc.annotate(
        f"20–26°N 最低\n{band_cov[idx] * 100:.1f}%",
        xy=(max(band_cov[idx] * 100, 2), band_lat[idx]),
        xytext=(38, band_lat[idx] + 1.5),
        fontsize=8,
        color="#8E2F2F",
        arrowprops={"arrowstyle": "->", "color": "#8E2F2F", "lw": 0.9},
    )

    fig.text(
        0.01,
        0.005,
        "注：纬向平均为同一纬度圈内有效海洋格点（非陆地/海冰）的算术平均，未乘 cosφ；\n"
        "覆盖率 = 有效格点数 / 401（0.25° 原生经度格点）。北端约 26°N 以北陆地与半封闭海域多，"
        "有效格点太少，曲线不绘制。",
        fontsize=8.5,
        color="0.35",
        va="bottom",
    )

    path = FIG_DIR / "sst_vs_latitude.png"
    save_png(fig, path)
    return path


def plot_heatmap(df: pd.DataFrame) -> Path:
    lat = df["latitude"].to_numpy(dtype=float)
    coverage = df["coverage"].to_numpy(dtype=float)
    usable = coverage >= COVERAGE_MIN
    months = np.arange(1, 13)
    data = np.vstack([df[f"sst_{m:02d}"].to_numpy(dtype=float) for m in months])
    masked = np.ma.masked_array(data, mask=(~np.isfinite(data)) | ~usable[None, :])

    fig, ax = plt.subplots(figsize=(11.6, 6.6))
    cmap = matplotlib.colormaps["RdYlBu_r"].copy()
    cmap.set_bad("#D9D9D9")
    filled = np.asarray(masked.filled(np.nan), dtype=float)
    vmin = np.floor(np.nanmin(filled) * 2) / 2
    vmax = np.ceil(np.nanmax(filled) * 2) / 2
    mesh = ax.pcolormesh(lat, months, masked, shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    cb = fig.colorbar(mesh, ax=ax, pad=0.015, aspect=30)
    cb.set_label("海表温度 SST（°C）")

    cs = ax.contour(lat, months, masked, levels=[25.0], colors="black", linewidths=0.9)
    ax.clabel(cs, fmt="25 °C", fontsize=8)

    ax.set_xlim(LAT_MIN - 0.125, LAT_MAX + 0.125)
    ax.set_ylim(12.5, 0.5)
    ax.set_yticks(months)
    ax.set_yticklabels([f"{m} 月" for m in months])
    ax.set_xticks(np.arange(-40, 31, 10))
    ax.xaxis.set_major_formatter(FuncFormatter(lat_formatter))
    ax.set_xlabel("纬度")
    ax.set_ylabel("月份")
    ax.set_title("印度洋纬向平均海表温度：月份 × 纬度\n（2016–2020 气候态，20°E–120°E，空白/灰色 = 覆盖率 < 10%）", fontsize=13)
    ax.grid(True, ls=":", lw=0.7, alpha=0.5, color="0.4")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

    path = FIG_DIR / "lat_month_heatmap.png"
    save_png(fig, path)
    return path


def main() -> int:
    setup_chinese_font()
    if not ZONAL_CSV.exists():
        raise SystemExit(f"未找到 {ZONAL_CSV}，请先运行 scripts/compute.py")
    df = pd.read_csv(ZONAL_CSV)
    print(f"读取 {ZONAL_CSV.name}：{len(df)} 行", flush=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_profiles(df)
    plot_heatmap(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
