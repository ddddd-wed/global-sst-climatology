#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""全球海表温度（SST）纬向结构与五大洋对比 —— 绘图脚本。

输出（真实数据就绪时写到 ``figures/``，否则自动降级到 ``figures/dev/``）：

1. ``global_sst_vs_latitude.png`` —— 90°S–90°N 年均 / 1 月 / 7 月曲线 + 覆盖率面板，
   年均 25 °C 等温线的**南北半球交点**同时标出；
2. ``basins_sst_vs_latitude.png`` —— 五大洋分盆年均曲线 + 旧印度洋结果对照 + 1/7 月面板；
3. ``global_sst_map.png`` —— 0.25° 全球年均 SST 等经纬投影地图（叠加 25 °C 等值线）；
4. ``lat_month_heatmap.png`` —— 全球 月份 × 纬度 热图；
5. ``indian_vs_global.png`` —— 印度洋旧廓线 vs 全球廓线 + 差值副图。

真实数据文件（``data/zonal_mean_global.csv``、``data/zonal_mean_basins.csv``、
``data/climatology_grid.npz``）不存在时，打印「真实数据未就绪，使用 fixture」，
在 ``.tmp/fixture/`` 自动生成合成数据并出图到 ``figures/dev/``，进程 0 退出。
合成数据只用于开发自测，图上会打「合成数据」水印，绝不会写入 ``data/``。

统计口径与数据侧一致：0.25° 原生格点、同纬圈非 NaN 海洋格点算术平均（不乘 cosφ）、
12 个月等权得年均。**绘图门槛用「海洋占比」``zonal_ocean_frac = valid_count / 1440`` ≥ 10%**；
``coverage = valid_count / ocean_grid_count`` 是「有数据比例」（非海冰/非缺测），只用于覆盖率
副图展示，不能拿来决定是否画曲线（否则窄海盆/极区会拿几十个格点画出看似正常的曲线）。
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
# 沙箱/只读 HOME 环境下 matplotlib 无法写入 ~/.matplotlib，缓存改放仓库内
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager, text as mtext  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
DEV_FIG_DIR = FIG_DIR / "dev"
FIXTURE_DIR = ROOT / ".tmp" / "fixture"

GLOBAL_CSV = DATA_DIR / "zonal_mean_global.csv"
BASINS_CSV = DATA_DIR / "zonal_mean_basins.csv"
GRID_NPZ = DATA_DIR / "climatology_grid.npz"
MASK_NPZ = DATA_DIR / "basin_mask.npz"
SUMMARY_JSON = DATA_DIR / "summary_metrics.json"
INDIAN_CSV = DATA_DIR / "zonal_mean.csv"

# 绘图门槛：海洋占比阈值（可配置常量，也支持 --coverage-min / 环境变量 SST_COVERAGE_MIN）
DEFAULT_COVERAGE_MIN = float(os.environ.get("SST_COVERAGE_MIN", "0.10"))
N_LON_GLOBAL = 1440          # 0.25° 全球经度格点数，zonal_ocean_frac 的分母
N_LON_INDIAN = 401           # 旧印度洋文件 0.25° 经度格点数，仅用于旧结果回归
LAT_MIN, LAT_MAX = -90.0, 90.0
ISOTHERM_LEVEL = 25.0
DPI = 200
FONT_NAME = "Microsoft YaHei"

# 海盆配色固定，便于跨图对照
BASIN_STYLE: dict[str, dict] = {
    "global": {"label": "全球", "color": "#111111"},
    "indian": {"label": "印度洋", "color": "#E67E22"},
    "pacific": {"label": "太平洋", "color": "#2E86C1"},
    "atlantic": {"label": "大西洋", "color": "#27AE60"},
    "arctic": {"label": "北冰洋", "color": "#8E44AD"},
    "southern": {"label": "南大洋", "color": "#C0392B"},
}
BASIN_ORDER = ("indian", "pacific", "atlantic", "arctic", "southern")

# 印度洋旧结果回归基准（README 现值，来自仓库已提交的 data/zonal_mean.csv）
INDIAN_BASELINE = {
    "sst_max": (29.14, 0.01),
    "lat_max": (2.125, 0.01),
    "isotherm_south": (-21.664, 0.01),
    "basin_mean": (26.163, 0.005),
}


# --------------------------------------------------------------------------- #
# 字体 / PNG 基础设施
# --------------------------------------------------------------------------- #
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
    """逐个检查图中实际用到的非 ASCII 字符是否有字形，防止豆腐块（□）。"""
    font_path = font_manager.findfont(font_manager.FontProperties(family=FONT_NAME))
    face = font_manager.get_font(font_path)
    chars = sorted(
        {ch for text in fig.findobj(mtext.Text) for ch in text.get_text() if ord(ch) > 127}
    )
    missing = [ch for ch in chars if face.get_char_index(ord(ch)) == 0]
    if missing:
        raise RuntimeError(f"字体 {FONT_NAME} 缺少字形：{''.join(missing)}")
    print(f"  字形检查通过：图中 {len(chars)} 个非 ASCII 字符在 {FONT_NAME} 中均有字形", flush=True)


def save_png(fig: plt.Figure, path: Path, dpi: int = DPI) -> tuple[int, int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    verify_glyphs(fig)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    missing = [str(w.message) for w in caught if "missing from font" in str(w.message)]
    plt.close(fig)
    if missing:
        raise RuntimeError("PNG 中存在缺失字形（豆腐块）：\n" + "\n".join(missing))
    with open(path, "rb") as fh:
        head = fh.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError(f"{path} 不是合法 PNG")
    width, height = struct.unpack(">II", head[16:24])
    size_kb = path.stat().st_size / 1024
    print(f"已生成 {path}：{width}×{height} px，{size_kb:.0f} KB", flush=True)
    return width, height, int(path.stat().st_size)


def add_synthetic_watermark(fig: plt.Figure) -> None:
    fig.text(
        0.5,
        0.5,
        "合成数据 · 非真实结果",
        fontsize=46,
        color="#C0392B",
        alpha=0.10,
        ha="center",
        va="center",
        rotation=18,
        zorder=50,
    )
    fig.text(
        0.995,
        0.005,
        "fixture（make_fixture.py 生成，仅开发自测）",
        fontsize=8,
        color="#B03A2E",
        ha="right",
        va="bottom",
    )


# --------------------------------------------------------------------------- #
# 通用小工具
# --------------------------------------------------------------------------- #
def lat_formatter(value: float, _pos=None) -> str:
    if abs(value) < 1e-9:
        return "0°"
    return f"{abs(value):.0f}°{'N' if value > 0 else 'S'}"


def isotherm_crossings(lat: np.ndarray, sst: np.ndarray, level: float = ISOTHERM_LEVEL) -> list[float]:
    """线性插值求等温线纬度，返回**所有**交点（升序），不再只取最后一个。"""
    crossings: list[float] = []
    for i in range(len(lat) - 1):
        y1, y2 = sst[i], sst[i + 1]
        if not (np.isfinite(y1) and np.isfinite(y2)) or y1 == y2:
            continue
        if (y1 - level) * (y2 - level) <= 0:
            t = (level - y1) / (y2 - y1)
            crossings.append(float(lat[i] + t * (lat[i + 1] - lat[i])))
    deduped: list[float] = []
    for c in crossings:
        if not deduped or abs(c - deduped[-1]) > 1e-6:
            deduped.append(c)
    return deduped


def zonal_ocean_fraction(frame: pd.DataFrame, n_lon: int = N_LON_GLOBAL) -> np.ndarray:
    """绘图门槛用的「海洋占比」：优先 zonal_ocean_frac，其次 valid_count/n_lon，最后退回 coverage。"""
    if "zonal_ocean_frac" in frame:
        return frame["zonal_ocean_frac"].to_numpy(dtype=float)
    if "valid_count" in frame:
        return frame["valid_count"].to_numpy(dtype=float) / n_lon
    return frame_coverage(frame)


def frame_coverage(frame: pd.DataFrame) -> np.ndarray:
    """「有数据比例」coverage：分母 ocean_grid_count；旧文件缺失该列时退回 valid_count/1440。

    注意：coverage 只用于覆盖率副图展示，**不能**作为是否绘制曲线的门槛。
    """
    if "coverage" in frame:
        return frame["coverage"].to_numpy(dtype=float)
    if "ocean_grid_count" in frame and "valid_count" in frame:
        ocean = frame["ocean_grid_count"].to_numpy(dtype=float)
        valid = frame["valid_count"].to_numpy(dtype=float)
        out = np.full(len(frame), np.nan, dtype=float)
        ok = ocean > 0
        out[ok] = valid[ok] / ocean[ok]
        return out
    if "valid_count" in frame:
        return frame["valid_count"].to_numpy(dtype=float) / N_LON_GLOBAL
    return np.full(len(frame), np.nan, dtype=float)


def coverage_ok(frame: pd.DataFrame, cov_min: float, n_lon: int = N_LON_GLOBAL) -> np.ndarray:
    """曲线是否绘制：看海洋占比（valid/n_lon），不看 coverage（冰情/缺测比例）。"""
    frac = zonal_ocean_fraction(frame, n_lon)
    return np.isfinite(frac) & (frac >= cov_min)


def cos_weighted_mean(
    lat: np.ndarray, sst: np.ndarray, ocean_frac: np.ndarray, cov_min: float
) -> float | None:
    """cosφ 加权均温（旧印度洋回归口径，仅用于旧结果，保持 26.16 °C 不变）。"""
    usable = np.isfinite(sst) & np.isfinite(ocean_frac) & (ocean_frac >= cov_min)
    if not usable.any():
        return None
    weights = np.cos(np.deg2rad(lat[usable]))
    return float(np.sum(weights * sst[usable]) / np.sum(weights))


def valid_weighted_mean(frame: pd.DataFrame, cov_min: float, n_lon: int = N_LON_GLOBAL) -> float | None:
    """新契约的 cos(lat)*valid_count 加权均温；缺 valid_count 时降级为 cosφ 加权。"""
    lat = frame["latitude"].to_numpy(dtype=float)
    sst = frame["sst_annual"].to_numpy(dtype=float)
    ocean_frac = zonal_ocean_fraction(frame, n_lon)
    usable = np.isfinite(sst) & np.isfinite(ocean_frac) & (ocean_frac >= cov_min)
    if not usable.any():
        return None
    weights = np.cos(np.deg2rad(lat[usable]))
    if "valid_count" in frame:
        weights = weights * np.maximum(frame["valid_count"].to_numpy(dtype=float)[usable], 0.0)
    if float(np.sum(weights)) <= 0:
        return None
    return float(np.sum(weights * sst[usable]) / np.sum(weights))


def bad_runs(lat: np.ndarray, bad: np.ndarray) -> list[tuple[float, float]]:
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


def shaded_bad_latitudes(ax: plt.Axes, lat: np.ndarray, bad: np.ndarray, cov_min: float, label: bool = True) -> None:
    shaded = False
    for y0, y1 in bad_runs(lat, bad):
        ax.axhspan(
            y0 - 0.125,
            y1 + 0.125,
            color="0.90",
            zorder=0,
            label=(f"海洋占比 < {cov_min:.0%}（不绘制）" if label and not shaded else None),
        )
        shaded = True


def load_basins(path: Path) -> dict[str, pd.DataFrame]:
    """读取分海盆 CSV，容忍列名大小写、首列无名与额外列。"""
    df = pd.read_csv(path)
    basin_col = None
    for col in df.columns:
        if str(col).strip().lower() == "basin":
            basin_col = col
            break
    if basin_col is None:
        first = df.columns[0]
        values = df[first].astype(str).str.strip().str.lower()
        known = set(BASIN_STYLE)
        if values.isin(known).mean() > 0.5:
            basin_col = first
    if basin_col is None:
        raise SystemExit(f"{path} 中找不到 basin 列（现有列：{list(df.columns)[:5]}…）")
    df[basin_col] = df[basin_col].astype(str).str.strip().str.lower()
    return {name: group.reset_index(drop=True) for name, group in df.groupby(basin_col, sort=False)}


def load_summary_means(path: Path) -> dict[str, float]:
    """从 summary_metrics.json 里尽力读取各海盆 cosφ 加权均温，字段名不固定。"""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    basins = data.get("basins") if isinstance(data, dict) else None
    if not isinstance(basins, dict):
        return {}
    keys = ("mean_sst_cos_weighted", "mean_sst", "sst_mean", "basin_mean", "annual_mean", "mean")
    out: dict[str, float] = {}
    for name, info in basins.items():
        value = None
        if isinstance(info, (int, float)):
            value = float(info)
        elif isinstance(info, dict):
            for key in keys:
                if isinstance(info.get(key), (int, float)):
                    value = float(info[key])
                    break
        if value is not None and np.isfinite(value):
            out[str(name).strip().lower()] = value
    return out


# --------------------------------------------------------------------------- #
# 数据定位：真实数据 vs fixture
# --------------------------------------------------------------------------- #
def resolve_inputs(force_fixture: bool, refresh_fixture: bool) -> dict:
    missing = [p.name for p in (GLOBAL_CSV, BASINS_CSV, GRID_NPZ) if not p.exists()]
    use_fixture = force_fixture or bool(missing)
    if use_fixture:
        if not force_fixture:
            print(f"真实数据未就绪（缺少：{', '.join(missing)}），使用 fixture。", flush=True)
        else:
            print("按 --fixture 强制使用 fixture。", flush=True)
        need_build = refresh_fixture or not (FIXTURE_DIR / "zonal_mean_global.csv").exists()
        if need_build:
            from make_fixture import build_fixture  # 延迟导入，避免真实模式多余开销

            print(f"生成合成 fixture -> {FIXTURE_DIR}", flush=True)
            build_fixture(FIXTURE_DIR)
        print("注意：以下图件全部基于合成数据，仅用于开发自测，不是观测结果。", flush=True)
        return {
            "mode": "fixture",
            "global_csv": FIXTURE_DIR / "zonal_mean_global.csv",
            "basins_csv": FIXTURE_DIR / "zonal_mean_basins.csv",
            "grid_npz": FIXTURE_DIR / "climatology_grid.npz",
            "mask_npz": FIXTURE_DIR / "basin_mask.npz",
            "summary_json": FIXTURE_DIR / "summary_metrics.json",
            "out_dir": DEV_FIG_DIR,
            "synthetic": True,
        }

    print("真实数据就绪：使用 data/ 下的全球结果。", flush=True)
    return {
        "mode": "real",
        "global_csv": GLOBAL_CSV,
        "basins_csv": BASINS_CSV,
        "grid_npz": GRID_NPZ,
        "mask_npz": MASK_NPZ,
        "summary_json": SUMMARY_JSON,
        "out_dir": FIG_DIR,
        "synthetic": False,
    }


def finalize(fig: plt.Figure, ctx: dict, name: str, dpi: int = DPI) -> Path:
    if ctx["synthetic"]:
        add_synthetic_watermark(fig)
    path = ctx["out_dir"] / name
    save_png(fig, path, dpi=dpi)
    return path


# --------------------------------------------------------------------------- #
# 图 1：全球纬向平均廓线 + 覆盖率面板
# --------------------------------------------------------------------------- #
def plot_global_profile(ctx: dict, cov_min: float) -> Path:
    df = pd.read_csv(ctx["global_csv"])
    lat = df["latitude"].to_numpy(dtype=float)
    coverage = frame_coverage(df)
    ocean_frac = zonal_ocean_fraction(df)
    annual = df["sst_annual"].to_numpy(dtype=float)
    usable = np.isfinite(annual) & np.isfinite(ocean_frac) & (ocean_frac >= cov_min)

    fig = plt.figure(figsize=(12.0, 7.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[4.4, 1.05], wspace=0.06)
    ax = fig.add_subplot(gs[0, 0])
    axc = fig.add_subplot(gs[0, 1], sharey=ax)
    axc.sharey(ax)

    shaded_bad_latitudes(ax, lat, ~usable, cov_min)
    shaded_bad_latitudes(axc, lat, ~usable, cov_min, label=False)

    series = [
        ("sst_annual", "年平均", "#111111", 2.8, "-", 3),
        ("sst_jan", "1 月", "#2E6DB4", 1.8, "-", 2),
        ("sst_jul", "7 月", "#C0392B", 1.8, "-", 2),
    ]
    valid_values: list[np.ndarray] = []
    for col, label, color, lw, ls, zorder in series:
        values = df[col].to_numpy(dtype=float)
        masked = np.where(usable & np.isfinite(values), values, np.nan)
        valid_values.append(masked)
        ax.plot(masked, lat, color=color, lw=lw, ls=ls, label=label, zorder=zorder)

    stacked = np.concatenate(valid_values)
    ax.set_xlim(np.floor(np.nanmin(stacked)) - 0.8, np.ceil(np.nanmax(stacked)) + 1.4)
    ax.set_ylim(LAT_MIN - 1.0, LAT_MAX + 1.0)

    ax.set_xlabel("海表温度 SST（°C）")
    ax.set_ylabel("纬度")
    ax.yaxis.set_major_formatter(FuncFormatter(lat_formatter))
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.grid(True, ls=":", lw=0.8, alpha=0.55)
    ax.set_axisbelow(True)
    ax.axvline(ISOTHERM_LEVEL, color="0.35", ls="--", lw=1.0, alpha=0.85, zorder=2)
    ax.text(ISOTHERM_LEVEL + 0.25, LAT_MIN + 1.5, "25 °C", color="0.30", fontsize=9, rotation=90, va="bottom")

    # 年均 25 °C 等温线：同时标出南、北半球交点（旧版只取 [-1] 会漏掉北半球）
    iso = isotherm_crossings(lat[usable], annual[usable])
    pos = [c for c in iso if c > 0]
    neg = [c for c in iso if c < 0]
    picks: list[float] = []
    if pos:
        picks.append(max(pos))
    if neg:
        picks.append(min(neg))
    if len(picks) < 2:  # 极端情况下补上剩余交点
        picks = picks or iso[:2]
    if iso:
        print(f"  年均 {ISOTHERM_LEVEL:.0f} °C 等温线交点（南北半球同时标注）："
              + "、".join(f"{c:.2f}°{'N' if c >= 0 else 'S'}" for c in picks), flush=True)
    else:
        print(f"  年均 {ISOTHERM_LEVEL:.0f} °C 等温线：数据范围内无交点", flush=True)
    for c in picks:
        ax.plot(ISOTHERM_LEVEL, c, "o", ms=6.5, color="#6A3D9A", zorder=5)
        ax.annotate(
            f"年均 25 °C 等温线\n{abs(c):.2f}°{'N' if c >= 0 else 'S'}",
            xy=(ISOTHERM_LEVEL, c),
            xytext=(ISOTHERM_LEVEL + 1.0, c + (7.0 if c >= 0 else -7.0)),
            fontsize=8.6,
            color="#4A2A6A",
            arrowprops={"arrowstyle": "->", "color": "#6A3D9A", "lw": 1.0},
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#6A3D9A", "alpha": 0.9},
            zorder=6,
        )

    ax.legend(loc="lower right", framealpha=0.92, fontsize=10)
    subtitle = "合成数据（fixture）" if ctx["synthetic"] else "2016–2020 气候态，0.25° 原生格点"
    ax.set_title(f"全球纬向平均海表温度随纬度变化\n（{subtitle}，90°S–90°N）", fontsize=13)

    # 覆盖率面板：蓝=海洋占比（绘图门槛），橙=有数据比例（冰情/缺测，仅展示）
    axc.fill_betweenx(lat, 0, ocean_frac * 100, color="#7FA6C9", alpha=0.55, lw=0)
    axc.plot(ocean_frac * 100, lat, color="#2E6DB4", lw=1.3, zorder=3)
    axc.plot(coverage * 100, lat, color="#E67E22", lw=1.3, ls="--", zorder=4)
    axc.axvline(cov_min * 100, color="#C0392B", ls="--", lw=1.1)
    axc.set_xlim(0, 100)
    axc.set_xticks([0, 50, 100])
    axc.tick_params(axis="x", labelsize=9)
    axc.set_xlabel("比例（%）", fontsize=9)
    axc.set_title(f"海洋占比（蓝，/{N_LON_GLOBAL}）\n有数据比例（橙，/ocean_grid_count）", fontsize=7.6, pad=8)
    axc.grid(True, axis="x", ls=":", lw=0.8, alpha=0.55)
    axc.set_axisbelow(True)
    plt.setp(axc.get_yticklabels(), visible=False)
    axc.text(cov_min * 100 + 1.5, LAT_MIN + 2.5, f"{cov_min:.0%} 门槛", fontsize=7.5,
             color="#8E2F2F", rotation=90, va="bottom")

    fig.text(
        0.01,
        0.005,
        "注：纬向平均为同一纬度圈内有效海洋格点（非陆地/海冰）的算术平均，未乘 cosφ；\n"
        f"海洋占比 zonal_ocean_frac = 有效格点数 / {N_LON_GLOBAL}，曲线只画 ≥ {cov_min:.0%} 的纬度；"
        "有数据比例 coverage = 有效格点数 / ocean_grid_count（非海冰/非缺测），仅用于副图展示。",
        fontsize=8.3,
        color="0.35",
        va="bottom",
    )
    return finalize(fig, ctx, "global_sst_vs_latitude.png")


# --------------------------------------------------------------------------- #
# 图 2：五大洋分盆对比 + 旧印度洋结果
# --------------------------------------------------------------------------- #
def plot_basins(ctx: dict, cov_min: float) -> Path:
    basins = load_basins(ctx["basins_csv"])
    summary_means = load_summary_means(ctx["summary_json"])
    names = [n for n in BASIN_ORDER if n in basins]
    names += [n for n in basins if n not in names and n != "global"]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14.4, 7.4))
    mean_text: list[str] = []

    for name in names:
        frame = basins[name]
        style = BASIN_STYLE.get(name, {"label": name, "color": None})
        lat = frame["latitude"].to_numpy(dtype=float)
        sst = frame["sst_annual"].to_numpy(dtype=float)
        usable = coverage_ok(frame, cov_min) & np.isfinite(sst)
        ax.plot(np.where(usable, sst, np.nan), lat, color=style["color"], lw=2.0, label=style["label"])
        mean = summary_means.get(name)
        if mean is None:
            mean = valid_weighted_mean(frame, cov_min)
        if mean is not None:
            mean_text.append(f"{style['label']} {mean:.2f} °C")

        # 1 月 / 7 月：实线 / 虚线（同色）
        for col, ls, alpha in (("sst_jan", "-", 0.95), ("sst_jul", "--", 0.75)):
            if col not in frame:
                continue
            values = frame[col].to_numpy(dtype=float)
            ok = coverage_ok(frame, cov_min) & np.isfinite(values)
            ax2.plot(np.where(ok, values, np.nan), lat, color=style["color"], lw=1.25, ls=ls, alpha=alpha)

    # 旧印度洋结果（仓库已提交的真实数据）作为对照；旧文件 coverage 即 valid/401
    if INDIAN_CSV.exists():
        old = pd.read_csv(INDIAN_CSV)
        lat_old = old["latitude"].to_numpy(dtype=float)
        sst_old = old["sst_annual"].to_numpy(dtype=float)
        frac_old = zonal_ocean_fraction(old, N_LON_INDIAN)
        ok = np.isfinite(sst_old) & np.isfinite(frac_old) & (frac_old >= cov_min)
        ax.plot(np.where(ok, sst_old, np.nan), lat_old, color="#7F8C8D", lw=2.0, ls=(0, (5, 2)),
                label="印度洋（旧结果 20°E–120°E/40°S–30°N）")

    for axis in (ax, ax2):
        axis.set_ylim(LAT_MIN - 1.0, LAT_MAX + 1.0)
        axis.set_yticks(np.arange(-90, 91, 30))
        axis.yaxis.set_major_formatter(FuncFormatter(lat_formatter))
        axis.grid(True, ls=":", lw=0.8, alpha=0.55)
        axis.set_axisbelow(True)
        axis.set_xlabel("海表温度 SST（°C）")
        axis.set_ylabel("纬度")
        axis.axvline(ISOTHERM_LEVEL, color="0.55", ls=":", lw=0.9)

    ax.set_title("五大洋年均 SST–纬度廓线\n（0.25° 原生格点，覆盖率 ≥ 10% 的纬度）", fontsize=12.5)
    ax.legend(loc="lower right", framealpha=0.93, fontsize=9)
    if mean_text:
        ax.text(
            0.015,
            0.985,
            "cosφ 加权均温：\n" + "\n".join(mean_text),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.4,
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "0.7", "alpha": 0.9},
        )

    ax2.set_title("分海盆 1 月（实线）与 7 月（虚线）\n南北半球季节反相", fontsize=12.5)
    proxy = [
        Line2D([0], [0], color="0.3", lw=1.4, ls="-", label="1 月"),
        Line2D([0], [0], color="0.3", lw=1.4, ls="--", label="7 月"),
    ]
    ax2.legend(handles=proxy, loc="lower right", framealpha=0.93, fontsize=9)

    note = (
        "注：分海盆边界为近似划分（印度洋 20°E–120°E，太平洋 120°E–280°E，其余归大西洋；"
        "66°N 以北为北冰洋，60°S 以南为南大洋）。"
        "曲线门槛统一用海洋占比 valid_count/1440 ≥ "
        f"{cov_min:.0%}（分盆文件无 zonal_ocean_frac 列时现算），"
        "coverage（valid_count/ocean_grid_count）只反映冰情/缺测，不用于门槛。"
    )
    if ctx["synthetic"]:
        note += "\n本图海盆曲线为合成 fixture；灰色点划线「印度洋（旧结果）」来自仓库已提交的真实 CSV。"
    fig.text(0.01, 0.005, note, fontsize=8.3, color="0.35", va="bottom")
    return finalize(fig, ctx, "basins_sst_vs_latitude.png")


# --------------------------------------------------------------------------- #
# 图 3：全球年均 SST 地图
# --------------------------------------------------------------------------- #
def plot_map(ctx: dict, _cov_min: float = DEFAULT_COVERAGE_MIN) -> Path:
    with np.load(ctx["grid_npz"]) as npz:
        lat = npz["lat"].astype(float)
        lon = npz["lon"].astype(float)
        annual = npz["sst_annual"].astype(float)
    if annual.shape != (len(lat), len(lon)):
        raise SystemExit(f"climatology_grid.npz 形状异常：{annual.shape} vs lat×lon")

    cmap = matplotlib.colormaps["RdYlBu_r"].copy()
    cmap.set_bad("#E6E6E6")
    fig, ax = plt.subplots(figsize=(13.2, 6.8))
    mesh = ax.pcolormesh(
        lon,
        lat,
        np.ma.masked_invalid(annual),
        shading="nearest",
        cmap=cmap,
        vmin=-2.0,
        vmax=32.0,
        rasterized=True,
    )
    cb = fig.colorbar(mesh, ax=ax, pad=0.012, aspect=28)
    cb.set_label("海表温度 SST（°C）", fontsize=11)
    cb.set_ticks([-2, 4, 10, 16, 22, 28, 32])

    cs = ax.contour(lon, lat, np.ma.masked_invalid(annual), levels=[ISOTHERM_LEVEL],
                    colors="black", linewidths=1.0)
    ax.clabel(cs, fmt="25 °C", fontsize=8)

    boundary_note = "未叠加海盆边界（basin_mask.npz 不存在）"
    if ctx["mask_npz"].exists():
        with np.load(ctx["mask_npz"]) as npz:
            keys = set(npz.files)
            if "basin_mask" in keys:
                basin_grid = npz["basin_mask"].astype(float)  # uint8: 0=陆地，1..5=海盆
                levels = [0.5, 1.5, 2.5, 3.5, 4.5]
                label = "basin_mask"
            elif "basin_id" in keys:
                basin_grid = npz["basin_id"].astype(float)    # int8: -1=陆地，0..4=海盆
                levels = [-0.5, 0.5, 1.5, 2.5, 3.5]
                label = "basin_id"
            else:
                basin_grid = None
                levels = []
                label = ""
        if basin_grid is not None and basin_grid.shape == annual.shape:
            step = 4
            ax.contour(lon[::step], lat[::step], basin_grid[::step, ::step],
                       levels=levels, colors="0.25", linewidths=0.35, alpha=0.8)
            boundary_note = f"细线为海盆边界与海岸线（{label}）"
        elif basin_grid is not None:
            boundary_note = "basin_mask.npz 形状不匹配，未画边界"

    ax.set_xlim(0, 360)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_xticks(np.arange(0, 361, 60))
    ax.set_xticklabels([f"{v:.0f}°" for v in np.arange(0, 361, 60)])
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.yaxis.set_major_formatter(FuncFormatter(lat_formatter))
    ax.set_xlabel("经度")
    ax.set_ylabel("纬度")
    subtitle = "合成数据（fixture）" if ctx["synthetic"] else "2016–2020 气候态"
    ax.set_title(f"全球年平均海表温度（0.25° 等经纬网格）\n（{subtitle}，黑色等值线 = 25 °C，灰色 = 陆地/海冰）", fontsize=13)
    ax.set_facecolor("#E6E6E6")

    fig.text(
        0.01,
        0.005,
        "注：未引入 cartopy，采用等经纬（Plate Carrée）投影的 pcolormesh；"
        f"色标固定 −2～32 °C；{boundary_note}。",
        fontsize=8.4,
        color="0.35",
        va="bottom",
    )
    return finalize(fig, ctx, "global_sst_map.png", dpi=180)


def plot_heatmap(ctx: dict, cov_min: float) -> Path:
    df = pd.read_csv(ctx["global_csv"])
    lat = df["latitude"].to_numpy(dtype=float)
    ocean_frac = zonal_ocean_fraction(df)
    usable = np.isfinite(ocean_frac) & (ocean_frac >= cov_min)
    months = np.arange(1, 13)
    data = np.vstack([df[f"sst_{m:02d}"].to_numpy(dtype=float) for m in months])
    masked = np.ma.masked_array(data, mask=(~np.isfinite(data)) | ~usable[None, :])

    fig, ax = plt.subplots(figsize=(12.4, 6.8))
    cmap = matplotlib.colormaps["RdYlBu_r"].copy()
    cmap.set_bad("#D9D9D9")
    filled = np.asarray(masked.filled(np.nan), dtype=float)
    vmin = np.floor(np.nanmin(filled) * 2) / 2
    vmax = np.ceil(np.nanmax(filled) * 2) / 2
    mesh = ax.pcolormesh(lat, months, masked, shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    cb = fig.colorbar(mesh, ax=ax, pad=0.015, aspect=30)
    cb.set_label("海表温度 SST（°C）")

    cs = ax.contour(lat, months, masked, levels=[ISOTHERM_LEVEL], colors="black", linewidths=0.9)
    ax.clabel(cs, fmt="25 °C", fontsize=8)

    ax.set_xlim(LAT_MIN - 0.125, LAT_MAX + 0.125)
    ax.set_ylim(12.5, 0.5)
    ax.set_yticks(months)
    ax.set_yticklabels([f"{m} 月" for m in months])
    ax.set_xticks(np.arange(-90, 91, 30))
    ax.xaxis.set_major_formatter(FuncFormatter(lat_formatter))
    ax.set_xlabel("纬度")
    ax.set_ylabel("月份")
    subtitle = "合成数据（fixture）" if ctx["synthetic"] else "2016–2020 气候态"
    ax.set_title(f"全球纬向平均海表温度：月份 × 纬度\n（{subtitle}，90°S–90°N，灰色 = 海洋占比 < {cov_min:.0%}）", fontsize=13)
    ax.grid(True, ls=":", lw=0.7, alpha=0.5, color="0.4")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

    fig.text(
        0.01,
        0.005,
        "注：每格为该纬度圈 12 个月气候态；12 个月等权平均得年均；"
        f"海洋占比 valid/{N_LON_GLOBAL} 不足的纬度置灰，不参与曲线与结论；"
        "灰色不代表冰情，coverage 见廓线副图。",
        fontsize=8.4,
        color="0.35",
        va="bottom",
    )
    return finalize(fig, ctx, "lat_month_heatmap.png")


# --------------------------------------------------------------------------- #
# 图 4：印度洋旧廓线 vs 全球廓线 + 差值
# --------------------------------------------------------------------------- #
def plot_indian_vs_global(ctx: dict, cov_min: float) -> Path:
    if not INDIAN_CSV.exists():
        raise SystemExit(f"缺少 {INDIAN_CSV}，无法做印度洋回归对照")
    old = pd.read_csv(INDIAN_CSV)
    glob = pd.read_csv(ctx["global_csv"])
    lat_old = old["latitude"].to_numpy(dtype=float)
    lat_glob = glob["latitude"].to_numpy(dtype=float)
    sst_old = old["sst_annual"].to_numpy(dtype=float)
    sst_glob = glob["sst_annual"].to_numpy(dtype=float)
    frac_old = zonal_ocean_fraction(old, N_LON_INDIAN)
    frac_glob = zonal_ocean_fraction(glob)
    ok_old = np.isfinite(sst_old) & np.isfinite(frac_old) & (frac_old >= cov_min)
    ok_glob = np.isfinite(sst_glob) & np.isfinite(frac_glob) & (frac_glob >= cov_min)

    # 全球廓线插值到印度洋纬度，仅用于重叠区差值
    common = (lat_glob >= lat_old.min()) & (lat_glob <= lat_old.max())
    lat_c = lat_glob[common]
    sst_c = sst_glob[common]
    ok_c = ok_glob[common]
    diff = sst_old - np.interp(lat_old, lat_c[ok_c], sst_c[ok_c]) if ok_c.any() else np.full_like(lat_old, np.nan)
    ok_diff = ok_old & np.isfinite(diff)

    fig = plt.figure(figsize=(11.8, 8.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.5, 1.0], hspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    axd = fig.add_subplot(gs[1, 0], sharex=ax)

    ax.plot(np.where(ok_glob, sst_glob, np.nan), lat_glob, color="#111111", lw=2.0, label="全球（90°S–90°N）")
    ax.plot(np.where(ok_old, sst_old, np.nan), lat_old, color="#E67E22", lw=2.2,
            label="印度洋（旧结果）")
    ax.set_ylabel("纬度")
    ax.yaxis.set_major_formatter(FuncFormatter(lat_formatter))
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.set_ylim(LAT_MIN - 1.0, LAT_MAX + 1.0)
    ax.grid(True, ls=":", lw=0.8, alpha=0.55)
    ax.set_axisbelow(True)
    ax.axvline(ISOTHERM_LEVEL, color="0.5", ls=":", lw=0.9)
    subtitle = "合成全球曲线（fixture）" if ctx["synthetic"] else "全球 2016–2020 气候态"
    ax.set_title(f"印度洋（旧结果）与全球年均 SST 廓线对照\n（{subtitle}；印度洋为 20°E–120°E/40°S–30°N）", fontsize=12.5)
    ax.legend(loc="lower right", framealpha=0.93, fontsize=9)
    ax.tick_params(labelbottom=False)

    axd.axhline(0.0, color="0.5", lw=0.9)
    axd.plot(np.where(ok_diff, diff, np.nan), lat_old, color="#7D3C98", lw=1.8)
    axd.fill_betweenx(lat_old, 0, np.where(ok_diff, diff, np.nan), color="#BB8FCE", alpha=0.3, lw=0)
    axd.set_ylabel("差值（°C）", fontsize=9)
    axd.set_xlabel("纬度")
    axd.xaxis.set_major_formatter(FuncFormatter(lat_formatter))
    axd.set_xlim(LAT_MIN - 1.0, LAT_MAX + 1.0)
    axd.set_xticks(np.arange(-90, 91, 30))
    axd.grid(True, ls=":", lw=0.8, alpha=0.55)
    axd.set_axisbelow(True)
    axd.set_title("印度洋 − 全球（重叠纬度，正值 = 印度洋更暖）", fontsize=9.5, pad=4)

    fig.text(
        0.01,
        0.005,
        "注：旧印度洋结果来自仓库已提交的 data/zonal_mean.csv（281 行，0.25°），其 coverage=valid/401 "
        "即海洋占比；全球曲线覆盖 90°S–90°N。差值只在两者均有有效值且海洋占比 ≥ "
        f"{cov_min:.0%} 的纬度计算。",
        fontsize=8.3,
        color="0.35",
        va="bottom",
    )
    return finalize(fig, ctx, "indian_vs_global.png")


# --------------------------------------------------------------------------- #
# 印度洋回归自检（证明全球版没有改变旧口径）
# --------------------------------------------------------------------------- #
def indian_regression_check() -> dict:
    if not INDIAN_CSV.exists():
        raise SystemExit(f"缺少 {INDIAN_CSV}，无法做印度洋回归自检")
    df = pd.read_csv(INDIAN_CSV)
    lat = df["latitude"].to_numpy(dtype=float)
    ocean_frac = zonal_ocean_fraction(df, N_LON_INDIAN)
    annual = df["sst_annual"].to_numpy(dtype=float)
    usable = np.isfinite(annual) & np.isfinite(ocean_frac) & (ocean_frac >= DEFAULT_COVERAGE_MIN)

    i_max = int(np.nanargmax(np.where(usable, annual, -np.inf)))
    isos = isotherm_crossings(lat[usable], annual[usable])
    iso_south = min(isos) if isos else float("nan")
    mean = cos_weighted_mean(lat, annual, ocean_frac, DEFAULT_COVERAGE_MIN)
    measured = {
        "sst_max": float(annual[i_max]),
        "lat_max": float(abs(lat[i_max])) * (1 if lat[i_max] >= 0 else -1),
        "isotherm_south": iso_south,
        "basin_mean": float(mean) if mean is not None else float("nan"),
    }
    print("印度洋回归自检（data/zonal_mean.csv，281 行）：", flush=True)
    failures: list[str] = []
    for key, (expected, tol) in INDIAN_BASELINE.items():
        got = measured[key]
        ok = np.isfinite(got) and abs(got - expected) <= tol
        if not ok:
            failures.append(f"{key}: 实测 {got:.4f} vs README {expected}")
        print(f"  {key:<15s} 实测 {got:8.3f}   期望 {expected:8.3f} ± {tol}   {'OK' if ok else 'FAIL'}",
              flush=True)
    print(f"  纬度=2.12°N、25 °C 等温线 21.66°S、海盆均值 26.16 °C 三项口径未变。", flush=True)
    if failures:
        raise SystemExit("印度洋回归失败：\n  " + "\n  ".join(failures))
    return measured


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
FIGURES = {
    "profile": plot_global_profile,
    "basins": plot_basins,
    "map": plot_map,
    "heatmap": plot_heatmap,
    "indian": plot_indian_vs_global,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="绘制全球 SST–纬度 图件（真实数据缺失时自动用 fixture）")
    parser.add_argument("--coverage-min", type=float, default=DEFAULT_COVERAGE_MIN,
                        help="绘图覆盖率阈值（默认 0.10）")
    parser.add_argument("--fixture", action="store_true", help="强制使用 .tmp/fixture 合成数据")
    parser.add_argument("--refresh-fixture", action="store_true", help="重新生成合成 fixture")
    parser.add_argument("--figures", default="", help="只画指定图，逗号分隔：profile,basins,map,heatmap,indian")
    args = parser.parse_args()

    cov_min = float(args.coverage_min)
    if not 0.0 < cov_min <= 1.0:
        raise SystemExit("--coverage-min 必须在 (0, 1] 内")

    setup_chinese_font()
    ctx = resolve_inputs(args.fixture, args.refresh_fixture)
    indian_regression_check()

    wanted = [k.strip() for k in args.figures.split(",") if k.strip()] or list(FIGURES)
    unknown = [k for k in wanted if k not in FIGURES]
    if unknown:
        raise SystemExit(f"未知图名：{unknown}，可选 {list(FIGURES)}")
    print(f"输出目录：{ctx['out_dir']}", flush=True)
    for key in wanted:
        FIGURES[key](ctx, cov_min)
    print(f"完成：{len(wanted)} 张图（模式：{ctx['mode']}）。", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
