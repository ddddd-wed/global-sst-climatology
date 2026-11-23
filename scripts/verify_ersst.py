#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Wave 2 (P3)：ERSST v5 数据与统计独立验收脚本。

验收内容：
1. data/raw_ersst/ 1982–2024 共 43 个年度文件（516 个月）与 manifest.json 完整性校验；
2. 6 份契约产物（ersst_zonal.csv、ersst_clim_1991_2020.csv、ersst_trend.csv、
   ersst_global_series.csv、ersst_vs_oisst_2016_2020.csv、ersst_summary.json）结构与数值自洽；
3. 纯标准库 Student-t 双侧 p-value / 95% 临界值与 OLS 拟合数值自检；
4. 科学基准断言：60°S–60°N 外部交叉校验偏差 < 0.5 °C、1982–2024 全球升温显著（p < 1e-6）、
   2023 或 2024 为最暖年份。
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from compute_ersst import (
    ERSST_CLIM_CSV,
    ERSST_N_LAT,
    ERSST_SERIES_CSV,
    ERSST_SUMMARY_JSON,
    ERSST_TREND_CSV,
    ERSST_VS_OISST_CSV,
    ERSST_YEAR_END,
    ERSST_YEAR_START,
    ERSST_ZONAL_CSV,
    RAW_ERSST_DIR,
    ols_trend,
    student_t_crit_95,
    student_t_two_sided_pvalue,
)


def check_raw_ersst(raw_dir: Path) -> None:
    years = list(range(ERSST_YEAR_START, ERSST_YEAR_END + 1))
    assert len(years) == 43
    manifest_path = raw_dir / "manifest.json"
    assert manifest_path.exists(), f"缺少 {manifest_path}"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest.get("files", [])) == 43, "manifest.json 年度文件数 != 43"

    h1 = b"time,depth,latitude,longitude,sst"
    h2 = b"UTC,m,degrees_north,degrees_east,degree_C"
    total_gz = 0
    for y in years:
        p = raw_dir / f"ersst_{y}.csv.gz"
        assert p.exists(), f"缺少 {p}"
        total_gz += p.stat().st_size
        with gzip.open(p, "rb") as fh:
            payload = fh.read()
        lines = payload.splitlines()
        assert len(lines) == 192_242, f"{p.name} 行数 {len(lines)} != 192242"
        assert lines[0] == h1 and lines[1] == h2, f"{p.name} 表头异常"
        assert lines[2].startswith(f"{y}-01-".encode()), f"{p.name} 首行非 01 月"
        assert lines[-1].startswith(f"{y}-12-".encode()), f"{p.name} 末行非 12 月"

    print(f"[1] data/raw_ersst 校验通过：{len(years)} 年 / {len(years)*12} 个月，gzip={total_gz/1e6:.2f} MB")


def check_math_and_ols() -> None:
    # 解析已知 Student-t 临界值自检（无需 scipy）
    assert abs(student_t_two_sided_pvalue(12.706204736174698, 1) - 0.05) < 1e-8
    assert abs(student_t_two_sided_pvalue(2.019540970, 41) - 0.05) < 1e-6
    assert abs(student_t_crit_95(41) - 2.019540970) < 1e-6
    assert abs(student_t_crit_95(1) - 12.706204736) < 1e-6

    x = np.arange(1982, 2025, dtype=np.float64)
    y_exact = 18.0 + 0.02 * (x - 1982.0)
    res_exact = ols_trend(x, y_exact)
    assert abs(float(res_exact["trend_per_decade"]) - 0.2) < 1e-12
    assert abs(float(res_exact["r2"]) - 1.0) < 1e-12
    assert float(res_exact["p_value"]) == 0.0

    # 带噪声样本自检：t_stat 与 p_value 一致性
    rng = np.random.default_rng(42)
    y_noisy = y_exact + rng.normal(0.0, 0.1, size=len(x))
    res_noisy = ols_trend(x, y_noisy)
    assert 0.0 < float(res_noisy["r2"]) < 1.0
    assert 0.0 <= float(res_noisy["p_value"]) < 1e-6
    assert float(res_noisy["ci95_halfwidth"]) > 0.0
    print("[2] Student-t (Betainc) 与 OLS 数值自检通过（df=1/41 临界值误差 < 1e-6）")


def check_outputs() -> None:
    # 1) ersst_zonal.csv
    df_z = pd.read_csv(ERSST_ZONAL_CSV)
    assert list(df_z.columns) == ["year", "latitude", "sst_annual", "sst_jan", "sst_jul", "valid_count"]
    assert len(df_z) == 43 * ERSST_N_LAT

    # 2) ersst_clim_1991_2020.csv
    df_c = pd.read_csv(ERSST_CLIM_CSV)
    assert list(df_c.columns) == ["latitude", "sst_annual", "sst_jan", "sst_jul", "sst_interannual_std", "valid_count"]
    assert len(df_c) == ERSST_N_LAT

    # 3) ersst_trend.csv
    df_t = pd.read_csv(ERSST_TREND_CSV)
    assert list(df_t.columns) == [
        "latitude",
        "trend_per_decade",
        "n_years",
        "r2",
        "p_value",
        "ci95_halfwidth",
        "sst_mean_1982_2024",
    ]
    assert len(df_t) == ERSST_N_LAT
    valid_t = df_t[df_t["n_years"] >= 30]
    assert len(valid_t) >= 75, f"有效趋势纬度过少：{len(valid_t)}"
    assert np.all(np.isfinite(valid_t["trend_per_decade"]))
    assert ((valid_t["r2"] >= 0.0) & (valid_t["r2"] <= 1.0)).all()
    assert ((valid_t["p_value"] >= 0.0) & (valid_t["p_value"] <= 1.0)).all()
    assert (valid_t["ci95_halfwidth"] > 0.0).all()

    # 4) ersst_global_series.csv
    df_s = pd.read_csv(ERSST_SERIES_CSV)
    assert list(df_s.columns) == [
        "year",
        "global_mean_sst",
        "nh_mean_sst",
        "sh_mean_sst",
        "tropics_mean_sst",
        "arctic_60n_mean_sst",
        "southern_60s_mean_sst",
    ]
    assert len(df_s) == 43
    assert df_s["year"].tolist() == list(range(1982, 2025))
    assert np.all(np.isfinite(df_s.to_numpy(dtype=float)))

    # 5) ersst_vs_oisst_2016_2020.csv
    df_cv = pd.read_csv(ERSST_VS_OISST_CSV)
    assert list(df_cv.columns) == [
        "latitude",
        "ersst_annual",
        "oisst_all_annual",
        "oisst_icefree_annual",
        "diff_all_minus_ersst",
        "diff_icefree_minus_ersst",
    ]
    assert len(df_cv) == ERSST_N_LAT
    m_all = np.isfinite(df_cv["ersst_annual"]) & np.isfinite(df_cv["oisst_all_annual"])
    assert np.allclose(
        df_cv.loc[m_all, "diff_all_minus_ersst"],
        df_cv.loc[m_all, "oisst_all_annual"] - df_cv.loc[m_all, "ersst_annual"],
        atol=5e-4,
    )
    m_if = np.isfinite(df_cv["ersst_annual"]) & np.isfinite(df_cv["oisst_icefree_annual"])
    assert np.allclose(
        df_cv.loc[m_if, "diff_icefree_minus_ersst"],
        df_cv.loc[m_if, "oisst_icefree_annual"] - df_cv.loc[m_if, "ersst_annual"],
        atol=5e-4,
    )

    # 6) ersst_summary.json & 科学合理性断言
    summary = json.loads(ERSST_SUMMARY_JSON.read_text(encoding="utf-8"))
    for k in (
        "dataset",
        "period",
        "clim_period",
        "global_mean_1991_2020_c",
        "global_trend_per_decade_c",
        "arctic_60n_trend_per_decade_c",
        "arctic_amplification_ratio",
        "tropics_trend_per_decade_c",
        "warmest_year",
        "coldest_year",
        "cross_validation_2016_2020",
        "pipeline",
    ):
        assert k in summary, f"ersst_summary.json 缺少 {k}"

    gt = summary["global_trend_per_decade_c"]
    assert gt["trend_per_decade_c"] > 0.0, f"全球趋势非正：{gt}"
    assert gt["p_value"] < 1e-6, f"全球趋势 p-value 未达到 < 1e-6：{gt['p_value']}"
    assert summary["warmest_year"]["year"] in (2023, 2024), f"最暖年份异常：{summary['warmest_year']}"

    cv60 = summary["cross_validation_2016_2020"]["60S_60N"]
    for mode in ("all_ocean", "ice_free"):
        assert abs(cv60[mode]["mean_bias_c"]) < 0.5, f"60S-60N {mode} bias 过大：{cv60[mode]}"
        assert cv60[mode]["rmse_c"] < 0.5, f"60S-60N {mode} RMSE 过大：{cv60[mode]}"
        assert cv60[mode]["pearson_r"] > 0.999, f"60S-60N {mode} r 过低：{cv60[mode]}"

    print(
        f"[3] 6 份 ERSST 契约产物与科学断言全通过：全球趋势 {gt['trend_per_decade_c']:+.4f} °C/decade "
        f"(p={gt['p_value']:.2e}), 最暖年 {summary['warmest_year']['year']}, "
        f"60°S–60°N bias={cv60['all_ocean']['mean_bias_c']:+.4f}°C / RMSE={cv60['all_ocean']['rmse_c']:.4f}°C"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="ERSST v5 (P3) 独立验收脚本")
    parser.add_argument("--raw-dir", type=Path, default=RAW_ERSST_DIR)
    args = parser.parse_args()

    check_raw_ersst(args.raw_dir)
    check_math_and_ols()
    check_outputs()
    print("\nERSST v5 (Wave 2 / P3) 全部验收通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
