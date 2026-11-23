# -*- coding: utf-8 -*-
"""scripts/verify.py 的关键逻辑单元测试（不依赖真实全球数据，不 import compute.py）。

运行::

    D:/PythonProject1/.venv/Scripts/python.exe -m unittest discover -s tests -v
"""

from __future__ import annotations

import gzip
import shutil
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import verify  # noqa: E402


def scratch_dir(name: str) -> Path:
    """测试用临时目录放在仓库 .tmp/ 下（沙箱允许写；不入库）。"""
    path = ROOT / ".tmp" / "test_verify" / name
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    return path


class TestIsotherm(unittest.TestCase):
    def test_both_hemispheres(self) -> None:
        lat = np.arange(-89.875, 90.0, 0.25)
        sst = -1.8 + 30.8 * np.cos(np.deg2rad(lat)) ** 2
        crossings = verify.isotherm_crossings(lat, sst, 25.0)
        self.assertGreaterEqual(len(crossings), 2)
        self.assertTrue(any(c > 0 for c in crossings), crossings)
        self.assertTrue(any(c < 0 for c in crossings), crossings)

    def test_no_false_crossing_on_flat_segment(self) -> None:
        lat = np.array([0.0, 1.0, 2.0])
        sst = np.array([10.0, 10.0, 10.0])
        self.assertEqual(verify.isotherm_crossings(lat, sst, 10.0), [])


class TestIndianRegression(unittest.TestCase):
    def test_readme_numbers(self) -> None:
        result = verify.check_indian_regression(verify.INDIAN_CSV)
        if not verify.INDIAN_CSV.exists():
            self.skipTest("仓库未提交 data/zonal_mean.csv")
        self.assertEqual(result["status"], "pass", result["detail"])
        measured = result["metrics"]["measured"]
        self.assertAlmostEqual(measured["sst_max"], 29.14, delta=0.01)
        self.assertAlmostEqual(measured["lat_max"], 2.125, delta=0.01)
        self.assertAlmostEqual(measured["isotherm_south"], -21.664, delta=0.01)
        self.assertAlmostEqual(measured["basin_mean"], 26.16, delta=0.01)


def _write_tiny_raw(raw_dir: Path, lats: np.ndarray, lons: np.ndarray) -> None:
    """写 12 个月 × 1 年、可解析的 ERDDAP 风格 CSV（很小，只用于测试重算逻辑）。"""
    for month in range(1, 13):
        rows = []
        for lat in lats:
            for i, lon in enumerate(lons):
                rows.append(
                    {
                        "time": f"2016-{month:02d}-16T12:00:00Z",
                        "zlev": 0.0,
                        "latitude": lat,
                        "longitude": lon,
                        "sst": float(month) + 0.1 * i,
                    }
                )
        df = pd.DataFrame(rows)
        path = raw_dir / f"sst_2016_{month:02d}.csv.gz"
        with gzip.open(path, "wt", encoding="utf-8", newline="") as fh:
            fh.write("time,zlev,latitude,longitude,sst\n")
            fh.write("UTC,m,degrees_north,degrees_east,degree_Celsius\n")
            fh.write(df.to_csv(index=False, header=False))


class TestRawRecompute(unittest.TestCase):
    def test_tiny_grid_roundtrip(self) -> None:
        lats = np.array([-0.375, -0.125, 0.125, 0.375])
        lons = np.array([0.125, 0.375, 0.625, 0.875])
        raw_dir = scratch_dir("raw") / "raw"
        raw_dir.mkdir()
        _write_tiny_raw(raw_dir, lats, lons)
        files = verify.list_raw_files(raw_dir)
        self.assertEqual(len(files), 12)
        lat, clim, annual, counts = verify.recompute_zonal_from_raw(files)
        np.testing.assert_allclose(lat, lats)
        # 第 1 月每个纬圈的期望均值 = 1 + 0.1 * mean(0..3) = 1.15
        np.testing.assert_allclose(clim[0], np.full(4, 1.15), atol=1e-12)
        np.testing.assert_allclose(annual, np.full(4, 6.5 + 0.15), atol=1e-12)
        np.testing.assert_allclose(counts[0], np.full(4, 4.0))
        self.assertEqual(counts.shape, (12, 4))

    def test_full_60_timestep_check_passes(self) -> None:
        """用 60 个「720 纬圈 × 1 经度」的小文件跑通 check_raw_recompute 全流程。"""
        lats = np.round(verify.LAT_FIRST + verify.LAT_STEP * np.arange(verify.N_LAT_GLOBAL), 3)
        annual = 27.0 - 0.15 * np.abs(lats)
        season = 8.0 * np.sin(np.deg2rad(lats))
        monthly = np.empty((12, len(lats)))
        for m in range(1, 13):
            monthly[m - 1] = annual + season * np.cos(2 * np.pi * (m - 8) / 12.0)

        raw_dir = scratch_dir("raw60") / "raw"
        raw_dir.mkdir()
        for year in range(2016, 2021):
            for month in range(1, 13):
                df = pd.DataFrame(
                    {
                        "time": f"{year}-{month:02d}-16T12:00:00Z",
                        "zlev": 0.0,
                        "latitude": lats,
                        "longitude": 0.125,
                        "sst": monthly[month - 1],
                    }
                )
                path = raw_dir / f"sst_{year}_{month:02d}.csv.gz"
                with gzip.open(path, "wt", encoding="utf-8", newline="") as fh:
                    fh.write("time,zlev,latitude,longitude,sst\n")
                    fh.write("UTC,m,degrees_north,degrees_east,degree_Celsius\n")
                    fh.write(df.to_csv(index=False, header=False))

        target = pd.DataFrame(
            {
                "latitude": lats,
                "valid_count": np.full(len(lats), 1.0),
                "coverage": np.ones(len(lats)),
                "sst_annual": monthly.mean(axis=0),
                "sst_jan": monthly[0],
                "sst_jul": monthly[6],
            }
        )
        for m in range(1, 13):
            target[f"sst_{m:02d}"] = monthly[m - 1]
        target = target[verify.REQUIRED_COLUMNS]

        self.assertEqual(len(verify.list_raw_files(raw_dir)), 60)
        result = verify.check_raw_recompute(raw_dir, target, "test::global")
        self.assertEqual(result["status"], "pass", result["detail"])
        self.assertLessEqual(result["metrics"]["max_abs_diff"], verify.NUMERIC_TOL)



class TestStructuralChecks(unittest.TestCase):
    @staticmethod
    def _frame() -> pd.DataFrame:
        lat = np.round(verify.LAT_FIRST + verify.LAT_STEP * np.arange(verify.N_LAT_GLOBAL), 3)
        annual = 27.0 - 0.15 * np.abs(lat)
        season = 8.0 * np.sin(np.deg2rad(lat))
        df = pd.DataFrame(
            {
                "latitude": lat,
                "valid_count": np.full(len(lat), 1440.0),
                "coverage": np.ones(len(lat)),
                "sst_annual": annual,
                "sst_jan": annual - season,
                "sst_jul": annual + season,
            }
        )
        for m in range(1, 13):
            df[f"sst_{m:02d}"] = annual
        return df[verify.REQUIRED_COLUMNS]

    @staticmethod
    def _new_frame() -> pd.DataFrame:
        """最终契约格式：coverage=valid/ocean_grid_count，zonal_ocean_frac=valid/1440，cov_MM 同 NaN。"""
        df = TestStructuralChecks._frame()
        ocean_count, valid = 800.0, 800.0
        df["ocean_grid_count"] = ocean_count
        df["valid_count"] = valid
        df["coverage"] = valid / ocean_count
        df["zonal_ocean_frac"] = valid / verify.N_LON_GLOBAL
        for m in range(1, 13):
            df[f"cov_{m:02d}"] = valid / ocean_count
        return df

    def test_valid_global_csv_passes(self) -> None:
        path = scratch_dir("valid") / "zonal_mean_global.csv"
        self._frame().to_csv(path, index=False)
        result = verify.check_global_zonal(path)
        self.assertEqual(result["status"], "pass", result["detail"])

    def test_new_contract_columns_pass(self) -> None:
        path = scratch_dir("newcols") / "zonal_mean_global.csv"
        self._new_frame().to_csv(path, index=False)
        result = verify.check_global_zonal(path)
        self.assertEqual(result["status"], "pass", result["detail"])
        self.assertEqual(result["metrics"]["coverage_denominator"], "ocean_grid_count")

    def test_coverage_denominator_mismatch_fails(self) -> None:
        df = self._new_frame()
        df.loc[10, "coverage"] = 0.9  # 应为 valid/ocean_grid_count = 1.0
        path = scratch_dir("badcov") / "zonal_mean_global.csv"
        df.to_csv(path, index=False)
        result = verify.check_global_zonal(path)
        self.assertEqual(result["status"], "fail")
        self.assertIn("coverage", result["detail"])

    def test_zonal_ocean_frac_mismatch_fails(self) -> None:
        df = self._new_frame()
        df.loc[10, "zonal_ocean_frac"] = 0.5  # 应为 800/1440
        path = scratch_dir("badfrac") / "zonal_mean_global.csv"
        df.to_csv(path, index=False)
        result = verify.check_global_zonal(path)
        self.assertEqual(result["status"], "fail")
        self.assertIn("zonal_ocean_frac", result["detail"])

    def test_cov_sst_nan_pattern_mismatch_fails(self) -> None:
        df = self._new_frame()
        df.loc[10, "cov_01"] = np.nan  # sst_01 仍为有限值
        path = scratch_dir("badnan") / "zonal_mean_global.csv"
        df.to_csv(path, index=False)
        result = verify.check_global_zonal(path)
        self.assertEqual(result["status"], "fail")
        self.assertIn("NaN 模式", result["detail"])

    def test_broken_coverage_fails(self) -> None:
        df = self._frame()
        df.loc[0, "coverage"] = 1.5
        path = scratch_dir("broken") / "zonal_mean_global.csv"
        df.to_csv(path, index=False)
        result = verify.check_global_zonal(path)
        self.assertEqual(result["status"], "fail")
        self.assertIn("coverage", result["detail"])


if __name__ == "__main__":
    unittest.main()
