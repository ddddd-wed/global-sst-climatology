#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""纯 NumPy 构建全球 0.25° 五大洋互斥掩膜并执行地理关键点校验。

坐标系约定：
- 纬度：-89.875 ~ 89.875（步长 0.25°，共 720 行）
- 经度：0.125 ~ 359.875（步长 0.25°，0-360° 东经体系，共 1440 列）

掩膜 ID（BASIN_IDS）：
- 0: land（陆地及被剔除的封闭内陆湖泊：里海、北美五大湖、拉多加湖）
- 1: pacific（太平洋）
- 2: atlantic（大西洋，含地中海、黑海等海洋边缘海）
- 3: indian（印度洋，北界止于波斯湾，含红海/波斯湾）
- 4: arctic（北冰洋，lat >= 66.5°）
- 5: southern（南大洋，lat < -60.0°）
"""

from __future__ import annotations

import numpy as np

from config import BASIN_IDS, BASINS


# 封闭内陆湖泊框：(键名, 中文名, lat_min, lat_max, lon_min, lon_max)
INLAND_LAKE_BOXES: tuple[tuple[str, str, float, float, float, float], ...] = (
    ("caspian_sea", "里海", 36.0, 48.0, 46.0, 56.0),
    ("great_lakes", "北美五大湖", 40.5, 50.0, 266.5, 286.0),
    ("lake_ladoga", "拉多加湖", 59.5, 62.0, 30.0, 33.2),
)


def inland_lake_mask(
    latitude: np.ndarray,
    longitude: np.ndarray,
    ocean_valid_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    """识别 OISST 格点中的封闭内陆湖泊，返回 (布尔掩膜, 格点数明细)。"""
    lat = np.asarray(latitude, dtype=np.float64)
    lon = np.asarray(longitude, dtype=np.float64)
    ocean = np.asarray(ocean_valid_mask, dtype=bool)
    lat_2d, lon_2d = np.meshgrid(lat, lon, indexing="ij")

    inland = np.zeros(ocean.shape, dtype=bool)
    breakdown: dict[str, int] = {}
    for key, _zh, lat0, lat1, lon0, lon1 in INLAND_LAKE_BOXES:
        box = ocean & (lat_2d >= lat0) & (lat_2d <= lat1) & (lon_2d >= lon0) & (lon_2d <= lon1)
        breakdown[key] = int(np.sum(box))
        inland |= box
    return inland, breakdown


def indian_pacific_boundary(lat: np.ndarray) -> np.ndarray:
    """计算印度洋与太平洋分界经线（0-360° 东经）。

    - -60.0 <= lat < -8.0 : lon = 120.0（与旧版印度洋东界一致）
    - -8.0 <= lat < -6.0  : (-8.0, 120.0) -> (-6.0, 105.5) 线性过渡
    - -6.0 <= lat < 1.5   : (-6.0, 105.5) -> (1.5, 103.5) 沿爪哇-苏门答腊线性过渡
    - 1.5 <= lat < 66.5   : lon = 100.5（经马来半岛切分，安达曼海归印度洋、南海归太平洋）
    """
    lat_arr = np.asarray(lat, dtype=np.float64)
    lon = np.full_like(lat_arr, 100.5, dtype=np.float64)

    m1 = lat_arr < -8.0
    m2 = (lat_arr >= -8.0) & (lat_arr < -6.0)
    m3 = (lat_arr >= -6.0) & (lat_arr < 1.5)

    lon[m1] = 120.0
    lon[m2] = 120.0 + (lat_arr[m2] - (-8.0)) * (105.5 - 120.0) / (-6.0 - (-8.0))
    lon[m3] = 105.5 + (lat_arr[m3] - (-6.0)) * (103.5 - 105.5) / (1.5 - (-6.0))
    return lon


def indian_atlantic_boundary(lat: np.ndarray) -> np.ndarray:
    """计算印度洋与大西洋分界经线（0-360° 东经）。

    - -60.0 <= lat < 30.0 : lon = 20.0（厄加勒斯角经线，红海/波斯湾归印度洋）
    - 30.0 <= lat < 66.5  : lon = 45.0（将地中海东段与黑海 20–45°E 归入大西洋）
    """
    lat_arr = np.asarray(lat, dtype=np.float64)
    return np.where(lat_arr < 30.0, 20.0, 45.0)


def pacific_atlantic_boundary(lat: np.ndarray) -> np.ndarray:
    """计算太平洋与大西洋的美洲分水岭分界经线（0-360° 东经）。

    - -60.0 <= lat < -55.0 : lon = 292.75（67.25°W，德雷克海峡/合恩角）
    - -55.0 <= lat < 8.5   : (-55.0, 290.0) -> (8.5, 280.0) 沿安第斯山脉内部
    - 8.5 <= lat < 19.0    : (8.5, 280.0) -> (19.0, 261.0) 沿中美洲陆桥至墨西哥
    - 19.0 <= lat < 66.5   : lon = 258.0（102°W，经北美大陆内部，墨西哥湾/加勒比海归大西洋）
    """
    lat_arr = np.asarray(lat, dtype=np.float64)
    lon = np.full_like(lat_arr, 258.0, dtype=np.float64)

    m1 = lat_arr < -55.0
    m2 = (lat_arr >= -55.0) & (lat_arr < 8.5)
    m3 = (lat_arr >= 8.5) & (lat_arr < 19.0)

    lon[m1] = 292.75
    lon[m2] = 290.0 + (lat_arr[m2] - (-55.0)) * (280.0 - 290.0) / (8.5 - (-55.0))
    lon[m3] = 280.0 + (lat_arr[m3] - 8.5) * (261.0 - 280.0) / (19.0 - 8.5)
    return lon


def build_basin_mask(
    latitude: np.ndarray,
    longitude: np.ndarray,
    ocean_valid_mask: np.ndarray,
) -> np.ndarray:
    """生成全球 0.25° 五大洋互斥 uint8 掩膜 (n_lat, n_lon)，并剔除封闭内陆湖泊。"""
    lat = np.asarray(latitude, dtype=np.float64)
    lon = np.asarray(longitude, dtype=np.float64)
    raw_ocean = np.asarray(ocean_valid_mask, dtype=bool)
    if raw_ocean.shape != (len(lat), len(lon)):
        raise ValueError(f"ocean_valid_mask 形状 {raw_ocean.shape} 与 ({len(lat)}, {len(lon)}) 不一致")

    inland, _ = inland_lake_mask(lat, lon, raw_ocean)
    ocean = raw_ocean & ~inland

    lat_2d, lon_2d = np.meshgrid(lat, lon, indexing="ij")
    mask = np.zeros(ocean.shape, dtype=np.uint8)

    # 极区两大洋
    southern = ocean & (lat_2d < -60.0)
    arctic = ocean & (lat_2d >= 66.5)
    mask[southern] = BASIN_IDS["southern"]
    mask[arctic] = BASIN_IDS["arctic"]

    # 中低纬三大洋 (-60.0 <= lat < 66.5)
    mid = ocean & (lat_2d >= -60.0) & (lat_2d < 66.5)
    b_ind_atl = indian_atlantic_boundary(lat_2d)
    b_ind_pac = indian_pacific_boundary(lat_2d)
    b_pac_atl = pacific_atlantic_boundary(lat_2d)

    indian = mid & (lon_2d >= b_ind_atl) & (lon_2d < b_ind_pac)
    pacific = mid & (lon_2d >= b_ind_pac) & (lon_2d < b_pac_atl)
    atlantic = mid & ((lon_2d < b_ind_atl) | (lon_2d >= b_pac_atl))

    mask[pacific] = BASIN_IDS["pacific"]
    mask[atlantic] = BASIN_IDS["atlantic"]
    mask[indian] = BASIN_IDS["indian"]

    # 互斥完整性校验：所有连通海洋格点必须且仅属于 1..5 之一，陆地与内陆湖均为 0
    if np.any(mask[ocean] == 0):
        raise AssertionError("存在未被分配海盆的有效海洋格点")
    if np.any(mask[~ocean] != 0):
        raise AssertionError("存在被错误分配海盆的陆地/内陆湖泊格点")
    return mask


# 关键地理坐标抽检点：(名称, 纬度, 经度0-360, 期望海盆key)
KEY_CHECKPOINTS: list[tuple[str, float, float, str]] = [
    ("南海", 15.0, 115.0, "pacific"),
    ("孟加拉湾", 15.0, 88.0, "indian"),
    ("墨西哥湾", 25.0, 270.0, "atlantic"),
    ("东热带太平洋", 10.0, 260.0, "pacific"),
    ("地中海", 35.0, 25.0, "atlantic"),
    ("德雷克海峡", -58.0, 295.0, "atlantic"),
    ("南大洋", -65.0, 150.0, "southern"),
    ("北冰洋", 75.0, 0.125, "arctic"),
    ("里海（内陆湖剔除）", 41.875, 50.125, "land"),
    ("北美五大湖（内陆湖剔除）", 45.125, 274.125, "land"),
    ("拉多加湖（内陆湖剔除）", 60.875, 31.375, "land"),
]


def verify_basin_mask(
    latitude: np.ndarray,
    longitude: np.ndarray,
    basin_mask: np.ndarray,
    ocean_valid_mask: np.ndarray,
) -> list[dict]:
    """对关键地理点、内陆湖泊剔除与海盆格点总和做自检断言，返回抽检明细。"""
    lat = np.asarray(latitude, dtype=np.float64)
    lon = np.asarray(longitude, dtype=np.float64)
    raw_ocean = np.asarray(ocean_valid_mask, dtype=bool)
    inland, _ = inland_lake_mask(lat, lon, raw_ocean)
    effective_ocean = raw_ocean & ~inland

    total_ocean = int(np.sum(effective_ocean))
    basin_sum = int(sum(np.sum(basin_mask == BASIN_IDS[k]) for k in ("pacific", "atlantic", "indian", "arctic", "southern")))
    if basin_sum != total_ocean:
        raise AssertionError(f"五大洋格点之和 ({basin_sum}) != 全球有效海洋格点数 ({total_ocean})")
    if np.any(basin_mask[inland] != 0):
        raise AssertionError("存在未被设为 0 的内陆湖泊格点")

    results: list[dict] = []
    for name, target_lat, target_lon, expected_key in KEY_CHECKPOINTS:
        i = int(np.argmin(np.abs(lat - target_lat)))
        j = int(np.argmin(np.abs(lon - (target_lon % 360.0))))
        actual_id = int(basin_mask[i, j])
        actual_key = BASINS[actual_id]["key"]
        if expected_key != "land" and not effective_ocean[i, j]:
            raise AssertionError(f"抽检点 {name} ({lat[i]:.3f}, {lon[j]:.3f}) 被判定为陆地")
        if actual_key != expected_key:
            raise AssertionError(
                f"抽检点 {name} ({lat[i]:.3f}, {lon[j]:.3f}) 归属为 {actual_key}，期望 {expected_key}"
            )
        results.append(
            {
                "name": name,
                "grid_lat": float(lat[i]),
                "grid_lon": float(lon[j]),
                "expected": expected_key,
                "actual": actual_key,
            }
        )
    return results
