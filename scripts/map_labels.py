# -*- coding: utf-8 -*-
"""全球/区域图与交互地图共用的中文地名、洋流标注表（经度 0–360°E）。

tier: 1=全球大字；2=全球+区域；3=仅在区域特写/HTML 放大时显示。
style: ocean/sea/current/cold_current。
"""

OCEAN_LABELS = [
    ("太平洋", 195.0, 18.0, 1, "ocean"),
    ("大西洋", 325.0, 18.0, 1, "ocean"),
    ("印度洋", 75.0, -20.0, 1, "ocean"),
    ("北冰洋", 180.0, 78.0, 1, "ocean"),
    ("南大洋", 180.0, -66.0, 1, "ocean"),
]

SEA_LABELS_T2 = [
    ("南海", 114.0, 14.0, "sea"),
    ("东海", 125.0, 29.0, "sea"),
    ("日本海", 135.0, 40.0, "sea"),
    ("鄂霍次克海", 150.0, 55.0, "sea"),
    ("白令海", 180.0, 57.0, "sea"),
    ("孟加拉湾", 88.0, 15.0, "sea"),
    ("阿拉伯海", 63.0, 18.0, "sea"),
    ("红海", 38.0, 20.0, "sea"),
    ("波斯湾", 51.5, 26.5, "sea"),
    ("地中海", 18.0, 36.0, "sea"),
    ("黑海", 34.0, 43.0, "sea"),
    ("墨西哥湾", 270.0, 25.0, "sea"),
    ("加勒比海", 285.0, 15.0, "sea"),
    ("西太平洋暖池", 148.0, 3.0, "sea"),
    ("东太平洋冷舌", 250.0, -1.0, "sea"),
]

SEA_LABELS_T3 = [
    ("渤海", 119.5, 39.0),
    ("黄海", 123.0, 36.0),
    ("台湾海峡", 119.5, 24.5),
    ("暹罗湾", 101.5, 10.0),
    ("爪哇海", 112.0, -5.5),
    ("苏拉威西海", 122.0, 3.0),
    ("班达海", 128.0, -5.5),
    ("珊瑚海", 155.0, -18.0),
    ("亚丁湾", 48.0, 12.5),
    ("安达曼海", 95.5, 11.0),
    ("爱琴海", 25.0, 38.0),
    ("亚得里亚海", 16.0, 43.0),
    ("北海", 4.0, 56.5),
    ("波罗的海", 19.5, 57.5),
    ("白令海峡", 191.0, 65.8),
]

CURRENT_LABELS = [
    ("黑潮", 141.0, 32.0, "warm"),
    ("湾流", 296.0, 37.0, "warm"),
    ("秘鲁寒流", 278.0, -16.0, "cold"),
    ("南极绕极流", 110.0, -53.0, "cold"),
    ("本格拉寒流", 17.0, -24.0, "cold"),
]


def _display_lon(lon: float, lon_min: float, lon_max: float) -> float:
    if lon_max > 360.0 and lon > 180.0:
        return lon - 360.0
    return lon


def labels_in_bbox(
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    include_tier1: bool = True,
    include_tier2: bool = True,
    include_tier3: bool = True,
    include_currents: bool = True,
    edge_margin: float = 0.0,
) -> list[dict]:
    """返回落在视口/区域内的标注，lon 已按跨 0° 区域转换为连续坐标。

    ``edge_margin`` > 0 时，距视口四边小于该角度（度）的标注不返回，避免字被裁半截。
    """
    out: list[dict] = []

    def add(name, lon, lat, kind, tier):
        x = _display_lon(float(lon), lon_min, lon_max)
        if not (lon_min + edge_margin - 1e-6 <= x <= lon_max - edge_margin + 1e-6):
            return
        if not (lat_min + edge_margin - 1e-6 <= lat <= lat_max - edge_margin + 1e-6):
            return
        out.append({"name": name, "lon": x, "lat": float(lat), "kind": kind, "tier": tier})

    if include_tier1:
        for name, lon, lat, tier, style in OCEAN_LABELS:
            add(name, lon, lat, style, tier)
    if include_tier2:
        for name, lon, lat, style in SEA_LABELS_T2:
            add(name, lon, lat, style, 2)
    if include_tier3:
        for name, lon, lat in SEA_LABELS_T3:
            add(name, lon, lat, "sea", 3)
    if include_currents:
        for name, lon, lat, kind in CURRENT_LABELS:
            add(name, lon, lat, kind, 2)
    return out
