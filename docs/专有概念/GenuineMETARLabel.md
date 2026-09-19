# Genuine METAR thunderstorm label

The supervised target is **whether VOTV’s METAR present-weather group reported a thunderstorm in a clock hour**, not a reanalysis weather code and not a precipitation proxy.

## 什么是该标签？

Decoded from Iowa Environmental Mesonet METAR archives (`dataset/raw_metar/`). Built by `dataset/build_thunderstorm_labels.py`.

**关键特征**

- Observation-based, not synthetic
- Unobserved hours stay `NaN` with `*_observed = 0` — never filled as 0
- Open-Meteo `weather_code` is **not** the label

README modelled window 2014-01-01T12:00Z → 2025-12-30T23:00Z: 82,018 hourly rows, 3,694 thunderstorm hours (~4.5%).

## 代码位置

| 方面 | 位置 |
|------|------|
| 构建 | `dataset/build_thunderstorm_labels.py` |
| 表 | `dataset/historical_thunderstorm_labels_votv.csv` |
| 元数据 | `historical_thunderstorm_labels_votv_metadata.json` |
| 说明 | `dataset/HISTORICAL_THUNDERSTORM_LABELS_README.md` |

## 不变量

1. **No imputation of negatives** for missing present weather.
2. **Clock-hour grid** after Phase 3 sync so a one-row shift is exactly one hour.
