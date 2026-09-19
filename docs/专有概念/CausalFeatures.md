# 28 causal features

Live scoring and training use the **same** 28 features of hours **≤ t** only. No future atmosphere, no target leak, no `weather_code`.

## Groups (from README / Phase 4)

| Group | Count | Contents |
|-------|-------|----------|
| Current state | 6 | temperature_2m, relative_humidity_2m, surface_pressure, wind_speed_10m, precipitation, cloud_cover |
| Time cyclic | 4 | hour_sin/cos, month_sin/cos |
| Wind dir cyclic | 2 | wind_direction_sin/cos |
| 1h and 3h deltas | 10 | T, RH, P, wind, precip |
| Rolling 3h/6h | 6 | precip acc, mean RH/P/T/wind |

Live path requires **6 preceding contiguous hours**.

## 代码位置

| 方面 | 位置 |
|------|------|
| 实现 | `dataset/feature_engineering_phase4.py` |
| 训练表 | `dataset/votv_thunderstorm_features_2014_2025.csv` |
| 实时 | `ml/predict_thunderstorm_nowcast.py` `build_feature_row` |
| 校验 | `dataset/verify_features_phase4.py` |
