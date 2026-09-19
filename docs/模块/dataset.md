# dataset/

Offline data pipeline: raw archives, labels, hourly sync, features, nowcast targets, and phase verifiers.

## 结构

```
dataset/
├── raw_metar/                 # yearly IEM METAR CSVs + combined raw
├── raw_openmeteo/             # yearly + hourly 2014–2025 archive
├── build_thunderstorm_labels.py
├── download_votv_metar_iem.py
├── fetch_votv_openmeteo_phase3.py
├── build_synchronized_dataset_phase3.py
├── feature_engineering_phase4.py
├── nowcast_targets_phase5.py
├── verify_*.py
├── votv_thunderstorm_*.csv    # synchronized / features / nowcast tables
├── historical_thunderstorm_labels_votv.csv
├── weather_data.csv, weather_data_with_code.csv   # older Phase 1 tables
├── _tmp_*.py, _probe_*.py     # scratch probes
└── data_collection.ipynb
```

## 关键产物

| 文件 | 角色 |
|------|------|
| `votv_thunderstorm_synchronized_2014_2025.csv` | clock-hour ruler + labels |
| `votv_thunderstorm_features_2014_2025.csv` | 28 features |
| `votv_thunderstorm_nowcast_2014_2025.csv` | targets 1h/2h/3h |

## 依赖

Open-Meteo archive, IEM METAR. Consumed by `ml/train_thunderstorm_nowcast_phase6.py` and live `feature_engineering_phase4`.
