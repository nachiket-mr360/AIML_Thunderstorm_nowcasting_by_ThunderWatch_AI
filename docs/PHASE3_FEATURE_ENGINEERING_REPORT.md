# Phase 3 — Multi-location causal feature engineering

**Date:** 2026-09-19  
**Repo:** V2 `C:\College\SIH2_v2`  
**V1 files:** not modified (hash check: True)  
**Flask / frontend / models:** not modified  
**Git:** no commit, no push  
**Training:** not run

---

## PHASE STATUS: **READY**

Causal probes passed. Missing METAR targets remain NA. No weather_code features.

---

## Design

Features at hour **T** use only Open-Meteo values at **T and earlier** on each station’s contiguous UTC hourly grid. Lags and rolling windows are **not** evaluated on a table compacted to observed labels (that would jump across missing-METAR hours).

| Rule | Applied |
|------|---------|
| No future predictors | `shift(h)` and trailing `rolling(h)` only |
| No future / current target as feature | `thunderstorm_target` copied, never used in formulas |
| No weather_code | refused if present |
| Missing labels | kept as NA; not filled with 0 |
| Lookback | first **24** hours per station dropped (max lag/roll), not imputed |
| Station isolation | features computed independently per ICAO |

Numeric feature count: **73** (V1 had 28). Identity columns (`station_id`, `city`, timestamps, target, `label_status`) are extra.

---

## Feature groups

| Group | Columns | Formula sketch |
|-------|---------|----------------|
| Current state | 6 | Open-Meteo hour T (raw wind direction not a model column) |
| Cyclic time | 4 | hour/month sin/cos (V1 identical) |
| Wind vector | 4 | dir sin/cos; u,v = −speed·sin/cos(dir) (met. FROM) |
| Lags | 30 | 6 variables × {1,3,6,12,24} h |
| Tendencies | 12 | Δ at 1h and 3h including cloud cover |
| Rolling | 14 | precip sum 3/6/12/24h; means 3h and 6h |
| Location | 3 | IEM lat, lon, elevation_m |

Explicit lag_1h/lag_3h coexist with change_1h/change_3h (change = current − lag). Both are kept so V1’s 28-feature set is a strict subset of formulas.

Full catalog: `dataset/multilocation/features/multilocation_features_2014_2025_metadata.json`.

---

## Per-station validation

| Station | Input rows | Output rows | Features | Lost (lookback) | Feature NaNs | Dup ts | Sorted | TS+ | Obs − | Unavailable | Causal probe | First valid T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VOTV | 105192 | 105168 | 73 | 24 | 0 | 0 | True | 3694 | 78313 | 23161 | PASS | 2014-01-02T00:00:00+00:00 |
| VECC | 105192 | 105168 | 73 | 24 | 0 | 0 | True | 3755 | 100282 | 1131 | PASS | 2014-01-02T00:00:00+00:00 |
| VIDP | 105192 | 105168 | 73 | 24 | 0 | 0 | True | 2031 | 99430 | 3707 | PASS | 2014-01-02T00:00:00+00:00 |
| VOCI | 105192 | 105168 | 73 | 24 | 0 | 0 | True | 2327 | 90179 | 12662 | PASS | 2014-01-02T00:00:00+00:00 |
| VABB | 105192 | 105168 | 73 | 24 | 0 | 0 | True | 1534 | 99100 | 4534 | PASS | 2014-01-02T00:00:00+00:00 |

First valid feature timestamp is **start + 24h** (`2014-01-01 00:00Z` → `2014-01-02 00:00Z`) because `lag_24h` and `precipitation_roll_sum_24h` need 24 hours of history. That is determined only by lookback, not by labels.

Pooled rows: **525840**. Unique `(station_id, timestamp_utc)`: **True**. Features computed within station only.

VOTV TS+ remains **3694** (the 24 dropped hours were all non-positive in this window). Unavailable labels are retained, not zero-filled.

---

## V1 28-feature comparison

V1 Phase 4 columns reproduced with the same formulas: current 6, cyclic 4, wind sin/cos, 1h/3h changes for T/RH/P/wind/precip, precip roll sum 3h/6h, 3h means for humidity/pressure/temperature/wind_speed.

V2 additions: lags 1–24h, cloud-cover change and roll, 6h/12h/24h rolling precip, 6h means, meteorological u/v, IEM location.

V1 dropped unsupervised hours and only needed 6h history (3 supervised rows dropped at the start of 2014). V2 **keeps unavailable targets** and drops **24** hours of each station grid.

Value overlap vs `dataset/votv_thunderstorm_features_2014_2025.csv`: **MATCH** (82007 overlapping timestamps; 0 mismatches at atol 1e-6 on all 28 V1 columns). V1 files were not rewritten.

---

## Artifacts

| Path | Role |
|------|------|
| `dataset/multilocation/feature_engineering_phase3.py` | Reusable builder |
| `dataset/multilocation/features/<icao>_features_2014_2025.csv` | Per station |
| `dataset/multilocation/features/multilocation_features_2014_2025.csv` | Pooled |
| `dataset/multilocation/features/multilocation_features_2014_2025_metadata.json` | Feature catalog + counts |

SHA256 pooled: `5e9af9245523b7b42296891f2a11f4571edb8758dcfad0056e88ff9b48c1e9a4`

---

## Not done

- No training / metrics
- No Flask/frontend
- No radar/satellite/lightning/NWP
- No V1 edits
- No git commit/push
- No t+1/t+2/t+3 targets
