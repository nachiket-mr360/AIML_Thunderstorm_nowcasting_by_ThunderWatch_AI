# Phase 14A — V2 Feature Builder Contract

**Date:** 2026-09-21  
**Repo:** SIH26072 V2  
**Scope:** historical/replay feature builder only. Not a live operational data pipeline.  
**Training / Flask / dashboard / live APIs:** not implemented.

**PHASE STATUS: PASS**

---

## 1. Objective

Provide a deterministic layer that converts chronological atmospheric + NWP history into the **exact 83 Model B features** required by the Phase 13B `V2InferenceEngine`, in Phase 13A order, using Phase 3 causal formulas.

This phase does **not** claim live data acquisition. Live sources and latency will be contracted later.

---

## 2. Source-of-truth implementation

| Item | Path |
|------|------|
| Feature formulas | `dataset/multilocation/feature_engineering_phase3.py` (`add_features`) |
| Canonical 83 names/order | `outputs/v2_inference/phase13a/inference_contract.json` → `model_b_features` |
| Builder | `src/features/v2_feature_builder.py` |
| Engine (unchanged) | `src/inference/v2_inference_engine.py` |

The builder **does not** hard-code a second 83-name list. It loads the contract. Intermediate columns may exist internally; the returned record is restricted to the 83.

---

## 3. Input contract

`build_features(history, timestamp_utc=None, station_id=None) → FeatureBuildResult`

History must be chronological hourly rows with:

- Identity: `station_id`, `timestamp_utc`
- Atmospheric (Open-Meteo archive units as trained): `temperature_2m`, `relative_humidity_2m`, `surface_pressure` (**Pa**), `wind_speed_10m`, `wind_direction_10m`, `precipitation`, `cloud_cover`
- NWP (as stored; no unit conversion): `nwp_cape`, `nwp_convective_inhibition`, `nwp_lifted_index`, `nwp_temperature_2m`, `nwp_relative_humidity_2m`, `nwp_surface_pressure` (**hPa**), `nwp_wind_speed_10m`, `nwp_wind_direction_10m`, `nwp_precipitation`, `nwp_cloud_cover`
- Location: `latitude`, `longitude`, `elevation_m`

Minimum causal context: **25 hourly rows** covering **T−24h … T** inclusive.

If `timestamp_utc` is omitted, T is the last row. Rows after T may be present; they are ignored.

---

## 4. Feature groups

| Group | Role |
|-------|------|
| Current atmospheric | 6 raw state fields at T (raw wind direction is encoded, not a model column) |
| Cyclic time | hour/month sin/cos |
| Wind vector | dir sin/cos; meteorological u/v |
| Lags | T−1, T−3, T−6, T−12, T−24 |
| Tendencies | T minus T−1; T minus T−3 |
| Rolling | trailing windows ending at T (current hour included) |
| Location | lat, lon, elevation |
| NWP | 10 fields valid at T |

Total: **83**.

---

## 5. Causal rules

For prediction timestamp **T**:

- Observations and NWP used as predictors are at **T or earlier**.
- Lags: T−1 / T−3 / T−6 / T−12 / T−24.
- Tendencies: T − T−1 and T − T−3.
- Rolling: windows ending at T.
- NWP: valid at T (Phase 8B `nwp_time_shift=false`).
- **Never** T+1 / T+2 / T+3 as predictors.

Tests mutate T+1…T+3 and assert the vector at T is unchanged.

---

## 6. NWP handling

NWP columns are copied from the T row **as stored**. No unit conversion.

Phase 13A: observed `surface_pressure` = **Pa**; `nwp_surface_pressure` = **hPa**.

Missing/non-finite NWP at T → UNAVAILABLE. No imputation.

---

## 7. Output contract

`FeatureBuildResult`:

- `status`: `COMPLETE` or `UNAVAILABLE`
- `reason`: documented fail-closed code when unavailable
- `features`: `dict` keyed in contract order, only on COMPLETE
- `feature_names`: 83 contract names
- Compatible with `V2InferenceEngine.predict(station_id, timestamp_utc, features, feature_names=…)`

---

## 8. Fail-closed rules

No fill of missing lags/rolls. No silent repair.

UNAVAILABLE when:

- fewer than 24 hours of causal history (need 25 rows T−24…T)
- missing required atmospheric observation
- missing required NWP at T
- duplicate timestamp
- non-monotonic timestamp
- non-hourly grid
- invalid/unknown station
- missing location
- NaN/inf in required source or derived 83

---

## 9. Integration comparison

Station **VOTV**, timestamp **2021-04-09T08:00:00+00:00**.

History: Phase 2B synchronized atmospheric hours + Phase 8B NWP overlap fields.  
Reference: corresponding row of `dataset/multilocation/features_nwp/multilocation_features_nwp_overlap_2021_2025.csv`.

| Check | Result |
|-------|--------|
| Names/order | exact 83, Phase 13A |
| Numeric | max abs difference **1.14e-13** (atol **1e-8**) |
| Mismatched columns | none |

Sample: `outputs/v2_inference/phase14a/feature_builder_sample.csv`.

---

## 10. Test results

`python -m pytest tests/test_v2_feature_builder_phase14a.py -q` → **11 passed**.

Covered: 83/order, Phase 3 formulas, lag/tendency/rolling causality, future-row invariance, insufficient history, missing NWP, duplicate ts, non-monotonic ts, engine compatibility, integration.

---

## 11. Limitations

- Replay/historical builder only; not live ingest.
- Requires contiguous UTC hourly history; gaps fail closed.
- Complete-case NWP (same as Model B training).
- Does not fetch satellite, radar, or lightning.
- Frozen Phase 8D models were not retrained.

---

## Model protection (SHA-256)

Unchanged before vs after validation (`integrity_unchanged: true`):

| Artifact | SHA-256 |
|----------|---------|
| 1h joblib | `a266c71bf901c4ea9392882b619e1a40905279af89db58123227a07f49733f21` |
| 2h joblib | `e99ff3e9fba4cef8a3e1d15e6c29cfdbee801200369230bd652a45a71b376c6c` |
| 3h joblib | `20692e504594ad17ea6fcc43dfe6e8899a5d5bdc2df93015df92327b97dc16ed` |
| Phase 13A contract | `a3aad93b211aff6a2da9b6a5e859c9649928a60ea2667f9888d854168d884940` |
| Phase 13B engine | `ef9c55e395df5dec1aac646ac447c96d2c449888930640c3303f5cfa43a6f3d3` |
| Phase 13C samples | `6d014924354a46cd068d1c1a4edb7820dca00f70cbc581a01913d66ec336fdb7` |
| Phase 13C validation | `edf55a4c04f449f23f2efa036368c02bdaa97619059342062227d5b5134022c9` |

---

## Not done (stop)

- No Flask
- No dashboard
- No live APIs
- No Phase 14B
