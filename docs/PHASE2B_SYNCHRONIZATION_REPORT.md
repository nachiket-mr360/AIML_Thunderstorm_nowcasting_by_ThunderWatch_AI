# Phase 2B — Multi-location dataset synchronization

**Date:** 2026-09-19  
**Repo:** V2 working copy `C:\College\SIH2_v2`  
**V1 files:** not modified (hash check: True)  
**Flask / frontend / models:** not modified  
**Git:** no commit, no push  
**Training / model metrics:** not run

Predictors are Open-Meteo atmospheric/reanalysis inputs. The thunderstorm target is **only** genuine IEM METAR `wxcodes`. Open-Meteo `weather_code` is not a target and was not joined as a label.

---

## PHASE STATUS: **READY**

No serious synchronization failures reported. Missing labels remain NA.

---

## Scientific rules applied

| Rule | Implementation |
|------|----------------|
| Target source | IEM `wxcodes` present-weather tokens (METAR + SPECI) |
| Positive | Any clock hour with ≥1 TS-family token (`TS`, `TSRA`, `-TSRA`, `VCTS`, …) |
| Observed negative | Hour has ≥1 **decodable** present-weather group and no TS-family token |
| Missing | No METAR in the hour, **or** reports exist but `wxcodes` is empty/`M` → `thunderstorm_target` NA |
| Never | Fill missing hours with 0; use Open-Meteo weather_code as target; t+1/t+2/t+3 targets |
| Join | `station_id` + UTC hour; keep rows with valid predictor timestamps (left join from predictors) |

`label_status`: `observed_thunderstorm` / `observed_non_thunderstorm` / `unavailable`.  
`target_observed`: 1 only when a 0/1 label is scientifically assignable.

---

## Per-station validation

| Station | City | Predictor rows | METAR obs | Matched rows | Unmatched pred | Observed targets | TS+ | Obs − | Missing target | Pos rate (obs) | Date range | Duplicates | Internal missing hours |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VOTV | Thiruvananthapuram | 105192 | 194907 | 105192 | 0 | 82021 | 3694 | 78327 | 23171 | 0.045037 | 2014-01-01T00:00:00+00:00 → 2025-12-31T23:00:00+00:00 | 0 | 0 |
| VECC | Kolkata | 105192 | 208000 | 105192 | 0 | 104061 | 3755 | 100306 | 1131 | 0.036085 | 2014-01-01T00:00:00+00:00 → 2025-12-31T23:00:00+00:00 | 0 | 0 |
| VIDP | Delhi | 105192 | 200210 | 105192 | 0 | 101485 | 2031 | 99454 | 3707 | 0.020013 | 2014-01-01T00:00:00+00:00 → 2025-12-31T23:00:00+00:00 | 0 | 0 |
| VOCI | Kochi | 105192 | 192744 | 105192 | 0 | 92517 | 2327 | 90190 | 12675 | 0.025152 | 2014-01-01T00:00:00+00:00 → 2025-12-31T23:00:00+00:00 | 0 | 0 |
| VABB | Mumbai | 105192 | 204279 | 105192 | 0 | 100658 | 1534 | 99124 | 4534 | 0.01524 | 2014-01-01T00:00:00+00:00 → 2025-12-31T23:00:00+00:00 | 0 | 0 |

Expected predictor hours in 2014–2025 UTC: **105192** per station (leap years included).

### Predictor missingness (NaN counts)

| Station | temperature_2m | relative_humidity_2m | surface_pressure | wind_speed_10m | wind_direction_10m | precipitation | cloud_cover |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VOTV | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| VECC | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| VIDP | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| VOCI | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| VABB | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

---

## VOTV vs V1 baseline

| Item | Value |
|------|--------|
| Phase 1B / V1 modelled-window VOTV TS hours | 3694 |
| Phase 2B VOTV `thunderstorm_target==1` | 3694 |
| Delta | 0 |
| Phase 2B VOTV observed labels | 82021 |
| Phase 1B observed hours (any report) | 102816 |

A small delta is expected if Phase 2B **does not** treat hours-with-reports-but-no-wx-group as negatives (Phase 1B / this phase: empty `wxcodes` is not a negative). Positive **hours** should still be close to 3694 because positives require a TS token, which implies a decodable wx group.

---

## Artifacts

| Path | Role |
|------|------|
| `dataset/multilocation/raw_metar/<icao>/` | IEM yearly + combined `wxcodes` extracts |
| `dataset/multilocation/synchronized/<icao>_synchronized_2014_2025.csv` | Per-station join |
| `dataset/multilocation/synchronized/multilocation_synchronized_2014_2025.csv` | Pooled five-station table |
| `dataset/multilocation/synchronized/multilocation_synchronized_2014_2025_metadata.json` | Machine-readable metadata |

Combined rows: **525960**. Combined SHA256: `31f652b9dc04969c0ff874c4ea143b3a14bb55a844d089777d6461e3baff5694`.

---

## What this phase did not do

- No V1 file edits
- No Flask/frontend changes
- No model training or skill metrics
- No radar / satellite / lightning / NWP
- No fabricated METAR
- No git commit or push
- No t+1/t+2/t+3 label columns
