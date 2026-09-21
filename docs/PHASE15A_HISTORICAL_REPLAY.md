# Phase 15A — V2 Historical Replay Engine

**Scope:** deterministic historical replay of the frozen Phase 13A → 13B → 14A pipeline.

**This phase demonstrates reproducible historical replay. It does not establish live, near-real-time, or operational forecasting capability.**

V2 remains a research/prototype system. Historical replay is the only supported prediction mode. No live APIs, Flask, dashboard, or UI are introduced here.

---

## 1. Purpose

Given a known ICAO `station_id` and a UTC prediction time `T`, the replay engine:

1. Loads historical atmospheric + NWP rows **at or before T**.
2. Builds the exact 83-feature Model B vector with the Phase 14A builder.
3. Runs the frozen Phase 13B inference engine.
4. Emits 1h / 2h / 3h thunderstorm **risk probabilities** and alerts.
5. **Only after inference**, looks up historical future labels for verification.
6. Labels the entire result `replay_mode = HISTORICAL_REPLAY`.

No model is trained or retuned. Thresholds stay at the Phase 13A values (0.065).

---

## 2. Why this is historical replay

The requested timestamp is always a **past** hour inside the validated NWP overlap (2021–2025). Features are built from archive rows already on disk. Future hours after T are never inputs. The engine does not fetch current weather and does not claim nowcasting skill from a single case.

Historical future labels (`target_1h` / `target_2h` / `target_3h`) are used **only for post-inference verification**. They are never supplied to the feature builder or the inference engine.

---

## 3. Data source

Validated overlap table (no new download):

`dataset/multilocation/features_nwp/multilocation_features_nwp_overlap_2021_2025.csv`

Required Phase 14A sources: atmospheric fields, location, and NWP (`nwp_cape`, `nwp_convective_inhibition`, precipitation, etc.). The overlap CSV stores `wind_direction_sin` / `wind_direction_cos` rather than `wind_direction_10m`; replay reconstructs the angle with `atan2` so the frozen builder can run. No other missing values are filled.

---

## 4. Prediction timestamp T

`timestamp_utc` is parsed as UTC. It is the hour at which a **research replay** prediction is issued. Leads are:

| Lead | Probability / alert at T | Historical label (after inference) |
| --- | --- | --- |
| 1h | thunderstorm risk at T for T+1 | `target_1h` on the row at T (storm at T+1) |
| 2h | risk at T for T+2 | `target_2h` on the row at T |
| 3h | risk at T for T+3 | `target_3h` on the row at T |

---

## 5. Causal history rule

Only rows with `timestamp_utc <= T` for the selected station are passed to Phase 14A.

The builder requires 25 hourly samples **T−24 … T** inclusive, strictly hourly, unique timestamps, finite atmospheric/location series, and finite required NWP at T.

Rows with `timestamp_utc > T` are never given to the builder or the engine. Changing or deleting those rows must not change the prediction at T.

---

## 6. 83-feature construction through Phase 14A

Module (unmodified): `src/features/v2_feature_builder.py`

Canonical names/order: `outputs/v2_inference/phase13a/inference_contract.json`.

Replay **drops** `target_1h` / `target_2h` / `target_3h` before `build_features`. Feature formulas are not reimplemented in this phase.

---

## 7. Frozen Phase 13B inference

Module (unmodified): `src/inference/v2_inference_engine.py`

Artifacts (unmodified):

- `models/v2/nwp_experiment/B_atmospheric_nwp_target_1h.joblib`
- `models/v2/nwp_experiment/B_atmospheric_nwp_target_2h.joblib`
- `models/v2/nwp_experiment/B_atmospheric_nwp_target_3h.joblib`

Output uses **risk / model probability** and **alert threshold**, not “confidence score”.

Alert: `lead_Xh_alert = 1` iff `lead_Xh_probability >= threshold_Xh`.

---

## 8. Target-label separation

1. Build features from causal history without target columns.
2. Call `V2InferenceEngine.predict`.
3. **Then** read `target_1h/2h/3h` from the historical row at T.

If a label is present: `lead_Xh_alert_hit = (predicted_alert == target)`.

If a label is missing: `alert_hit` is null.

A single replay case is an event-level demonstration. It is **not** aggregate accuracy, F1, or precision.

---

## 9. Fail-closed rules

Return `data_status = UNAVAILABLE` (null probabilities/alerts) if:

- unknown station (not in VOTV, VECC, VIDP, VOCI, VABB)
- invalid timestamp
- timestamp outside the validated NWP overlap
- insufficient T−24…T history
- missing required atmospheric or NWP input (including CIN or precipitation)
- non-hourly or duplicate timestamps
- non-finite values

Do **not** zero-fill, forward-fill, backward-fill, interpolate, invent NWP, call an API, or swap models.

---

## 10. Example replay command

```text
python -m src.replay.v2_historical_replay --station VOTV --timestamp 2023-06-15T12:00:00Z
```

Python:

```python
from src.replay.v2_historical_replay import run_historical_replay

run_historical_replay("VOTV", "2023-06-15T12:00:00Z")
```

Sample output: `outputs/v2_replay/phase15a/replay_sample.json`

Tests: `python -m pytest tests/test_v2_historical_replay_phase15a.py -q`

---

## 11. Limitations

- Historical overlap only (not live data).
- Same frozen Model B and thresholds as Phase 8D / 13A.
- No satellite, radar, or lightning predictors.
- No operational SLA, latency, or dissemination.
- `alert_hit` on one timestamp is not a performance claim.

This phase demonstrates reproducible historical replay. It does not establish live, near-real-time, or operational forecasting capability.
