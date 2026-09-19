# Nowcast vs precipitation proxy

Two models in `models/` are **different products**. Mixing their metrics is a scientific error.

## 1-hour thunderstorm nowcast

- Artifact: `thunderstorm_nowcast_1h.joblib`
- HTTP: `GET /api/prediction`
- Target: genuine METAR thunderstorm at **t+1h** (`target_1h`)
- Threshold: **0.0775** (validation F2, then locked)
- Score is **not** a calibrated probability (`class_weight='balanced'`)

Test split (README): recall 0.6040, precision 0.1332, ROC-AUC 0.8453, 11,597 rows.

## Phase 1 surrogate

- Artifact: `storm_risk_model.joblib`
- HTTP: `GET /api/prediction/proxy`
- Target: high accumulated precipitation (training 90th percentile over next 3h) — **not** thunderstorm or lightning
- Loaded lazily so both forests do not always share one 512 MiB instance

## 2h / 3h nowcast

Trained in Phase 6; **not served**. In **this V2 copy**, the `.joblib` files are 134 bytes (incomplete). Metadata JSON remains.
