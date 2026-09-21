# Phase 13B — V2 Inference Engine

**Scope:** reusable Python inference only. No Flask, no dashboard, no data fetch, no retraining.

**Model version:** `V2-MODEL-B-MULTILEAD-PHASE8D-RESEARCH`

---

## Objective

Implement the Phase 13A contract as `V2InferenceEngine`:

1. Accept one prepared observation (identity + 83 features).
2. Validate against Phase 13A.
3. Load the three frozen Phase 8D Model B joblibs **only if** validation passes.
4. Emit 1h/2h/3h RF `predict_proba[:, 1]` scores.
5. Apply frozen thresholds **0.065**.
6. Return the exact 13A output schema.

---

## Frozen models (read-only)

| lead | path | threshold |
| --- | --- | --- |
| 1h | `models/v2/nwp_experiment/B_atmospheric_nwp_target_1h.joblib` | 0.065 |
| 2h | `models/v2/nwp_experiment/B_atmospheric_nwp_target_2h.joblib` | 0.065 |
| 3h | `models/v2/nwp_experiment/B_atmospheric_nwp_target_3h.joblib` | 0.065 |

SHA-256 unchanged after tests:

- 1h `a266c71bf901c4ea9392882b619e1a40905279af89db58123227a07f49733f21`
- 2h `e99ff3e9fba4cef8a3e1d15e6c29cfdbee801200369230bd652a45a71b376c6c`
- 3h `20692e504594ad17ea6fcc43dfe6e8899a5d5bdc2df93015df92327b97dc16ed`

Feature order is loaded from `outputs/v2_inference/phase13a/inference_contract.json` (`model_b_features`). No second hand-maintained list.

---

## Fail-closed

If station is missing/unknown, timestamp missing, feature count ≠ 83, names/order ≠ contract, or any value is null / non-numeric / NaN / inf:

- `data_status = "UNAVAILABLE"`
- all six lead probability/alert fields = `null`
- **RF is not called**
- no zero-fill, no imputation

`data_completeness` = fraction of the 83 columns that are present and finite.

Alert: `probability >= 0.065` → `alert = 1` (including equality).

---

## Tests

`python -m pytest tests/test_v2_inference_engine.py -q`

**11 passed.**

Covered: valid complete vector; missing / NaN / inf; wrong count and wrong order; unknown station; threshold 0.065; output schema; model version; frozen hashes.

---

## Artifacts

| path | role |
| --- | --- |
| `src/inference/v2_inference_engine.py` | engine |
| `tests/test_v2_inference_engine.py` | unit tests |
| `outputs/v2_inference/phase13b/inference_test_results.json` | case log |
| `outputs/v2_inference/phase13b/inference_engine_validation.json` | final validation |
| `docs/PHASE13B_INFERENCE_ENGINE.md` | this report |

Phase 13A contract JSON was not modified. No training. No HTTP.

---

## PHASE STATUS

**READY** — Phase 13B complete.

**STOP.** Do not start Phase 13C. Do not implement Flask or the dashboard.
