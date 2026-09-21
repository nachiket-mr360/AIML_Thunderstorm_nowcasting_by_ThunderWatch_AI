# Phase 15B — Multi-case Historical Replay Evaluation

**Phase 15B evaluates the frozen V2 Model B through historical replay. It does not retrain, retune, or modify the model.**

**Historical replay evaluation does not establish live, near-real-time, or operational forecasting capability.**

---

## 1. Purpose

Repeat the Phase 15A replay workflow across many deterministic timestamps `T` and all five stations, then aggregate 1h / 2h / 3h **risk probabilities** against historical labels using the **frozen** 0.065 alert threshold.

This is an evaluation of the existing Phase 13B Model B, not a new training experiment.

---

## 2. Relationship to Phase 15A

Phase 15A implemented `src/replay/v2_historical_replay.py` for a single `(station_id, T)` replay.

Phase 15B (`src/replay/v2_historical_replay_eval_phase15b.py`) runs the same causal rules at scale:

- observations at or before `T` only
- no `target_*` in the inference vector
- labels read **after** prediction
- fail-closed (no fill / interpolate)

---

## 3. Evaluation period

Validated NWP overlap: **2021-04-01 00:00 UTC** through **2025-12-31 23:00 UTC**.

First candidate hour: `2021-04-01T00:00:00+00:00`  
Last candidate hour: `2025-12-31T18:00:00+00:00` (18:00 is the last 6-hour grid point).

Only timestamps with a complete causal window `T−24…T` are scored as `COMPLETE`.

---

## 4. Station coverage

All five contract stations, equal candidate grids (6944 each):

VOTV, VECC, VIDP, VOCI, VABB

Results are reported **pooled** and **per station**. Stations are not ranked.

---

## 5. Sampling strategy

**Interval: every 6 hours** (UTC `hour % 6 == 0`: 00, 06, 12, 18).

No random sampling and no `random_state` for timestamp selection.

Hourly replay would require tens of thousands of Phase 14A 25-row rebuilds (~0.12 s each). Six-hour sampling still spans the full overlap and all stations. When the overlap CSV already stores the Phase 13A 83-vector (Phase 14A-validated), evaluation uses that vector **after** a fail-closed `T−24…T` hourly/NWP check. Synthetic tests and missing-vector cases still call Phase 14A `build_features`.

---

## 6. Causal feature rule

For each candidate `T`:

- require 25 consecutive hourly rows ending at `T`
- require finite required NWP at `T` (including CIN and precipitation)
- never pass rows with `timestamp > T`
- never impute

---

## 7. Target separation

`target_1h` / `target_2h` / `target_3h` are excluded from the feature vector.

After inference, labels on the row at `T` are attached (storm at T+1 / T+2 / T+3).

If a lead’s label is missing, that lead is **omitted from that lead’s metrics** only. A missing 3h label does not drop a valid 1h case. Missing labels are **not** treated as negatives.

---

## 8. Frozen models

Unmodified:

- `outputs/v2_inference/phase13a/inference_contract.json`
- `src/inference/v2_inference_engine.py`
- `src/features/v2_feature_builder.py`
- `src/replay/v2_historical_replay.py` (Phase 15A)
- `models/v2/nwp_experiment/B_atmospheric_nwp_target_{1,2,3}h.joblib`

Version: `V2-MODEL-B-MULTILEAD-PHASE8D-RESEARCH`

---

## 9. Frozen thresholds

Exactly Phase 13A / 8D:

| Lead | Threshold |
| --- | --- |
| 1h | 0.065 |
| 2h | 0.065 |
| 3h | 0.065 |

Alert: `probability >= 0.065`. Thresholds were **not** re-selected on the 15B sample.

---

## 10. Metrics

Per lead, on rows with both probability and label:

sample count, positives/negatives, positive rate, PR-AUC (`average_precision_score`), ROC-AUC, precision, recall, F1, F2 (β=2), alert rate, TP/FP/FN/TN.

If only one class is present, PR-AUC and ROC-AUC are null.

**Pooled (threshold 0.065):**

| Lead | n | pos | pos. rate | PR-AUC | ROC-AUC | P | R | F1 | F2 | alert rate | TP | FP | FN | TN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1h | 31949 | 946 | 0.0296 | 0.478 | 0.911 | 0.164 | 0.735 | 0.268 | 0.433 | 0.133 | 695 | 3552 | 251 | 27451 |
| 2h | 31876 | 969 | 0.0304 | 0.472 | 0.914 | 0.169 | 0.724 | 0.273 | 0.437 | 0.131 | 702 | 3463 | 267 | 27444 |
| 3h | 31581 | 952 | 0.0301 | 0.496 | 0.909 | 0.158 | 0.726 | 0.260 | 0.422 | 0.138 | 691 | 3679 | 261 | 26950 |

CSV: `outputs/v2_replay/phase15b/metrics_pooled.csv`

Confusion counts satisfy TP+FP+FN+TN = n on every lead.

---

## 11. Data-quality accounting

| Quantity | Count |
| --- | ---: |
| Candidate timestamps (5 stations × 6-hour grid) | 34720 |
| Successful replay (`COMPLETE`) | 34130 |
| Unavailable | 590 |
| Unavailable — missing T−24…T history | 20 |
| Unavailable — missing NWP | 570 |
| Unavailable — invalid input | 0 |
| `target_1h` missing among COMPLETE | 2181 |
| `target_2h` missing among COMPLETE | 2254 |
| `target_3h` missing among COMPLETE | 2549 |
| Eval n 1h / 2h / 3h | 31949 / 31876 / 31581 |

JSON: `outputs/v2_replay/phase15b/replay_data_quality.json`  
Predictions: `outputs/v2_replay/phase15b/replay_predictions.csv`

---

## 12. Station-level evaluation

See `outputs/v2_replay/phase15b/metrics_by_station.csv`. Values are descriptive only; no station is declared better or worse.

Yearly **successful** replay counts (not concentrated in one year):

| Year | COMPLETE cases |
| --- | ---: |
| 2021 | 5480 |
| 2022 | 7015 |
| 2023 | 7015 |
| 2024 | 7320 |
| 2025 | 7300 |

2021 is shorter because overlap starts 1 April.

---

## 13. Phase 8D comparison

`outputs/v2_replay/phase15b/phase8d_vs_15b_comparison.csv`

| | Phase 8D | Phase 15B |
| --- | --- | --- |
| Role | Model B **held-out test split** | Historical replay over **full overlap** |
| Sample | Complete-case stored features, time split | 6-hour grid, fail-closed window |
| Years | Test years only (Phase 6 cutoffs) | 2021–2025 including train/val hours |
| Threshold | 0.065 frozen | 0.065 frozen (not re-tuned) |

Numeric differences (including higher 15B PR-AUC) are **not** an accuracy improvement. 15B is not a held-out test and is not interchangeable with 8D.

---

## 14. Reproducibility

- Deterministic UTC 6-hour grid
- Metrics recomputed twice from the same prediction table: identical (`reproducibility_check.json`)
- Unit tests re-run the evaluator on a fixed synthetic history and require identical probabilities

---

## 15. Limitations

- Historical archive only; not live or operational forecasting.
- 6-hour sampling, not every hour.
- Full-period replay **leaks training-period hours** relative to Phase 8D test metrics.
- Single-threshold operating point (0.065); no new threshold search.
- No satellite / radar / lightning.
- Event-level `alert_hit` on one Phase 15A case is not this evaluation; 15B still does not claim operational skill.

---

## Commands

```text
python -m pytest tests/test_v2_historical_replay_phase15b.py -q
python -m src.replay.v2_historical_replay_eval_phase15b --interval-hours 6
```

Implementation: `src/replay/v2_historical_replay_eval_phase15b.py`
