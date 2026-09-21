# Phase 15C — Historical Error and Event Analysis

**Phase 15C is an observational analysis of the frozen V2 Model B replay outputs. It does not modify the model.**

**Observed differences across stations, months, years, or leads are descriptive and should not be interpreted as causal explanations.**

**Historical replay analysis does not establish live, near-real-time, or operational forecasting capability.**

---

## 1. Purpose

Answer: *What does the frozen V2 Model B actually do when replayed over historical data?*

This phase does not train, retune, or change thresholds. It classifies Phase 15B replay outputs at the frozen 0.065 operating point.

---

## 2. Input data

Primary: `outputs/v2_replay/phase15b/replay_predictions.csv`

| | |
| --- | ---: |
| Input rows | 34720 |
| COMPLETE | 34130 |
| UNAVAILABLE | 590 |

Valid labeled cases (COMPLETE + finite probability + finite target; missing labels excluded, not set to 0):

| Lead | n |
| --- | ---: |
| 1h | 31949 |
| 2h | 31876 |
| 3h | 31581 |

These n match Phase 15B `evaluation_sample_count_per_lead`.

---

## 3. Frozen model / threshold

Model version `V2-MODEL-B-MULTILEAD-PHASE8D-RESEARCH`. Alert iff `probability >= 0.065` on each lead. Alerts in the table match this rule. Probabilities lie in [0, 1]. No duplicate `(station_id, timestamp)` rows.

---

## 4. Confusion / event analysis

FACT (threshold 0.065; TP+FP+FN+TN = n):

| Lead | n | TP | FP | FN | TN |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1h | 31949 | 695 | 3552 | 251 | 27451 |
| 2h | 31876 | 702 | 3463 | 267 | 27444 |
| 3h | 31581 | 691 | 3679 | 261 | 26950 |

INTERPRETATION: Most labeled hours are true negatives. Alerts are much more common than labeled storms; false positives outnumber true positives at this threshold.

Among 1h alerts (TP+FP = 4247), FP are 3552 (**83.6% of 1h alerts** in this sample). Missed storms (FN) are 251 of 946 positives (**26.5% of 1h labeled storms not alerted**).

---

## 5. Probability behavior

Predicted-probability **bins** (not a calibration study):

1h positive rate rises across bins: 0.37% in `[0, 0.025)` → 6.7% in `[0.065, 0.100)` → **99.3%** in `[0.500, 1.000]` (285 cases, 283 positives). The same monotonic pattern appears at 2h and 3h.

Most mass sits below 0.025 (e.g. 22220 / 31949 of 1h cases). High scores are rare and, in this sample, almost always labeled positive.

---

## 6. False positives

| Lead | FP | min | median | mean | max | median distance above 0.065 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1h | 3552 | 0.065 | 0.10 | 0.124 | 0.55 | 0.035 |
| 2h | 3463 | 0.065 | 0.10 | 0.117 | 0.71 | 0.035 |
| 3h | 3679 | 0.065 | 0.10 | 0.115 | 0.69 | 0.035 |

Station counts (1h FP): VECC 1248, VOTV 690, VOCI 645, VABB 589, VIDP 380. These are **counts, not rankings**. Month counts for 1h FP are higher in May–September than in January–February (observed seasonality of **alerts and labels**, not a causal monsoon claim).

---

## 7. False negatives

| Lead | FN | min | median | mean | max | n with 0.065−p ≤ 0.02 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1h | 251 | 0.00 | 0.035 | 0.033 | 0.06 | 67 |
| 2h | 267 | 0.00 | 0.035 | 0.036 | 0.06 | 88 |
| 3h | 261 | 0.00 | 0.035 | 0.035 | 0.06 | 75 |

INTERPRETATION: Many missed events have scores well below 0.065 (median 0.035), not only “just under” the cut. A minority sit within 0.02 of the threshold.

---

## 8. Lead relationships

Among 34130 COMPLETE rows with all three alerts defined:

| Pattern (1h 2h 3h) | count | % |
| --- | ---: | ---: |
| 000 | 27332 | 80.08 |
| 111 | 2997 | 8.78 |
| 001 | 1148 | 3.36 |
| 011 | 727 | 2.13 |
| 100 | 718 | 2.10 |
| 110 | 573 | 1.68 |
| 010 | 349 | 1.02 |
| 101 | 286 | 0.84 |

Alert frequency: 1h 13.40%, 2h 13.61%, 3h 15.11%.  
1h≠2h transitions: 2080 (6.09%). 2h≠3h: 2356 (6.90%).

This is **model-output co-occurrence**, not storm duration.

---

## 9. Monthly / seasonal behavior

1h month-of-year (FACT): labeled positive rate is lower in Jan/Feb/Dec (~0.5%) than in May (~5.6%) and October (~4.6%). Alert rate is also higher in May–September (e.g. June 1h alert rate 25.0%) than in January (1.6%).

Do **not** read this as “monsoon causes better/worse model skill.” It is co-variation of labels, alerts, and calendar month in this replay sample.

---

## 10. Station-wise descriptive results

All five stations appear. Example 1h (not a ranking):

| Station | n | pos. rate | alert rate | TP | FP | FN | precision | recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| VOTV | 5561 | 0.038 | 0.156 | 178 | 690 | 36 | 0.205 | 0.832 |
| VECC | 6776 | 0.042 | 0.217 | 222 | 1248 | 61 | 0.151 | 0.784 |
| VIDP | 6674 | 0.024 | 0.069 | 83 | 380 | 78 | 0.179 | 0.516 |
| VOCI | 6237 | 0.027 | 0.122 | 119 | 645 | 49 | 0.156 | 0.708 |
| VABB | 6701 | 0.018 | 0.102 | 93 | 589 | 27 | 0.136 | 0.775 |

No station is “best.” No geographic cause is inferred.

---

## 11. Yearly behavior

1h valid n is spread across 2021–2025. 2021 1h FN = 0 in this 6-hour labeled subset (FACT); later years have non-zero FN. That is a sample observation, not a claim that 2021 was easier physically.

---

## 12. Representative cases

Deterministic: TP/TN = first row by `station_id`, timestamp; FP/FN = labeled row with smallest `|p − 0.065|`. All exist (`status=ok`). Examples (1h):

| Class | station | time | p | alert | target | p−0.065 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| TP | VABB | 2021-04-27T18:00Z | 0.595 | 1 | 1 | — |
| FP | VABB | 2021-07-08T00:00Z | 0.065 | 1 | 0 | 0.000 |
| FN | VABB | 2024-06-11T00:00Z | 0.060 | 0 | 1 | −0.005 |
| TN | VABB | 2021-04-02T00:00Z | 0.000 | 0 | 0 | — |

These do **not** explain physical causes.

---

## 13. Near-threshold behavior

`|p − 0.065| ≤ 0.02` (not used to pick a new threshold):

| Lead | n | pos | pos. rate | TP | FP | FN | TN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1h | 2439 | 141 | 0.058 | 74 | 1101 | 67 | 1197 |
| 2h | 2672 | 182 | 0.068 | 94 | 1166 | 88 | 1324 |
| 3h | 2753 | 155 | 0.056 | 80 | 1217 | 75 | 1381 |

A large share of near-cut cases are FP or TN, i.e. the operating point sits in a dense low-positive region.

---

## 14. Data-quality validation

`outputs/v2_replay/phase15c/analysis_validation.json`:

- missing targets excluded
- per-lead n matches 15B
- UNAVAILABLE not in labeled metrics
- no duplicates
- probabilities in [0,1]
- alerts ≡ 0.065 rule
- bins partition labeled probabilities
- five stations present
- analysis deterministic

---

## 15. Scientific interpretation

Frozen Model B, 6-hour historical replay, 0.065:

- Default output is **no alert** (~80% pattern 000).
- When it alerts, **most alerts are false positives** at this threshold in this sample.
- When a storm is labeled, **most are caught** (1h recall 0.735 in 15B; FN 251 vs TP 695).
- Very high scores (>0.5) are uncommon and almost always positive here.
- 1h/2h/3h alerts often agree on 000 or 111, with ~6–7% lead-to-lead flips.
- Calendar month and station **counts differ**; causes are not identified.

---

## 16. Limitations

- 6-hour sampling; full-period replay includes train/val years (see 15B vs 8D).
- Bins are not a reliability diagram / ECE study.
- No meteorological covariates joined for error cases.
- Threshold frozen; near-threshold counts are not a tuning recommendation.

---

## 17. What this does NOT prove

- Not live, NRT, or operational skill.
- Not that 1h is the “best” lead.
- Not that any station is best/worst.
- Not that monsoon or humidity “causes” errors.
- Not that a new threshold should be chosen.

---

## Outputs

`outputs/v2_replay/phase15c/*.csv` and `analysis_validation.json`  
Code: `src/replay/v2_historical_replay_analysis_phase15c.py`  
Tests: `python -m pytest tests/test_v2_historical_replay_phase15c.py -q`
