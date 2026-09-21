# Phase 8C — NWP experiment readiness audit (V2)

**Generated:** 2026-09-20T16:14:24.175824+00:00
**Script:** `dataset/multilocation/audit_nwp_readiness_phase8c.py`
**Scope:** readiness counts only. No training, no imputation, no models, no new split dates.

Phase 6 artifacts were not modified.

## Frozen Phase 6 / Phase 4 cutoffs (unchanged)

Assignment matches Phase 6: `train` if `timestamp <= train_end`; `validation` if `train_end < timestamp <= validation_end`; else `test`.

| partition | Phase 6 start UTC | Phase 6 end UTC |
| --- | --- | --- |
| train | 2014-01-02T00:00:00+00:00 | 2022-05-27T01:00:00+00:00 |
| validation | 2022-05-27T02:00:00+00:00 | 2024-03-14T11:00:00+00:00 |
| test | 2024-03-14T12:00:00+00:00 | 2025-12-30T23:00:00+00:00 |

**Train cutoff starts before the NWP overlap:** `True`  
NWP overlap clock: `2021-04-01T00:00:00+00:00` → `2025-12-31T23:00:00+00:00`  
Phase 6 train start is 2014-01-02; NWP only exists from 2021-04-01. Cutoffs were **not** rewritten.

## Resulting overlap windows (same cutoffs, truncated by data)

| split | overlap start | overlap end | rows |
| --- | --- | --- | --- |
| train | 2021-04-01T00:00:00+00:00 | 2022-05-27T01:00:00+00:00 | 50530 |
| validation | 2022-05-27T02:00:00+00:00 | 2024-03-14T11:00:00+00:00 | 78890 |
| test | 2024-03-14T12:00:00+00:00 | 2025-12-31T23:00:00+00:00 | 78900 |

Test rows after Phase 6 reported `test_end` (2025-12-30 23:00) remain in **test** because Phase 6 uses an open-ended `else test` rule, not a new cutoff.

## Pooled counts by lead

### target_1h

| split | total | observed | pos | neg | missing target | pos rate |
| --- | --- | --- | --- | --- | --- | --- |
| train | 50530 | 46501 | 1721 | 44780 | 4029 | 0.0370 |
| validation | 78890 | 72558 | 1965 | 70593 | 6332 | 0.0271 |
| test | 78900 | 73694 | 2172 | 71522 | 5206 | 0.0295 |

### target_2h

| split | total | observed | pos | neg | missing target | pos rate |
| --- | --- | --- | --- | --- | --- | --- |
| train | 50530 | 46500 | 1721 | 44779 | 4030 | 0.0370 |
| validation | 78890 | 72559 | 1965 | 70594 | 6331 | 0.0271 |
| test | 78900 | 73689 | 2172 | 71517 | 5211 | 0.0295 |

### target_3h

| split | total | observed | pos | neg | missing target | pos rate |
| --- | --- | --- | --- | --- | --- | --- |
| train | 50530 | 46499 | 1721 | 44778 | 4031 | 0.0370 |
| validation | 78890 | 72560 | 1965 | 70595 | 6330 | 0.0271 |
| test | 78900 | 73684 | 2172 | 71512 | 5216 | 0.0295 |

## Counts by station

### target_1h by station

| station | split | total | observed | pos | neg | missing target | pos rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VOTV | train | 10106 | 7687 | 519 | 7168 | 2419 | 0.0675 |
| VOTV | validation | 15778 | 11886 | 489 | 11397 | 3892 | 0.0411 |
| VOTV | test | 15780 | 12899 | 509 | 12390 | 2881 | 0.0395 |
| VECC | train | 10106 | 10009 | 479 | 9530 | 97 | 0.0479 |
| VECC | validation | 15778 | 15665 | 512 | 15153 | 113 | 0.0327 |
| VECC | test | 15780 | 15627 | 637 | 14990 | 153 | 0.0408 |
| VIDP | train | 10106 | 9950 | 222 | 9728 | 156 | 0.0223 |
| VIDP | validation | 15778 | 15393 | 390 | 15003 | 385 | 0.0253 |
| VIDP | test | 15780 | 15269 | 364 | 14905 | 511 | 0.0238 |
| VOCI | train | 10106 | 8920 | 302 | 8618 | 1186 | 0.0339 |
| VOCI | validation | 15778 | 14139 | 354 | 13785 | 1639 | 0.0250 |
| VOCI | test | 15780 | 14535 | 361 | 14174 | 1245 | 0.0248 |
| VABB | train | 10106 | 9935 | 199 | 9736 | 171 | 0.0200 |
| VABB | validation | 15778 | 15475 | 220 | 15255 | 303 | 0.0142 |
| VABB | test | 15780 | 15364 | 301 | 15063 | 416 | 0.0196 |

### target_2h by station

| station | split | total | observed | pos | neg | missing target | pos rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VOTV | train | 10106 | 7686 | 519 | 7167 | 2420 | 0.0675 |
| VOTV | validation | 15778 | 11887 | 489 | 11398 | 3891 | 0.0411 |
| VOTV | test | 15780 | 12898 | 509 | 12389 | 2882 | 0.0395 |
| VECC | train | 10106 | 10009 | 479 | 9530 | 97 | 0.0479 |
| VECC | validation | 15778 | 15665 | 512 | 15153 | 113 | 0.0327 |
| VECC | test | 15780 | 15626 | 637 | 14989 | 154 | 0.0408 |
| VIDP | train | 10106 | 9950 | 222 | 9728 | 156 | 0.0223 |
| VIDP | validation | 15778 | 15393 | 390 | 15003 | 385 | 0.0253 |
| VIDP | test | 15780 | 15268 | 364 | 14904 | 512 | 0.0238 |
| VOCI | train | 10106 | 8920 | 302 | 8618 | 1186 | 0.0339 |
| VOCI | validation | 15778 | 14139 | 354 | 13785 | 1639 | 0.0250 |
| VOCI | test | 15780 | 14534 | 361 | 14173 | 1246 | 0.0248 |
| VABB | train | 10106 | 9935 | 199 | 9736 | 171 | 0.0200 |
| VABB | validation | 15778 | 15475 | 220 | 15255 | 303 | 0.0142 |
| VABB | test | 15780 | 15363 | 301 | 15062 | 417 | 0.0196 |

### target_3h by station

| station | split | total | observed | pos | neg | missing target | pos rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VOTV | train | 10106 | 7685 | 519 | 7166 | 2421 | 0.0675 |
| VOTV | validation | 15778 | 11888 | 489 | 11399 | 3890 | 0.0411 |
| VOTV | test | 15780 | 12897 | 509 | 12388 | 2883 | 0.0395 |
| VECC | train | 10106 | 10009 | 479 | 9530 | 97 | 0.0479 |
| VECC | validation | 15778 | 15665 | 512 | 15153 | 113 | 0.0327 |
| VECC | test | 15780 | 15625 | 637 | 14988 | 155 | 0.0408 |
| VIDP | train | 10106 | 9950 | 222 | 9728 | 156 | 0.0223 |
| VIDP | validation | 15778 | 15393 | 390 | 15003 | 385 | 0.0253 |
| VIDP | test | 15780 | 15267 | 364 | 14903 | 513 | 0.0238 |
| VOCI | train | 10106 | 8920 | 302 | 8618 | 1186 | 0.0339 |
| VOCI | validation | 15778 | 14139 | 354 | 13785 | 1639 | 0.0250 |
| VOCI | test | 15780 | 14533 | 361 | 14172 | 1247 | 0.0248 |
| VABB | train | 10106 | 9935 | 199 | 9736 | 171 | 0.0200 |
| VABB | validation | 15778 | 15475 | 220 | 15255 | 303 | 0.0142 |
| VABB | test | 15780 | 15362 | 301 | 15061 | 418 | 0.0196 |

## Readiness notes

- Same chronological UTC boundaries as Phase 6; no random split; no new dates.
- Missing targets left as missing; not converted to 0.
- NWP overlap train set is shorter than Phase 6 train because 2014–2021-03 has no GFS Historical Forecast in this collection.
- Validation and test fall entirely inside the NWP window.
- This audit does not decide whether NWP helps.

## Artifacts

| path | role |
| --- | --- |
| `docs/PHASE8C_READINESS_AUDIT.md` | this report |
| `dataset/multilocation/features_nwp/phase8c_readiness_audit.json` | counts JSON |

## PHASE STATUS

**READY** — readiness audit only. Do not proceed to Phase 8D in this task.
