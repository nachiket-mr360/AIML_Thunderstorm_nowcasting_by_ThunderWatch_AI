# Phase 5 — Multi-lead thunderstorm target construction (V2)

**Generated:** 2026-09-19T17:38:34.853996+00:00  
**Script:** `dataset/multilocation/build_multilead_targets_phase5.py`  
**PHASE STATUS:** **READY**

V1, Flask, frontend, and Phase 3/4 feature CSVs were not modified. No models were trained.

---

## Task

Phase 4 predicts thunderstorm at the **same hour T** as the features. That is a baseline.

Phase 5 attaches genuine **future** METAR thunderstorm observations:

| target | definition |
| --- | --- |
| `target_1h` | genuine METAR thunderstorm at **T+1 hour** |
| `target_2h` | genuine METAR thunderstorm at **T+2 hours** |
| `target_3h` | genuine METAR thunderstorm at **T+3 hours** |

Features remain at time **T**. Only the label is shifted.

Resolution is **hourly**. No 30-minute target is claimed.

## Label source

**Only** the genuine METAR `thunderstorm_target` from Phase 3/4 feature tables.

Not used: Open-Meteo `weather_code`, precipitation-derived labels, model predictions, future atmospheric variables.

## Causality

- Predictors at T are functions of Open-Meteo at T and earlier (Phase 3).
- `target_Lh` is METAR at T+L on the **same station**.
- Shift is `label.shift(-L)` on a **contiguous 1-hour grid per ICAO**.
- Compacted (drop-NA) shifting is refused: that would skip missing hours and invent a variable lead.
- If T+L has no usable METAR: `target_Lh = NA` and `target_Lh_observed = 0`. **Never written as 0.**

## Missing-label policy

Unavailable future observations stay NA. They must be excluded from training later; they are not negative examples.

Last 1 / 2 / 3 feature hours of each station have no T+L on the table, so those leads are NA even if METAR exists after the feature window end.

## Station isolation

Shift is computed independently for VOTV, VECC, VIDP, VOCI, VABB. A row never receives another station’s label.

## Validation summary

| Station | Horizon | Feature rows | Observed target | Positive | Negative | Unavailable | Positive rate | First valid T | Final valid T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VOTV | 1h | 105168 | 82006 | 3694 | 78312 | 23162 | 0.045045 | 2014-01-02T00:00:00+00:00 | 2025-12-30T22:00:00+00:00 |
| VOTV | 2h | 105168 | 82005 | 3694 | 78311 | 23163 | 0.045046 | 2014-01-02T00:00:00+00:00 | 2025-12-30T21:00:00+00:00 |
| VOTV | 3h | 105168 | 82004 | 3694 | 78310 | 23164 | 0.045047 | 2014-01-02T00:00:00+00:00 | 2025-12-30T20:00:00+00:00 |
| VECC | 1h | 105168 | 104036 | 3755 | 100281 | 1132 | 0.036093 | 2014-01-02T00:00:00+00:00 | 2025-12-30T22:00:00+00:00 |
| VECC | 2h | 105168 | 104035 | 3755 | 100280 | 1133 | 0.036094 | 2014-01-02T00:00:00+00:00 | 2025-12-30T21:00:00+00:00 |
| VECC | 3h | 105168 | 104034 | 3755 | 100279 | 1134 | 0.036094 | 2014-01-02T00:00:00+00:00 | 2025-12-30T20:00:00+00:00 |
| VIDP | 1h | 105168 | 101461 | 2031 | 99430 | 3707 | 0.020018 | 2014-01-02T00:00:00+00:00 | 2025-12-30T22:00:00+00:00 |
| VIDP | 2h | 105168 | 101460 | 2031 | 99429 | 3708 | 0.020018 | 2014-01-02T00:00:00+00:00 | 2025-12-30T21:00:00+00:00 |
| VIDP | 3h | 105168 | 101459 | 2031 | 99428 | 3709 | 0.020018 | 2014-01-02T00:00:00+00:00 | 2025-12-30T20:00:00+00:00 |
| VOCI | 1h | 105168 | 92505 | 2327 | 90178 | 12663 | 0.025155 | 2014-01-02T00:00:00+00:00 | 2025-12-30T22:00:00+00:00 |
| VOCI | 2h | 105168 | 92504 | 2327 | 90177 | 12664 | 0.025156 | 2014-01-02T00:00:00+00:00 | 2025-12-30T21:00:00+00:00 |
| VOCI | 3h | 105168 | 92503 | 2327 | 90176 | 12665 | 0.025156 | 2014-01-02T00:00:00+00:00 | 2025-12-30T20:00:00+00:00 |
| VABB | 1h | 105168 | 100633 | 1534 | 99099 | 4535 | 0.015244 | 2014-01-02T00:00:00+00:00 | 2025-12-30T22:00:00+00:00 |
| VABB | 2h | 105168 | 100632 | 1534 | 99098 | 4536 | 0.015244 | 2014-01-02T00:00:00+00:00 | 2025-12-30T21:00:00+00:00 |
| VABB | 3h | 105168 | 100631 | 1534 | 99097 | 4537 | 0.015244 | 2014-01-02T00:00:00+00:00 | 2025-12-30T20:00:00+00:00 |

Checks:

- no station mixing (each input file has a single `station_id`)
- no timestamp errors (unique, sorted, exactly 1 h steps)
- no future feature leakage (feature columns not rewritten; only labels shifted)
- direction self-test: `target_Lh(T) == thunderstorm_target(T+L)` on the same station, including NA
- unobserved future hours are never 0
- same-hour vs `target_1h` agreement is well below 1 (shift is real)

## Alignment examples

### VOTV (Thiruvananthapuram)
- **T** `2014-01-02T00:00:00+00:00`  
  features timestamp = `2014-01-02T00:00:00+00:00`  
  target_1h timestamp = `2014-01-02T01:00:00+00:00` value=0.0  
  target_2h timestamp = `2014-01-02T02:00:00+00:00` value=0.0  
  target_3h timestamp = `2014-01-02T03:00:00+00:00` value=0.0  
  same-hour METAR at T = 0.0
- **T** `2014-01-02T05:00:00+00:00`  
  features timestamp = `2014-01-02T05:00:00+00:00`  
  target_1h timestamp = `2014-01-02T06:00:00+00:00` value=None  
  target_2h timestamp = `2014-01-02T07:00:00+00:00` value=None  
  target_3h timestamp = `2014-01-02T08:00:00+00:00` value=None  
  same-hour METAR at T = 0.0
- **T** `2014-01-14T18:00:00+00:00`  
  features timestamp = `2014-01-14T18:00:00+00:00`  
  target_1h timestamp = `2014-01-14T19:00:00+00:00` value=1.0  
  target_2h timestamp = `2014-01-14T20:00:00+00:00` value=1.0  
  target_3h timestamp = `2014-01-14T21:00:00+00:00` value=None  
  same-hour METAR at T = 0.0
- **T** `2025-12-31T20:00:00+00:00`  
  features timestamp = `2025-12-31T20:00:00+00:00`  
  target_1h timestamp = `2025-12-31T21:00:00+00:00` value=None  
  target_2h timestamp = `2025-12-31T22:00:00+00:00` value=None  
  target_3h timestamp = `2025-12-31T23:00:00+00:00` value=None  
  same-hour METAR at T = None

### VECC (Kolkata)
- **T** `2014-01-02T00:00:00+00:00`  
  features timestamp = `2014-01-02T00:00:00+00:00`  
  target_1h timestamp = `2014-01-02T01:00:00+00:00` value=0.0  
  target_2h timestamp = `2014-01-02T02:00:00+00:00` value=0.0  
  target_3h timestamp = `2014-01-02T03:00:00+00:00` value=0.0  
  same-hour METAR at T = 0.0
- **T** `2014-01-02T20:00:00+00:00`  
  features timestamp = `2014-01-02T20:00:00+00:00`  
  target_1h timestamp = `2014-01-02T21:00:00+00:00` value=None  
  target_2h timestamp = `2014-01-02T22:00:00+00:00` value=0.0  
  target_3h timestamp = `2014-01-02T23:00:00+00:00` value=0.0  
  same-hour METAR at T = 0.0
- **T** `2014-03-23T11:00:00+00:00`  
  features timestamp = `2014-03-23T11:00:00+00:00`  
  target_1h timestamp = `2014-03-23T12:00:00+00:00` value=1.0  
  target_2h timestamp = `2014-03-23T13:00:00+00:00` value=1.0  
  target_3h timestamp = `2014-03-23T14:00:00+00:00` value=1.0  
  same-hour METAR at T = 0.0
- **T** `2025-12-31T20:00:00+00:00`  
  features timestamp = `2025-12-31T20:00:00+00:00`  
  target_1h timestamp = `2025-12-31T21:00:00+00:00` value=None  
  target_2h timestamp = `2025-12-31T22:00:00+00:00` value=None  
  target_3h timestamp = `2025-12-31T23:00:00+00:00` value=None  
  same-hour METAR at T = None

### VIDP (Delhi)
- **T** `2014-01-02T00:00:00+00:00`  
  features timestamp = `2014-01-02T00:00:00+00:00`  
  target_1h timestamp = `2014-01-02T01:00:00+00:00` value=0.0  
  target_2h timestamp = `2014-01-02T02:00:00+00:00` value=0.0  
  target_3h timestamp = `2014-01-02T03:00:00+00:00` value=0.0  
  same-hour METAR at T = None
- **T** `2014-01-03T02:00:00+00:00`  
  features timestamp = `2014-01-03T02:00:00+00:00`  
  target_1h timestamp = `2014-01-03T03:00:00+00:00` value=None  
  target_2h timestamp = `2014-01-03T04:00:00+00:00` value=0.0  
  target_3h timestamp = `2014-01-03T05:00:00+00:00` value=None  
  same-hour METAR at T = 0.0
- **T** `2014-01-21T03:00:00+00:00`  
  features timestamp = `2014-01-21T03:00:00+00:00`  
  target_1h timestamp = `2014-01-21T04:00:00+00:00` value=1.0  
  target_2h timestamp = `2014-01-21T05:00:00+00:00` value=1.0  
  target_3h timestamp = `2014-01-21T06:00:00+00:00` value=1.0  
  same-hour METAR at T = 0.0
- **T** `2025-12-31T20:00:00+00:00`  
  features timestamp = `2025-12-31T20:00:00+00:00`  
  target_1h timestamp = `2025-12-31T21:00:00+00:00` value=None  
  target_2h timestamp = `2025-12-31T22:00:00+00:00` value=None  
  target_3h timestamp = `2025-12-31T23:00:00+00:00` value=None  
  same-hour METAR at T = None

### VOCI (Kochi)
- **T** `2014-01-02T00:00:00+00:00`  
  features timestamp = `2014-01-02T00:00:00+00:00`  
  target_1h timestamp = `2014-01-02T01:00:00+00:00` value=0.0  
  target_2h timestamp = `2014-01-02T02:00:00+00:00` value=0.0  
  target_3h timestamp = `2014-01-02T03:00:00+00:00` value=0.0  
  same-hour METAR at T = 0.0
- **T** `2014-01-02T06:00:00+00:00`  
  features timestamp = `2014-01-02T06:00:00+00:00`  
  target_1h timestamp = `2014-01-02T07:00:00+00:00` value=None  
  target_2h timestamp = `2014-01-02T08:00:00+00:00` value=None  
  target_3h timestamp = `2014-01-02T09:00:00+00:00` value=None  
  same-hour METAR at T = 0.0
- **T** `2014-03-08T12:00:00+00:00`  
  features timestamp = `2014-03-08T12:00:00+00:00`  
  target_1h timestamp = `2014-03-08T13:00:00+00:00` value=1.0  
  target_2h timestamp = `2014-03-08T14:00:00+00:00` value=0.0  
  target_3h timestamp = `2014-03-08T15:00:00+00:00` value=None  
  same-hour METAR at T = 0.0
- **T** `2025-12-31T20:00:00+00:00`  
  features timestamp = `2025-12-31T20:00:00+00:00`  
  target_1h timestamp = `2025-12-31T21:00:00+00:00` value=None  
  target_2h timestamp = `2025-12-31T22:00:00+00:00` value=None  
  target_3h timestamp = `2025-12-31T23:00:00+00:00` value=None  
  same-hour METAR at T = None

### VABB (Mumbai)
- **T** `2014-01-02T00:00:00+00:00`  
  features timestamp = `2014-01-02T00:00:00+00:00`  
  target_1h timestamp = `2014-01-02T01:00:00+00:00` value=0.0  
  target_2h timestamp = `2014-01-02T02:00:00+00:00` value=0.0  
  target_3h timestamp = `2014-01-02T03:00:00+00:00` value=0.0  
  same-hour METAR at T = 0.0
- **T** `2014-01-03T20:00:00+00:00`  
  features timestamp = `2014-01-03T20:00:00+00:00`  
  target_1h timestamp = `2014-01-03T21:00:00+00:00` value=None  
  target_2h timestamp = `2014-01-03T22:00:00+00:00` value=0.0  
  target_3h timestamp = `2014-01-03T23:00:00+00:00` value=0.0  
  same-hour METAR at T = 0.0
- **T** `2014-04-21T15:00:00+00:00`  
  features timestamp = `2014-04-21T15:00:00+00:00`  
  target_1h timestamp = `2014-04-21T16:00:00+00:00` value=1.0  
  target_2h timestamp = `2014-04-21T17:00:00+00:00` value=0.0  
  target_3h timestamp = `2014-04-21T18:00:00+00:00` value=0.0  
  same-hour METAR at T = 0.0
- **T** `2025-12-31T20:00:00+00:00`  
  features timestamp = `2025-12-31T20:00:00+00:00`  
  target_1h timestamp = `2025-12-31T21:00:00+00:00` value=None  
  target_2h timestamp = `2025-12-31T22:00:00+00:00` value=None  
  target_3h timestamp = `2025-12-31T23:00:00+00:00` value=None  
  same-hour METAR at T = None


Example interpretation: if T is `2014-01-02T00:00:00+00:00`, then target_1h is the METAR thunderstorm flag at `2014-01-02T01:00:00+00:00`, target_2h at `02:00Z`, target_3h at `03:00Z`. Feature columns still describe `00:00Z`.

## Hourly resolution

Predictors and METAR labels are on the **clock hour**. These leads are 1, 2, and 3 **hours**, not 30 minutes. Do not describe this table as sub-hourly nowcasting.

## Artifacts

| Path | Role |
| --- | --- |
| `dataset/multilocation/targets/<icao>_nowcast_targets_2014_2025.csv` | Per-station leads |
| `dataset/multilocation/targets/multilocation_nowcast_targets_2014_2025.csv` | Pooled |
| `dataset/multilocation/targets/multilocation_nowcast_targets_2014_2025_metadata.json` | Counts |
| `dataset/multilocation/targets/multilocation_nowcast_targets_2014_2025.csv` | pooled CSV |

Join to Phase 3 features on `(station_id, timestamp_utc)`. Feature vectors are **not** duplicated here.

## Not done

- No 1h/2h/3h model training
- No Flask / frontend / V1 edits
- No NWP / lightning / satellite / radar
- No git commit or push
