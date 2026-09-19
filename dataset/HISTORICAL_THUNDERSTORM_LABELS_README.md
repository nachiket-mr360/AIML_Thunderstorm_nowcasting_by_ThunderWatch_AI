# Historical Thunderstorm Labels for VOTV (Phase 2 data acquisition)

**File:** `dataset/historical_thunderstorm_labels_votv.csv`
**Station:** VOTV - Thiruvananthapuram International Airport, Kerala, India (8.4667 N, 76.9500 E)
**Source:** Iowa Environmental Mesonet (IEM) ASOS/METAR archive
**Status:** Phase 2 label acquisition only. Not merged into the ML dataset, no model retrained.

---

## 1. What this dataset is

A per-hour **observed** thunderstorm label table on exactly the same UTC hourly grid as
`dataset/weather_data_with_code.csv` (227,928 hours, 2000-01-01 00:00Z to 2025-12-31 23:00Z).

It is a *labels* table. It deliberately does not contain features, and it was deliberately not
joined onto the existing Open-Meteo dataset.

## 2. Source and access

| Item | Value |
|---|---|
| Source | IEM ASOS/METAR archive request service (Iowa State University) |
| Endpoint | `https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py` |
| Request | `station=VOTV`, `data=all`, `tz=UTC`, `format=onlycomma`, `report_type=3` (routine) + `report_type=4` (SPECI) |
| Why | The export carries a dedicated, pre-parsed **observed present-weather** column, `wxcodes` |
| Download script | `dataset/download_votv_metar_iem.py` |
| Build script | `dataset/build_thunderstorm_labels.py` |
| Docs script | `dataset/make_label_docs.py` |

Raw records downloaded: **273,139** across 26 yearly chunks.
Raw range actually obtained: **2000-01-01 02:40:00+00:00 -> 2025-12-30 23:30:00+00:00**.
The requested 2000-2025 range was covered except for 2025-12-31 (no reports in the archive yet).

## 3. Label definition

**Primary evidence is the `wxcodes` field, never a text search of the METAR body.**

* `thunderstorm_label = 1` - at least one valid observed report inside hour `H` carries a TS-family
  present-weather code at the station: `TS`, `TSRA`, `+TSRA`, `-TSRA`, `-TS`, `+TS`, `TSGR`,
  `TSDZ`, `TS/HZ`, `TS BR`, `5000TS`-style malformed but parsed variants, ... or a vicinity code `VCTS`.
* `thunderstorm_label = 0` - the hour has at least one valid observed report and none carries such a code.
* `thunderstorm_label = NaN` - the hour has **no report at all**. Absence of a report is never read as
  absence of a thunderstorm.

Explicitly **not** used as a label: Open-Meteo `weather_code`, rainfall thresholds, humidity or cloud
thresholds, raw-text `TS` search, and `TS` appearing only in `TEMPO`/`BECMG`/`NOSIG`/`PROBnn` trend
groups, and `RETS` (recent thunderstorm, i.e. not occurring at observation time).

A report at valid time `t` is assigned to hour `H = floor(t)`, so hour `H` covers `[H, H+1)`.
The label is therefore a **same-hour** target; any lead-time shift is Phase 3's job.

### Audit columns

| Column | Meaning |
|---|---|
| `thunderstorm_label` | primary label, 0 / 1 / NaN |
| `thunderstorm_label_strict` | as primary, but `VCTS`-only hours are 0 |
| `thunderstorm_label_recovered` | audit variant that also flags malformed pre-2012 bodies with a demonstrable observed TS that IEM failed to parse |
| `wxcodes_observed` | distinct observed present-weather groups present in the hour |
| `n_reports` | valid observed reports in the hour |
| `n_reports_full_decode` | of those, reports with both altimeter and temperature groups |
| `n_reports_with_wx` | of those, reports carrying a decodable present-weather group |
| `wx_field_observed` | 1 when `n_reports_with_wx > 0` |
| `n_ts_reports` | reports in the hour carrying a TS-family code |

## 4. Headline numbers

| Metric | Value |
|---|---|
| Raw METAR/SPECI records downloaded | 273,139 |
| Records used after cleaning | 273,125 |
| Duplicate-timestamp rows dropped | 13 (none conflicting) |
| Distinct report timestamps | 273,125 |
| Hourly grid rows | 227,928 |
| Observed hours (report exists) | 170,869 (74.966%) |
| Missing / unobserved hours (NaN) | 57,059 |
| Observed thunderstorm reports (`wxcodes`) | 6,717 |
| - of which station TS only | 6,711 |
| - of which `VCTS` only | 6 |
| Positive hours | **4,604** |
| Positive % of observed hours | **2.694%** |
| Positive % of all grid hours | 2.02% |
| Hours with a decodable present-weather group | 110,939 |
| 0-labels that rest on no present-weather group | 59,930 |

## 5. Counts by year

`pos%` columns: **observed** = positives / observed hours; **evidence** = positives / observed hours
that contain a decodable present-weather group (the recommended view).

| Year | Grid h | Observed h | Missing h | Positives | Pos% observed | Present-wx coverage of reports % | Pos% evidence |
|---|---|---|---|---|---|---|---|
| 2000 | 8784 | 3099 | 5685 | 0 | 0.0 | 0.0 | - |
| 2001 | 8760 | 2590 | 6170 | 0 | 0.0 | 0.04 | 0.0 |
| 2002 | 8760 | 2782 | 5978 | 0 | 0.0 | 0.0 | - |
| 2003 | 8760 | 3830 | 4930 | 36 | 0.94 | 28.64 | 3.276 |
| 2004 | 8784 | 4574 | 4210 | 2 | 0.044 | 1.77 | 2.469 |
| 2005 | 8760 | 5068 | 3692 | 84 | 1.657 | 38.56 | 4.299 |
| 2006 | 8760 | 5188 | 3572 | 6 | 0.116 | 6.26 | 1.846 |
| 2007 | 8760 | 5106 | 3654 | 79 | 1.547 | 46.91 | 3.299 |
| 2008 | 8784 | 5152 | 3632 | 106 | 2.057 | 46.89 | 4.389 |
| 2009 | 8760 | 5607 | 3153 | 88 | 1.569 | 53.48 | 2.946 |
| 2010 | 8760 | 5724 | 3036 | 135 | 2.358 | 64.79 | 3.613 |
| 2011 | 8760 | 3643 | 5117 | 52 | 1.427 | 54.87 | 2.37 |
| 2012 | 8784 | 7732 | 1052 | 152 | 1.966 | 70.65 | 2.728 |
| 2013 | 8760 | 7958 | 802 | 170 | 2.136 | 76.18 | 2.76 |
| 2014 | 8760 | 8189 | 571 | 279 | 3.407 | 85.54 | 3.92 |
| 2015 | 8760 | 7952 | 808 | 397 | 4.992 | 74.9 | 6.429 |
| 2016 | 8784 | 8568 | 216 | 149 | 1.739 | 81.02 | 2.09 |
| 2017 | 8760 | 8694 | 66 | 291 | 3.347 | 83.36 | 3.944 |
| 2018 | 8760 | 8649 | 111 | 415 | 4.798 | 81.09 | 5.81 |
| 2019 | 8760 | 8680 | 80 | 293 | 3.376 | 75.57 | 4.359 |
| 2020 | 8784 | 8672 | 112 | 297 | 3.425 | 71.57 | 4.665 |
| 2021 | 8760 | 8696 | 64 | 426 | 4.899 | 74.12 | 6.423 |
| 2022 | 8760 | 8669 | 91 | 296 | 3.414 | 72.71 | 4.559 |
| 2023 | 8760 | 8556 | 204 | 331 | 3.869 | 73.83 | 5.103 |
| 2024 | 8784 | 8756 | 28 | 320 | 3.655 | 84.75 | 4.241 |
| 2025 | 8760 | 8735 | 25 | 200 | 2.29 | 76.8 | 2.927 |

## 6. Counts by month (UTC)

| Month | Observed h | Positives | Pos% observed | Pos% evidence |
|---|---|---|---|---|
| 1 | 14564 | 57 | 0.391 | 0.634 |
| 2 | 13192 | 54 | 0.409 | 0.694 |
| 3 | 14276 | 280 | 1.961 | 3.318 |
| 4 | 13422 | 996 | 7.421 | 12.065 |
| 5 | 14120 | 1005 | 7.118 | 10.957 |
| 6 | 13804 | 283 | 2.05 | 2.921 |
| 7 | 14231 | 61 | 0.429 | 0.591 |
| 8 | 14572 | 70 | 0.48 | 0.719 |
| 9 | 14427 | 182 | 1.262 | 1.985 |
| 10 | 15002 | 790 | 5.266 | 7.961 |
| 11 | 14491 | 664 | 4.582 | 6.693 |
| 12 | 14768 | 162 | 1.097 | 1.699 |

The modern-era shape is the physically expected bi-modal Thiruvananthapuram pattern (pre-monsoon
April-May peak ~8.7-10.3%, post-monsoon October-November ~5.8-6.8%), which is a useful sanity check
that the `wxcodes` label is real signal and not an artefact.

## 7. Temporal sampling and report density - the key caveat

| Year | Raw reports | Reports/day | Observed hour coverage % | Present-wx coverage % |
|---|---|---|---|---|
| 2000 | 3108 | 8.56 | 35.28 | 0.0 |
| 2001 | 2595 | 7.21 | 29.57 | 0.04 |
| 2002 | 2783 | 7.86 | 31.76 | 0.0 |
| 2003 | 3837 | 10.63 | 43.72 | 28.64 |
| 2004 | 4574 | 12.53 | 52.07 | 1.77 |
| 2005 | 5068 | 13.92 | 57.85 | 38.56 |
| 2006 | 5188 | 14.25 | 59.22 | 6.26 |
| 2007 | 5106 | 14.03 | 58.29 | 46.91 |
| 2008 | 5153 | 14.12 | 58.65 | 46.89 |
| 2009 | 5662 | 15.55 | 64.01 | 53.48 |
| 2010 | 6305 | 17.51 | 65.34 | 64.79 |
| 2011 | 5285 | 16.06 | 41.59 | 54.87 |
| 2012 | 11310 | 30.99 | 88.02 | 70.65 |
| 2013 | 12244 | 33.64 | 90.84 | 76.18 |
| 2014 | 13423 | 36.98 | 93.48 | 85.54 |
| 2015 | 12452 | 34.21 | 90.78 | 74.9 |
| 2016 | 15456 | 42.35 | 97.54 | 81.02 |
| 2017 | 17093 | 46.96 | 99.25 | 83.36 |
| 2018 | 17064 | 47.01 | 98.73 | 81.09 |
| 2019 | 17087 | 46.94 | 99.09 | 75.57 |
| 2020 | 15681 | 42.96 | 98.72 | 71.57 |
| 2021 | 17213 | 47.29 | 99.27 | 74.12 |
| 2022 | 17170 | 47.17 | 98.96 | 72.71 |
| 2023 | 17221 | 47.44 | 97.67 | 73.83 |
| 2024 | 17661 | 48.39 | 99.68 | 84.75 |
| 2025 | 17386 | 47.76 | 99.71 | 76.8 |

Three distinct reporting regimes exist:

* **2000-2005** - ~7-14 reports/day, almost all at `:40`. Hour coverage 30-58%.
* **2006-2011** - ~14-18 reports/day, `:40` and `:10`. Hour coverage 46-66%, and 2011 has 36
  missing days plus a coverage regression against 2010.
* **2012-2015** - ~31-37 reports/day, mixed minutes. Hour coverage 88-94%.
* **2016-2025** - ~42-48 reports/day, `:00`/`:30` plus SPECI. Hour coverage 98-100%.

### The present-weather field is missing, not just sparse, in the early years

2000-2002 bodies are old-style `AUTO` records such as
`VOTV 010240Z AUTO 00000KT 2 1/2SM 28/22 A2985 RMK T02800220 IEM_DS3505` - there is **no present-weather
group anywhere in 8,486 reports**, so `wxcodes` is `M` for every single one and no thunderstorm can
possibly be detected. That is why 2000, 2001 and 2002 report **0 positives**: it is an encoding gap,
not a thunderstorm-free climate. The field stays unreliable through 2013 (1.8% coverage in 2004, 6.3%
in 2006, then 47-65% from 2007-2011) and only becomes stable from 2014 (>=71.6% every year).

### Label rate by era

| Era | Observed h | Positives | Pos% observed |
|---|---|---|---|
| 2000-2005 | 21943 | 122 | 0.56 |
| 2006-2011 | 30420 | 466 | 1.53 |
| 2012-2013 | 15690 | 322 | 2.05 |
| 2014-2015 | 16141 | 676 | 4.19 |
| 2016-2025 | 86675 | 3018 | 3.48 |

The 0.56% -> 1.53% -> 2.05% -> 4.19% -> 3.48% progression is a detection/encoding trend, not a climate
trend: the same physical seasonality is present in every era, just muted 2-2.5x before 2014.

## 8. Is 2000-2025 suitable for modelling?

**No - not as a whole.** Use a shorter stable window:

| Window | Grid h | Observed h | Hours with present-wx group | Positives (all observed) | Positives (evidence only) | Pos% (evidence) |
|---|---|---|---|---|---|---|
| 2000-2025 | 227928 | 170869 | 110939 (48.67%) | 4604 | 4604 | 4.15 |
| 2012-2025 | 122736 | 118506 | 93752 (76.39%) | 4016 | 4016 | 4.284 |
| 2014-2025 | 105192 | 102816 | 82021 (77.97%) | 3694 | 3694 | 4.504 |
| 2016-2025 | 87672 | 86675 | 68728 (78.39%) | 3018 | 3018 | 4.391 |
| 2017-2025 | 78888 | 78107 | 61600 (78.09%) | 2869 | 2869 | 4.657 |

* **Recommended: 2014-2025** - every year has >=71.6% present-weather coverage, the label rate is
  stable at 3.3-5.0%, observed-hour coverage is 91-100%, and the raw cadence is near-hourly.
* **Stricter alternative: 2016-2025** - the most homogeneous sub-window (98-100% hour coverage,
  42-48 reports/day).
* **Do not use 2000-2013** for supervised learning: the label is not observable in a stable way.

Recommended masking for any consumer:

```python
import pandas as pd
df = pd.read_csv("dataset/historical_thunderstorm_labels_votv.csv", parse_dates=["timestamp_utc"])
df = df[(df.timestamp_utc.dt.year >= 2014) & (df.wx_field_observed == 1)
        & (df.thunderstorm_label.notna())]
```

## 9. Suspicious / inconsistent records found

| Finding | Count | Treatment |
|---|---|---|
| `TS` in raw body only inside `TEMPO`/`BECMG`/trend groups | 1,740 | Excluded. Bodies were cut at the first trend marker before inspection. Never labelled 1. |
| Observed `TS` in body that IEM left out of `wxcodes` (malformed pre-2012 encoding) | 111 | Isolated in `thunderstorm_label_recovered`. Affects 76 extra hours. Not in the primary label. |
| `RETS` / `RETSRA` (recent thunderstorm, not present weather) | 14 | Deliberately **not** labelled 1. |
| Duplicate `valid` timestamps (all Jan 2013, identical duplicates) | 13 rows / 12 stamps | De-duplicated, first kept. No conflicting pairs. |
| Non-observation product (24-hour forecast body) | 1 | Excluded from the report pool. |
| Reports with no altimeter/temperature (truncated `AUTO`) | 21,632 | Kept as reports (the report exists) but exposed via `n_reports_full_decode`; 21,269 hours rest only on such reports. |
| `VCTS`-only reports | 6 | In the primary label (observed present-weather field), isolated in the strict column. |
| 2025-12-31 | 24 hours | Entirely unobserved -> NaN. |

## 10. Limitations

- Observations only describe the aerodrome point; a thunderstorm over the city or the catchment that is not within the station's vicinity is not labelled.
- METAR present weather is a human/sensor judgement of 'thunderstorm observed at the station'; its detection range is finite and reporting practice varies between observers.
- The hour label aggregates reports within [H, H+1), so it is a same-hour (nowcast-style) target, not a lead-time-shifted target. Phase 3 must apply any lead-time shift explicitly.
- Pre-2012 reports use a legacy, frequently malformed encoding in which the present-weather group was sometimes glued to the visibility group ('5000TS') or written with slashes ('TS/-DZ'), causing IEM to leave wxcodes empty. 111 such observed TS reports were recovered into an audit-only column; they are not in the primary label.
- ~1,740 reports mention TS only in TEMPO/BECMG trend groups. These never produce a positive label. They were verified by cutting each body at the first trend marker.
- The dataset is a label table only. It is deliberately NOT merged into the Open-Meteo features and no model was retrained in this phase.
- IEM is a redistribution archive; no formal SLA and occasional ingestion duplicates exist (12 duplicate timestamps in Jan 2013 were de-duplicated, none conflicting).
- Coverage ends 2025-12-30 23:30Z, so 2025-12-31 is entirely unobserved (NaN).

## 11. Reproduce

```
python dataset/download_votv_metar_iem.py     # raw IEM dump -> dataset/raw_metar/
python dataset/build_thunderstorm_labels.py   # labels -> dataset/historical_thunderstorm_labels_votv.csv
python dataset/make_label_docs.py             # metadata + this README
```

## 12. Validation results

All checks PASS (see `historical_thunderstorm_labels_votv_metadata.json` -> `validation`):
timestamps parse, are UTC, unique and strictly increasing; labels contain only 0/1/NaN; the three label
variants are properly nested; no future information is used; station coordinates are constant; the
source field is preserved for audit; the hourly grid is identical to `weather_data_with_code.csv`; and
all pre-existing project files are byte-identical to their pre-phase state.
