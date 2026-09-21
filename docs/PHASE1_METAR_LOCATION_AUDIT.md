# Phase 1A — Multi-location genuine thunderstorm METAR audit

**Date:** 2026-09-19  
**Repo:** V2 working copy `C:\College\SIH2_v2` (remote `AIML_Thunderstorm_nowcasting_v2`)  
**V1:** not modified  
**V2 application/code/datasets/models:** not modified (this file only)  
**No training, no V2 dataset build, no git commit/push**

This is a **data-availability audit**, not a ranking of forecast skill.

---

## Exact source

| Item | Value |
|------|--------|
| Archive | Iowa Environmental Mesonet (IEM) **IN__ASOS** network (Indian ASOS/METAR) |
| Station metadata | `https://mesonet.agron.iastate.edu/api/1/network/IN__ASOS.json` (145 stations) |
| Observation extract | `https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py` — same service as V1 `dataset/download_votv_metar_iem.py` |
| Present-weather field | IEM **`wxcodes`** (observed present-weather group, not TEMPO/BECMG) |
| VOTV full-history comparison | Existing V1 files `dataset/historical_thunderstorm_labels_votv_stats.json` and `HISTORICAL_THUNDERSTORM_LABELS_README.md` (already on disk; not re-downloaded) |

Open-Meteo `weather_code`, precipitation, cloud cover, and NWP were **not** used as targets.

---

## Retrieval method (lightweight)

1. **Existence and archive window** from IEM `IN__ASOS` metadata (`id`, `name`, `online`, `archive_begin`, `archive_end`). `archive_end` is null for all candidates (listed as still online).
2. **2024 UTC calendar year**, `data=wxcodes` only, `report_type=3` (METAR) and `4` (SPECI), `tz=UTC`, `format=onlycomma`, `missing=M`. One year × 11 stations, wxcodes-only (~0.4 MB each), not a multi-year dump.
3. **VOTV multi-year counts** taken only from the existing V1 label stats (2000–2025), not from a new download.

2024 row counts are **report-level** (typically ~30 min), not hourly labels. Hourly `NaN` vs `0` was **not** built for non-VOTV stations.

---

## Thunderstorm identification rule (V1-aligned)

Used only on **`wxcodes`**:

- A report is thunderstorm-positive if any token (split on space/`/`) contains METAR **TS** present weather, including intensity (`-TSRA`, `+TS`) and vicinity **`VCTS`**.
- `wxcodes` empty or `M` = **no present-weather group** (not a negative thunderstorm observation).
- No conversion of missing hours/reports into label `0`.
- Trend/forecast groups are not in `wxcodes` (V1 design). Raw METAR body text was not searched in this audit.

---

## Station | City | Exists | Data Period | Approx Observations | Thunderstorm Observations | Coverage Quality | Status

| Station | City | Exists | Data Period | Approx Observations | Thunderstorm Observations | Coverage Quality | Status |
|---------|------|--------|-------------|---------------------|---------------------------|------------------|--------|
| VOTV | Thiruvananthapuram | Yes (`IN__ASOS`, online) | IEM metadata begin **1996-01-31**; V1 dump **2000-01-01 → 2025-12-30** | V1: **273,125** unique reports; hourly grid 227,928 h, **170,869** observed, **57,059** missing | V1: **6,717** TS reports in `wxcodes`; **4,604** positive **hours** (2.694% of observed hours). 2024 sample: 17,661 reports, 529 TS reports, 14,968 with a present-weather group | High for 2014–2025 pipeline (V1). Early 2000s present-weather often missing | **REFERENCE** |
| VOBG | Bengaluru (HAL / older Bangalore id) | Yes | Metadata begin **2001-01-31**; 2024 reports 01-01 → 12-30 | 2024: **16,513** reports | 2024: **268** TS reports. Present-weather group: 3,942; `M`/empty: **12,571** | Station exists; 2024 cadence OK but **present-weather often absent** | **CONDITIONAL** |
| VOBL | Bengaluru (Kempegowda) | Yes | Metadata begin **2008-07-21**; 2024 01-01 → 12-30 | 2024: **17,521** reports | 2024: **157** TS reports. Present-weather: 4,375; `M`/empty: **13,146** | Same issue as VOBG: frequent missing `wxcodes` | **CONDITIONAL** |
| VABB | Mumbai | Yes | Metadata begin **1944-12-31**; 2024 01-01 → 12-30 | 2024: **17,397** reports | 2024: **262** TS reports. Present-weather: 17,380; `M`: **17** | 2024 almost complete `wxcodes`. Multi-year hourly labels **not** built | **CONDITIONAL** |
| VOHY | Hyderabad | Yes | Metadata begin **1944-12-31**; 2024 first 01-01 01:30 last 12-30 18:30 | 2024: **9,802** reports (sparser than peers) | 2024: **127** TS reports. Present-weather: 5,992; `M`: 3,810 | Exists; fewer reports and more missing wx than VABB/VECC. IEM also lists **VOHS** (RGIA, begin 2008-07-21) — not in this candidate list | **CONDITIONAL** |
| VOMM | Chennai (IEM name: Madras) | Yes | Metadata begin **1944-12-31**; 2024 01-01 → 12-30 | 2024: **17,499** reports | 2024: **259** TS reports. Present-weather: 10,368; `M`: 7,131 | Dense reports; moderate missing present-weather | **CONDITIONAL** |
| VIDD | Delhi (IEM name: Delhi) | Yes (id exists) | Metadata begin **1944-12-31** | **2024: 154 reports only** (07 Jan–29 Dec, sparse) | 2024: **4** TS reports | Metadata claims a long archive; **2024 volume is not a usable aerodrome series**. IGI is **VIDP** (in `IN__ASOS`, begin 1975-03-07) — not requested here | **REJECT** |
| VECC | Kolkata | Yes | Metadata begin **1944-12-31**; 2024 01-01 → 12-30 | 2024: **17,594** reports | 2024: **565** TS reports. Present-weather: 17,269; `M`: 325 | Dense 2024 reports and frequent genuine TS in `wxcodes`. Multi-year labels not built | **CONDITIONAL** |
| VOCI | Kochi | Yes | Metadata begin **2005-02-09**; 2024 01-01 → 12-30 | 2024: **16,428** reports | 2024: **342** TS reports. Present-weather: 15,383; `M`: 1,045 | Backup; 2024 looks usable pending full years | **CONDITIONAL** |
| VOCB | Coimbatore | Yes | Metadata begin **1973-01-01**; 2024 01-01 → 12-30 | 2024: **15,840** reports | 2024: **226** TS reports. Present-weather: 11,051; `M`: 4,789 | Backup; genuine TS present | **CONDITIONAL** |
| VOML | Mangalore | Yes | Metadata begin **1972-12-31**; 2024 01-01 → 12-30 | 2024: **16,845** reports | 2024: **794** TS reports. Present-weather: 6,928; `M`: **9,917** | Backup; many TS tokens in 2024 but **high missing `wxcodes`** | **CONDITIONAL** |

**Status meanings used**

- **REFERENCE** — VOTV V1 baseline; do not replace.
- **TRAINING-READY** — none besides the V1 VOTV series, because no other station has a completed multi-year hourly genuine-label table in this audit.
- **CONDITIONAL** — station is in IEM; 2024 `wxcodes` contain genuine TS; full-period hourly labels and missingness rules are not yet built.
- **REJECT** — 2024 archive is not a viable observation series (VIDD).

---

## Comparison notes (data readiness only)

- All **eleven ICAO ids exist** in `IN__ASOS`.
- **VIDD is not a substitute for Delhi IGI.** 154 reports in 2024 vs ~17k at VOTV/VABB/VECC.
- **VABB** and **VECC** had the most complete 2024 present-weather fields among primaries (almost all rows had `wxcodes` ≠ `M`).
- **VOBG / VOBL** have good report cadence but **most 2024 rows lack a present-weather group**, so many hours cannot be labeled 0 or 1 without violating “no imputed negatives.”
- **VOML** 2024 TS **report** count is high (794) but so is missing wx; positives are real, zeros would be weakly supported.
- **VOHY** vs **VOHS**: two Hyderabad ids; candidate list used VOHY only.
- 2024 TS **report** counts are not hourly event rates and are not comparable to V1’s 4,604 positive **hours** without the same hour-grid aggregation.

---

## Recommended next candidates (readiness only, not “best climate”)

Based **only** on 2024 IEM `wxcodes` density + genuine TS presence + metadata existence, if Phase 1B builds labels like V1:

1. **VECC** (Kolkata) — dense reports, dense present-weather, 565 TS reports in 2024.
2. **VABB** (Mumbai) — dense reports, nearly complete `wxcodes`, 262 TS reports in 2024.
3. **VOCI** (Kochi, backup) — dense present-weather, 342 TS reports in 2024.
4. **VOMM** (Chennai) — dense reports, 259 TS reports; more missing wx than VABB/VECC.
5. **VOCB** (Coimbatore, backup) — usable 2024 volume; more missing wx.

Keep **VOTV** as reference; do not drop it.

Do **not** advance **VIDD** until a different ICAO (e.g. **VIDP**) is explicitly in scope and audited the same way.

Treat **VOBG / VOBL / VOML / VOHY** as second-wave: existence is proven; present-weather completeness needs a multi-year `wxcodes` missingness audit before training.

---

## Limitations

- Non-VOTV **multi-year** observation and thunderstorm-hour counts were **not** downloaded (by design). Table numbers for those stations are **2024 report-level** only.
- IEM `archive_begin` (e.g. 1944) does **not** mean continuous, decoded present-weather back to that year (V1 VOTV already showed empty wx in 2000–2002).
- No SPECI vs METAR split beyond requesting both types.
- No hourly grid, no `NaN` policy application except the rule stated.
- Bengaluru has two sites (VOBG and VOBL); they are not interchangeable.
- Hyderabad VOHS and Delhi VIDP appeared in metadata but were not candidate rows.
- Network JSON `IncompleteRead` occurred on urllib; metadata was completed via curl to a temp file that was **not** kept in the repo.

---

## PHASE 1A STATUS: **NEEDS REVIEW**

Not **BLOCKED** (IEM is reachable; genuine `wxcodes` TS exists at multiple stations).  
Not **READY** (only VOTV has a full genuine hourly label series; VIDD is unusable; others need multi-year label construction and missingness rules before any V2 training set).
