# Phase 1B — Multi-year genuine METAR thunderstorm coverage audit

**Date:** 2026-09-19  
**Repo:** V2 working copy `C:\College\SIH2_v2`  
**V1:** not modified  
**V2 application / datasets / models:** not modified (this file only)  
**No training, no pooled dataset build, no Open-Meteo download, no git commit/push**

This is a **coverage and label-availability audit**, not a ranking of climate or forecast skill.

---

## Exact source and window

| Item | Value |
|------|--------|
| Archive | Iowa Environmental Mesonet (IEM) **IN__ASOS** |
| Extract | `https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py` (same service as V1 `dataset/download_votv_metar_iem.py`) |
| Fields | `data=wxcodes` only (no METAR body, no Open-Meteo) |
| Report types | `3` METAR + `4` SPECI |
| Time | UTC, `format=onlycomma`, `missing=M` |
| Window | **2014-01-01 → 2025-12-31** request; last rows typically **2025-12-30 23:00 UTC** |
| Delhi | **VIDP** (IGI), not VIDD |

Open-Meteo `weather_code`, precipitation, and cloud cover were **not** used as targets.

---

## Thunderstorm identification (V1-aligned)

Copied from `dataset/build_thunderstorm_labels.py` present-weather rules, applied **only** to IEM `wxcodes`:

- Tokenise on whitespace and `/`.
- Intensity prefixes `+` / `-` stripped.
- Positive if any token is station **TS…** (`TS`, `TSRA`, `-TSRA`, `TSGR`, …) or vicinity **VCTS**.
- Empty / `M` = **no present-weather group**, not a negative label.
- Hours with **no report** are **not** counted as `0`.
- Hourly normalisation: floor `valid` to UTC clock hour; a TS hour is any hour with ≥1 TS-family report.
- Positive-event rate = **TS hours / hours that contain ≥1 report** (observed hours only).
- “Years with TS” = calendar years with **≥20 unique TS hours** (meaningful annual sample, not a climate ranking).

Body-recovery of malformed pre-2012 METAR was **not** used (window starts 2014).

---

## Station \| City \| Years Covered \| METAR Reports \| TS Observations \| TS Hours \| Years With TS \| Major Gaps \| Readiness

| Station | City | Years Covered | METAR Reports | TS Observations | TS Hours | Years With TS | Major Gaps | Readiness |
|---------|------|---------------|---------------|-----------------|----------|---------------|------------|-----------|
| VOTV | Thiruvananthapuram | 2014–2025 (first 2014-01-01, last 2025-12-30) | 194,908 (wx group 151,829; 77.9%) | 5,708 | 3,694 (3.59% of 102,816 observed hours) | 12 / 12 | none ≥7 days | **REFERENCE** |
| VECC | Kolkata | 2014–2025 | 208,003 (wx 206,542; **99.3%**) | 6,128 | 3,755 (3.60% of 104,375 h) | 12 / 12 | none ≥7 days | **TRAINING-READY** |
| VABB | Mumbai | 2014–2025 | 204,281 (wx 197,349; **96.6%**) | 2,514 | 1,534 (1.48% of 103,849 h) | 12 / 12 | none ≥7 days | **TRAINING-READY** |
| VIDP | Delhi (IGI) | 2014–2025 | 200,211 (wx 197,610; **98.7%**) | 3,139 | 2,031 (1.98% of 102,531 h) | 12 / 12 | none ≥7 days | **TRAINING-READY** |
| VOCI | Kochi | 2014–2025 | 192,751 (wx 171,598; 89.0%) | 3,426 | 2,327 (2.27% of 102,710 h) | 12 / 12 | one ~8.6 d (2018-08-15 → 2018-08-24) | **TRAINING-READY** |
| VOMM | Chennai | 2014–2025 | 205,087 (wx 96,465; **47.0%**) | 3,981 | 2,519 (2.41% of 104,367 h) | 12 / 12 | none ≥7 days | **CONDITIONAL** |
| VOCB | Coimbatore | 2014-02-21 – 2025-12-30 | 181,518 (wx 108,595; 59.8%) | 1,947 | 1,302 (1.29% of 100,884 h) | 12 / 12 | none ≥7 days | **CONDITIONAL** |
| VOBL | Bengaluru (Kempegowda) | 2014–2025 | 208,731 (wx 44,302; **21.2%**) | 2,711 | 1,551 (1.49% of 104,307 h) | 12 / 12 | none ≥7 days | **CONDITIONAL** |
| VOHY | Hyderabad | 2014–2025 (sparser; last 2025-12-30 16:00) | 106,394 (wx 57,931; 54.4%) | 1,632 | 1,065 (1.71% of 62,406 h) | 11 / 12 (≥20 h); 12 with any TS | none ≥7 days (coverage is thin, not one long hole) | **CONDITIONAL** |
| VOML | Mangalore | **2015-05-21 – 2025-12-30** (2014: 0 reports) | 171,190 (wx 56,362; **32.9%**) | 4,922 | 3,116 (3.47% of 89,831 h) | 11 / 12 | none ≥7 days after start | **CONDITIONAL** |
| VOBG | Bengaluru (HAL) | **2015-12-07 – 2025-12-30** (2014: 0; 2015: 661 reports, 0 TS) | 122,879 (wx 26,066; **21.2%**) | 1,826 | 1,439 (2.02% of 71,352 h) | 10 / 12 | two 2023 holes ~10.9 d and ~8.0 d | **CONDITIONAL** |

**Readiness meanings**

- **REFERENCE** — V1 modelling station; keep as the baseline, do not replace.
- **TRAINING-READY** — 2014–2025 IEM series has dense reports, high present-weather completeness, genuine TS in every year, and no year-scale archive hole. Still needs a V1-style hourly label table before training.
- **CONDITIONAL** — genuine TS exists, but missing `wxcodes`, late archive start, or sparse hours would weaken negatives if used like V1.
- **REJECT** — none in this VIDP-corrected candidate list.

VOTV **3,694** TS hours in 2014–2025 matches the V1 modelled-window figure in `README.md`.

---

## Yearly TS hours (unique UTC hours)

| Year | VOTV | VECC | VABB | VIDP | VOCI | VOMM | VOCB | VOBL | VOHY | VOML | VOBG |
|------|------|------|------|------|------|------|------|------|------|------|------|
| 2014 | 279 | 247 | 72 | 153 | 169 | 101 | 83 | 118 | 19 | 0 | 0 |
| 2015 | 397 | 318 | 73 | 104 | 141 | 260 | 107 | 149 | 25 | 154 | 0 |
| 2016 | 149 | 332 | 74 | 213 | 107 | 151 | 82 | 57 | 80 | 169 | 72 |
| 2017 | 291 | 318 | 156 | 168 | 227 | 246 | 126 | 148 | 106 | 247 | 169 |
| 2018 | 415 | 314 | 66 | 125 | 230 | 143 | 176 | 150 | 76 | 256 | 173 |
| 2019 | 293 | 269 | 159 | 106 | 243 | 259 | 115 | 146 | 100 | 271 | 128 |
| 2020 | 297 | 327 | 211 | 150 | 166 | 266 | 87 | 150 | 101 | 186 | 138 |
| 2021 | 426 | 420 | 202 | 194 | 230 | 304 | 72 | 162 | 120 | 385 | 149 |
| 2022 | 296 | 254 | 144 | 183 | 218 | 165 | 113 | 176 | 101 | 329 | 194 |
| 2023 | 331 | 316 | 76 | 243 | 217 | 196 | 114 | 92 | 104 | 319 | 126 |
| 2024 | 320 | 336 | 156 | 206 | 232 | 167 | 141 | 96 | 89 | 486 | 161 |
| 2025 | 200 | 304 | 145 | 186 | 147 | 261 | 86 | 107 | 144 | 314 | 129 |

---

## 1. Stations suitable for the first V2 pooled training experiment

Data-readiness only (complete present-weather + multi-year genuine TS hours):

1. **VOTV** — REFERENCE; include, do not drop.
2. **VECC** — densest `wxcodes`, 3,755 TS hours.
3. **VIDP** — Delhi IGI; 2,031 TS hours, 98.7% wx completeness (VIDD remains unusable from Phase 1A).
4. **VOCI** — 2,327 TS hours, 89% wx completeness; one short 2018 gap.
5. **VABB** — 1,534 TS hours, 96.6% wx completeness (lower positive rate than VOTV/VECC, not a quality ranking).

These five have 12 years of reports, no missing 2014, and present-weather groups on most rows, so V1-style `1` / `0` / `NaN` hourly labels can be constructed without imputing missing hours as negatives.

---

## 2. Stations that require further investigation

| Station | Why not first-wave |
|---------|--------------------|
| **VOMM** | Report cadence is dense, but **~53% of rows lack `wxcodes`**. Positives are real; many hours cannot be labeled `0` without violating “no imputed negatives.” |
| **VOCB** | Usable TS every year; ~40% missing present-weather; starts 2014-02-21. |
| **VOBL** | Full 2014–2025 cadence, but **~79% of reports have no wx group**. |
| **VOHY** | Only ~62k observed hours (vs ~103k peers); 2014–2015 very sparse; IEM also lists **VOHS** (not audited). |
| **VOML** | High TS hour count, but **~67% missing wx** and **no 2014** archive in this extract. |
| **VOBG** | No 2014; 2015 almost empty; HAL vs Kempegowda (**VOBL**) are not interchangeable; wx mostly missing. |

Investigation means: missingness-by-hour (can a `0` be assigned when reports exist but wx is `M`?), SPECI vs METAR mix, and whether VOHS should replace VOHY.

---

## 3. Stations that should not yet be used

- **None of the eleven are REJECT** after switching Delhi to **VIDP**.
- Do **not** treat **VOBG 2014–2015** or **VOML 2014** as training years (empty / near-empty IEM extract).
- Do **not** pool **VOBG + VOBL** as one city without a site-identity decision.
- **VIDD** stays out of scope (Phase 1A REJECT).

---

## Limitations

- Audit used **`wxcodes` only**; no raw METAR body, no recovered-TS column.
- Hourly grid is report-floor hours, **not** yet joined to Open-Meteo feature hours.
- Gaps are holes between consecutive **observed** hours ≥7 days; overnight silence of a few hours is not listed.
- 2025 is incomplete through 30 Dec in the IEM files retrieved on 2026-09-19.
- IEM cache lived in OS temp only; nothing was added under `dataset/`.

---

## PHASE 1B STATUS: **NEEDS REVIEW**

Not **BLOCKED** (IEM reachable; multi-year genuine TS hours exist at all eleven IDs).  
Not **READY** (no V2 hourly label tables, no pooled dataset, no training).  

A first pooled experiment can proceed **after** V1-style label construction for **VOTV + VECC + VIDP + VOCI + VABB** only.
)
