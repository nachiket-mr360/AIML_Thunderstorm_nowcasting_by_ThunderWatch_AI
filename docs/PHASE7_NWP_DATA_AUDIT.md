# Phase 7 — NWP / model-data availability audit (V2)

**Date:** 2026-09-19  
**Repo:** V2 working copy `C:\College\SIH2_v2`  
**V1:** not modified  
**V2 datasets / models / Flask:** not modified  
**No accounts created, no credentials requested, no bulk download, no training, no NWP features added**

This is a **reproducibility and access audit**, not a ranking of forecast skill. Sources are described by documented trade-offs only.

V2 baseline (for overlap): five ICAO locations **VOTV, VECC, VIDP, VOCI, VABB**; atmospheric features and METAR labels **2014-01-01 → 2025-12-31 UTC** (Phase 2A/5/6). Phase 6 models are 1h/2h/3h Random Forests on **causal Open-Meteo archive surface features + genuine METAR targets**.

---

## PHASE STATUS

**READY**

Genuine historical **model / reanalysis** fields can be added later in a documented, student-reproducible way **without** changing V1 or the current V2 tables. No source covers every priority convective variable for the **full** 2014–2025 window at hourly resolution without an account. A future experiment can proceed after the operator chooses a source and (for Copernicus) creates a CDS account.

---

## Method (lightweight only)

| Check | What was done |
|-------|----------------|
| Open-Meteo Historical Forecast API | Unauthenticated GET, 1-day windows at VOTV and the other four IEM points |
| Open-Meteo Previous Runs API | Same; documented start-date range from API error body |
| Open-Meteo Archive API | Same; already used in V2 Phase 2A for surface predictors (not NWP) |
| Copernicus CDS catalogue | Unauthenticated STAC collection JSON for ERA5 single-level and pressure-level datasets |
| Other sources | Public documentation only (ECMWF Open Data, NOAA GFS archives) — no download |

Request coordinates (same as Phase 2A): VOTV 8.482, 76.920; VECC 22.6547, 88.4467; VIDP 28.5667, 77.1167; VOCI 10.15, 76.4; VABB 19.1005, 72.8585.

---

## Source comparison

| Source | Variables | Historical coverage | Resolution | Access | Authentication | Spatial characteristics | V2 suitability | Limitations |
|--------|-----------|---------------------|------------|--------|----------------|-------------------------|----------------|-------------|
| **Open-Meteo Historical Forecast API** (`historical-forecast-api.open-meteo.com/v1/forecast`) | Hourly: CAPE, CIN (`convective_inhibition`), lifted index, boundary-layer height, 2 m T, RH, surface pressure, 10 m u/v, geopotential height (e.g. 500 hPa), precipitation, cloud cover (total/low/mid/high). Availability **depends on underlying model**. | API-enforced range **2016-01-01 → ~present** (probe: start before 2016 → 400). Convective indices at VOTV were **null Jan–Mar 2021**, **populated from 2021-04-01** in a 1-day probe. Surface T/precip exist earlier than CAPE. | Hourly. Native grid set by `models=` (GFS ~0.11–0.25°, IFS 0.25°/0.4°, ICON ~0.125°, UKMO ~10 km). | HTTPS GET, no key for non-commercial fair use. Programmatic download permitted. | **None** for the public endpoint used here. | Point query; API returns **served cell** (VOTV: 8.471, 76.933 for `best_match`). All five V2 sites returned 200. `icon_eu` **400** (Europe domain). | **Candidate for a future NWP-style experiment** if the study window is allowed to start **~Apr 2021** (or later) and the chosen `models=` is frozen. Not a drop-in for full 2014–2025. | Past **forecast** fields, not observations. `best_match` can change which model fills a variable over time. Early years in 2016–2021 have holes for CAPE/CIN/LI/z500. Not the same product as V2 Phase 2A archive. Fair-use rate limits. |
| **Open-Meteo Previous Runs API** (`previous-runs-api.open-meteo.com/v1/forecast`) | Same family as historical forecast (CAPE, CIN, LI, BLH, T, RH, pressure, wind, z500, precip, cloud) when the archived run contains them. | Same API range **2016-01-01 → ~2026-10-04**. Probe: CAPE **null** on 2021-03-01; **populated** on 2024-01-01 and 2024-06-01 at VOTV. | Hourly archived NWP **runs** (lead-time structure; previous-run semantics). Spatial grid = chosen model. | HTTPS GET, no key. Programmatic. | **None**. | Same point→cell mapping as other Open-Meteo APIs. Five sites queryable. | **Candidate** when the goal is **true previous-cycle NWP** (closer to operational nowcast inputs) rather than a homogenized historical-forecast series. | Shorter effective convective-variable coverage than ERA5. Need to document `previous_day` / run vs valid-time to stay causal. Same fair-use limits. Not 2014–2020 complete for CAPE. |
| **Open-Meteo Archive API** (`archive-api.open-meteo.com/v1/archive`) — **already in V2 Phase 2A** | Surface: T2m, RH, pressure, wind, precip, cloud, **BLH**. CAPE / CIN / LI / z500 accepted in the query but returned **`undefined` units and all-null** (2014-01-01 and 2024-06-01 probes). | **1940–present** (ERA5-based reanalysis through Open-Meteo). V2 already holds **2014–2025 hourly** surface series for five sites. | Hourly. ERA5 ~0.25° (land products finer). | HTTPS GET, no key. Already used in this repo. | **None**. | Served cell documented in Phase 2A metadata. | **Not a new NWP source.** Suitable to **keep as the current feature backbone**. BLH could be added later from the **same** archive without a new vendor. | **Reanalysis**, not a forecast. Convective indices **not usable** via this endpoint in the probes. Adding archive BLH is feature engineering on reanalysis, not NWP. |
| **Copernicus ERA5 / CDS** (`reanalysis-era5-single-levels`, `reanalysis-era5-pressure-levels`) | Single levels: CAPE, CIN, BLH, 2 m T, 2 m dewpoint (RH derivable), surface/MSLP, 10 m u/v, total precip, total/low/mid/high cloud, etc. Pressure levels: geopotential, T, u/v, RH/q at 1000–1 hPa. Lifted index **not** a standard ERA5 single-level parameter (can be computed from soundings). | **1940–present**, hourly. Updated with ~5-day latency (CDS collection text). Fully covers **2014–2025**. | Hourly. ~**0.25°** (31 km). Ensemble 3-hourly. | CDS web + **cdsapi**. Programmatic download **permitted after** licence accept. | **Yes:** Copernicus CDS **account + API key/token**. Licence must be accepted. **Not created in this phase.** | Global lat/lon. Five V2 points are inside the domain; extract as nearest cell or small bbox. | **Candidate** when the experiment needs **full 2014–2025** overlap and physically consistent convective/reanalysis fields. Distinct from Open-Meteo forecast APIs. | **Reanalysis (analysis+short forecast)**, not live NWP. Using ERA5 at time T with METAR at T+1h is **not** the same as using a real-time model run. Volume explodes if full grids are stored; **point/bbox extracts** stay small. Student must register. CDS queue times. Terms of use. |
| **ECMWF Open Data** (IFS 0.25°, documented public NWP) | CAPE and many IFS fields on recent open-data catalogue; not a 2014–2025 research archive. | Operational **open data is recent** (order of last ~days to ~few years depending on product), **not** a drop-in 2014–2025 archive. | Hourly/stepped forecast; ~0.25°. | HTTPS / AWS, no CDS account for the open subset. | Typically **none** for the open subset. | Global. Five sites in domain. | **Poor fit for historical V2 training**; useful later for **live** NWP if a short-history experiment is designed separately. | Does not cover the V2 training decade. Changing catalogue/retention. |
| **NOAA GFS historical / AWS NOMADS** | CAPE, CIN, LI, precip, T, RH, winds, heights, PBL — GFS native. | Multi-year GFS archives exist (NOMADS / AWS `noaa-gfs-bdp-pds` and similar), but **run-based GRIB**, not a simple point API. Coverage of older cycles is incomplete vs ERA5. | 3-hourly older GFS; 1-hourly more recent; ~0.25°. | Public HTTP/S3. Programmatic. | **None** for public buckets. | Global. Point extract requires GRIB tooling. | Possible for a **GFS-only** student project if someone accepts GRIB pipelines. Heavier than Open-Meteo. | Large volume if not subsetted. Reproducibility needs pinned cycle, fxxx lead, and grid. Not lightweight. |

---

## Probe notes (not a full dataset)

| Probe | Result |
|-------|--------|
| Historical Forecast `start_date=2014-01-01` | 400: allowed range **2016-01-01 to 2026-10-04** |
| Historical Forecast CAPE at VOTV | Null 2021-01…03; **non-null 2021-04-01** (1760 J/kg) and later sample days including **2025-12-31** at VIDP |
| Historical Forecast all five sites | HTTP 200; each maps to a nearby model cell |
| Previous Runs CAPE 2024-01-01 VOTV | Non-null hourly series (e.g. 890…190 J/kg) |
| Archive CAPE/CIN/LI/z500 | Units `undefined`, values **null** |
| Archive BLH 2014-01-01 VOTV | Non-null (e.g. 255 m at 00 UTC) |
| CDS ERA5 collections | Public STAC 200; hourly 1940–present described; download API not exercised (would need auth) |

Approximate volume **if** a future phase extracts **point** hourly series only (5 sites × ~12 variables × ~10 years): **tens of MB CSV**, not a “huge dataset”. Full ERA5 or GFS grids would be **orders of magnitude larger** and are **not** recommended.

---

## Recommended candidate source(s) for a future experiment

Do **not** treat this as a skill ranking.

**A. Open-Meteo Historical Forecast (and/or Previous Runs), frozen `models=`**

- Trade-off: **no account**, same vendor as live V1/V2 Open-Meteo usage, five sites already work, convective variables exist in **recent** years.
- Trade-off: **does not cover 2014–early 2021 CAPE/CIN/LI**; mixing `best_match` across years is a hidden model change.
- Use Previous Runs if the scientific claim is “features a forecaster could have had from an NWP cycle,” and Historical Forecast if the claim is “archived forecast valid at T.”

**B. Copernicus ERA5 via CDS (point extract)**

- Trade-off: **full overlap with 2014–2025**, CAPE/CIN/BLH/pressure-level heights documented, standard in student climate/NWP papers.
- Trade-off: **reanalysis ≠ operational NWP**; **account + licence** required; not identical to the Open-Meteo archive already in V2 (do not silently mix without a join spec).

A reasonable **future experiment design** (not executed here): keep Phase 3 Open-Meteo archive features as the baseline; add **either** (A) forecast CAPE/CIN/LI on the **overlapping sub-period** **or** (B) ERA5 CAPE/CIN/BLH on the **full** period; never both without an explicit ablation.

---

## Exact variables to request later (when collection is approved)

**If Open-Meteo Historical Forecast / Previous Runs** (hourly, UTC, one year per request, frozen `models`, e.g. `gfs_global` **or** `ecmwf_ifs025` — pick one and do not switch):

- `cape`
- `convective_inhibition`
- `lifted_index`
- `boundary_layer_height`
- `temperature_2m`
- `relative_humidity_2m`
- `surface_pressure`
- `wind_u_component_10m`
- `wind_v_component_10m`
- `geopotential_height_500hPa` (and optionally 850/700 if the chosen model returns them)
- `precipitation`
- `cloud_cover`, `cloud_cover_low`, `cloud_cover_mid`, `cloud_cover_high`

Do **not** use Open-Meteo `weather_code` as a thunderstorm label (V1/V2 rule).

**If ERA5 CDS** (single levels + optional pressure levels):

- Single: `convective_available_potential_energy`, `convective_inhibition`, `boundary_layer_height`, `2m_temperature`, `2m_dewpoint_temperature`, `surface_pressure` / `mean_sea_level_pressure`, `10m_u_component_of_wind`, `10m_v_component_of_wind`, `total_precipitation`, `total_cloud_cover` (and low/mid/high if needed)
- Pressure: `geopotential` at 500/700/850 hPa; optional T, u, v, q
- Lifted index: **derive** from profile if required; do not assume a native CDS field

---

## Exact overlapping period with V2

| Source | Overlap with V2 2014-01-01 … 2025-12-31 |
|--------|----------------------------------------|
| Open-Meteo Archive (current features) | **Full** (already collected) |
| Open-Meteo Historical Forecast / Previous Runs (API window) | **2016-01-01 … 2025-12-31** calendar, but **convective indices not filled until ~2021-04** at the probed VOTV cell |
| Practical Open-Meteo NWP-variable experiment | **2021-04-01 … 2025-12-31** (verify per model/variable before a full pull) |
| ERA5 CDS | **Full 2014-01-01 … 2025-12-31** (reanalysis, ~5-day production lag does not affect this closed historical window) |

Phase 6 splits (train through 2022-05-27, val through 2024-03-14, test from 2024-03-14) remain usable only if the **new** variables exist on both sides of each cut. An Open-Meteo-CAPE experiment that starts in 2021 **shortens train** relative to Phase 6.

---

## Account / token

| Source | Required now? | Required for a future full collect? |
|--------|----------------|--------------------------------------|
| Open-Meteo Historical Forecast / Previous Runs / Archive | No | No (public HTTPS; respect fair use) |
| Copernicus ERA5 | **Yes** (CDS account, API key, accepted dataset licence) | Operator must register; this audit did **not** |
| ECMWF Open Data / NOAA GFS public | No | No, but tooling (ecCodes/cfgrib) is on the operator |

---

## What the project operator must do manually (if anything)

1. **Decide** experiment claim: archived **forecast** (Open-Meteo) vs **reanalysis** (ERA5) vs **previous NWP cycle** (Previous Runs).
2. If ERA5: create a **Copernicus CDS** account, accept the ERA5 licence, store the API key **outside the git repo**, install `cdsapi`.
3. Freeze **model id** (Open-Meteo) or **dataset version / grid** (ERA5) in a future collector script — do not use silent `best_match` across a decade.
4. Keep V1 and current V2 feature/label CSVs unchanged; write any NWP extract under a **new** directory in a later phase.
5. Do not pull full global GRIB/NetCDF; request **five points or small boxes** only.

Nothing else is required to close **this** audit phase.

---

## What this phase did not do

- No CDS login, no Open-Meteo commercial key, no bulk download  
- No change to `dataset/multilocation/` or models  
- No training, no feature join, no V1 edits, no git commit/push  

---

## PHASE STATUS (repeat)

**READY**
)
