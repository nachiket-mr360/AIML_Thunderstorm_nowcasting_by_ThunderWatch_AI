# Phase 14B — V2 Data-Source & Latency Audit

**Date:** 2026-09-21  
**Project:** SIH26072 V2  
**Scope:** audit only. No live system, no simulated feeds, no fake timestamps, no satellite/radar/lightning fabrication, no training, no Flask, no dashboard, no Phase 14C.

**PHASE STATUS: AUDIT COMPLETE**

This is **not** an operational readiness claim.

---

## 1. Objective

Record, with evidence already in the V2 tree and prior documentation-only audits, the **availability, timestamp semantics, retrieval method, and latency suitability** of every data layer that a future replay / demo / live architecture might use.

Historical data is **not** treated as real-time data.

---

## 2. Audit methodology

| Rule | Application |
|------|-------------|
| Evidence | Existing Phase 1B, 2A/3, 7, 8, 9, 12A, 13A, 14A docs and collectors. No large downloads. |
| Latency | If the source does not document a number: **NOT DOCUMENTED**. No invented delays. |
| Access | If login/grant was not performed: **NOT VERIFIED**, **ACCESS_PENDING**, or **INSTITUTIONAL_ACCESS_REQUIRED**. |
| Classification | Exactly one of: REPLAY_READY, NEAR_REAL_TIME_CANDIDATE, OPERATIONAL_CANDIDATE, ACCESS_PENDING, INSTITUTIONAL_ACCESS_REQUIRED, REFERENCE_ONLY, REJECTED, UNKNOWN. |
| Prediction-time | YES / NO / CONDITIONAL / UNKNOWN — can the source provide information **legitimately available at or before T**? |
| No ranking | Sources are not ordered by “best skill.” |

Artifacts: `outputs/v2_data_sources/phase14b/data_source_audit.csv`, `data_source_audit.json`.

---

## 3. Atmospheric observations

**Training / replay source:** Open-Meteo **Historical Weather Archive** (`archive-api.open-meteo.com/v1/archive`), Phase 2A/3.

**Required variables (Phase 3 / 14A):** `temperature_2m`, `relative_humidity_2m`, `surface_pressure` (Pa), `wind_speed_10m`, `wind_direction_10m`, `precipitation`, `cloud_cover`. `weather_code` may exist in raw pulls and is **forbidden** as a predictor.

| Item | Finding |
|------|---------|
| Timestamp | UTC hourly valid time of archive hour **T** (`timezone=GMT` in collectors). |
| Historical | In-repo **2014-01-01 → 2025-12-31** at five ICAO points. |
| Current/recent | Archive is **HISTORICAL**. Suitability of “latest archive hour” as a live observation is **NOT VERIFIED**. |
| Retrieval | Unauthenticated HTTPS GET. |
| Latency | **NOT DOCUMENTED**. ERA5-class archives are not a live publication contract (CDS ERA5 text cited in Phase 7 as ~5-day class for CDS itself; that number is **not** assigned to Open-Meteo archive here). |
| Represent T? | **YES** for replay of that archived hour. **CONDITIONAL** if used as wall-clock live observation. |
| Class | **HISTORICAL**. Classification: **REPLAY_READY**. |

**Separate product — Open-Meteo Forecast API** (`api.open-meteo.com`), used by the copied V1 live nowcast:

- **NEAR_REAL_TIME** product family, not the training archive.
- Served cell / values may differ from Phase 2A archive cells.
- Latency **NOT DOCUMENTED**.
- Prediction-time: **CONDITIONAL**.
- Classification: **NEAR_REAL_TIME_CANDIDATE** (demo only). **Not LIVE/OPERATIONAL** as an observation network.

---

## 4. NWP

**Validated Model B source (do not replace):** Open-Meteo **Historical Forecast** `models=gfs_global`, overlap **2021-04-01 → 2025-12-31**. Forecast fields, **not** observations, **not** ERA5.

Variables: CAPE, CIN, lifted index, T, RH, pressure (**hPa**), wind, precipitation, cloud cover.

| Item | Finding |
|------|---------|
| Init/run | Homogenized **valid-time** series. Phase 8B `nwp_time_shift=false`. |
| Valid time | Same UTC hour T as the atmospheric row. |
| Latency | **NOT DOCUMENTED**. Phase 13A: valid-time Historical Forecast is **not** an operational latency contract. |
| Available at/before T? | **YES** for historical replay (field labeled valid at T). **NO** as proof the cycle was published and received before wall-clock T. |
| Classification | **REPLAY_READY**. |

**Open-Meteo Previous Runs API:** same vendor; run vs valid-time must be frozen to stay causal. Classification: **NEAR_REAL_TIME_CANDIDATE**. Prediction-time: **CONDITIONAL** (only if the chosen previous run was issued ≤ T). Not in Model B.

**NOAA GFS / ECMWF Open Data:** documentation-only in Phase 7. Possible **OPERATIONAL_CANDIDATE** for a future live extract. Not collected. Prediction-time: **CONDITIONAL**. Must not overwrite the validated table.

ERA5 CDS remains reanalysis, **not** live NWP (account not created).

---

## 5. Satellite

**Intended:** MOSDAC / INSAT-3DR Cloud Mask **`3RIMG_L2B_CMK`**.

Phase 10 was **not** executed. MOSDAC account **not created**. No download.

| Item | Finding |
|------|---------|
| Temporal / spatial resolution | **NOT VERIFIED** this phase. |
| Timestamp | Product scan/valid time on MOSDAC; hourly alignment would need documented resample of scans **≤ T**. |
| Access | Registration/approval typical. **ACCESS_PENDING**. |
| Latency | **NOT DOCUMENTED**. |
| Available at/before T? | **UNKNOWN**. |
| Classification | **ACCESS_PENDING**. |

INSAT imagers are **not** lightning mappers (Phase 9A). No satellite feature columns.

---

## 6. Radar

**Intended:** IMD radar ecosystem / Radar Data Supply Portal.

Phase 11 pending. Prior V2 design: **college/university letter and undertaking**.

| Item | Finding |
|------|---------|
| Products / resolution / timestamps | **NOT VERIFIED** (no portal login). |
| Historical / NRT | **NOT VERIFIED**. |
| Station-centric features | Possible **in principle** after access; not implementable now. |
| Latency | **NOT DOCUMENTED**. |
| Available at/before T? | **UNKNOWN**. |
| Classification | **INSTITUTIONAL_ACCESS_REQUIRED**. |

Do **not** scrape public map tiles. No synthetic radar.

---

## 7. Real-time lightning

| Source | Type | Access | Classification | At/before T? |
|--------|------|--------|----------------|--------------|
| IITM LLN (Damini ≠ bulk API) | Ground OBSERVATION | Research grant | **INSTITUTIONAL_ACCESS_REQUIRED** | **CONDITIONAL** after grant |
| IMD LLN / public nowcast pages | Observation internally; public mostly maps/alerts | MoU / request | **INSTITUTIONAL_ACCESS_REQUIRED** | **CONDITIONAL** after grant |
| NASA TRMM LIS / ISS-LIS flashes | Overpass OBSERVATION | Earthdata (not used here) | **REFERENCE_ONLY** | **CONDITIONAL** on overpass time ≤ T only; **NO** as 24/7 hourly |
| Combined LIS annual thunder hours | Climatology in-repo | File present | **REFERENCE_ONLY** | **NO** |
| WWLLN | Global VLF OBSERVATION | Agreement / typical cost | **INSTITUTIONAL_ACCESS_REQUIRED** | **CONDITIONAL** if licensed |

**Do not** promote `annual_thunder_hours` to real-time lightning (Phase 12B). No synthetic lightning. No TRAINING-READY open flash archive (Phase 9A).

---

## 8. Historical labels

**Source:** IEM IN__ASOS METAR+SPECI present weather (`wxcodes`), genuine TS-family tokens (V1/Phase 1B). Not Open-Meteo `weather_code`.

- `thunderstorm_target` at hour T; multi-lead `target_1h/2h/3h` = METAR TS at **T+L**.
- Hourly floor of report `valid` time UTC.
- Missing/unavailable hours stay **NA**, never filled with 0.
- **TRAINING / EVALUATION LABEL, not a prediction input.**
- Available at/before T as a **predictor**: **NO**.
- Classification: **REFERENCE_ONLY**.

---

## 9. Prediction-time availability matrix

| Layer | Source (current) | At or before T? | Semantics |
|-------|------------------|-----------------|-----------|
| Atmosphere archive | Open-Meteo archive | **YES** (replay) | Archive hour T |
| Atmosphere forecast API | Open-Meteo forecast | **CONDITIONAL** | Different product; lag NOT DOCUMENTED |
| NWP Historical Forecast | gfs_global valid time | **YES** replay / **NO** live contract | Valid time = T, not issuance time |
| NWP Previous Runs | Open-Meteo | **CONDITIONAL** | Run issued ≤ T required |
| Satellite CMK | MOSDAC | **UNKNOWN** | ACCESS_PENDING |
| Radar | IMD portal | **UNKNOWN** | INSTITUTIONAL_ACCESS_REQUIRED |
| Lightning LLN | IITM/IMD | **CONDITIONAL** | No data in repo |
| LIS thunder hours | NASA combined | **NO** | Annual climatology |
| METAR targets | IEM | **NO** as input | Labels at T or T+L |

---

## 10. Replay mode (MODE A)

**Legitimately possible: YES.**

In-repo Open-Meteo archive + Historical Forecast NWP + Phase 14A builder + Phase 13B engine. METAR only for scoring after the fact.

This is a **historical replay**, not live operations.

---

## 11. Near-real-time mode (MODE B)

**Not claimed.** Evidence is insufficient.

Forecast-API atmosphere and Previous-Runs NWP are **candidates** only, with product mismatch and **NOT DOCUMENTED** latency. Satellite/radar/lightning are blocked.

---

## 12. Operational / live mode (MODE C)

**Not claimed.** No operational NWP latency contract, no MOSDAC/radar/LLN access, Model B is a research complete-case prototype.

---

## 13. Access blockers

- MOSDAC account / approval (**ACCESS_PENDING**)
- IMD radar: university letter / portal (**INSTITUTIONAL_ACCESS_REQUIRED**)
- IITM/IMD/WWLLN lightning extracts (**INSTITUTIONAL_ACCESS_REQUIRED**)
- Live NWP: missing init-time vs valid-time contract
- Atmosphere: archive ≠ forecast API

---

## 14. Recommended next data-source actions

Documentation / access only (not implementation in this phase):

1. Keep Mode A replay as the only supported path for Model B.
2. If a demo is designed later: measure Forecast API vs archive and Previous Runs vs Historical Forecast; record latency only if measured.
3. MOSDAC registration for `3RIMG_L2B_CMK` metadata (no bulk pull until a later phase).
4. Institutional radar letter if radar remains in scope.
5. Lightning only via written IITM/IMD grant; otherwise leave absent.
6. Never treat METAR T+L or LIS climatology as live predictors.

---

## 15. Scientific limitations

- No invented latency numbers.
- Historical Forecast valid-time ≠ operational availability.
- Archive reanalysis-class atmosphere ≠ METAR and ≠ live AWS.
- Complete-case Model B fails closed on missing NWP.
- Five ICAO sites only.
- This audit does **not** modify the ML pipeline.

---

## Integrity

SHA-256 unchanged vs Phase 14A/13:

| Artifact | SHA-256 |
|----------|---------|
| 13A contract | `a3aad93b211aff6a2da9b6a5e859c9649928a60ea2667f9888d854168d884940` |
| 13B engine | `ef9c55e395df5dec1aac646ac447c96d2c449888930640c3303f5cfa43a6f3d3` |
| 14A builder | `05aa33c092758e728bc4f2d97f605b95038e4814457b1fdec244ab3a9629c0cf` |
| 1h model | `a266c71bf901c4ea9392882b619e1a40905279af89db58123227a07f49733f21` |
| 2h model | `e99ff3e9fba4cef8a3e1d15e6c29cfdbee801200369230bd652a45a71b376c6c` |
| 3h model | `20692e504594ad17ea6fcc43dfe6e8899a5d5bdc2df93015df92327b97dc16ed` |

---

## Final decision matrix

| Layer | Current status | Prediction-time usable? | Future action |
|-------|----------------|-------------------------|---------------|
| Atmospheric | REPLAY_READY (archive); Forecast API NEAR_REAL_TIME_CANDIDATE | YES (replay); CONDITIONAL (forecast API) | Do not conflate products |
| NWP | REPLAY_READY Historical Forecast; Previous Runs candidate; GRIB operational candidate not collected | YES (replay valid time); CONDITIONAL (live/previous run); NO (live Historical Forecast contract) | Do not replace Phase 8 table |
| Satellite | ACCESS_PENDING | UNKNOWN | MOSDAC approval; no columns yet |
| Radar | INSTITUTIONAL_ACCESS_REQUIRED | UNKNOWN | College letter; no tiles |
| Real-time lightning | INSTITUTIONAL_ACCESS_REQUIRED / REFERENCE_ONLY | NO now; CONDITIONAL after grant; NO for climatology | No synthetic lightning |
| Historical labels | REFERENCE_ONLY | NO as input | Keep genuine METAR definition |

**STOP.** Phase 14C not started. No Flask. No dashboard. No large downloads.
