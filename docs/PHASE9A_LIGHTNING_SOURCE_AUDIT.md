# Phase 9A — Lightning data source audit (V2)

**Date:** 2026-09-20  
**Repo:** V2 working copy `C:\College\SIH2_v2`  
**V1:** not modified  
**Phase 3–8 artifacts / V2 datasets / models / Flask:** not modified  
**No accounts created, no data requested, no bulk download, no training, no lightning features**

This is an **access and observation-identity audit**, not an implementation plan and not a ranking of nowcast skill.

V2 modelling window (for overlap only): five ICAO locations **VOTV, VECC, VIDP, VOCI, VABB**; atmospheric features and METAR labels **2014-01-01 → 2025-12-31 UTC**.

**Required product class:** genuine **historical lightning observations** (flashes / strokes / events, or counts derived from those events).

**Explicitly not observations:** thunderstorm forecasts, lightning forecasts, weather warnings, METAR thunderstorm codes (`TS`/`TSRA`/…), weather-code fields, NWP lightning probability, satellite convective proxies without a lightning sensor.

---

## PHASE STATUS

**AUDIT COMPLETE — NO TRAINING-READY OPEN ARCHIVE**

Genuine ground-based lightning **observations** exist over India (IITM / IMD Lightning Location Network and related operational services). They are **not** obtainable as a student-reproducible historical flash catalog through a documented public API or open bulk download.

Satellite lightning imagers that **do** observe lightning (NASA LIS / ISS-LIS) cover India only on **overpasses**, not as a continuous 24/7 field at the five aerodromes. They are useful as **reference / climatology**, not as V2 hourly predictors.

**No source is declared TRAINING-READY.** Future lightning work would be **CONDITIONAL** on a formal IITM/IMD (or equivalent) data grant, or would use **REFERENCE** satellite overpass data with a much weaker nowcast claim.

This phase **did not** collect lightning data.

---

## Method (documentation only)

| Check | What was done |
|-------|----------------|
| IMD / IITM lightning services | Public programme pages, Damini app descriptions, published LLN papers — **no login, no request** |
| MOSDAC / ISRO INSAT catalogues | Public product lists for INSAT-3D / 3DR / 3DS — **no download** |
| NDEM / NRSC | Public NDEM / Bhuvan hazard-layer descriptions — **no download** |
| NASA GHRC LIS / ISS-LIS | Public dataset landing pages (Earthdata) — **no order, no account created** |
| Commercial / volunteer networks | Public licence/access notes only |
| Open-Meteo / METAR / NWP | Identified as **non-observation** for this audit |

No HTTP bulk probes of lightning archives were run (those APIs are not public). Optional metadata: `docs/phase9a_lightning_source_audit.json`.

Station coordinates (same as Phase 2A): VOTV 8.482, 76.920; VECC 22.6547, 88.4467; VIDP 28.5667, 77.1167; VOCI 10.15, 76.4; VABB 19.1005, 72.8585. All five are inside India.

---

## Observation vs forecast vs derived

| Class | Meaning in this audit | Examples |
|-------|----------------------|----------|
| **OBSERVATION** | Instrument-detected lightning (ground TOA/MDF network, or spaceborne lightning mapper) | IITM/IMD LLN strokes; NASA LIS flashes |
| **FORECAST** | Predicted lightning or thunderstorm in the future | IMD thunderstorm warnings; Damini *alerts* that are nowcast products; NWP lightning density |
| **DERIVED** | Convective/thunderstorm *proxy* without a lightning sensor | INSAT IR brightness / cooling rates; METAR `TS`; weather_code; CAPE-based lightning probability |

A source may **operate** an observation network and still only **publish** forecasts or maps. Access is judged on the **historical observation archive**, not on the existence of an app.

---

## Sources (priority: official / reputable)

### 1. IMD Lightning Location Network / IMD lightning services

| Field | Finding |
|-------|---------|
| Source name | IMD Lightning Detection / Location Network products; IMD thunderstorm & lightning web services |
| Organization | India Meteorological Department (MoES) |
| Class | **OBSERVATION** at the sensor network; **FORECAST / nowcast** in most public web products |
| Historical availability | Operational lightning mapping is used in IMD nowcasting. A **public, documented historical flash catalog (2014–2025)** for bulk research download was **not found**. |
| Earliest useful date | Not documented as an open time series. Network density increased through the 2010s; climatology papers use later, denser years more confidently. |
| Latest available date | Near-real-time operational use (Damini / IMD nowcast pages). Historical dump: **not public**. |
| Temporal resolution | Network: individual strokes/flashes (sub-second). Public products: typically maps / alerts, not hourly station extracts. |
| Spatial resolution | Ground-network location accuracy typically on the order of **hundreds of metres to a few km** (sensor geometry dependent) — **not independently verified here**. |
| Individual events? | Yes **inside IMD/IITM systems**. **No** public event API found. |
| Flash counts / density? | Derivable **if** event data were granted. Public layers are often already aggregated. |
| India coverage | National operational intent; detection efficiency varies with sensor density. |
| Five locations? | All five ICAO sites are in India; **coverage is plausible** once the network is dense, but **not demonstrated with data in this phase**. |
| Access method | IMD website / nowcast pages / Damini; research data via **institutional request**, not a documented open REST archive. |
| API / download | **No** documented open historical lightning API comparable to Open-Meteo. |
| Registration | Yes for any research extract (IMD data request / MoU). **Not created.** |
| Approval | **Yes**, expected. |
| Payment | Official research extracts may be free or charged under IMD data policy; **not confirmed** (no request filed). |
| Licence | IMD data policy / government copyright; redistribution usually restricted. |
| Reproducibility | **Low** for an SIH GitHub prototype: third parties cannot re-download the same flashes. |
| SIH academic use | **Conditional** only after a written grant and a **non-redistribution** constraint. |

**Status: CONDITIONAL** (observations exist; archive not openly obtainable).

---

### 2. IITM Lightning Location Network (LLN) and Damini

| Field | Finding |
|-------|---------|
| Source name | IITM Lightning Location Network (LLN); Damini Lightning Alert app |
| Organization | Indian Institute of Tropical Meteorology, Pune (MoES) |
| Class | **OBSERVATION** (VLF/LF ground network). Damini **displays observations** and also issues **nowcast alerts** (mixed UI). |
| Historical availability | LLN build-out from **~2011**, expanding sensors through the 2010s. Research papers use multi-year IITM stroke databases. **No public bulk historical dump.** |
| Earliest useful date | For a *national* student extract, treat **~2014–2016** as the earliest *possible* research window (network still growing). Exact start for all five cities is **unknown without the archive**. |
| Latest available | Ongoing operations (Damini). |
| Temporal resolution | Individual strokes/flashes. |
| Spatial resolution | Network location error typically **~km scale** in published IITM work (varies). |
| Individual events? | Yes in the IITM database. Damini is **not** a historical API. |
| Flash counts / density? | Yes if events are granted (hourly counts in radii around VOTV/VECC/VIDP/VOCI/VABB). |
| India coverage | Designed as an **India** network. Coastal Kerala (VOTV, VOCI), Mumbai (VABB), Delhi (VIDP), Kolkata (VECC) are in-scope geographically. |
| Five locations? | **Geographically yes**; detection efficiency per site/year **not verified**. |
| Access | Research collaboration / data request to IITM; Damini is end-user, not bulk history. |
| API / download | **No** public historical API found. |
| Registration / approval / payment | Registration + **approval** typical. Payment unknown. **No account created.** |
| Licence | Institutional; usually **no open republish**. |
| Reproducibility | **Low** without a citable grant ID and frozen extract. |
| SIH academic use | Best **Indian observation** candidate **if** SIH/institute obtains data. Not usable in-repo until then. |

**Overlap with 2014-01-01 → 2025-12-31:** **unknown fraction of years** until a grant specifies the extract. Literature implies **partial-to-full overlap is physically possible** (network existed in this decade). **Do not assume 12 years of uniform detection efficiency.**

**Status: CONDITIONAL**

---

### 3. MOSDAC / ISRO INSAT-3D, INSAT-3DR, INSAT-3DS

| Field | Finding |
|-------|---------|
| Source name | MOSDAC INSAT imager / sounder and derived weather products |
| Organization | ISRO / Space Applications Centre; MOSDAC |
| Class | **DERIVED** convective/cloud products. INSAT-3D/3DR/3DS imagers are **not** Geostationary Lightning Mappers (unlike GOES-R GLM). **No lightning-flash observation product** was identified on the public MOSDAC catalogues reviewed at audit time. |
| Historical availability | INSAT-3D from **2013**, 3DR **2016**, 3DS **2024** — for **radiance / derived meteorology**, not lightning events. |
| Individual lightning events? | **No** |
| Flash density from lightning? | **No** (unless a future dedicated lightning payload is documented; none found for these missions as public lightning catalogs). |
| Five locations? | Satellite FOV covers India, but that is **not lightning**. |
| Access | MOSDAC registration for many datasets. **Not created.** |
| SIH academic use | Useful later as **satellite convection**, **out of scope** as lightning observations. |

**Status: REJECT** (for lightning *observations*; keep as possible future satellite branch, not Phase 9 lightning).

---

### 4. NDEM / NRSC lightning-related products

| Field | Finding |
|-------|---------|
| Source name | NDEM (National Database for Emergency Management) / NRSC Bhuvan hazard layers |
| Organization | NRSC / ISRO; MHA partnership for NDEM |
| Class | Typically **aggregated hazard / vulnerability / climatology maps**, not an event-level 2014–2025 flash archive. If lightning layers exist, they are **derived or summarised**, not a research LLN dump. |
| Individual events? | **Not documented** as downloadable stroke catalogs. |
| Five locations? | National maps would include the five cities as pixels, not as station time series. |
| Access | NDEM often requires **authorised user** registration. **Not created.** |
| SIH academic use | Maps for context only. |

**Status: REJECT** (no demonstrated historical flash observations for ML).

---

### 5. Other documented Indian government / research observation networks

| Network | Notes | Status |
|---------|-------|--------|
| **IITM + IMD joint operational lightning nowcast** | Same observation family as (1)–(2); public face is Damini / IMD nowcast. | CONDITIONAL (duplicate access path) |
| **State disaster-management lightning dashboards** | Usually Damini/IMD feeds or warnings — **FORECAST/display**, not independent archives. | REJECT as primary observation archive |
| **NCMRWF / IMD NWP lightning diagnostics** | **MODEL-DERIVED**, not observations. | REJECT |

No additional **open** Indian government lightning **event archive** (CSV/NetCDF of flashes, 2014–2025, five sites) was found.

---

## Backup (external research — not primary)

### 6. NASA TRMM LIS and ISS-LIS (GHRC)

| Field | Finding |
|-------|---------|
| Source name | Lightning Imaging Sensor (TRMM LIS; ISS-LIS) |
| Organization | NASA / GHRC / JAXA (TRMM) |
| Class | **OBSERVATION** (spaceborne optical lightning mapper) |
| Historical period | TRMM LIS **1997-12 → 2015-04**; ISS-LIS **2017-03 → 2023-11** (mission end). **Gap 2015–2017.** No 2024–2025 ISS-LIS. |
| Overlap with 2014–2025 | TRMM: **2014-01-01 → 2015-04** only. ISS-LIS: **2017-03 → 2023-11**. **Not** 2016, **not** 2024–2025. |
| Temporal / spatial | Overpass (~90 min ISS), **not continuous**. Nadir-class optical resolution ~**4–8 km**. |
| Individual flashes? | Yes (science products). |
| Density? | Orbit-period density only; **cannot** form a continuous “last 1 h flash count” at VOTV comparable to LLN. |
| India / five sites? | Overpasses **do** cross India; all five sites can appear in some orbits. **Not 24/7.** |
| Access | NASA Earthdata **registration**; data **free** for research; programmatic (Earthdata). **Account not created.** |
| Licence | NASA open data (cite GHRC). Redistribution generally allowed with citation. |
| Reproducibility | **High** for the satellite files; **low utility** for hourly station nowcast features. |
| SIH use | Climatology / case studies / **REFERENCE**. Not a drop-in V2 predictor. |

**Status: REFERENCE**

GOES-R **GLM** and Meteosat **MTG LI** are lightning **observations** but **do not cover India** (Americas / Europe–Africa). **REJECT** for this geography.

### 7. WWLLN (World Wide Lightning Location Network)

| Field | Finding |
|-------|---------|
| Class | **OBSERVATION** (global VLF) |
| Coverage | Global including India; **lower detection efficiency** than dense national LF networks. |
| History | Multi-year, including 2014–2025 in principle. |
| Access | University of Washington / host institutions; **typically research agreement / cost**. Not an open dump. |
| SIH | Backup if IITM/IMD refused; still not open. |

**Status: CONDITIONAL** (backup, not official Indian network).

### 8. Commercial networks (Vaisala GLD360, ENTLN, etc.)

Paid, licensed, not student-reproducible without a contract. **Status: REJECT** for this SIH prototype.

### 9. Volunteer / undocumented (Blitzortung, Kaggle dumps, scraped maps)

Not official, quality and licence unsuitable as V2 training truth. **Status: REJECT**

### 10. Non-observations already in V2 (do not relabel)

| Product | Class | Status |
|---------|-------|--------|
| METAR present weather (`TS`, …) | Station **weather** observation, **not** lightning flashes | Already V2 **labels**, not lightning predictors |
| Open-Meteo `weather_code` / CAPE | **DERIVED** / reanalysis / NWP | REJECT as lightning |
| IMD thunderstorm warnings | **FORECAST** | REJECT |
| Phase 7–8 NWP CAPE/CIN | **MODEL** | REJECT |

---

## Historical overlap with V2 (2014-01-01 → 2025-12-31)

Lightning history **need not** equal the full atmospheric window. Actual overlap:

| Source | Overlap with 2014–2025 | Continuous hourly at 5 sites? |
|--------|------------------------|-------------------------------|
| IITM / IMD LLN | **Physically possible** for much of the decade; **zero bytes obtainable** in this audit | Unknown without grant |
| INSAT-3D/3DR/3DS | N/A (no lightning sensor catalog) | No |
| NDEM/NRSC | N/A (no event series found) | No |
| TRMM LIS | **~16 months** (2014-01 → 2015-04), overpass only | No |
| ISS-LIS | **~6.7 years** (2017-03 → 2023-11), overpass only; **missing 2014–2016 and 2024–2025** | No |
| WWLLN | Likely full-period **if licensed** | Sparse efficiency |
| GLM / MTG LI | **None** over India | No |

**Documented open observation overlap suitable for V2-style hourly features: none.**

---

## Target use case (not implemented)

Possible future predictors (flash count, density, recent activity, trend, distance to recent lightning) **require event-level or high-frequency gridded observations**. Only IITM/IMD LLN (or a paid global network) could support that **if** data were granted. Satellite LIS **cannot** support continuous “recent activity” at one aerodrome.

**Not implemented in Phase 9A.**

---

## Comparison table

| Source | Observation? | Historical period | Resolution | India coverage | Five locations? | Access | Registration | Reproducibility | Status |
|--------|--------------|-------------------|------------|----------------|-----------------|--------|--------------|-----------------|--------|
| IMD lightning services / LLN products | Yes (network); public pages mostly nowcast | Operational; **no open 2014–2025 catalog** | Stroke / map | National (variable DE) | Geographically yes; unverified | Request / MoU | Yes | Low | **CONDITIONAL** |
| IITM LLN + Damini | Yes (LLN); Damini is display/alert | ~2011+ network; **no public dump** | Stroke (sub-second) | India | Geographically yes; unverified | Research grant | Yes + approval | Low | **CONDITIONAL** |
| MOSDAC INSAT-3D/3DR/3DS | **No** (imager, not lightning mapper) | 2013 / 2016 / 2024 radiance | km-scale IR/VIS | Full disk India | N/A | MOSDAC | Often yes | High for radiance | **REJECT** |
| NDEM / NRSC | Not as flash archive | Hazard maps, not events | Map / admin | National maps | As pixels only | Authorised portal | Yes | Low | **REJECT** |
| NCMRWF / NWP lightning | **No** (derived) | Model-dependent | Grid | Yes | Yes as grid | Various | Varies | Medium | **REJECT** |
| NASA TRMM LIS | Yes (satellite) | 1997–2015-04; overlap **2014–2015 only** | Overpass, ~4–8 km | Overpasses | Occasional | Earthdata | Yes (free) | High | **REFERENCE** |
| NASA ISS-LIS | Yes (satellite) | 2017-03–2023-11 | Overpass, ~4–8 km | Overpasses | Occasional | Earthdata | Yes (free) | High | **REFERENCE** |
| GOES GLM / MTG LI | Yes, **wrong GEO** | N/A for India | Continuous in their FOV | **No** | **No** | Open (NASA/EUMETSAT) | Often yes | High | **REJECT** |
| WWLLN | Yes | Multi-year incl. 2014–2025 (if licensed) | Stroke, lower DE | Yes | Likely | Agreement / fee | Yes | Medium | **CONDITIONAL** (backup) |
| Vaisala GLD360 / ENTLN | Yes | Commercial archives | Stroke | Yes | Yes | Contract | Yes + pay | Low | **REJECT** |
| Blitzortung / Kaggle / scrapes | Unreliable | Undocumented | Variable | Partial | Unknown | Informal | N/A | Very low | **REJECT** |
| METAR TS / weather_code | **Not lightning** | 2014–2025 in V2 | Hourly station | Five sites | Yes | Already in V2 | No | High | **REJECT** (wrong variable) |

**TRAINING-READY count: 0**

---

## Recommendation (audit only — do not collect)

1. **Do not** treat any current V2 field as lightning.
2. **Primary path** if SIH later needs lightning predictors: **IITM LLN / IMD** written data request for **event lists** (time, lat, lon, optionally peak current) in bounding boxes around the five sites. Until approval, lightning branch stays **blocked**.
3. **Open reproducible path:** NASA **ISS-LIS + TRMM LIS** for **REFERENCE** case studies only — **not** hourly five-station training.
4. **Do not** scrape Damini or IMD map tiles (ToS, completeness, reproducibility).

---

## What this phase did **not** do

- Download historical lightning
- Create accounts or request data
- Build `features_lightning` or join to METAR
- Train models
- Modify Phase 3–8 or V1

---

## Optional metadata

See `docs/phase9a_lightning_source_audit.json` (summary only).

---

PHASE 9A COMPLETE — LIGHTNING SOURCE AUDIT READY
