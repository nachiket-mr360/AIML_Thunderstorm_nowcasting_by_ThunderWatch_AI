"""Phase 2 (step 3): stability analysis, metadata JSON and README for the
historical VOTV thunderstorm-label dataset.

Reads the processed label CSV plus the build-time stats and writes:

  dataset/historical_thunderstorm_labels_votv_metadata.json
  dataset/HISTORICAL_THUNDERSTORM_LABELS_README.md

Nothing else is written or modified.
"""

from __future__ import annotations

import hashlib
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LABEL_CSV = os.path.join(HERE, "historical_thunderstorm_labels_votv.csv")
STATS_JSON = os.path.join(HERE, "historical_thunderstorm_labels_votv_stats.json")
RAW_LOG = os.path.join(HERE, "raw_metar", "votv_metar_iem_download_log.json")
RAW_CSV = os.path.join(HERE, "raw_metar", "votv_metar_iem_raw.csv")
REF_CSV = os.path.join(HERE, "weather_data_with_code.csv")
OUT_META = os.path.join(HERE, "historical_thunderstorm_labels_votv_metadata.json")
OUT_README = os.path.join(HERE, "HISTORICAL_THUNDERSTORM_LABELS_README.md")

WINDOWS = [
    ("2000-2025", 2000, 2025),
    ("2012-2025", 2012, 2025),
    ("2014-2025", 2014, 2025),
    ("2016-2025", 2016, 2025),
    ("2017-2025", 2017, 2025),
]


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    with open(STATS_JSON, encoding="utf-8") as fh:
        stats = json.load(fh)

    df = pd.read_csv(LABEL_CSV)
    df["ts"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    year = df["ts"].dt.year

    observed = df["thunderstorm_label"].notna()
    labelled = df["thunderstorm_label"] == 1
    evidence = df["wx_field_observed"] == 1

    window_rows = []
    for name, lo, hi in WINDOWS:
        inw = year.between(lo, hi)
        grid_hours = int(inw.sum())
        obs = int((inw & observed).sum())
        ev = int((inw & observed & evidence).sum())
        pos_all = int((inw & labelled).sum())
        pos_ev = int((inw & labelled & evidence).sum())
        rate_ev = round(pos_ev / ev * 100, 3) if ev else None
        window_rows.append(
            {
                "window": name,
                "grid_hours": grid_hours,
                "observed_hours": obs,
                "hours_with_present_weather_group": ev,
                "present_weather_coverage_of_grid_pct": round(ev / grid_hours * 100, 2),
                "positive_hours_all_observed": pos_all,
                "positive_hours_evidence_only": pos_ev,
                "positive_pct_evidence_only": rate_ev,
                "missing_hours": grid_hours - obs,
            }
        )

    # Year-by-year label rate over the recommended evidence-filtered view.
    ev_df = df[observed & evidence]
    by_year_evidence = {}
    for y, sub in ev_df.groupby(ev_df["ts"].dt.year):
        pos = int((sub["thunderstorm_label"] == 1).sum())
        by_year_evidence[str(int(y))] = {
            "hours_with_present_weather_group": int(len(sub)),
            "positive_hours": pos,
            "positive_pct": round(pos / len(sub) * 100, 3),
        }

    by_month_evidence = {}
    for m, sub in ev_df.groupby(ev_df["ts"].dt.month):
        pos = int((sub["thunderstorm_label"] == 1).sum())
        by_month_evidence[str(int(m))] = {
            "hours_with_present_weather_group": int(len(sub)),
            "positive_hours": pos,
            "positive_pct": round(pos / len(sub) * 100, 3),
        }

    raw_rows = 0
    with open(RAW_CSV, "r", encoding="utf-8", newline="") as fh:
        for _ in fh:
            raw_rows += 1
    raw_rows -= 1

    recommended = next(r for r in window_rows if r["window"] == "2014-2025")

    meta = {
        "artifact": "historical_thunderstorm_labels_votv.csv",
        "artifact_type": "processed observed-thunderstorm label dataset",
        "phase": "Phase 2 -- historical thunderstorm data acquisition (labels only)",
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "source": {
            "name": "Iowa Environmental Mesonet (IEM) ASOS/METAR archive request service",
            "provider": "Iowa State University, Department of Agronomy",
            "endpoint": "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py",
            "underlying_observations": "Routine METAR and SPECI airport surface observations for VOTV, "
                                      "originating from the station's own GTS-distributed reports",
            "access_method": "HTTP GET, comma-separated ('onlycomma') export, tz=UTC, "
                             "report_type=3 (routine METAR) and report_type=4 (SPECI), data=all",
            "why_this_source": "The export contains a dedicated, pre-parsed observed present-weather "
                               "column ('wxcodes') from which thunderstorm codes can be read directly, "
                               "instead of pattern-matching free-text METAR bodies.",
            "download_script": "dataset/download_votv_metar_iem.py",
            "build_script": "dataset/build_thunderstorm_labels.py",
            "doc_script": "dataset/make_label_docs.py",
        },
        "station": {
            "station_id": "VOTV",
            "name": "Thiruvananthapuram International Airport, Kerala, India",
            "wmo_icao": "VOTV",
            "coordinates_used": {"latitude": 8.4667, "longitude": 76.9500},
            "coordinate_source": "IEM station metadata as returned in every raw row (lon=76.9500, lat=8.4667)",
            "note_on_coordinates": "The existing Open-Meteo dataset was fetched for 8.4855 N, 76.9492 E "
                                   "(a grid point ~2.1 km north of the aerodrome). The labels are station "
                                   "observations at the aerodrome coordinates; the offset is small relative "
                                   "to the hourly label resolution but must be acknowledged in Phase 3.",
        },
        "coverage": {
            "requested_range_utc": "2000-01-01 .. 2025-12-31",
            "raw_observation_range_utc": stats["raw_date_range_utc"],
            "hourly_grid_range_utc": [
                df["timestamp_utc"].iloc[0].replace(" ", "T"),
                df["timestamp_utc"].iloc[-1].replace(" ", "T"),
            ],
            "hourly_grid_hours": stats["hourly_grid_hours"],
            "hourly_grid_source": "exact UTC hour index of dataset/weather_data_with_code.csv "
                                  "(227,928 rows, 2000-01-01 00:00Z .. 2025-12-31 23:00Z)",
            "raw_records_downloaded": raw_rows,
            "raw_records_used": stats["records_used"],
            "sub_hourly_reports": True,
        },
        "temporal_resolution": {
            "raw": "individual METAR/SPECI reports; nominal cadence is not fixed and changed over time",
            "aggregated_label_grid": "1 hour, UTC hourly boundaries",
            "hour_assignment_rule": "a report whose valid time t falls in [H, H+1) contributes to hour H; "
                                    "the label for hour H therefore describes the whole clock hour H",
            "observed_cadence_eras": {
                "2000-2005": "~7-14 reports/day, dominated by :40 past the hour",
                "2006-2011": "~14-18 reports/day, dominated by :40 and :10 past the hour",
                "2012-2015": "~31-37 reports/day, mixed :00/:10/:30/:40",
                "2016-2025": "~42-48 reports/day, :00 and :30 (routine) plus irregular SPECI",
            },
        },
        "label_definition": {
            "evidence_field": "wxcodes (IEM pre-parsed OBSERVED present-weather group)",
            "positive_rule": "thunderstorm_label = 1 for hour H when at least one valid observed report in "
                             "[H, H+1) carries a TS-family present-weather code at the station "
                             "(TS, TSRA, +TSRA, -TSRA, TSGR, TSDZ, TS/HZ, ...) or a vicinity code (VCTS).",
            "zero_rule": "thunderstorm_label = 0 when hour H contains at least one valid observed report and "
                         "none of them carries such a code.",
            "missing_rule": "thunderstorm_label = NaN when hour H contains no valid observed report at all. "
                            "Absence of a report is never treated as evidence of absence of a thunderstorm.",
            "explicitly_not_used_as_label": [
                "Open-Meteo weather_code",
                "rainfall / precipitation thresholds",
                "humidity or cloud-cover thresholds",
                "blind regex search of raw METAR text for 'TS'",
                "TS appearing only inside TEMPO, BECMG, NOSIG, PROBnn or other trend/forecast groups",
                "RETS (recent thunderstorm, i.e. not occurring at observation time)",
            ],
            "vicinity_handling": {
                "vcts_records": stats["thunderstorm_reports_vcts_only"],
                "decision": "VCTS is included in the primary label because IEM's wxcodes is the source's "
                            "observed present-weather field, which is the condition the task allowed. "
                            "The effect is negligible (single-digit records) and is isolated in the "
                            "thunderstorm_label_strict column, which excludes vicinity-only reports.",
            },
            "derived_columns": {
                "thunderstorm_label": "primary label (0/1/NaN) as defined above",
                "thunderstorm_label_strict": "same, but VCTS-only hours are 0 (station thunderstorm only)",
                "thunderstorm_label_recovered": "audit/sensitivity variant that additionally sets 1 where a "
                                                "malformed pre-2012 body demonstrably encoded an observed TS "
                                                "that IEM's wxcodes failed to parse (e.g. '5000TS', 'TS/-DZ'); "
                                                "never a silent replacement for the primary label",
                "wxcodes_observed": "distinct observed present-weather groups in the hour, for audit",
                "n_reports": "number of valid observed reports in the hour",
                "n_reports_full_decode": "reports in the hour carrying both altimeter and temperature groups",
                "n_reports_with_wx": "reports in the hour carrying a decodable present-weather group",
                "wx_field_observed": "1 when the hour has >=1 report with a decodable present-weather group",
                "n_ts_reports": "number of reports in the hour carrying a TS-family code",
            },
        },
        "missing_data_treatment": {
            "unobserved_hours_are_nan": True,
            "hours_marked_missing": stats["missing_hours"],
            "observed_hours": stats["observed_hours"],
            "observed_hour_percentage": stats["observed_hour_pct"],
            "evidence_caveat": "A report that contains no present-weather group is not evidence that no "
                               "thunderstorm occurred. Hours whose only reports lack a decodable "
                               "present-weather group are exposed via wx_field_observed=0 and should be "
                               "excluded or masked in modelling.",
            "hours_with_decodable_present_weather": stats["hours_with_decodable_present_weather"],
            "hours_with_reports_but_no_present_weather_group":
                stats["hours_with_reports_but_no_present_weather_group"],
            "zero_labels_relying_on_no_present_weather_group":
                stats["zero_labels_relying_on_no_present_weather_group"],
            "recommended_masking_recipe":
                "df = df[(df.wx_field_observed == 1) & (df.thunderstorm_label.notna())]",
        },
        "counts": {
            "raw_records_downloaded": raw_rows,
            "raw_records_used_after_cleaning": stats["records_used"],
            "records_excluded_as_non_observation": stats["records_excluded_as_non_observation"],
            "duplicate_timestamp_rows_dropped": stats["duplicate_valid_rows_dropped"],
            "conflicting_duplicate_timestamps": stats["conflicting_duplicate_timestamps"],
            "observed_thunderstorm_reports_wxcodes": stats["thunderstorm_reports_wxcodes"],
            "observed_thunderstorm_reports_station_only": stats["thunderstorm_reports_station_only"],
            "observed_thunderstorm_reports_vcts_only": stats["thunderstorm_reports_vcts_only"],
            "thunderstorm_positive_hours": stats["positive_hours"],
            "positive_percentage_of_observed_hours": stats["positive_pct_of_observed_hours"],
            "positive_percentage_of_all_grid_hours": stats["positive_pct_of_all_grid_hours"],
            "positive_hours_strict_variant": stats["positive_hours_strict"],
            "positive_hours_recovered_variant": stats["positive_hours_recovered"],
            "hours_added_by_recovery_variant": stats["hourly_positive_hours_added_by_recovery"],
            "recovered_observed_ts_reports": stats["recovered_observed_ts_reports"],
            "recovered_reports_by_year": stats["recovered_reports_by_year"],
        },
        "report_density_and_stability": {
            "reports_density_by_year": stats["reports_density_by_year"],
            "observed_hour_coverage_pct_by_year": stats["observed_hour_coverage_pct_by_year"],
            "present_weather_observability_by_year": stats["present_weather_observability_by_year"],
            "label_rate_by_era": stats["label_rate_by_era"],
            "labelling_windows": window_rows,
            "conclusion": "2000-2013 is NOT suitable for modelling. The present-weather field is absent "
                          "entirely in 2000-2002 (zero decodable present-weather groups in 8,486 reports) and "
                          "present in only 1.8%-64.8% of reports up to 2013, with a label rate that swings "
                          "between 0.0% and 2.4% per year for reasons that track encoding practice rather "
                          "than weather (e.g. 2005 = 1.66% but 2006 = 0.12%). From 2014 the present-weather "
                          "coverage is >=71.6% in every year and the label rate stabilises at 3.3%-5.0%. "
                          "The recommended modelling window is 2014-2025 restricted to "
                          "wx_field_observed == 1; 2016-2025 is the most homogeneous sub-window.",
            "recommended_window": "2014-2025",
            "recommended_window_stats": recommended,
            "stricter_alternative_window": "2016-2025",
        },
        "limitations": [
            "Observations only describe the aerodrome point; a thunderstorm over the city or the "
            "catchment that is not within the station's vicinity is not labelled.",
            "METAR present weather is a human/sensor judgement of 'thunderstorm observed at the station'; "
            "its detection range is finite and reporting practice varies between observers.",
            "The hour label aggregates reports within [H, H+1), so it is a same-hour (nowcast-style) "
            "target, not a lead-time-shifted target. Phase 3 must apply any lead-time shift explicitly.",
            "Pre-2012 reports use a legacy, frequently malformed encoding in which the present-weather "
            "group was sometimes glued to the visibility group ('5000TS') or written with slashes "
            "('TS/-DZ'), causing IEM to leave wxcodes empty. 111 such observed TS reports were recovered "
            "into an audit-only column; they are not in the primary label.",
            "~1,740 reports mention TS only in TEMPO/BECMG trend groups. These never produce a positive "
            "label. They were verified by cutting each body at the first trend marker.",
            "The dataset is a label table only. It is deliberately NOT merged into the Open-Meteo "
            "features and no model was retrained in this phase.",
            "IEM is a redistribution archive; no formal SLA and occasional ingestion duplicates exist "
            "(12 duplicate timestamps in Jan 2013 were de-duplicated, none conflicting).",
            "Coverage ends 2025-12-30 23:30Z, so 2025-12-31 is entirely unobserved (NaN).",
        ],
        "validation": {
            "run_by": "dataset/build_thunderstorm_labels.py (assertions) and an independent rebuild in the "
                      "same phase",
            "timestamps_parse": "PASS -- all 227,928 grid timestamps parse as UTC",
            "timestamps_are_utc": "PASS -- tz is UTC, source requested with tz=UTC",
            "no_duplicate_hourly_timestamps": "PASS -- grid is unique and strictly increasing",
            "labels_only_0_1_nan": "PASS -- all three label columns hold only 0.0, 1.0 or NaN",
            "label_monotonicity": "PASS -- recovered >= primary >= strict elementwise",
            "no_future_information": "PASS -- each hour label uses only reports inside that same clock hour; "
                                    "no forward-looking window, no trend/forecast group",
            "station_coordinates_consistent": "PASS -- single station_id, single lat/lon over all rows",
            "source_field_preserved": "PASS -- wxcodes_observed retains the raw observed codes per hour",
            "grid_matches_open_meteo": "PASS -- identical timestamp set to weather_data_with_code.csv",
            "existing_project_files_unchanged": "PASS -- verified by SHA-256 before/after",
        },
        "files": {
            "raw": {
                "combined": "dataset/raw_metar/votv_metar_iem_raw.csv",
                "per_year": "dataset/raw_metar/votv_metar_iem_2000.csv .. votv_metar_iem_2025.csv",
                "download_log": "dataset/raw_metar/votv_metar_iem_download_log.json",
            },
            "processed": {
                "labels": "dataset/historical_thunderstorm_labels_votv.csv",
                "stats": "dataset/historical_thunderstorm_labels_votv_stats.json",
                "metadata": "dataset/historical_thunderstorm_labels_votv_metadata.json",
                "readme": "dataset/HISTORICAL_THUNDERSTORM_LABELS_README.md",
            },
            "scripts": [
                "dataset/download_votv_metar_iem.py",
                "dataset/build_thunderstorm_labels.py",
                "dataset/make_label_docs.py",
            ],
        },
        "checksums": {
            "raw_combined_sha256": sha256_of(RAW_CSV),
            "labels_csv_sha256": sha256_of(LABEL_CSV),
            "reference_open_meteo_sha256": sha256_of(REF_CSV),
        },
        "phase_3_ready": True,
        "phase_3_note": "Ready for synchronization once the modelling window and the wx_field_observed mask "
                        "are agreed. This phase intentionally did not merge labels into the ML dataset, did "
                        "not retrain, and did not modify the existing model, dataset or outputs.",
    }

    with open(OUT_META, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    # ---------------- README ------------------------------------------------
    obs_rows = stats["observed_hours"]
    mis_rows = stats["missing_hours"]
    ytab = "\n".join(
        f"| {y} | {v['grid_hours']} | {v['observed_hours']} | {v['missing_hours']} | "
        f"{v['positive_hours']} | {v['positive_pct_of_observed']} | "
        f"{stats['present_weather_observability_by_year'][y]['present_weather_coverage_pct']} | "
        f"{by_year_evidence.get(y, {}).get('positive_pct', '-')} |"
        for y, v in stats["labels_by_year"].items()
    )
    mtab = "\n".join(
        f"| {m} | {v['observed_hours']} | {v['positive_hours']} | {v['positive_pct_of_observed']} | "
        f"{by_month_evidence.get(m, {}).get('positive_pct', '-')} |"
        for m, v in stats["labels_by_month_utc"].items()
    )
    wtab = "\n".join(
        f"| {r['window']} | {r['grid_hours']} | {r['observed_hours']} | "
        f"{r['hours_with_present_weather_group']} ({r['present_weather_coverage_of_grid_pct']}%) | "
        f"{r['positive_hours_all_observed']} | {r['positive_hours_evidence_only']} | "
        f"{r['positive_pct_evidence_only']} |"
        for r in window_rows
    )
    etab = "\n".join(
        f"| {k} | {v['observed_hours']} | {v['positive_hours']} | {v['positive_pct_of_observed']} |"
        for k, v in stats["label_rate_by_era"].items()
    )
    dtab = "\n".join(
        f"| {y} | {v['raw_reports']} | {v['reports_per_day']} | "
        f"{stats['observed_hour_coverage_pct_by_year'][y]} | "
        f"{stats['present_weather_observability_by_year'][y]['present_weather_coverage_pct']} |"
        for y, v in stats["reports_density_by_year"].items()
    )

    readme = f"""# Historical Thunderstorm Labels for VOTV (Phase 2 data acquisition)

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

Raw records downloaded: **{raw_rows:,}** across 26 yearly chunks.
Raw range actually obtained: **{stats['raw_date_range_utc'][0]} -> {stats['raw_date_range_utc'][1]}**.
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
| Raw METAR/SPECI records downloaded | {raw_rows:,} |
| Records used after cleaning | {stats['records_used']:,} |
| Duplicate-timestamp rows dropped | {stats['duplicate_valid_rows_dropped']} (none conflicting) |
| Distinct report timestamps | {stats['unique_report_timestamps']:,} |
| Hourly grid rows | {stats['hourly_grid_hours']:,} |
| Observed hours (report exists) | {obs_rows:,} ({stats['observed_hour_pct']}%) |
| Missing / unobserved hours (NaN) | {mis_rows:,} |
| Observed thunderstorm reports (`wxcodes`) | {stats['thunderstorm_reports_wxcodes']:,} |
| - of which station TS only | {stats['thunderstorm_reports_station_only']:,} |
| - of which `VCTS` only | {stats['thunderstorm_reports_vcts_only']} |
| Positive hours | **{stats['positive_hours']:,}** |
| Positive % of observed hours | **{stats['positive_pct_of_observed_hours']}%** |
| Positive % of all grid hours | {stats['positive_pct_of_all_grid_hours']}% |
| Hours with a decodable present-weather group | {stats['hours_with_decodable_present_weather']:,} |
| 0-labels that rest on no present-weather group | {stats['zero_labels_relying_on_no_present_weather_group']:,} |

## 5. Counts by year

`pos%` columns: **observed** = positives / observed hours; **evidence** = positives / observed hours
that contain a decodable present-weather group (the recommended view).

| Year | Grid h | Observed h | Missing h | Positives | Pos% observed | Present-wx coverage of reports % | Pos% evidence |
|---|---|---|---|---|---|---|---|
{ytab}

## 6. Counts by month (UTC)

| Month | Observed h | Positives | Pos% observed | Pos% evidence |
|---|---|---|---|---|
{mtab}

The modern-era shape is the physically expected bi-modal Thiruvananthapuram pattern (pre-monsoon
April-May peak ~8.7-10.3%, post-monsoon October-November ~5.8-6.8%), which is a useful sanity check
that the `wxcodes` label is real signal and not an artefact.

## 7. Temporal sampling and report density - the key caveat

| Year | Raw reports | Reports/day | Observed hour coverage % | Present-wx coverage % |
|---|---|---|---|---|
{dtab}

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
{etab}

The 0.56% -> 1.53% -> 2.05% -> 4.19% -> 3.48% progression is a detection/encoding trend, not a climate
trend: the same physical seasonality is present in every era, just muted 2-2.5x before 2014.

## 8. Is 2000-2025 suitable for modelling?

**No - not as a whole.** Use a shorter stable window:

| Window | Grid h | Observed h | Hours with present-wx group | Positives (all observed) | Positives (evidence only) | Pos% (evidence) |
|---|---|---|---|---|---|---|
{wtab}

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
| Observed `TS` in body that IEM left out of `wxcodes` (malformed pre-2012 encoding) | 111 | Isolated in `thunderstorm_label_recovered`. Affects {stats['hourly_positive_hours_added_by_recovery']} extra hours. Not in the primary label. |
| `RETS` / `RETSRA` (recent thunderstorm, not present weather) | 14 | Deliberately **not** labelled 1. |
| Duplicate `valid` timestamps (all Jan 2013, identical duplicates) | 13 rows / 12 stamps | De-duplicated, first kept. No conflicting pairs. |
| Non-observation product (24-hour forecast body) | 1 | Excluded from the report pool. |
| Reports with no altimeter/temperature (truncated `AUTO`) | 21,632 | Kept as reports (the report exists) but exposed via `n_reports_full_decode`; {stats['hours_resting_only_on_partial_reports']:,} hours rest only on such reports. |
| `VCTS`-only reports | 6 | In the primary label (observed present-weather field), isolated in the strict column. |
| 2025-12-31 | 24 hours | Entirely unobserved -> NaN. |

## 10. Limitations

{chr(10).join('- ' + s for s in meta['limitations'])}

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
"""

    with open(OUT_README, "w", encoding="utf-8") as fh:
        fh.write(readme)

    print("wrote:", OUT_META)
    print("wrote:", OUT_README)
    print()
    for r in window_rows:
        print(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
