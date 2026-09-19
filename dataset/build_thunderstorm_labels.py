"""Phase 2 (step 2): build the historical VOTV thunderstorm-label dataset.

Reads the RAW IEM METAR dump produced by download_votv_metar_iem.py and
aggregates OBSERVED thunderstorm reports onto the exact hourly UTC grid of
dataset/weather_data_with_code.csv.

Label rules (see dataset/HISTORICAL_THUNDERSTORM_LABELS_README.md):
  * The primary evidence is IEM's ``wxcodes`` column, which holds the OBSERVED
    present-weather group. Trend/forecast groups (TEMPO/BECMG/NOSIG/PROBnn) are
    never used, so a TS that appears only in a trend group can never produce 1.
  * thunderstorm_label = 1  if >=1 valid observed report in the hour carries a
    TS-family present-weather code at the station (TS, TSRA, +/-TSRA, TSGR, ...)
    or a vicinity code (VCTS), i.e. the source's observed present-weather field.
    0  if the hour contains >=1 report and none carries such a code.
    NaN if the hour contains no report at all (absence of report is NOT
    evidence of absence of thunderstorm).
  * thunderstorm_label_strict excludes vicinity-only (VCTS) reports.
  * thunderstorm_label_recovered additionally recovers TS that the IEM parser
    dropped in malformed pre-2012 bodies (e.g. ``5000TS``, ``TS/-DZ``); it is an
    audit/sensitivity column only, never a silent replacement.

Only NEW files are written. No existing project file is modified.
"""

from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_CSV = os.path.join(HERE, "raw_metar", "votv_metar_iem_raw.csv")
REF_CSV = os.path.join(HERE, "weather_data_with_code.csv")
OUT_CSV = os.path.join(HERE, "historical_thunderstorm_labels_votv.csv")
OUT_STATS = os.path.join(HERE, "historical_thunderstorm_labels_votv_stats.json")

STATION = "VOTV"
# IEM station coordinates for VOTV (all raw rows agree).
STATION_LAT = 8.4667
STATION_LON = 76.9500

# Trend/remark markers: everything after the first one is forecast, not observed.
TREND_CUT_RE = re.compile(r"\b(?:TEMPO|BECMG|NOSIG|RMK|PROB\d{2}|EMPO|TMPO|BECMG)\b")
# Misspellings of TEMPO that still introduce a trend group.
TREND_TYPO_RE = re.compile(r"\b(?:EMPO|TMPO)\b")
RECENT_WEATHER_RE = re.compile(r"\bRE[A-Z]{2,}\b")
# A routine decode carries both an altimeter group and a temperature/dewpoint group.
QNH_RE = re.compile(r"\bQ\d{4}\b")
TEMP_RE = re.compile(r"\bM?\d{2}/M?\d{2}\b")
# A forecast product starts with ddhhmmZ followed by a ddhhmm valid period.
FORECAST_PRODUCT_RE = re.compile(r"^\S+\s+\d{6}Z\s+\d{6}\s")


def split_wx_codes(value: str) -> list[str]:
    """Tokenise a present-weather field on whitespace and '/' separators."""
    if value is None:
        return []
    raw = str(value).strip()
    if raw == "" or raw.upper() == "M":
        return []
    return [tok for tok in re.split(r"[\/\s]+", raw) if tok]


def has_decodable_wx(value: str) -> bool:
    """True when the report carries an actual present-weather group.

    IEM writes 'M' when the METAR body contains no present-weather group at all,
    so an 'M' row is NOT evidence that no weather occurred -- early VOTV bodies
    (2000-2002 especially) are AUTO records with no weather field whatsoever.
    """
    return len(split_wx_codes(value)) > 0


def classify_token(token: str) -> str:
    """Return 'ts' (at station), 'vcts' (in vicinity), or 'other'."""
    tok = token.strip().upper()
    if not tok:
        return "other"
    tok = tok.lstrip("+-")  # intensity prefix
    if tok.startswith("VC"):
        body = tok[2:].lstrip("+-")
        return "vcts" if body.startswith("TS") else "other"
    return "ts" if tok.startswith("TS") else "other"


def code_family(value: str) -> tuple[bool, bool]:
    """(has station TS, has vicinity TS) for a present-weather field."""
    station = vicinity = False
    for tok in split_wx_codes(value):
        kind = classify_token(tok)
        if kind == "ts":
            station = True
        elif kind == "vcts":
            vicinity = True
    return station, vicinity


def recover_body_ts(body: str) -> tuple[bool, bool]:
    """Detect clearly-observed TS in the observation body that IEM's wxcodes missed.

    Returns (confident, ambiguous). Only the portion of the body before the first
    trend/remark marker is examined, ``RETS`` (recent thunderstorm) is excluded
    because it is not present weather, and a detection that survives only because
    of a misspelled trend marker is reported as ambiguous rather than confident.
    """
    if not isinstance(body, str) or not body:
        return False, False
    allowed = TREND_CUT_RE.split(body, maxsplit=1)[0]
    # Compare against a version that also cuts at misspelled trend markers.
    strict = TREND_TYPO_RE.split(body, maxsplit=1)[0] if TREND_TYPO_RE.search(body) else allowed

    def has_present_ts(text: str) -> bool:
        for tok in re.split(r"[\/\s]+", text):
            tok = tok.strip().upper()
            if not tok:
                continue
            # Strip a leading visibility/units numeric token glued to the wx code.
            tok = re.sub(r"^\d+", "", tok)
            tok = tok.lstrip("+-")
            if not tok or tok.startswith("RE"):
                continue
            if tok.startswith("TS"):
                return True
            if tok.startswith("VC"):
                b = tok[2:].lstrip("+-")
                if b.startswith("TS"):
                    return True
        return False

    return has_present_ts(allowed), has_present_ts(allowed) and not has_present_ts(strict)


def main() -> int:
    grid = pd.read_csv(REF_CSV, usecols=["date"])
    grid_ts = pd.to_datetime(grid["date"], utc=True, format="mixed")
    if grid_ts.duplicated().any():
        raise SystemExit("reference grid has duplicate timestamps")
    if not grid_ts.is_monotonic_increasing:
        grid_ts = grid_ts.sort_values().reset_index(drop=True)
    grid_label_str = grid_ts.dt.strftime("%Y-%m-%d %H:%M:%S+00:00")
    grid_hour = grid_ts.dt.floor("h")

    raw = pd.read_csv(RAW_CSV, dtype=str, keep_default_na=False, na_values=[])
    n_raw = len(raw)

    raw["valid_dt"] = pd.to_datetime(raw["valid"], format="%Y-%m-%d %H:%M", utc=True)
    raw["hour"] = raw["valid_dt"].dt.floor("h")
    raw["wxcodes"] = raw["wxcodes"].fillna("").astype(str)

    # --- data-quality flagging (does not change the label definition) -------
    raw["has_qnh"] = raw["metar"].str.contains(QNH_RE, regex=True, na=False)
    raw["has_temp"] = raw["metar"].str.contains(TEMP_RE, regex=True, na=False)
    raw["full_decode"] = raw["has_qnh"] & raw["has_temp"]
    raw["wx_decodable"] = raw["wxcodes"].map(has_decodable_wx)

    fc_mask = raw["metar"].str.match(FORECAST_PRODUCT_RE, na=False)
    non_obs = fc_mask & ~raw["full_decode"]
    n_excluded_non_obs = int(non_obs.sum())
    work = raw.loc[~non_obs].copy()

    # --- duplicate timestamps ---------------------------------------------
    conf = (
        work.groupby("valid_dt")["wxcodes"]
        .apply(lambda s: sorted(set(s)))
        .reset_index(name="distinct_wx")
    )
    n_dup_rows = int(work["valid_dt"].duplicated().sum())
    conflicting = conf[conf["distinct_wx"].apply(len) > 1]
    n_dup_conflicting = int(len(conflicting))
    n_distinct_before = int(work["valid_dt"].nunique())
    work = work.sort_values(["valid_dt", "metar"]).drop_duplicates(subset=["valid_dt"], keep="first")
    n_distinct_after = int(len(work))

    # --- thunderstorm extraction ------------------------------------------
    fam = work["wxcodes"].map(code_family)
    work["ts_station"] = [a for a, _ in fam]
    work["ts_vicinity"] = [b for _, b in fam]
    work["ts_any_wx"] = work["ts_station"] | work["ts_vicinity"]

    rec = work["metar"].map(recover_body_ts)
    work["ts_recovered_confident"] = [a for a, _ in rec]
    work["ts_recovered_ambiguous"] = [b for _, b in rec]
    # Only recover where the dedicated field failed to report any TS family.
    work["ts_recover_added"] = work["ts_recovered_confident"] & ~work["ts_any_wx"]
    work["ts_recover_ambiguous_only"] = (
        work["ts_recovered_ambiguous"] & ~work["ts_any_wx"] & ~work["ts_recovered_confident"]
    )

    n_ts_reports_wx = int(work["ts_any_wx"].sum())
    n_ts_reports_station = int(work["ts_station"].sum())
    n_ts_reports_vcts_only = int((work["ts_vicinity"] & ~work["ts_station"]).sum())
    n_recovered = int(work["ts_recover_added"].sum())
    n_recovered_ambiguous = int(work["ts_recover_ambiguous_only"].sum())

    # --- hourly aggregation ------------------------------------------------
    g = work.groupby("hour")
    agg = pd.DataFrame(
        {
            "n_reports": g.size(),
            "n_reports_full_decode": g["full_decode"].sum(),
            "n_reports_with_wx": g["wx_decodable"].sum(),
            "ts_station_any": g["ts_station"].any(),
            "ts_any_any": g["ts_any_wx"].any(),
            "ts_recovered_any": g["ts_recover_added"].any(),
            "wx_codes": g["wxcodes"].apply(lambda s: " | ".join(sorted({c for c in s if c and c != "M"}))),
        }
    )
    agg["n_ts_reports"] = g["ts_any_wx"].sum()

    out = pd.DataFrame({"timestamp_utc": grid_label_str, "hour": grid_hour})
    out = out.merge(agg, left_on="hour", right_index=True, how="left")
    out["n_reports"] = out["n_reports"].fillna(0).astype(int)
    out["n_reports_full_decode"] = out["n_reports_full_decode"].fillna(0).astype(int)
    out["n_reports_with_wx"] = out["n_reports_with_wx"].fillna(0).astype(int)
    out["n_ts_reports"] = out["n_ts_reports"].fillna(0).astype(int)
    out["wx_codes"] = out["wx_codes"].fillna("")
    out["wx_field_observed"] = (out["n_reports_with_wx"] > 0).astype(int)

    observed = out["n_reports"] > 0
    lab = pd.Series(np.nan, index=out.index, dtype="float64")
    lab[observed] = 0.0
    lab[observed & out["ts_any_any"].fillna(False).astype(bool)] = 1.0
    out["thunderstorm_label"] = lab

    lab_s = pd.Series(np.nan, index=out.index, dtype="float64")
    lab_s[observed] = 0.0
    lab_s[observed & out["ts_station_any"].fillna(False).astype(bool)] = 1.0
    out["thunderstorm_label_strict"] = lab_s

    lab_r = lab.copy()
    rec_hours = out["ts_recovered_any"].fillna(False).astype(bool)
    lab_r[rec_hours] = 1.0
    out["thunderstorm_label_recovered"] = lab_r

    out["station_id"] = STATION
    out["latitude"] = STATION_LAT
    out["longitude"] = STATION_LON

    cols = [
        "timestamp_utc", "station_id", "latitude", "longitude",
        "thunderstorm_label", "thunderstorm_label_strict", "thunderstorm_label_recovered",
        "wxcodes_observed", "n_reports", "n_reports_full_decode", "n_reports_with_wx",
        "wx_field_observed", "n_ts_reports",
    ]
    out = out.rename(columns={"wx_codes": "wxcodes_observed"})[cols].sort_values("timestamp_utc")
    out.to_csv(OUT_CSV, index=False)

    # --- statistics --------------------------------------------------------
    n_hours = len(out)
    n_observed = int(observed.sum())
    n_missing = n_hours - n_observed
    n_pos = int((out["thunderstorm_label"] == 1).sum())
    n_pos_strict = int((out["thunderstorm_label_strict"] == 1).sum())
    n_pos_rec = int((out["thunderstorm_label_recovered"] == 1).sum())

    ts_ts = pd.to_datetime(out["timestamp_utc"], utc=True)
    obs_hours = out.loc[observed]

    def counts_by(period: pd.Series, data: pd.DataFrame) -> dict:
        res: dict[str, dict] = {}
        for key, sub in data.groupby(period, dropna=True):
            pos = int((sub["thunderstorm_label"] == 1).sum())
            res[str(key)] = {
                "observed_hours": int(len(sub)),
                "positive_hours": pos,
                "positive_pct_of_observed": round(pos / len(sub) * 100, 3) if len(sub) else None,
            }
        return res

    by_year = counts_by(ts_ts.dt.year, obs_hours)
    by_month = counts_by(ts_ts.dt.month, obs_hours)

    grid_year = ts_ts.dt.year
    year_missing = {}
    for key, cnt in grid_year.value_counts().sort_index().items():
        obs_cnt = int((grid_year[observed] == key).sum())
        year_missing[str(key)] = int(cnt - obs_cnt)
    for key in by_year:
        by_year[key]["grid_hours"] = int((grid_year == int(key)).sum())
        by_year[key]["missing_hours"] = year_missing.get(key, 0)

    dense = (
        work.groupby(work["valid_dt"].dt.year)
        .agg(reports=("valid_dt", "size"), days=("valid_dt", lambda s: s.dt.date.nunique()))
    )
    density_by_year = {
        str(int(y)): {
            "raw_reports": int(r["reports"]),
            "reports_per_day": round(r["reports"] / r["days"], 2) if r["days"] else None,
        }
        for y, r in dense.iterrows()
    }
    hour_cov = {}
    for y, cnt in grid_year.value_counts().sort_index().items():
        o = int((grid_year[observed] == y).sum())
        hour_cov[str(int(y))] = round(o / cnt * 100, 2)

    recovered = work.loc[work["ts_recover_added"] | work["ts_recover_ambiguous_only"], "valid_dt"].dt.year

    # Evidence quality: does a report in the hour actually carry a present-weather
    # group, i.e. could a thunderstorm have been detected at all?
    wq = work.groupby(work["valid_dt"].dt.year).agg(
        reports=("valid_dt", "size"),
        reports_with_wx=("wx_decodable", "sum"),
        reports_full_decode=("full_decode", "sum"),
    )
    observability_by_year = {
        str(int(y)): {
            "reports": int(r["reports"]),
            "reports_with_present_weather_group": int(r["reports_with_wx"]),
            "present_weather_coverage_pct": round(r["reports_with_wx"] / r["reports"] * 100, 2) if r["reports"] else None,
            "reports_with_full_routine_decode_pct": round(r["reports_full_decode"] / r["reports"] * 100, 2) if r["reports"] else None,
        }
        for y, r in wq.iterrows()
    }

    era_bins = [1999, 2005, 2011, 2013, 2015, 2025]
    era_labels = ["2000-2005", "2006-2011", "2012-2013", "2014-2015", "2016-2025"]
    era = pd.cut(ts_ts.dt.year, bins=era_bins, labels=era_labels)
    era_rows = {}
    for name, sub in obs_hours.groupby(era[observed], observed=True):
        n = len(sub)
        pos = int((sub["thunderstorm_label"] == 1).sum())
        era_rows[str(name)] = {
            "observed_hours": n,
            "positive_hours": pos,
            "positive_pct_of_observed": round(pos / n * 100, 2) if n else None,
        }

    stats = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "raw_records": n_raw,
        "records_excluded_as_non_observation": n_excluded_non_obs,
        "records_used": int(len(work)),
        "duplicate_valid_rows_dropped": n_dup_rows,
        "unique_report_timestamps": n_distinct_after,
        "report_timestamps_after_dedup_check": {"before": n_distinct_before, "after": n_distinct_after},
        "conflicting_duplicate_timestamps": n_dup_conflicting,
        "raw_date_range_utc": [str(raw["valid_dt"].min()), str(raw["valid_dt"].max())],
        "hourly_grid_hours": n_hours,
        "observed_hours": n_observed,
        "missing_hours": n_missing,
        "observed_hour_pct": round(n_observed / n_hours * 100, 3),
        "thunderstorm_reports_wxcodes": n_ts_reports_wx,
        "thunderstorm_reports_station_only": n_ts_reports_station,
        "thunderstorm_reports_vcts_only": n_ts_reports_vcts_only,
        "positive_hours": n_pos,
        "positive_pct_of_observed_hours": round(n_pos / n_observed * 100, 3),
        "positive_pct_of_all_grid_hours": round(n_pos / n_hours * 100, 3),
        "positive_hours_strict": n_pos_strict,
        "positive_hours_recovered": n_pos_rec,
        "hourly_positive_hours_added_by_recovery": n_pos_rec - n_pos,
        "recovered_observed_ts_reports": n_recovered,
        "recovered_ambiguous_ts_reports": n_recovered_ambiguous,
        "recovered_reports_by_year": {str(int(k)): int(v) for k, v in recovered.value_counts().sort_index().items()},
        "hours_resting_only_on_partial_reports": int(
            ((out["n_reports"] > 0) & (out["n_reports_full_decode"] == 0)).sum()
        ),
        "hours_with_decodable_present_weather": int((out["wx_field_observed"] == 1).sum()),
        "hours_with_reports_but_no_present_weather_group": int(
            ((out["n_reports"] > 0) & (out["wx_field_observed"] == 0)).sum()
        ),
        "zero_labels_relying_on_no_present_weather_group": int(
            ((out["thunderstorm_label"] == 0) & (out["wx_field_observed"] == 0)).sum()
        ),
        "present_weather_observability_by_year": observability_by_year,
        "label_rate_by_era": era_rows,
        "labels_by_year": by_year,
        "labels_by_month_utc": by_month,
        "reports_density_by_year": density_by_year,
        "observed_hour_coverage_pct_by_year": hour_cov,
    }
    with open(OUT_STATS, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)

    # --- validations -------------------------------------------------------
    problems = []
    if not ts_ts.is_monotonic_increasing:
        problems.append("timestamps not sorted")
    if ts_ts.duplicated().any():
        problems.append("duplicate hourly timestamps")
    if ts_ts.isna().any():
        problems.append("unparseable timestamps")
    if not (ts_ts == ts_ts.dt.tz_convert("UTC")).all() or str(ts_ts.dt.tz) != "UTC":
        problems.append("timestamps not UTC")
    vals = set(out["thunderstorm_label"].dropna().unique())
    if not vals <= {0.0, 1.0}:
        problems.append(f"unexpected label values {vals}")
    for c in ("thunderstorm_label_strict", "thunderstorm_label_recovered"):
        if not set(out[c].dropna().unique()) <= {0.0, 1.0}:
            problems.append(f"unexpected values in {c}")
    if not (out["thunderstorm_label_recovered"].fillna(-1) >= out["thunderstorm_label"].fillna(-1)).all():
        problems.append("recovered label not a superset of primary label")
    if not (out["thunderstorm_label"].fillna(-1) >= out["thunderstorm_label_strict"].fillna(-1)).all():
        problems.append("primary label not a superset of strict label")
    if (out["n_ts_reports"] > 0).sum() != n_pos:
        problems.append("positive hours != hours with >=1 TS report")
    if out["latitude"].nunique() != 1 or out["longitude"].nunique() != 1:
        problems.append("station coordinates vary")
    if out["station_id"].nunique() != 1:
        problems.append("station_id varies")
    if not (out["n_reports"] >= out["n_reports_full_decode"]).all():
        problems.append("full-decode count exceeds report count")
    if not (out["n_reports"] >= out["n_reports_with_wx"]).all():
        problems.append("present-weather count exceeds report count")
    if not set(out["wx_field_observed"].unique()) <= {0, 1}:
        problems.append("wx_field_observed is not 0/1")
    if (out["wx_field_observed"] == 1).sum() != int((out["n_reports_with_wx"] > 0).sum()):
        problems.append("wx_field_observed inconsistent with n_reports_with_wx")
    grid_set = set(grid_ts.dt.strftime("%Y-%m-%d %H:%M:%S+00:00"))
    if set(out["timestamp_utc"]) != grid_set:
        problems.append("hourly grid differs from weather_data_with_code.csv")

    print("raw records:", n_raw)
    print("records used:", len(work), "| excluded non-observation:", n_excluded_non_obs)
    print("dup valid rows dropped:", n_dup_rows, "| conflicting dup timestamps:", n_dup_conflicting)
    print("raw range:", stats["raw_date_range_utc"][0], "->", stats["raw_date_range_utc"][1])
    print("grid hours:", n_hours, "| observed:", n_observed, "| missing:", n_missing)
    print("TS reports (wxcodes):", n_ts_reports_wx, "| vcts-only:", n_ts_reports_vcts_only)
    print("recovered body TS:", n_recovered, "| ambiguous:", n_recovered_ambiguous)
    print("positive hours:", n_pos, f"({stats['positive_pct_of_observed_hours']}% of observed)")
    print("hours with a decodable present-weather group:", stats["hours_with_decodable_present_weather"])
    print("0-labels with no present-weather group in the hour:",
          stats["zero_labels_relying_on_no_present_weather_group"])
    print("label rate by era:", era_rows)
    print("validation problems:", problems if problems else "NONE")
    print("wrote:", OUT_CSV)
    print("wrote:", OUT_STATS)
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
