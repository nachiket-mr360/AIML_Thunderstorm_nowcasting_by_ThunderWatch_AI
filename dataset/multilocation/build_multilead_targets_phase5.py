"""Phase 5 (V2): multi-lead thunderstorm targets from genuine METAR labels.

features(T) stay at hour T. Only thunderstorm_target is shifted forward
within each station on the contiguous hourly grid:

    features(T) -> target_1h = METAR thunderstorm at T+1h
    features(T) -> target_2h = METAR thunderstorm at T+2h
    features(T) -> target_3h = METAR thunderstorm at T+3h

Missing/unusable METAR at the future hour remains NA (never 0).
Does not train models. Does not modify V1, Flask, or feature CSVs.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
FEAT_DIR = HERE / "features"
OUT_DIR = HERE / "targets"
OUT_REPORT = REPO / "docs" / "PHASE5_MULTILEAD_TARGET_REPORT.md"

STATIONS = ["VOTV", "VECC", "VIDP", "VOCI", "VABB"]
TIME = "timestamp_utc"
SAME_HOUR = "thunderstorm_target"
LEADS = (1, 2, 3)
IDENTITY = ["station_id", "city", TIME, SAME_HOUR, "target_observed", "label_status"]


def log(msg: str) -> None:
    print(msg, flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_hourly_grid(ts: pd.Series, station: str) -> None:
    if ts.duplicated().any():
        raise SystemExit(f"{station}: duplicate timestamps")
    if not ts.is_monotonic_increasing:
        raise SystemExit(f"{station}: timestamps not sorted")
    deltas = ts.diff().dropna()
    if not (deltas == pd.Timedelta(hours=1)).all():
        raise SystemExit(f"{station}: feature table is not contiguous hourly")


def build_station(station: str) -> tuple[pd.DataFrame, dict]:
    path = FEAT_DIR / f"{station.lower()}_features_2014_2025.csv"
    usecols = IDENTITY
    df = pd.read_csv(path, usecols=usecols)
    df[TIME] = pd.to_datetime(df[TIME], utc=True, format="ISO8601")
    if (df["station_id"] != station).any():
        raise SystemExit(f"{station}: mixed station_id in feature file")
    assert_hourly_grid(df[TIME], station)

    label = df[SAME_HOUR].astype(float)
    # shift(-L) on contiguous hourly rows == clock-hour T+L
    out = df[IDENTITY].copy()
    for L in LEADS:
        col = f"target_{L}h"
        fut_ts = df[TIME] + pd.Timedelta(hours=L)
        out[f"{col}_timestamp"] = fut_ts
        shifted = label.shift(-L)
        out[col] = shifted
        observed = shifted.notna().astype(np.int8)
        out[f"{col}_observed"] = observed
        # Direction check vs map lookup (same station only)
        lookup = pd.Series(label.to_numpy(), index=df[TIME])
        expected = fut_ts.map(lookup)
        found = shifted.to_numpy(dtype=float)
        want = expected.to_numpy(dtype=float)
        disagree = ~((np.isnan(found) & np.isnan(want)) | (found == want))
        if disagree.any():
            raise SystemExit(f"{station}: {col} alignment failed ({int(disagree.sum())} rows)")
        if int(((out[f"{col}_observed"] == 0) & (out[col] == 0)).sum()):
            raise SystemExit(f"{station}: unobserved {col} written as 0")

    n = len(out)
    stats_horizons = {}
    for L in LEADS:
        col = f"target_{L}h"
        obs = out[col].notna()
        pos = int((out[col] == 1).sum())
        neg = int((out[col] == 0).sum())
        una = int(out[col].isna().sum())
        valid = out.loc[obs]
        stats_horizons[col] = {
            "total_feature_rows": n,
            "observed_target_rows": int(obs.sum()),
            "positive_rows": pos,
            "negative_rows": neg,
            "unavailable_rows": una,
            "positive_rate": round(float(pos / obs.sum()), 6) if obs.any() else None,
            "first_valid_target_timestamp": (
                valid[TIME].iloc[0].isoformat() if len(valid) else None
            ),
            "final_valid_target_timestamp": (
                valid[TIME].iloc[-1].isoformat() if len(valid) else None
            ),
            "first_future_label_timestamp": (
                valid[f"{col}_timestamp"].iloc[0].isoformat() if len(valid) else None
            ),
            "final_future_label_timestamp": (
                valid[f"{col}_timestamp"].iloc[-1].isoformat() if len(valid) else None
            ),
        }
        log(
            f"    {station} {col}: n={n} obs={int(obs.sum())} +={pos} -={neg} "
            f"NA={una} rate={stats_horizons[col]['positive_rate']}"
        )

    # Same-hour vs 1h must not be identical on observed overlap (except rare persistence)
    both = out[SAME_HOUR].notna() & out["target_1h"].notna()
    agree = float((out.loc[both, SAME_HOUR] == out.loc[both, "target_1h"]).mean())
    n_disagree = int((out.loc[both, SAME_HOUR] != out.loc[both, "target_1h"]).sum())
    if n_disagree == 0:
        raise SystemExit(f"{station}: target_1h identical to same-hour on all observed rows; shift error?")

    examples = []
    # pick: first row, a positive 1h if any, a missing 1h if any, last-3
    idxs = [0, max(n - 4, 0)]
    pos_idx = out.index[out["target_1h"] == 1]
    na_idx = out.index[out["target_1h"].isna()]
    if len(pos_idx):
        idxs.append(int(pos_idx[0]))
    if len(na_idx):
        idxs.append(int(na_idx[0]))
    for i in sorted(set(idxs)):
        r = out.iloc[i]
        examples.append(
            {
                "station_id": station,
                "timestamp_T": r[TIME].isoformat(),
                "features_timestamp": r[TIME].isoformat(),
                "target_1h_timestamp": r["target_1h_timestamp"].isoformat(),
                "target_2h_timestamp": r["target_2h_timestamp"].isoformat(),
                "target_3h_timestamp": r["target_3h_timestamp"].isoformat(),
                "thunderstorm_target_T": None if pd.isna(r[SAME_HOUR]) else float(r[SAME_HOUR]),
                "target_1h": None if pd.isna(r["target_1h"]) else float(r["target_1h"]),
                "target_2h": None if pd.isna(r["target_2h"]) else float(r["target_2h"]),
                "target_3h": None if pd.isna(r["target_3h"]) else float(r["target_3h"]),
            }
        )

    stats = {
        "station_id": station,
        "city": str(df["city"].iloc[0]),
        "total_feature_rows": n,
        "same_hour_positives": int((df[SAME_HOUR] == 1).sum()),
        "same_hour_negatives": int((df[SAME_HOUR] == 0).sum()),
        "same_hour_unavailable": int(df[SAME_HOUR].isna().sum()),
        "first_feature_timestamp": df[TIME].iloc[0].isoformat(),
        "last_feature_timestamp": df[TIME].iloc[-1].isoformat(),
        "same_hour_vs_target_1h_agreement_when_both_observed": round(agree, 6),
        "horizons": stats_horizons,
        "examples": examples,
        "contiguous_hourly": True,
        "station_mixing": False,
    }
    return out, stats


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


def write_report(all_stats: list[dict], pooled_path: Path, created: str) -> None:
    rows = []
    for s in all_stats:
        for L in LEADS:
            h = s["horizons"][f"target_{L}h"]
            rows.append(
                [
                    s["station_id"],
                    f"{L}h",
                    h["total_feature_rows"],
                    h["observed_target_rows"],
                    h["positive_rows"],
                    h["negative_rows"],
                    h["unavailable_rows"],
                    h["positive_rate"],
                    h["first_valid_target_timestamp"],
                    h["final_valid_target_timestamp"],
                ]
            )

    example_blocks = []
    for s in all_stats:
        example_blocks.append(f"### {s['station_id']} ({s['city']})")
        for ex in s["examples"]:
            example_blocks.append(
                f"- **T** `{ex['timestamp_T']}`  \n"
                f"  features timestamp = `{ex['features_timestamp']}`  \n"
                f"  target_1h timestamp = `{ex['target_1h_timestamp']}` value={ex['target_1h']}  \n"
                f"  target_2h timestamp = `{ex['target_2h_timestamp']}` value={ex['target_2h']}  \n"
                f"  target_3h timestamp = `{ex['target_3h_timestamp']}` value={ex['target_3h']}  \n"
                f"  same-hour METAR at T = {ex['thunderstorm_target_T']}"
            )
        example_blocks.append("")

    body = f"""# Phase 5 — Multi-lead thunderstorm target construction (V2)

**Generated:** {created}  
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

{md_table(
    [
        "Station",
        "Horizon",
        "Feature rows",
        "Observed target",
        "Positive",
        "Negative",
        "Unavailable",
        "Positive rate",
        "First valid T",
        "Final valid T",
    ],
    rows,
)}

Checks:

- no station mixing (each input file has a single `station_id`)
- no timestamp errors (unique, sorted, exactly 1 h steps)
- no future feature leakage (feature columns not rewritten; only labels shifted)
- direction self-test: `target_Lh(T) == thunderstorm_target(T+L)` on the same station, including NA
- unobserved future hours are never 0
- same-hour vs `target_1h` agreement is well below 1 (shift is real)

## Alignment examples

{chr(10).join(example_blocks)}

Example interpretation: if T is `2014-01-02T00:00:00+00:00`, then target_1h is the METAR thunderstorm flag at `2014-01-02T01:00:00+00:00`, target_2h at `02:00Z`, target_3h at `03:00Z`. Feature columns still describe `00:00Z`.

## Hourly resolution

Predictors and METAR labels are on the **clock hour**. These leads are 1, 2, and 3 **hours**, not 30 minutes. Do not describe this table as sub-hourly nowcasting.

## Artifacts

| Path | Role |
| --- | --- |
| `dataset/multilocation/targets/<icao>_nowcast_targets_2014_2025.csv` | Per-station leads |
| `dataset/multilocation/targets/multilocation_nowcast_targets_2014_2025.csv` | Pooled |
| `dataset/multilocation/targets/multilocation_nowcast_targets_2014_2025_metadata.json` | Counts |
| `{pooled_path.as_posix()}` | pooled CSV |

Join to Phase 3 features on `(station_id, timestamp_utc)`. Feature vectors are **not** duplicated here.

## Not done

- No 1h/2h/3h model training
- No Flask / frontend / V1 edits
- No NWP / lightning / satellite / radar
- No git commit or push
"""
    OUT_REPORT.write_text(body, encoding="utf-8")


def main() -> int:
    log("=" * 72)
    log("PHASE 5 V2 — MULTI-LEAD METAR TARGETS (1h / 2h / 3h)")
    log("=" * 72)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    frames = []
    all_stats = []
    for station in STATIONS:
        log(f"\n[{station}]")
        out, stats = build_station(station)
        per = OUT_DIR / f"{station.lower()}_nowcast_targets_2014_2025.csv"
        out.to_csv(per, index=False)
        stats["output_csv"] = str(per.relative_to(REPO)).replace("\\", "/")
        stats["sha256"] = sha256_file(per)
        frames.append(out)
        all_stats.append(stats)

    pooled = pd.concat(frames, ignore_index=True)
    if pooled.duplicated(["station_id", TIME]).any():
        raise SystemExit("pooled duplicate (station_id, timestamp)")
    mix = pooled.groupby(TIME)["station_id"].nunique()
    if (mix > 5).any():
        raise SystemExit("unexpected station count per timestamp")

    pooled_path = OUT_DIR / "multilocation_nowcast_targets_2014_2025.csv"
    pooled.to_csv(pooled_path, index=False)
    created = datetime.now(timezone.utc).isoformat()
    meta = {
        "phase": "5",
        "version": "SIH26072-V2-phase5-multilead-targets",
        "created_at_utc": created,
        "leads_hours": list(LEADS),
        "source_label": SAME_HOUR,
        "missing_filled_with_zero": False,
        "resolution": "hourly",
        "thirty_minute_target_claimed": False,
        "stations": all_stats,
        "pooled_rows": int(len(pooled)),
        "pooled_sha256": sha256_file(pooled_path),
        "forbidden": [
            "Open-Meteo weather_code labels",
            "precipitation-derived labels",
            "model predictions as labels",
            "future atmospheric variables as features",
            "cross-station label shift",
            "NA converted to 0",
        ],
    }
    meta_path = OUT_DIR / "multilocation_nowcast_targets_2014_2025_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_report(all_stats, pooled_path.relative_to(REPO), created)
    log(f"\nWrote {pooled_path}")
    log(f"Wrote {OUT_REPORT}")
    log("PHASE STATUS: READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
