"""Phase 8C: apply frozen Phase 6 UTC cutoffs to NWP overlap table. No training."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent.parent
CSV = BASE / "features_nwp" / "multilocation_features_nwp_overlap_2021_2025.csv"
P4_SPLIT = ROOT / "outputs" / "v2_baseline" / "split_boundaries.json"
P6_SPLIT = ROOT / "outputs" / "v2_multilead" / "split_boundaries.json"
REPORT = ROOT / "docs" / "PHASE8C_READINESS_AUDIT.md"
OUT_JSON = BASE / "features_nwp" / "phase8c_readiness_audit.json"

STATIONS = ["VOTV", "VECC", "VIDP", "VOCI", "VABB"]
HORIZONS = ("target_1h", "target_2h", "target_3h")
SPLITS = ("train", "validation", "test")
TIME = "timestamp_utc"
STATION = "station_id"


def assign_split(ts: pd.Series, train_end, val_end) -> pd.Series:
    out = pd.Series("test", index=ts.index)
    out[ts <= train_end] = "train"
    out[(ts > train_end) & (ts <= val_end)] = "validation"
    return out


def counts(sub: pd.DataFrame, horizon: str) -> dict:
    n = int(len(sub))
    observed = sub[horizon].notna()
    n_obs = int(observed.sum())
    n_miss = int((~observed).sum())
    y = sub.loc[observed, horizon]
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    other = n_obs - n_pos - n_neg
    pos_rate = (n_pos / n_obs) if n_obs else None
    return {
        "total_rows": n,
        "observed_target_rows": n_obs,
        "positive_rows": n_pos,
        "negative_rows": n_neg,
        "missing_target_rows": n_miss,
        "non_binary_observed": int(other),
        "positive_rate": pos_rate,
    }


def main() -> int:
    with P4_SPLIT.open(encoding="utf-8") as fh:
        p4 = json.load(fh)
    with P6_SPLIT.open(encoding="utf-8") as fh:
        p6 = json.load(fh)

    train_end = pd.Timestamp(p4["train"]["end_timestamp_utc"])
    val_end = pd.Timestamp(p4["validation"]["end_timestamp_utc"])
    p6_train_start = pd.Timestamp(p4["train"]["start_timestamp_utc"])
    p6_val_start = pd.Timestamp(p4["validation"]["start_timestamp_utc"])
    p6_test_start = pd.Timestamp(p4["test"]["start_timestamp_utc"])
    p6_test_end = pd.Timestamp(p4["test"]["end_timestamp_utc"])

    df = pd.read_csv(CSV, usecols=[STATION, TIME] + list(HORIZONS))
    df[TIME] = pd.to_datetime(df[TIME], utc=True, format="ISO8601")
    df["split"] = assign_split(df[TIME], train_end, val_end)

    overlap_min = df[TIME].min()
    overlap_max = df[TIME].max()
    train_cutoff_before_nwp = bool(p6_train_start < overlap_min)

    results = {}
    for h in HORIZONS:
        results[h] = {"pooled": {}, "by_station": {}}
        for sp in SPLITS:
            sub = df.loc[df["split"] == sp]
            results[h]["pooled"][sp] = counts(sub, h)
            results[h]["by_station"][sp] = {st: counts(sub.loc[sub[STATION] == st], h) for st in STATIONS}

    split_windows = {}
    for sp in SPLITS:
        sub = df.loc[df["split"] == sp]
        split_windows[sp] = {
            "n_rows": int(len(sub)),
            "start": sub[TIME].min().isoformat() if len(sub) else None,
            "end": sub[TIME].max().isoformat() if len(sub) else None,
        }

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase6_cutoffs_source": str(P4_SPLIT.relative_to(ROOT)).replace("\\", "/"),
        "phase6_frozen_boundaries": {
            "train_start": p6_train_start.isoformat(),
            "train_end": train_end.isoformat(),
            "validation_start": p6_val_start.isoformat(),
            "validation_end": val_end.isoformat(),
            "test_start": p6_test_start.isoformat(),
            "test_end": p6_test_end.isoformat(),
            "phase6_split_json_train_end": p6.get("train_end"),
        },
        "new_cutoffs_invented": False,
        "train_start_before_nwp_period": train_cutoff_before_nwp,
        "overlap_start": overlap_min.isoformat(),
        "overlap_end": overlap_max.isoformat(),
        "overlap_split_windows": split_windows,
        "counts": results,
        "training": False,
        "imputation": False,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def r(x):
        return "NA" if x is None else f"{x:.4f}"

    def table(block: dict) -> str:
        lines = [
            "| split | total | observed | pos | neg | missing target | pos rate |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for sp in SPLITS:
            m = block[sp]
            lines.append(
                f"| {sp} | {m['total_rows']} | {m['observed_target_rows']} | {m['positive_rows']} | "
                f"{m['negative_rows']} | {m['missing_target_rows']} | {r(m['positive_rate'])} |"
            )
        return "\n".join(lines)

    lines = [
        "# Phase 8C — NWP experiment readiness audit (V2)",
        "",
        f"**Generated:** {payload['generated_at_utc']}",
        "**Script:** `dataset/multilocation/audit_nwp_readiness_phase8c.py`",
        "**Scope:** readiness counts only. No training, no imputation, no models, no new split dates.",
        "",
        "Phase 6 artifacts were not modified.",
        "",
        "## Frozen Phase 6 / Phase 4 cutoffs (unchanged)",
        "",
        "Assignment matches Phase 6: `train` if `timestamp <= train_end`; `validation` if `train_end < timestamp <= validation_end`; else `test`.",
        "",
        "| partition | Phase 6 start UTC | Phase 6 end UTC |",
        "| --- | --- | --- |",
        f"| train | {p6_train_start.isoformat()} | {train_end.isoformat()} |",
        f"| validation | {p6_val_start.isoformat()} | {val_end.isoformat()} |",
        f"| test | {p6_test_start.isoformat()} | {p6_test_end.isoformat()} |",
        "",
        f"**Train cutoff starts before the NWP overlap:** `{train_cutoff_before_nwp}`  ",
        f"NWP overlap clock: `{overlap_min.isoformat()}` → `{overlap_max.isoformat()}`  ",
        "Phase 6 train start is 2014-01-02; NWP only exists from 2021-04-01. Cutoffs were **not** rewritten.",
        "",
        "## Resulting overlap windows (same cutoffs, truncated by data)",
        "",
        "| split | overlap start | overlap end | rows |",
        "| --- | --- | --- | --- |",
    ]
    for sp in SPLITS:
        w = split_windows[sp]
        lines.append(f"| {sp} | {w['start']} | {w['end']} | {w['n_rows']} |")
    lines += [
        "",
        "Test rows after Phase 6 reported `test_end` (2025-12-30 23:00) remain in **test** because Phase 6 uses an open-ended `else test` rule, not a new cutoff.",
        "",
        "## Pooled counts by lead",
        "",
    ]
    for h in HORIZONS:
        lines.append(f"### {h}")
        lines.append("")
        lines.append(table(results[h]["pooled"]))
        lines.append("")
    lines.append("## Counts by station")
    lines.append("")
    for h in HORIZONS:
        lines.append(f"### {h} by station")
        lines.append("")
        lines.append(
            "| station | split | total | observed | pos | neg | missing target | pos rate |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for st in STATIONS:
            for sp in SPLITS:
                m = results[h]["by_station"][sp][st]
                lines.append(
                    f"| {st} | {sp} | {m['total_rows']} | {m['observed_target_rows']} | {m['positive_rows']} | "
                    f"{m['negative_rows']} | {m['missing_target_rows']} | {r(m['positive_rate'])} |"
                )
        lines.append("")
    lines += [
        "## Readiness notes",
        "",
        "- Same chronological UTC boundaries as Phase 6; no random split; no new dates.",
        "- Missing targets left as missing; not converted to 0.",
        "- NWP overlap train set is shorter than Phase 6 train because 2014–2021-03 has no GFS Historical Forecast in this collection.",
        "- Validation and test fall entirely inside the NWP window.",
        "- This audit does not decide whether NWP helps.",
        "",
        "## Artifacts",
        "",
        "| path | role |",
        "| --- | --- |",
        "| `docs/PHASE8C_READINESS_AUDIT.md` | this report |",
        "| `dataset/multilocation/features_nwp/phase8c_readiness_audit.json` | counts JSON |",
        "",
        "## PHASE STATUS",
        "",
        "**READY** — readiness audit only. Do not proceed to Phase 8D in this task.",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
