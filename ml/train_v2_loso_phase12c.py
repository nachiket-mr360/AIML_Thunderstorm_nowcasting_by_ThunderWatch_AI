"""Phase 12C: Leave-one-station-out generalization of Model B.

Experiment only. Does not modify Phase 3/6/8/12B artifacts.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

SEED = 42
F_BETA = 2.0
ALERT_RATE_CAP = 0.20

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_CSV = (
    BASE_DIR
    / "dataset"
    / "multilocation"
    / "features_nwp"
    / "multilocation_features_nwp_overlap_2021_2025.csv"
)
PHASE4_SPLIT = BASE_DIR / "outputs" / "v2_baseline" / "split_boundaries.json"
OUT_DIR = BASE_DIR / "outputs" / "v2_generalization" / "phase12c"
REPORT_PATH = BASE_DIR / "docs" / "PHASE12C_LOSO_GENERALIZATION_REPORT.md"

TIME = "timestamp_utc"
STATION = "station_id"
STATIONS = ["VOTV", "VECC", "VIDP", "VOCI", "VABB"]
HORIZONS = ("target_1h", "target_2h", "target_3h")

LOCATION_FEATURES = ["latitude", "longitude", "elevation_m"]
ATMOSPHERIC_FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "precipitation",
    "cloud_cover",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "wind_direction_sin",
    "wind_direction_cos",
    "wind_u_10m",
    "wind_v_10m",
    "temperature_lag_1h",
    "temperature_lag_3h",
    "temperature_lag_6h",
    "temperature_lag_12h",
    "temperature_lag_24h",
    "humidity_lag_1h",
    "humidity_lag_3h",
    "humidity_lag_6h",
    "humidity_lag_12h",
    "humidity_lag_24h",
    "pressure_lag_1h",
    "pressure_lag_3h",
    "pressure_lag_6h",
    "pressure_lag_12h",
    "pressure_lag_24h",
    "wind_speed_lag_1h",
    "wind_speed_lag_3h",
    "wind_speed_lag_6h",
    "wind_speed_lag_12h",
    "wind_speed_lag_24h",
    "precipitation_lag_1h",
    "precipitation_lag_3h",
    "precipitation_lag_6h",
    "precipitation_lag_12h",
    "precipitation_lag_24h",
    "cloud_cover_lag_1h",
    "cloud_cover_lag_3h",
    "cloud_cover_lag_6h",
    "cloud_cover_lag_12h",
    "cloud_cover_lag_24h",
    "temperature_change_1h",
    "temperature_change_3h",
    "humidity_change_1h",
    "humidity_change_3h",
    "pressure_change_1h",
    "pressure_change_3h",
    "wind_speed_change_1h",
    "wind_speed_change_3h",
    "precipitation_change_1h",
    "precipitation_change_3h",
    "cloud_cover_change_1h",
    "cloud_cover_change_3h",
    "precipitation_roll_sum_3h",
    "precipitation_roll_sum_6h",
    "precipitation_roll_sum_12h",
    "precipitation_roll_sum_24h",
    "humidity_roll_mean_3h",
    "humidity_roll_mean_6h",
    "pressure_roll_mean_3h",
    "pressure_roll_mean_6h",
    "temperature_roll_mean_3h",
    "temperature_roll_mean_6h",
    "wind_speed_roll_mean_3h",
    "wind_speed_roll_mean_6h",
    "cloud_cover_roll_mean_3h",
    "cloud_cover_roll_mean_6h",
]
MODEL_B_FEATURES = ATMOSPHERIC_FEATURES + LOCATION_FEATURES + [
    "nwp_cape",
    "nwp_convective_inhibition",
    "nwp_lifted_index",
    "nwp_temperature_2m",
    "nwp_relative_humidity_2m",
    "nwp_surface_pressure",
    "nwp_wind_speed_10m",
    "nwp_wind_direction_10m",
    "nwp_precipitation",
    "nwp_cloud_cover",
]
NWP_FEATURES = MODEL_B_FEATURES[-10:]

MODEL_PARAMS = {
    "n_estimators": 200,
    "criterion": "gini",
    "max_features": "sqrt",
    "class_weight": "balanced",
    "random_state": SEED,
    "n_jobs": -1,
}

FORBIDDEN_COLS = {"annual_thunder_hours"}
PROTECTED = [
    BASE_DIR / "dataset" / "multilocation" / "features" / "multilocation_features_2014_2025.csv",
    DATA_CSV,
    BASE_DIR / "docs" / "PHASE8D_NWP_EXPERIMENT_REPORT.md",
    BASE_DIR / "docs" / "PHASE12B_LIGHTNING_CLIMATOLOGY_EXPERIMENT.md",
    BASE_DIR / "outputs" / "v2_multimodal" / "phase12b" / "experiment_config.json",
    BASE_DIR / "ml" / "train_v2_nwp_experiment_phase8d.py",
]


def log(msg: str) -> None:
    print(msg, flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def metrics_at_threshold(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    roc = float(roc_auc_score(y_true, proba)) if len(np.unique(y_true)) > 1 else None
    pr = float(average_precision_score(y_true, proba)) if len(np.unique(y_true)) > 1 else None
    n = len(y_true)
    pos = int(np.sum(y_true == 1))
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=F_BETA, zero_division=0)),
        "roc_auc": roc,
        "pr_auc": pr,
        "predicted_alert_rate": float(np.mean(y_pred)) if n else None,
        "n_samples": int(n),
        "positives": pos,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def threshold_sweep(y_true: np.ndarray, proba: np.ndarray, beta: float = F_BETA) -> pd.DataFrame:
    order = np.argsort(-proba, kind="stable")
    ys = y_true[order]
    ps = proba[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    run_end = np.empty(len(ps), dtype=bool)
    run_end[:-1] = ps[:-1] != ps[1:]
    run_end[-1] = True
    tp, fp, ps = tp[run_end], fp[run_end], ps[run_end]
    n = len(y_true)
    positives = int(np.sum(y_true))
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / positives if positives else np.zeros(len(ps))
    beta_sq = beta * beta
    denom = beta_sq * precision + recall
    fbeta = np.where(denom > 0, (1 + beta_sq) * precision * recall / np.maximum(denom, 1e-12), 0.0)
    return pd.DataFrame(
        {
            "threshold": ps,
            "fbeta": fbeta,
            "predicted_positive_rate": (tp + fp) / n,
        }
    )


def select_threshold(sweep: pd.DataFrame, cap: float = ALERT_RATE_CAP) -> dict:
    inside = sweep[sweep["predicted_positive_rate"] <= cap]
    if inside.empty:
        raise SystemExit("no candidate threshold satisfies the alert-rate guardrail")
    best_fbeta = float(inside["fbeta"].max())
    tied = inside[np.isclose(inside["fbeta"], best_fbeta)]
    chosen = tied.iloc[int(tied["threshold"].to_numpy().argmin())]
    unconstrained = sweep.iloc[int(sweep["fbeta"].to_numpy().argmax())]
    return {
        "selected_threshold": float(chosen["threshold"]),
        "threshold_selected_on": "chronological_validation_training_stations_only",
        "test_station_used_for_threshold_selection": False,
        "guardrail_binding": bool(unconstrained["predicted_positive_rate"] > cap),
        "selected_validation_fbeta": float(chosen["fbeta"]),
        "selected_validation_predicted_positive_rate": float(chosen["predicted_positive_rate"]),
    }


def assign_split(ts: pd.Series, train_end, val_end) -> pd.Series:
    out = pd.Series("test", index=ts.index)
    out[ts <= train_end] = "train"
    out[(ts > train_end) & (ts <= val_end)] = "validation"
    return out


def r(x):
    return "NA" if x is None else f"{x:.4f}"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    before = {str(p.relative_to(BASE_DIR)): sha256_file(p) for p in PROTECTED if p.exists()}

    with PHASE4_SPLIT.open(encoding="utf-8") as fh:
        p4 = json.load(fh)
    train_end = pd.Timestamp(p4["train"]["end_timestamp_utc"])
    val_end = pd.Timestamp(p4["validation"]["end_timestamp_utc"])

    if len(MODEL_B_FEATURES) != 83:
        raise SystemExit(f"STOP: Model B feature count {len(MODEL_B_FEATURES)} != 83")
    if any("target" in c.lower() or "label" in c.lower() or "weather_code" in c.lower() for c in MODEL_B_FEATURES):
        raise SystemExit("STOP: target leakage in features")
    if "annual_thunder_hours" in MODEL_B_FEATURES:
        raise SystemExit("STOP: lightning climatology in Model B")

    usecols = [STATION, TIME] + MODEL_B_FEATURES + list(HORIZONS)
    log(f"Loading {DATA_CSV}")
    df = pd.read_csv(DATA_CSV, usecols=usecols)
    extra = FORBIDDEN_COLS.intersection(df.columns)
    if extra:
        raise SystemExit(f"STOP: forbidden columns present: {extra}")
    sat_radar = [c for c in df.columns if "satellite" in c.lower() or "radar" in c.lower()]
    if sat_radar:
        raise SystemExit(f"STOP: satellite/radar columns: {sat_radar}")

    df[TIME] = pd.to_datetime(df[TIME], utc=True, format="ISO8601")
    stations = sorted(df[STATION].astype(str).unique().tolist())
    if stations != sorted(STATIONS):
        raise SystemExit(f"STOP: expected 5 stations {STATIONS}, got {stations}")
    if int(df.duplicated([STATION, TIME]).sum()):
        raise SystemExit("STOP: duplicate keys")

    complete = df[NWP_FEATURES].notna().all(axis=1)
    df_cc = df.loc[complete].copy()
    if int(df_cc[NWP_FEATURES].isna().sum().sum()):
        raise SystemExit("STOP: NWP NA after complete-case")
    log(f"Complete-case rows: {len(df_cc)} / {len(df)}")

    records = []
    summaries = []

    for hold in STATIONS:
        train_stations = [s for s in STATIONS if s != hold]
        log(f"LOSO hold={hold} train={train_stations}")
        for horizon in HORIZONS:
            observed = df_cc[df_cc[horizon].notna()].copy()
            observed[horizon] = observed[horizon].astype(int)
            observed["split"] = assign_split(observed[TIME], train_end, val_end)

            pool = observed[observed[STATION].isin(train_stations)]
            held = observed[observed[STATION] == hold]
            train_df = pool.loc[pool["split"] == "train"]
            val_df = pool.loc[pool["split"] == "validation"]
            test_df = held.loc[held["split"] == "test"]

            if hold in set(train_df[STATION].unique()) or hold in set(val_df[STATION].unique()):
                raise SystemExit(f"STOP: held-out {hold} leaked into train/val")
            if set(test_df[STATION].unique()) != {hold}:
                raise SystemExit(f"STOP: test stations != {{{hold}}}")
            if set(train_df[STATION].unique()) != set(train_stations):
                raise SystemExit(f"STOP: train station coverage {set(train_df[STATION].unique())}")
            if train_df.empty or val_df.empty or test_df.empty:
                raise SystemExit(f"STOP: empty split hold={hold} {horizon}")
            if train_df[TIME].max() >= val_df[TIME].min():
                raise SystemExit("STOP: train/val chronological leak")
            if val_df[TIME].max() >= test_df[TIME].min():
                raise SystemExit("STOP: validation not entirely before held-out test")

            y_train = train_df[horizon].to_numpy(dtype=int)
            y_val = val_df[horizon].to_numpy(dtype=int)
            y_test = test_df[horizon].to_numpy(dtype=int)
            X_train = train_df[MODEL_B_FEATURES].to_numpy(dtype=np.float32)
            X_val = val_df[MODEL_B_FEATURES].to_numpy(dtype=np.float32)
            X_test = test_df[MODEL_B_FEATURES].to_numpy(dtype=np.float32)
            if not np.isfinite(X_train).all():
                raise SystemExit("STOP: non-finite train features")

            log(f"  fit {horizon} n_train={len(train_df)} n_val={len(val_df)} n_test={len(test_df)}")
            clf = RandomForestClassifier(**MODEL_PARAMS)
            clf.fit(X_train, y_train)
            proba_val = clf.predict_proba(X_val)[:, 1]
            thr_info = select_threshold(threshold_sweep(y_val, proba_val))
            threshold = float(thr_info["selected_threshold"])
            m_val = metrics_at_threshold(y_val, proba_val, threshold)
            m_test = metrics_at_threshold(y_test, clf.predict_proba(X_test)[:, 1], threshold)
            del clf, X_train, X_val, X_test

            rec = {
                "experiment": "LEAVE-ONE-STATION-OUT GENERALIZATION",
                "held_out_station": hold,
                "train_stations": ",".join(train_stations),
                "lead": horizon,
                "n_features": len(MODEL_B_FEATURES),
                "n_train": int(len(train_df)),
                "n_validation": int(len(val_df)),
                "positives_train": int(y_train.sum()),
                "positives_validation": int(y_val.sum()),
                "n_test": m_test["n_samples"],
                "positives_test": m_test["positives"],
                "threshold": threshold,
                "pr_auc": m_test["pr_auc"],
                "roc_auc": m_test["roc_auc"],
                "precision": m_test["precision"],
                "recall": m_test["recall"],
                "f1": m_test["f1"],
                "f2": m_test["f2"],
                "alert_rate": m_test["predicted_alert_rate"],
                "val_f2": m_val["f2"],
                "val_alert_rate": m_val["predicted_alert_rate"],
                "train_max_utc": train_df[TIME].max().isoformat(),
                "val_max_utc": val_df[TIME].max().isoformat(),
                "test_min_utc": test_df[TIME].min().isoformat(),
            }
            records.append(rec)
            summaries.append(
                {
                    "held_out_station": hold,
                    "lead": horizon,
                    "n_test": rec["n_test"],
                    "positives_test": rec["positives_test"],
                    "threshold": rec["threshold"],
                    "pr_auc": rec["pr_auc"],
                    "roc_auc": rec["roc_auc"],
                    "f2": rec["f2"],
                    "alert_rate": rec["alert_rate"],
                }
            )

    after = {str(p.relative_to(BASE_DIR)): sha256_file(p) for p in PROTECTED if p.exists()}
    if before != after:
        raise SystemExit("STOP: protected artifact hash changed")

    metrics_df = pd.DataFrame(records)
    metrics_df.to_csv(OUT_DIR / "loso_metrics.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUT_DIR / "loso_station_summary.csv", index=False)

    config = {
        "phase": "12C",
        "experiment_label": "LEAVE-ONE-STATION-OUT GENERALIZATION",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "not_a_replacement_for_phase8d_pooled_benchmark": True,
        "model": "B atmospheric + location_temporal + nwp",
        "n_features": 83,
        "feature_list": MODEL_B_FEATURES,
        "rf": MODEL_PARAMS,
        "complete_case_nwp": True,
        "annual_thunder_hours": False,
        "satellite": False,
        "radar": False,
        "train_end_utc": p4["train"]["end_timestamp_utc"],
        "validation_end_utc": p4["validation"]["end_timestamp_utc"],
        "threshold": "max F2 on chronological validation of TRAIN stations only; alert rate <= 0.20",
        "held_out_evaluation_window": "Phase 6/8 test timestamps only (after validation_end)",
        "protected_hashes_unchanged": True,
        "stations": STATIONS,
    }
    with (OUT_DIR / "loso_config.json").open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)

    def rows_for(lead: str) -> list[dict]:
        return [x for x in records if x["lead"] == lead]

    lines = [
        "# Phase 12C — Leave-one-station-out generalization",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        "**Script:** `ml/train_v2_loso_phase12c.py`",
        "**Label:** LEAVE-ONE-STATION-OUT GENERALIZATION",
        "**Scope:** experiment only. Not a replacement for the frozen Phase 8D pooled chronological benchmark.",
        "",
        "Phase 3 / 6 / 8A–8E / 12B artifacts were not modified. Frozen models were not overwritten.",
        "",
        "## 1. Objective",
        "",
        "Evaluate whether validated V2 **Model B** (atmospheric + location/temporal + NWP) transfers to a station that never appears in training or threshold selection.",
        "",
        "## 2. Dataset",
        "",
        "| item | value |",
        "| --- | --- |",
        "| CSV | `dataset/multilocation/features_nwp/multilocation_features_nwp_overlap_2021_2025.csv` |",
        f"| complete-case NWP rows | {len(df_cc)} / {len(df)} |",
        "| missing targets | excluded, not filled with 0 |",
        "| lightning / satellite / radar | not used |",
        "",
        "## 3. Feature groups",
        "",
        f"Model B: **{len(MODEL_B_FEATURES)}** columns = Phase 3 atmospheric + location/temporal (73) + 10 `nwp_*` (Phase 8D).",
        "",
        "## 4. LOSO methodology",
        "",
        "Five independent experiments. For each held-out ICAO:",
        "",
        "- Fit RF on the other four stations, **train** timestamps only (`timestamp <= train_end`).",
        "- Select threshold on the other four stations, **validation** timestamps only.",
        "- Evaluate on the held-out station, **test** timestamps only (`timestamp > validation_end`).",
        "",
        "This keeps chronological validation **entirely before** held-out test evaluation, and keeps the held-out station out of fitting and thresholding.",
        "",
        "This is **spatial transfer**, not the Phase 8D pooled test. Do not treat LOSO metrics as beating or replacing 8D.",
        "",
        "## 5. Station split table",
        "",
        "| held-out test | train stations |",
        "| --- | --- |",
        "| VOTV | VECC, VIDP, VOCI, VABB |",
        "| VECC | VOTV, VIDP, VOCI, VABB |",
        "| VIDP | VOTV, VECC, VOCI, VABB |",
        "| VOCI | VOTV, VECC, VIDP, VABB |",
        "| VABB | VOTV, VECC, VIDP, VOCI |",
        "",
        "## 6. Threshold methodology",
        "",
        "- Same Phase 8D rule: maximize validation **F2**, predicted alert rate **≤ 0.20**, tie-break lower threshold.",
        "- Validation pool = training stations only.",
        "- Frozen threshold applied to the unseen station’s test window.",
        "- Test station unused for threshold selection.",
        "",
        f"- Train end UTC: `{p4['train']['end_timestamp_utc']}`",
        f"- Validation end UTC: `{p4['validation']['end_timestamp_utc']}`",
        "",
    ]

    for lead, title in (("target_1h", "7. 1h results"), ("target_2h", "8. 2h results"), ("target_3h", "9. 3h results")):
        lines += [
            f"## {title}",
            "",
            "| held-out | n_test | pos | threshold | PR-AUC | ROC-AUC | P | R | F1 | F2 | alert rate |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for rec in rows_for(lead):
            m = rec
            lines.append(
                f"| {m['held_out_station']} | {m['n_test']} | {m['positives_test']} | {m['threshold']:.6f} | "
                f"{r(m['pr_auc'])} | {r(m['roc_auc'])} | {r(m['precision'])} | {r(m['recall'])} | "
                f"{r(m['f1'])} | {r(m['f2'])} | {r(m['alert_rate'])} |"
            )
        lines.append("")

    lines += [
        "## 10. Per-station results",
        "",
        "Pooled station × lead table (same numbers as above; not a ranking).",
        "",
        "| station | lead | n_test | pos | threshold | PR-AUC | ROC-AUC | P | R | F1 | F2 | alert rate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for rec in records:
        lines.append(
            f"| {rec['held_out_station']} | {rec['lead']} | {rec['n_test']} | {rec['positives_test']} | "
            f"{rec['threshold']:.6f} | {r(rec['pr_auc'])} | {r(rec['roc_auc'])} | {r(rec['precision'])} | "
            f"{r(rec['recall'])} | {r(rec['f1'])} | {r(rec['f2'])} | {r(rec['alert_rate'])} |"
        )
    lines += [
        "",
        "Stations are **not** ranked. No best/worst location is declared.",
        "",
        "## 11. Limitations",
        "",
        "- Five aerodromes only; climate regimes differ (coastal vs inland monsoon).",
        "- Location features (lat/lon/elev) of the held-out site were never seen in training; the RF may not interpolate geography.",
        "- NWP overlap from 2021-04-01; complete-case drops CIN/precip holes.",
        "- Held-out evaluation uses the chronological **test** window only, so sample size is smaller than all-years LOSO.",
        "- Alert-rate cap is enforced on **training-station validation**, so held-out alert rate may exceed 20%.",
        "- Not comparable 1:1 to Phase 8D pooled test (that test still saw the station in training).",
        "",
        "## 12. Interpretation",
        "",
        "LOSO measures **cross-station transfer** of Model B. High pooled 8D skill can coexist with weak transfer if the forest relies on station-specific climate encoded in location/NWP climatology.",
        "",
        "These results do **not** claim operational superiority, production readiness, or that a new site can be added without local labels.",
        "",
        "## Integrity",
        "",
        "- Five stations; held-out absent from train/val: **True**",
        "- Feature count 83; no targets/lightning/satellite/radar: **True**",
        "- Complete-case NWP as Phase 8D: **True**",
        "- Validation max timestamp < held-out test min: **True**",
        "- Protected hashes unchanged: **True**",
        "",
        "## Artifacts",
        "",
        "| path | role |",
        "| --- | --- |",
        "| `outputs/v2_generalization/phase12c/loso_metrics.csv` | full metrics |",
        "| `outputs/v2_generalization/phase12c/loso_station_summary.csv` | compact station × lead |",
        "| `outputs/v2_generalization/phase12c/loso_config.json` | config |",
        "| `docs/PHASE12C_LOSO_GENERALIZATION_REPORT.md` | this report |",
        "",
        "## PHASE STATUS",
        "",
        "**READY** — LOSO generalization experiment complete. Stop; do not start Phase 12D.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    log(f"Wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
