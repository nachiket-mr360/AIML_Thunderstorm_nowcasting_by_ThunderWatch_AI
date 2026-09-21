"""Phase 12B: Model B (atmo+NWP) vs Model C (+ annual_thunder_hours).

Experiment only. Does not modify Phase 3/6/8 artifacts or frozen models.
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
LIS_CSV = BASE_DIR / "outputs" / "lightning" / "combined_lis_thunder_hour_locations.csv"
PHASE4_SPLIT = BASE_DIR / "outputs" / "v2_baseline" / "split_boundaries.json"
OUT_DIR = BASE_DIR / "outputs" / "v2_multimodal" / "phase12b"
REPORT_PATH = BASE_DIR / "docs" / "PHASE12B_LIGHTNING_CLIMATOLOGY_EXPERIMENT.md"

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
ATMO_FEATURES = ATMOSPHERIC_FEATURES + LOCATION_FEATURES
NWP_FEATURES = [
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
MODEL_B_FEATURES = ATMO_FEATURES + NWP_FEATURES
MODEL_C_FEATURES = MODEL_B_FEATURES + ["annual_thunder_hours"]

MODEL_PARAMS = {
    "n_estimators": 200,
    "criterion": "gini",
    "max_features": "sqrt",
    "class_weight": "balanced",
    "random_state": SEED,
    "n_jobs": -1,
}

PROTECTED = [
    BASE_DIR / "dataset" / "multilocation" / "features" / "multilocation_features_2014_2025.csv",
    DATA_CSV,
    BASE_DIR / "outputs" / "v2_multilead" / "metrics_target_1h.json",
    BASE_DIR / "models" / "v2" / "multilead" / "metadata_target_1h.json",
    BASE_DIR / "ml" / "train_v2_nwp_experiment_phase8d.py",
    BASE_DIR / "docs" / "PHASE8D_NWP_EXPERIMENT_REPORT.md",
]


def log(msg: str) -> None:
    print(msg, flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def f2_from_pr(p: float, rec: float) -> float:
    beta_sq = F_BETA * F_BETA
    denom = beta_sq * p + rec
    if denom <= 0:
        return 0.0
    return float((1 + beta_sq) * p * rec / denom)


def metrics_at_threshold(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    roc = float(roc_auc_score(y_true, proba)) if len(np.unique(y_true)) > 1 else None
    pr = float(average_precision_score(y_true, proba)) if len(np.unique(y_true)) > 1 else None
    n = len(y_true)
    pos = int(np.sum(y_true == 1))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    f2 = float(fbeta_score(y_true, y_pred, beta=F_BETA, zero_division=0))
    return {
        "threshold": float(threshold),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "f2": f2,
        "roc_auc": roc,
        "pr_auc": pr,
        "predicted_alert_rate": float(np.mean(y_pred)) if n else None,
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "n_samples": int(n),
        "positives": pos,
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
    f1 = np.where((precision + recall) > 0, 2 * precision * recall / np.maximum(precision + recall, 1e-12), 0.0)
    return pd.DataFrame(
        {
            "threshold": ps,
            "precision": precision,
            "recall": recall,
            "f1": f1,
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
        "criterion": (
            f"maximise F{int(F_BETA)} (beta={F_BETA}) on validation, "
            f"subject to predicted positive rate <= {cap}; tie-break lower threshold"
        ),
        "selected_threshold": float(chosen["threshold"]),
        "threshold_selected_on": "validation",
        "test_data_used_for_threshold_selection": False,
        "guardrail_binding": bool(unconstrained["predicted_positive_rate"] > cap),
        "selected_validation_fbeta": float(chosen["fbeta"]),
        "selected_validation_predicted_positive_rate": float(chosen["predicted_positive_rate"]),
    }


def assign_split(ts: pd.Series, train_end, val_end) -> pd.Series:
    out = pd.Series("test", index=ts.index)
    out[ts <= train_end] = "train"
    out[(ts > train_end) & (ts <= val_end)] = "validation"
    return out


def fit_eval(name: str, horizon: str, feats: list[str], train_df, val_df, test_df) -> dict:
    y_train = train_df[horizon].to_numpy(dtype=int)
    y_val = val_df[horizon].to_numpy(dtype=int)
    y_test = test_df[horizon].to_numpy(dtype=int)
    X_train = train_df[feats].to_numpy(dtype=np.float32)
    X_val = val_df[feats].to_numpy(dtype=np.float32)
    X_test = test_df[feats].to_numpy(dtype=np.float32)
    if not np.isfinite(X_train).all():
        raise SystemExit(f"{name}/{horizon}: non-finite train features")
    leaked = [c for c in feats if any(s in c.lower() for s in ("target", "label", "weather_code"))]
    if leaked:
        raise SystemExit(f"forbidden features: {leaked}")

    log(f"Fitting {name} {horizon} n={len(train_df)} n_feat={len(feats)}")
    clf = RandomForestClassifier(**MODEL_PARAMS)
    clf.fit(X_train, y_train)
    proba_val = clf.predict_proba(X_val)[:, 1]
    thr_info = select_threshold(threshold_sweep(y_val, proba_val))
    threshold = float(thr_info["selected_threshold"])
    m_val = metrics_at_threshold(y_val, proba_val, threshold)
    proba_test = clf.predict_proba(X_test)[:, 1]
    m_test = metrics_at_threshold(y_test, proba_test, threshold)
    loc = {}
    for st in STATIONS:
        mask = test_df[STATION].to_numpy() == st
        loc[st] = metrics_at_threshold(y_test[mask], proba_test[mask], threshold)
    payload = {
        "experiment": name,
        "horizon": horizon,
        "n_features": len(feats),
        "feature_list": feats,
        "threshold": threshold,
        "threshold_selection": thr_info,
        "n_train": int(len(train_df)),
        "n_validation": int(len(val_df)),
        "n_test": int(len(test_df)),
        "positives_train": int(y_train.sum()),
        "positives_validation": int(y_val.sum()),
        "positives_test": int(y_test.sum()),
        "metrics_validation": m_val,
        "metrics_test": m_test,
        "metrics_test_by_location": loc,
    }
    del clf, X_train, X_val, X_test
    return payload


def r(x):
    return "NA" if x is None else f"{x:.4f}"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    before = {str(p.relative_to(BASE_DIR)): sha256_file(p) for p in PROTECTED if p.exists()}

    with PHASE4_SPLIT.open(encoding="utf-8") as fh:
        p4 = json.load(fh)
    train_end = pd.Timestamp(p4["train"]["end_timestamp_utc"])
    val_end = pd.Timestamp(p4["validation"]["end_timestamp_utc"])

    lis = pd.read_csv(LIS_CSV, usecols=["station", "annual_thunder_hours"])
    if lis["station"].duplicated().any():
        raise SystemExit("duplicate lightning station")
    if set(lis["station"]) != set(STATIONS):
        raise SystemExit("lightning station set mismatch")
    if lis["annual_thunder_hours"].isna().any():
        raise SystemExit("missing annual_thunder_hours")

    usecols = [STATION, TIME] + ATMO_FEATURES + NWP_FEATURES + list(HORIZONS)
    log(f"Loading {DATA_CSV}")
    df = pd.read_csv(DATA_CSV, usecols=usecols)
    df[TIME] = pd.to_datetime(df[TIME], utc=True, format="ISO8601")
    if int(df.duplicated([STATION, TIME]).sum()):
        raise SystemExit("duplicate station/timestamp")
    if set(df[STATION].unique()) != set(STATIONS):
        raise SystemExit("station set mismatch")

    n_before_join = len(df)
    df = df.merge(lis, how="left", left_on=STATION, right_on="station")
    df = df.drop(columns=["station"])
    if len(df) != n_before_join:
        raise SystemExit("STOP: lightning join changed row count")
    if df["annual_thunder_hours"].isna().any():
        raise SystemExit("STOP: lightning join introduced NA")

    nunique = df.groupby(STATION)["annual_thunder_hours"].nunique()
    if (nunique != 1).any():
        raise SystemExit(f"STOP: annual_thunder_hours not constant per station: {nunique.to_dict()}")
    log(f"Lightning constant-per-station OK: {nunique.to_dict()}")

    complete = df[NWP_FEATURES].notna().all(axis=1)
    df_cc = df.loc[complete].copy()
    if int(df_cc[NWP_FEATURES].isna().sum().sum()):
        raise SystemExit("NWP NA remaining after complete-case filter")
    log(f"Complete-case rows: {len(df_cc)} / {len(df)}")

    all_results = {}
    sanity = {}
    for horizon in HORIZONS:
        observed = df_cc[df_cc[horizon].notna()].copy()
        observed[horizon] = observed[horizon].astype(int)
        if not set(observed[horizon].unique()).issubset({0, 1}):
            raise SystemExit(f"{horizon}: non-binary")
        observed["split"] = assign_split(observed[TIME], train_end, val_end)
        train_df = observed.loc[observed["split"] == "train"]
        val_df = observed.loc[observed["split"] == "validation"]
        test_df = observed.loc[observed["split"] == "test"]
        if train_df[TIME].max() >= val_df[TIME].min():
            raise SystemExit(f"{horizon}: train/val leak")
        if val_df[TIME].max() >= test_df[TIME].min():
            raise SystemExit(f"{horizon}: val/test leak")

        keys_train = list(zip(train_df[STATION].tolist(), train_df[TIME].astype(str).tolist()))
        keys_val = list(zip(val_df[STATION].tolist(), val_df[TIME].astype(str).tolist()))
        keys_test = list(zip(test_df[STATION].tolist(), test_df[TIME].astype(str).tolist()))

        b = fit_eval("B_atmospheric_nwp", horizon, MODEL_B_FEATURES, train_df, val_df, test_df)
        c = fit_eval("C_atmospheric_nwp_lis", horizon, MODEL_C_FEATURES, train_df, val_df, test_df)

        if b["n_train"] != c["n_train"] or b["n_validation"] != c["n_validation"] or b["n_test"] != c["n_test"]:
            raise SystemExit(f"STOP {horizon}: B/C row counts differ")
        if b["positives_train"] != c["positives_train"] or b["positives_test"] != c["positives_test"]:
            raise SystemExit(f"STOP {horizon}: B/C positives differ")
        if set(train_df[STATION].unique()) != set(STATIONS):
            raise SystemExit(f"STOP {horizon}: station coverage")

        sanity[horizon] = {
            "identical_train_keys": True,
            "identical_val_keys": True,
            "identical_test_keys": True,
            "n_train": len(keys_train),
            "n_val": len(keys_val),
            "n_test": len(keys_test),
            "identical_targets": True,
            "identical_splits": True,
            "nwp_missing_in_used_rows": 0,
            "lis_missing_in_used_rows": 0,
            "stations": STATIONS,
        }
        all_results[horizon] = {"B": b, "C": c}

    after = {str(p.relative_to(BASE_DIR)): sha256_file(p) for p in PROTECTED if p.exists()}
    if before != after:
        raise SystemExit("protected artifact hash changed")

    metric_keys = ["pr_auc", "roc_auc", "precision", "recall", "f1", "f2", "predicted_alert_rate"]
    rows = []
    for h in HORIZONS:
        mb = all_results[h]["B"]["metrics_test"]
        mc = all_results[h]["C"]["metrics_test"]
        row = {
            "lead": h,
            "n_train": all_results[h]["B"]["n_train"],
            "n_validation": all_results[h]["B"]["n_validation"],
            "n_test": all_results[h]["B"]["n_test"],
            "threshold_B": all_results[h]["B"]["threshold"],
            "threshold_C": all_results[h]["C"]["threshold"],
        }
        for k in metric_keys:
            row[f"B_{k}"] = mb[k]
            row[f"C_{k}"] = mc[k]
            row[f"delta_C_minus_B_{k}"] = (None if mb[k] is None or mc[k] is None else mc[k] - mb[k])
        rows.append(row)
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(OUT_DIR / "model_b_vs_c_metrics.csv", index=False)

    st_rows = []
    for h in HORIZONS:
        for st in STATIONS:
            m = all_results[h]["C"]["metrics_test_by_location"][st]
            st_rows.append(
                {
                    "lead": h,
                    "station": st,
                    "n_test": m["n_samples"],
                    "positives": m["positives"],
                    "threshold_C": all_results[h]["C"]["threshold"],
                    "pr_auc": m["pr_auc"],
                    "roc_auc": m["roc_auc"],
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "f1": m["f1"],
                    "f2": m["f2"],
                    "alert_rate": m["predicted_alert_rate"],
                }
            )
    pd.DataFrame(st_rows).to_csv(OUT_DIR / "model_c_station_metrics.csv", index=False)

    config = {
        "phase": "12B",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_csv": str(DATA_CSV.relative_to(BASE_DIR)).replace("\\", "/"),
        "lightning_csv": str(LIS_CSV.relative_to(BASE_DIR)).replace("\\", "/"),
        "complete_case_nwp": True,
        "imputation": False,
        "rf": MODEL_PARAMS,
        "seed": SEED,
        "alert_rate_cap": ALERT_RATE_CAP,
        "f_beta": F_BETA,
        "train_end_utc": p4["train"]["end_timestamp_utc"],
        "validation_end_utc": p4["validation"]["end_timestamp_utc"],
        "model_b_features": MODEL_B_FEATURES,
        "model_c_features": MODEL_C_FEATURES,
        "n_features_B": len(MODEL_B_FEATURES),
        "n_features_C": len(MODEL_C_FEATURES),
        "lightning_nunique_per_station": {k: 1 for k in STATIONS},
        "sanity": sanity,
        "protected_hashes_unchanged": True,
        "frozen_models_overwritten": False,
        "results": {
            h: {
                "B": {
                    "threshold": all_results[h]["B"]["threshold"],
                    "n_train": all_results[h]["B"]["n_train"],
                    "n_validation": all_results[h]["B"]["n_validation"],
                    "n_test": all_results[h]["B"]["n_test"],
                    "metrics_test": all_results[h]["B"]["metrics_test"],
                },
                "C": {
                    "threshold": all_results[h]["C"]["threshold"],
                    "n_train": all_results[h]["C"]["n_train"],
                    "n_validation": all_results[h]["C"]["n_validation"],
                    "n_test": all_results[h]["C"]["n_test"],
                    "metrics_test": all_results[h]["C"]["metrics_test"],
                },
            }
            for h in HORIZONS
        },
    }
    with (OUT_DIR / "experiment_config.json").open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)

    lines = [
        "# Phase 12B — Lightning climatology fusion experiment",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        "**Script:** `ml/train_v2_lightning_climatology_phase12b.py`",
        "**Scope:** experiment only. Not production. No Phase 3/6/8 mutation. No frozen model overwrite.",
        "",
        "## Objective",
        "",
        "Compare Model B (atmospheric + location/temporal + NWP) to Model C (B + `annual_thunder_hours`) on identical complete-case NWP-overlap rows, frozen Phase 8D RF/split/threshold protocol.",
        "",
        "## Data sources",
        "",
        "| source | path |",
        "| --- | --- |",
        "| NWP overlap | `dataset/multilocation/features_nwp/multilocation_features_nwp_overlap_2021_2025.csv` |",
        "| Lightning | `outputs/lightning/combined_lis_thunder_hour_locations.csv` (`station`, `annual_thunder_hours` only) |",
        "",
        "Complete-case: all 10 NWP fields present. Missing METAR targets excluded, not zero-filled.",
        "",
        "## Lightning join validation",
        "",
        "- Join: `station_id` = `station` (not timestamp).",
        "- Row count after left join: unchanged vs overlap table.",
        "- `annual_thunder_hours` missing after join: **0**.",
        "- Unique values per station: **1** (static climatology).",
        "- Not hourly lightning; not nowcast observations.",
        "",
        "## Identical-row verification",
        "",
        "For each lead, Model B and Model C used the same `station_id` + `timestamp_utc` rows, same targets, same train/validation/test membership.",
        "",
        "| lead | train n (pos) | val n (pos) | test n (pos) | B vs C rows identical |",
        "| --- | --- | --- | --- | --- |",
    ]
    for h in HORIZONS:
        b = all_results[h]["B"]
        lines.append(
            f"| {h} | {b['n_train']} ({b['positives_train']}) | "
            f"{b['n_validation']} ({b['positives_validation']}) | "
            f"{b['n_test']} ({b['positives_test']}) | True |"
        )
    lines += [
        "",
        "## Model B configuration",
        "",
        f"- Features: {len(MODEL_B_FEATURES)} (Phase 3 73 + 10 `nwp_*`).",
        "- RF: n_estimators=200, gini, max_features=sqrt, class_weight=balanced, random_state=42.",
        "- Threshold: max validation F2, alert rate ≤ 0.20; frozen for test.",
        "",
        "## Model C configuration",
        "",
        f"- Features: Model B + `annual_thunder_hours` only ({len(MODEL_C_FEATURES)} columns).",
        "- Same RF, split, complete-case, threshold protocol.",
        "",
        "## Split verification",
        "",
        f"- Train if timestamp ≤ `{p4['train']['end_timestamp_utc']}`.",
        f"- Validation if ≤ `{p4['validation']['end_timestamp_utc']}`.",
        "- Else test. No shuffle. Train max < val min; val max < test min.",
        "",
    ]
    for h in HORIZONS:
        lines += [
            f"## {h} results (test)",
            "",
            "| model | threshold | PR-AUC | ROC-AUC | P | R | F1 | F2 | alert rate |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for lab, key in (("B atmo+NWP", "B"), ("C + thunder hours", "C")):
            m = all_results[h][key]["metrics_test"]
            thr = all_results[h][key]["threshold"]
            lines.append(
                f"| {lab} | {thr:.6f} | {r(m['pr_auc'])} | {r(m['roc_auc'])} | {r(m['precision'])} | "
                f"{r(m['recall'])} | {r(m['f1'])} | {r(m['f2'])} | {r(m['predicted_alert_rate'])} |"
            )
        lines.append("")

    lines += [
        "## Station-level Model C results (test, same threshold as pooled C)",
        "",
        "| lead | station | n | pos | PR-AUC | ROC-AUC | P | R | F1 | F2 | alert rate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for h in HORIZONS:
        for st in STATIONS:
            m = all_results[h]["C"]["metrics_test_by_location"][st]
            lines.append(
                f"| {h} | {st} | {m['n_samples']} | {m['positives']} | {r(m['pr_auc'])} | {r(m['roc_auc'])} | "
                f"{r(m['precision'])} | {r(m['recall'])} | {r(m['f1'])} | {r(m['f2'])} | {r(m['predicted_alert_rate'])} |"
            )
    lines += [
        "",
        "Stations are not ranked; no “best location” is declared.",
        "",
        "## Metric deltas C − B (test)",
        "",
        "| lead | Δ PR-AUC | Δ ROC-AUC | Δ P | Δ R | Δ F1 | Δ F2 | Δ alert rate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for h in HORIZONS:
        mb = all_results[h]["B"]["metrics_test"]
        mc = all_results[h]["C"]["metrics_test"]
        def d(k):
            if mb[k] is None or mc[k] is None:
                return "NA"
            return f"{mc[k] - mb[k]:+.4f}"
        lines.append(
            f"| {h} | {d('pr_auc')} | {d('roc_auc')} | {d('precision')} | {d('recall')} | "
            f"{d('f1')} | {d('f2')} | {d('predicted_alert_rate')} |"
        )

    # Classification based on results — filled after we know numbers; written in this same run
    pr_d = [all_results[h]["C"]["metrics_test"]["pr_auc"] - all_results[h]["B"]["metrics_test"]["pr_auc"] for h in HORIZONS]
    roc_d = [all_results[h]["C"]["metrics_test"]["roc_auc"] - all_results[h]["B"]["metrics_test"]["roc_auc"] for h in HORIZONS]
    f2_d = [all_results[h]["C"]["metrics_test"]["f2"] - all_results[h]["B"]["metrics_test"]["f2"] for h in HORIZONS]
    mean_pr = float(np.mean(pr_d))
    mean_roc = float(np.mean(roc_d))
    mean_f2 = float(np.mean(f2_d))
    # Heuristic: retain if PR-AUC improves on majority of leads by > 0.002 without F2 collapse
    n_pr_up = sum(1 for x in pr_d if x > 0.002)
    n_pr_down = sum(1 for x in pr_d if x < -0.002)
    if n_pr_down >= 2 and mean_pr < 0:
        classification = "REMOVE_FROM_FUSION"
        class_why = (
            "Model C is worse than B on PR-AUC for most leads; the static climatology does not justify fusion cost."
        )
    elif n_pr_up >= 2 and mean_pr > 0.002:
        classification = "RETAIN_AS_STATIC_COVARIATE"
        class_why = (
            "Small but consistent ranking-metric gain vs B on the same rows. Keep only as a labelled static station covariate, not as lightning nowcast."
        )
    else:
        classification = "REFERENCE_ONLY"
        class_why = (
            "Deltas vs B are negligible or mixed relative to NWP already encoding location climate. "
            "Do not treat as operational lightning. Keep as a documented reference feature, not a default fusion layer."
        )

    lines += [
        "",
        "## Proxy / interpretation caveat",
        "",
        "`annual_thunder_hours` is **constant within each station**. Any Model C change vs B can only come from a **station-level climate/location offset**, not from time-varying lightning. Location is already in Model B (`latitude`, `longitude`, `elevation_m`). LIS thunder hours are **not** observed lightning at prediction time and must not be described as nowcast lightning.",
        "",
        "## Scientific limitations",
        "",
        "- Combined LIS climatology mixes TRMM (1998–2013) and ISS LIS (2017–2023); not 2014–2025 hourly flashes.",
        "- Complete-case NWP overlap only (from 2021-04-01); CIN/precip holes dropped as in Phase 8D.",
        "- RF can use a constant-per-station column as a surrogate station ID.",
        "- Thresholds chosen on validation only; test unused for tuning.",
        "",
        "## Decision for future multimodal fusion",
        "",
        f"**Classification: `{classification}`**",
        "",
        class_why,
        "",
        f"Mean test Δ PR-AUC (C−B) = {mean_pr:+.4f}; mean Δ ROC-AUC = {mean_roc:+.4f}; mean Δ F2 = {mean_f2:+.4f}.",
        "",
        "Satellite and radar remain future groups (Phases 10–11). This experiment does not add them.",
        "",
        "## Artifacts",
        "",
        "| path | role |",
        "| --- | --- |",
        "| `outputs/v2_multimodal/phase12b/model_b_vs_c_metrics.csv` | pooled B vs C + deltas |",
        "| `outputs/v2_multimodal/phase12b/model_c_station_metrics.csv` | Model C by station |",
        "| `outputs/v2_multimodal/phase12b/experiment_config.json` | config + sanity |",
        "| `docs/PHASE12B_LIGHTNING_CLIMATOLOGY_EXPERIMENT.md` | this report |",
        "",
        "## PHASE STATUS",
        "",
        f"**READY** — classification `{classification}`. Stop; do not proceed to Phase 12C in this task.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    with (OUT_DIR / "experiment_config.json").open("w", encoding="utf-8") as fh:
        config["classification"] = classification
        json.dump(config, fh, indent=2)
    log(f"Wrote {REPORT_PATH}")
    log(f"Classification: {classification}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
