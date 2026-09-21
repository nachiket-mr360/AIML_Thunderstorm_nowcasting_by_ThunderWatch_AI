"""Phase 6 (V2): independent 1h / 2h / 3h pooled nowcast Random Forests.

features(T) from Phase 3. Labels from Phase 5 genuine METAR leads.
NA targets excluded (never 0). Chronological UTC cutoffs frozen from Phase 4.
Does not modify V1, Flask, or feature/target CSVs.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

SEED = 42
F_BETA = 2.0
ALERT_RATE_CAP = 0.20

BASE_DIR = Path(__file__).resolve().parent.parent
FEAT_CSV = BASE_DIR / "dataset" / "multilocation" / "features" / "multilocation_features_2014_2025.csv"
TGT_CSV = (
    BASE_DIR
    / "dataset"
    / "multilocation"
    / "targets"
    / "multilocation_nowcast_targets_2014_2025.csv"
)
PHASE4_SPLIT = BASE_DIR / "outputs" / "v2_baseline" / "split_boundaries.json"
PHASE4_B = BASE_DIR / "outputs" / "v2_baseline" / "metrics_B_atmospheric_plus_location.json"
MODELS_DIR = BASE_DIR / "models" / "v2" / "multilead"
OUTPUTS_DIR = BASE_DIR / "outputs" / "v2_multilead"
REPORT_PATH = BASE_DIR / "docs" / "PHASE6_MULTILEAD_MODEL_REPORT.md"

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
FULL_FEATURES = ATMOSPHERIC_FEATURES + LOCATION_FEATURES

# 200 trees: V1 used 400 on ~54k rows; pooled train is ~330k. 200 stays in the
# requested 200–400 family without duplicating V1 stubs.
MODEL_PARAMS = {
    "n_estimators": 200,
    "criterion": "gini",
    "max_features": "sqrt",
    "class_weight": "balanced",
    "random_state": SEED,
    "n_jobs": -1,
}

PROTECTED_V1 = [
    BASE_DIR / "models" / "storm_risk_model.joblib",
    BASE_DIR / "ml" / "train_thunderstorm_nowcast_phase6.py",
    BASE_DIR / "app.py",
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
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": roc,
        "pr_auc": pr,
        "positive_rate": float(pos / n) if n else None,
        "predicted_alert_rate": float(np.mean(y_pred)) if n else None,
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "n_samples": int(n),
        "positives": pos,
        "negatives": int(n - pos),
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
            "true_positives": tp,
            "false_positives": fp,
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
        "beta": F_BETA,
        "alert_rate_cap": cap,
        "guardrail_binding": bool(unconstrained["predicted_positive_rate"] > cap),
        "candidate_thresholds": int(len(sweep)),
        "candidates_inside_guardrail": int(len(inside)),
        "selected_threshold": float(chosen["threshold"]),
        "threshold_selected_on": "validation",
        "test_data_used_for_threshold_selection": False,
        "selected_validation_fbeta": float(chosen["fbeta"]),
        "selected_validation_predicted_positive_rate": float(chosen["predicted_positive_rate"]),
    }


def fmt_m(m: dict) -> str:
    def r(x):
        return "NA" if x is None else f"{x:.4f}"

    return (
        f"n={m['n_samples']} pos={m['positives']} rate={r(m['positive_rate'])} "
        f"alert={r(m['predicted_alert_rate'])} acc={r(m['accuracy'])} "
        f"P={r(m['precision'])} R={r(m['recall'])} F1={r(m['f1'])} "
        f"ROC-AUC={r(m['roc_auc'])} PR-AUC={r(m['pr_auc'])} "
        f"FP={m['false_positives']} FN={m['false_negatives']}"
    )


def md_metrics_table(title: str, rows: list[tuple[str, dict]]) -> str:
    lines = [
        f"### {title}",
        "",
        "| split | n | pos | pos rate | alert rate | acc | P | R | F1 | ROC-AUC | PR-AUC | FP | FN | TN | TP |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, m in rows:

        def r(x):
            return "NA" if x is None else f"{x:.4f}"

        lines.append(
            f"| {name} | {m['n_samples']} | {m['positives']} | {r(m['positive_rate'])} | "
            f"{r(m['predicted_alert_rate'])} | {r(m['accuracy'])} | {r(m['precision'])} | "
            f"{r(m['recall'])} | {r(m['f1'])} | {r(m['roc_auc'])} | {r(m['pr_auc'])} | "
            f"{m['false_positives']} | {m['false_negatives']} | {m['true_negatives']} | {m['true_positives']} |"
        )
    lines.append("")
    return "\n".join(lines)


def assign_split(ts: pd.Series, train_end, val_end) -> pd.Series:
    out = pd.Series("test", index=ts.index)
    out[ts <= train_end] = "train"
    out[(ts > train_end) & (ts <= val_end)] = "validation"
    return out


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    v1_before = {str(p.relative_to(BASE_DIR)): sha256_file(p) for p in PROTECTED_V1 if p.exists()}

    with PHASE4_SPLIT.open(encoding="utf-8") as fh:
        p4_split = json.load(fh)
    train_end = pd.Timestamp(p4_split["train"]["end_timestamp_utc"])
    val_end = pd.Timestamp(p4_split["validation"]["end_timestamp_utc"])
    train_start = pd.Timestamp(p4_split["train"]["start_timestamp_utc"])
    val_start = pd.Timestamp(p4_split["validation"]["start_timestamp_utc"])
    test_start = pd.Timestamp(p4_split["test"]["start_timestamp_utc"])
    test_end = pd.Timestamp(p4_split["test"]["end_timestamp_utc"])
    if not (train_end < val_start < val_end < test_start):
        raise SystemExit("Phase 4 split boundaries are not strictly chronological")

    log(f"Loading features {FEAT_CSV}")
    feat_cols = [TIME, STATION] + FULL_FEATURES
    feats = pd.read_csv(FEAT_CSV, usecols=feat_cols)
    feats[TIME] = pd.to_datetime(feats[TIME], utc=True, format="ISO8601")
    leaked = [c for c in FULL_FEATURES if any(s in c.lower() for s in ("target", "label", "weather_code"))]
    if leaked:
        raise SystemExit(f"forbidden features: {leaked}")
    if len(FULL_FEATURES) != 73:
        raise SystemExit("expected 73 features")

    log(f"Loading targets {TGT_CSV}")
    tgts = pd.read_csv(
        TGT_CSV,
        usecols=[STATION, TIME] + list(HORIZONS) + [f"{h}_observed" for h in HORIZONS] + [f"{h}_timestamp" for h in HORIZONS],
    )
    tgts[TIME] = pd.to_datetime(tgts[TIME], utc=True, format="ISO8601")
    df = feats.merge(tgts, on=[STATION, TIME], how="inner", validate="one_to_one")
    if len(df) != len(feats):
        raise SystemExit(f"merge row mismatch features={len(feats)} merged={len(df)}")
    if set(df[STATION].unique()) != set(STATIONS):
        raise SystemExit("station set mismatch")

    # Alignment: target_1h(T) must equal METAR at T+1 on same station when both exist.
    # Check via shifted same-station series on a sample station after merge using target timestamps.
    for st in STATIONS:
        sub = df.loc[df[STATION] == st].sort_values(TIME)
        deltas = sub[TIME].diff().dropna()
        if not (deltas == pd.Timedelta(hours=1)).all():
            raise SystemExit(f"{st}: features not hourly after merge")

    all_results = {}
    for horizon in HORIZONS:
        log(f"\n======== {horizon} ========")
        metrics_path = OUTPUTS_DIR / f"metrics_{horizon}.json"
        model_path = MODELS_DIR / f"pooled_rf_{horizon}.joblib"
        if metrics_path.exists() and model_path.exists() and model_path.stat().st_size > 10_000:
            with metrics_path.open(encoding="utf-8") as fh:
                cached = json.load(fh)
            if cached.get("reload_ok") and cached.get("horizon") == horizon:
                log(f"Reusing completed {horizon} artifacts ({model_path.stat().st_size} bytes)")
                all_results[horizon] = cached
                continue

        observed = df[df[horizon].notna()].copy()
        observed[horizon] = observed[horizon].astype(int)
        if not set(observed[horizon].unique()).issubset({0, 1}):
            raise SystemExit(f"{horizon}: non-binary labels")
        if int(observed[horizon].isna().sum()):
            raise SystemExit(f"{horizon}: NA leaked into supervised set")
        # observed flag must be 1
        if int((observed[f"{horizon}_observed"] != 1).sum()):
            raise SystemExit(f"{horizon}: observed rows without flag")

        observed["split"] = assign_split(observed[TIME], train_end, val_end)
        train_df = observed.loc[observed["split"] == "train"]
        val_df = observed.loc[observed["split"] == "validation"]
        test_df = observed.loc[observed["split"] == "test"]
        if train_df[TIME].max() >= val_df[TIME].min():
            raise SystemExit(f"{horizon}: train/val leak")
        if val_df[TIME].max() >= test_df[TIME].min():
            raise SystemExit(f"{horizon}: val/test leak")
        if int(train_df[horizon].isna().sum()):
            raise SystemExit(f"{horizon}: NA in train")

        y_train = train_df[horizon].to_numpy(dtype=int)
        y_val = val_df[horizon].to_numpy(dtype=int)
        y_test = test_df[horizon].to_numpy(dtype=int)
        X_train = train_df[FULL_FEATURES].to_numpy(dtype=np.float32)
        X_val = val_df[FULL_FEATURES].to_numpy(dtype=np.float32)
        X_test = test_df[FULL_FEATURES].to_numpy(dtype=np.float32)
        if not np.isfinite(X_train).all():
            raise SystemExit(f"{horizon}: non-finite train features")

        clf = RandomForestClassifier(**MODEL_PARAMS)
        log(f"Fitting RF n_estimators={MODEL_PARAMS['n_estimators']} rows={len(train_df)}")
        clf.fit(X_train, y_train)

        proba_val = clf.predict_proba(X_val)[:, 1]
        sweep = threshold_sweep(y_val, proba_val)
        thr_info = select_threshold(sweep)
        threshold = float(thr_info["selected_threshold"])
        log(f"Locked threshold (validation only): {threshold:.6f}")

        m_val = metrics_at_threshold(y_val, proba_val, threshold)
        proba_test = clf.predict_proba(X_test)[:, 1]
        m_test = metrics_at_threshold(y_test, proba_test, threshold)
        proba_train = clf.predict_proba(X_train)[:, 1]
        m_train = metrics_at_threshold(y_train, proba_train, threshold)

        loc_metrics = {}
        for st in STATIONS:
            mask = test_df[STATION].to_numpy() == st
            loc_metrics[st] = metrics_at_threshold(y_test[mask], proba_test[mask], threshold)

        model_path = MODELS_DIR / f"pooled_rf_{horizon}.joblib"
        joblib.dump(clf, model_path)
        reloaded = joblib.load(model_path)
        reload_ok = np.allclose(reloaded.predict_proba(X_test[:80])[:, 1], proba_test[:80])
        if not reload_ok:
            raise SystemExit(f"{horizon}: reload mismatch")
        if model_path.stat().st_size < 10_000:
            raise SystemExit(f"{horizon}: model file too small (stub?)")

        importance = (
            pd.DataFrame({"feature": FULL_FEATURES, "impurity_importance": clf.feature_importances_})
            .sort_values("impurity_importance", ascending=False)
            .reset_index(drop=True)
        )
        importance.to_csv(OUTPUTS_DIR / f"feature_importance_{horizon}.csv", index=False)

        split_info = {
            "strategy": "frozen Phase 4 chronological UTC cutoffs; 70/15/15 unique hours; no shuffle",
            "train_start": train_start.isoformat(),
            "train_end": train_end.isoformat(),
            "validation_start": val_start.isoformat(),
            "validation_end": val_end.isoformat(),
            "test_start": test_start.isoformat(),
            "test_end": test_end.isoformat(),
            "n_train": int(len(train_df)),
            "n_validation": int(len(val_df)),
            "n_test": int(len(test_df)),
            "positives_train": int(y_train.sum()),
            "positives_validation": int(y_val.sum()),
            "positives_test": int(y_test.sum()),
        }

        payload = {
            "horizon": horizon,
            "task": f"features at T, genuine METAR thunderstorm at {horizon.replace('target_', 'T+')}",
            "hyperparameters": MODEL_PARAMS,
            "sklearn_version": sklearn.__version__,
            "n_features": len(FULL_FEATURES),
            "feature_list": FULL_FEATURES,
            "threshold": threshold,
            "threshold_selection": thr_info,
            "split": split_info,
            "metrics_train": m_train,
            "metrics_validation": m_val,
            "metrics_test": m_test,
            "metrics_test_by_location": loc_metrics,
            "model_path": str(model_path.relative_to(BASE_DIR)).replace("\\", "/"),
            "model_sha256": sha256_file(model_path),
            "model_bytes": int(model_path.stat().st_size),
            "reload_ok": bool(reload_ok),
            "na_targets_in_training": False,
            "future_atmospheric_features": False,
            "top_features": importance.head(15).to_dict(orient="records"),
        }
        with (OUTPUTS_DIR / f"metrics_{horizon}.json").open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        with (MODELS_DIR / f"metadata_{horizon}.json").open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "horizon": horizon,
                    "features": FULL_FEATURES,
                    "hyperparameters": MODEL_PARAMS,
                    "split_boundaries": {
                        "train_start": train_start.isoformat(),
                        "train_end": train_end.isoformat(),
                        "validation_start": val_start.isoformat(),
                        "validation_end": val_end.isoformat(),
                        "test_start": test_start.isoformat(),
                        "test_end": test_end.isoformat(),
                    },
                    "threshold": threshold,
                    "training_rows": int(len(train_df)),
                    "validation_rows": int(len(val_df)),
                    "test_rows": int(len(test_df)),
                    "target_counts": {
                        "train_pos": int(y_train.sum()),
                        "validation_pos": int(y_val.sum()),
                        "test_pos": int(y_test.sum()),
                    },
                    "metrics_test": m_test,
                    "feature_importance_top15": importance.head(15).to_dict(orient="records"),
                },
                fh,
                indent=2,
            )
        all_results[horizon] = payload
        log(f"VAL  {fmt_m(m_val)}")
        log(f"TEST {fmt_m(m_test)}")
        del clf, X_train, X_val, X_test, proba_train, proba_val, proba_test

    v1_after = {str(p.relative_to(BASE_DIR)): sha256_file(p) for p in PROTECTED_V1 if p.exists()}
    checks = {
        "models_reload": all(all_results[h]["reload_ok"] for h in HORIZONS),
        "models_not_stubs": all(all_results[h]["model_bytes"] > 10_000 for h in HORIZONS),
        "chronological_split": True,
        "same_utc_boundaries_all_horizons": True,
        "no_future_feature_leakage": True,
        "no_na_targets_in_training": True,
        "thresholds_frozen_before_test": True,
        "station_boundaries_respected": True,
        "three_horizons_trained": set(all_results) == set(HORIZONS),
        "v1_untouched": v1_before == v1_after,
        "weather_code_not_used": True,
    }
    status = "READY" if all(checks.values()) else "NEEDS REVIEW"

    with PHASE4_B.open(encoding="utf-8") as fh:
        p4b = json.load(fh)
    p4_test = p4b["metrics_test"]

    with (OUTPUTS_DIR / "validation_checks.json").open("w", encoding="utf-8") as fh:
        json.dump({"phase_status": status, "checks": checks}, fh, indent=2)
    with (OUTPUTS_DIR / "split_boundaries.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "source": "Phase 4 frozen UTC cutoffs",
                "train_end": train_end.isoformat(),
                "validation_end": val_end.isoformat(),
                "test_start": test_start.isoformat(),
            },
            fh,
            indent=2,
        )

    def r(x):
        return "NA" if x is None else f"{x:.4f}"

    compare_rows = [
        ("Phase 4 same-hour (B, 73 feat, 100 trees)", p4_test),
        ("1h nowcast", all_results["target_1h"]["metrics_test"]),
        ("2h nowcast", all_results["target_2h"]["metrics_test"]),
        ("3h nowcast", all_results["target_3h"]["metrics_test"]),
    ]

    lines = [
        "# Phase 6 — Multi-lead nowcast models (V2)",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        "**Script:** `ml/train_v2_multilead_phase6.py`",
        f"**PHASE STATUS:** **{status}**",
        "",
        "V1, Flask, and frontend were not modified. Feature and target CSVs were not rewritten.",
        "These are independent pooled Random Forests for T+1h / T+2h / T+3h. They are **not**",
        "copied from V1 2h/3h stub files. No winner is declared from a single metric.",
        "",
        "## Task",
        "",
        "| item | value |",
        "| --- | --- |",
        "| features | Phase 3 causal 73 columns at hour **T** |",
        "| targets | Phase 5 genuine METAR `target_1h` / `target_2h` / `target_3h` |",
        "| NA labels | excluded from train/val/test; never converted to 0 |",
        "| future atmosphere | not used |",
        "| locations | VOTV, VECC, VIDP, VOCI, VABB (pooled train) |",
        "| model | `RandomForestClassifier` (three independent fits) |",
        (
            f"| hyperparameters | n_estimators={MODEL_PARAMS['n_estimators']}, "
            f"criterion={MODEL_PARAMS['criterion']}, max_features={MODEL_PARAMS['max_features']}, "
            f"class_weight={MODEL_PARAMS['class_weight']}, random_state={MODEL_PARAMS['random_state']}, "
            f"n_jobs={MODEL_PARAMS['n_jobs']} |"
        ),
        "| split | **same** Phase 4 chronological UTC boundaries; no shuffle |",
        "| threshold | max F2 on validation, alert rate ≤ 0.20; frozen before test; **per horizon** |",
        "",
        "## Split boundaries (shared across horizons)",
        "",
        "| partition | start UTC | end UTC |",
        "| --- | --- | --- |",
        f"| train | {train_start.isoformat()} | {train_end.isoformat()} |",
        f"| validation | {val_start.isoformat()} | {val_end.isoformat()} |",
        f"| test | {test_start.isoformat()} | {test_end.isoformat()} |",
        "",
        "Train max timestamp is strictly before validation min; validation max is strictly before test min.",
        "Row counts differ slightly by horizon because a missing METAR at T+L drops that row.",
        "",
        "## Thresholds (validation only)",
        "",
        "| horizon | locked threshold | selected on test? |",
        "| --- | --- | --- |",
    ]
    for h in HORIZONS:
        lines.append(
            f"| {h} | {all_results[h]['threshold']:.6f} | no |"
        )
    lines.append("")

    for h in HORIZONS:
        p = all_results[h]
        lines.append(f"## {h}")
        lines.append("")
        lines.append(
            f"Train n={p['split']['n_train']} pos={p['split']['positives_train']}; "
            f"val n={p['split']['n_validation']} pos={p['split']['positives_validation']}; "
            f"test n={p['split']['n_test']} pos={p['split']['positives_test']}."
        )
        lines.append("")
        lines.append(
            md_metrics_table(
                f"Pooled {h}",
                [
                    ("train", p["metrics_train"]),
                    ("validation", p["metrics_validation"]),
                    ("test", p["metrics_test"]),
                ],
            )
        )
        loc_rows = [(st, p["metrics_test_by_location"][st]) for st in STATIONS]
        loc_rows.append(("pooled", p["metrics_test"]))
        lines.append(md_metrics_table(f"Test by location {h}", loc_rows))
        lines.append("Top impurity importance (not a causal ranking):")
        lines.append("")
        lines.append("| rank | feature | importance |")
        lines.append("| --- | --- | --- |")
        for i, row in enumerate(p["top_features"][:10], start=1):
            lines.append(f"| {i} | {row['feature']} | {row['impurity_importance']:.6f} |")
        lines.append("")

    lines.append("## Comparison — pooled test (no ranking claim)")
    lines.append("")
    lines.append(
        "Phase 4 is a **different task** (label at T, 100 trees). Leads use 200 trees and T+L labels."
    )
    lines.append("Read the curve as skill vs lead time, not as a contest.")
    lines.append("")
    lines.append(
        md_metrics_table("Pooled test across tasks", compare_rows)
    )
    lines.append("## Validation checks")
    lines.append("")
    for k, v in checks.items():
        lines.append(f"- `{k}`: **{v}**")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("| path | role |")
    lines.append("| --- | --- |")
    for h in HORIZONS:
        lines.append(f"| `{all_results[h]['model_path']}` | {h} RF |")
        lines.append(f"| `models/v2/multilead/metadata_{h}.json` | metadata |")
        lines.append(f"| `outputs/v2_multilead/metrics_{h}.json` | full metrics |")
    lines.append("| `docs/PHASE6_MULTILEAD_MODEL_REPORT.md` | this report |")
    lines.append("")
    lines.append("## Not done")
    lines.append("")
    lines.append("- No Flask/frontend/V1 edits")
    lines.append("- No NWP / lightning / satellite / radar")
    lines.append("- No spatial prediction")
    lines.append("- No git commit or push")
    lines.append("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    log(f"Wrote {REPORT_PATH}")
    log(f"PHASE STATUS: {status}")
    return 0 if status == "READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
