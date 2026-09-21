"""Phase 8E: train-only median imputation robustness vs Phase 8D complete-case.

Does not retrain A or complete-case B. Does not modify Phase 8D artifacts.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "ml"))
from train_v2_nwp_experiment_phase8d import (  # noqa: E402
    ALERT_RATE_CAP,
    ATMO_FEATURES,
    HORIZONS,
    MODEL_PARAMS,
    NWP_FEATURES,
    NWP_FEATURES_SET,
    PHASE4_SPLIT,
    STATION,
    STATIONS,
    TIME,
    assign_split,
    log,
    metrics_at_threshold,
    select_threshold,
    sha256_file,
    threshold_sweep,
)

DATA_CSV = (
    BASE_DIR
    / "dataset"
    / "multilocation"
    / "features_nwp"
    / "multilocation_features_nwp_overlap_2021_2025.csv"
)
P8D_DIR = BASE_DIR / "outputs" / "v2_nwp_experiment"
MODELS_DIR = BASE_DIR / "models" / "v2" / "nwp_experiment" / "phase8e"
OUTPUTS_DIR = BASE_DIR / "outputs" / "v2_nwp_experiment" / "phase8e"
REPORT_PATH = BASE_DIR / "docs" / "PHASE8E_NWP_ROBUSTNESS_REPORT.md"

IMPUTE_COLS = ["nwp_convective_inhibition", "nwp_precipitation"]

PROTECTED = [
    P8D_DIR / "metrics_B_atmospheric_nwp_target_1h.json",
    P8D_DIR / "metrics_B_atmospheric_nwp_target_2h.json",
    P8D_DIR / "metrics_B_atmospheric_nwp_target_3h.json",
    P8D_DIR / "comparison.json",
    BASE_DIR / "ml" / "train_v2_nwp_experiment_phase8d.py",
    BASE_DIR / "models" / "v2" / "multilead" / "metadata_target_1h.json",
    BASE_DIR / "dataset" / "multilocation" / "features" / "multilocation_features_2014_2025.csv",
]


def r(x):
    return "NA" if x is None else f"{x:.4f}"


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    before = {str(p): sha256_file(p) for p in PROTECTED if p.exists()}

    with PHASE4_SPLIT.open(encoding="utf-8") as fh:
        p4 = json.load(fh)
    train_end = pd.Timestamp(p4["train"]["end_timestamp_utc"])
    val_end = pd.Timestamp(p4["validation"]["end_timestamp_utc"])

    usecols = [STATION, TIME] + ATMO_FEATURES + NWP_FEATURES + list(HORIZONS)
    log(f"Loading {DATA_CSV}")
    df = pd.read_csv(DATA_CSV, usecols=usecols)
    df[TIME] = pd.to_datetime(df[TIME], utc=True, format="ISO8601")
    if int(df.duplicated([STATION, TIME]).sum()):
        raise SystemExit("duplicate keys")

    p8d = {}
    for h in HORIZONS:
        with (P8D_DIR / f"metrics_B_atmospheric_nwp_{h}.json").open(encoding="utf-8") as fh:
            p8d[h] = json.load(fh)

    all_results = {}
    for horizon in HORIZONS:
        observed = df[df[horizon].notna()].copy()
        observed[horizon] = observed[horizon].astype(int)
        observed["split"] = assign_split(observed[TIME], train_end, val_end)
        train_df = observed.loc[observed["split"] == "train"].copy()
        val_df = observed.loc[observed["split"] == "validation"].copy()
        test_df = observed.loc[observed["split"] == "test"].copy()
        if train_df[TIME].max() >= val_df[TIME].min() or val_df[TIME].max() >= test_df[TIME].min():
            raise SystemExit(f"{horizon}: chronological leak")

        medians = {}
        imputed = {"train": {}, "validation": {}, "test": {}}
        for col in IMPUTE_COLS:
            med = float(train_df[col].median())
            if not np.isfinite(med):
                raise SystemExit(f"{horizon}: train median non-finite for {col}")
            medians[col] = med
            for name, part in (("train", train_df), ("validation", val_df), ("test", test_df)):
                n_miss = int(part[col].isna().sum())
                imputed[name][col] = n_miss
                part[col] = part[col].fillna(med)
        leftover = int(pd.concat([train_df, val_df, test_df])[NWP_FEATURES].isna().sum().sum())
        if leftover:
            raise SystemExit(f"{horizon}: NWP NA remain after impute: {leftover}")

        feats = NWP_FEATURES_SET
        y_train = train_df[horizon].to_numpy(dtype=int)
        y_val = val_df[horizon].to_numpy(dtype=int)
        y_test = test_df[horizon].to_numpy(dtype=int)
        X_train = train_df[feats].to_numpy(dtype=np.float32)
        X_val = val_df[feats].to_numpy(dtype=np.float32)
        X_test = test_df[feats].to_numpy(dtype=np.float32)
        if not np.isfinite(X_train).all():
            raise SystemExit(f"{horizon}: non-finite train")

        log(f"Fitting imputed NWP {horizon} n={len(train_df)}")
        clf = RandomForestClassifier(**MODEL_PARAMS)
        clf.fit(X_train, y_train)
        proba_val = clf.predict_proba(X_val)[:, 1]
        thr_info = select_threshold(threshold_sweep(y_val, proba_val), ALERT_RATE_CAP)
        threshold = float(thr_info["selected_threshold"])
        m_val = metrics_at_threshold(y_val, proba_val, threshold)
        m_test = metrics_at_threshold(y_test, clf.predict_proba(X_test)[:, 1], threshold)

        model_path = MODELS_DIR / f"B_nwp_imputed_{horizon}.joblib"
        joblib.dump(clf, model_path)
        payload = {
            "experiment": "B_atmospheric_nwp_train_median_impute",
            "horizon": horizon,
            "impute_columns": IMPUTE_COLS,
            "train_medians": medians,
            "imputed_value_counts": imputed,
            "n_train": int(len(train_df)),
            "n_validation": int(len(val_df)),
            "n_test": int(len(test_df)),
            "positives_train": int(y_train.sum()),
            "positives_validation": int(y_val.sum()),
            "positives_test": int(y_test.sum()),
            "threshold": threshold,
            "threshold_selection": thr_info,
            "metrics_validation": m_val,
            "metrics_test": m_test,
            "model_path": str(model_path.relative_to(BASE_DIR)).replace("\\", "/"),
            "phase8d_complete_case_test": p8d[horizon]["metrics_test"],
            "phase8d_n_train": p8d[horizon]["n_train"],
            "phase8d_n_validation": p8d[horizon]["n_validation"],
            "phase8d_n_test": p8d[horizon]["n_test"],
        }
        with (OUTPUTS_DIR / f"metrics_{horizon}.json").open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        all_results[horizon] = payload
        del clf, X_train, X_val, X_test

    after = {str(p): sha256_file(p) for p in PROTECTED if p.exists()}
    if before != after:
        raise SystemExit("protected artifact changed")

    lines = [
        "# Phase 8E — NWP missing-value robustness (train-only median)",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        "**Script:** `ml/train_v2_nwp_experiment_phase8e.py`",
        "**Scope:** robustness check only. Phase 8D complete-case remains the reference. Not a production model.",
        "",
        "Phase 3/4/5/6/8D artifacts and V1 were not modified. Complete-case and atmospheric-only models were not retrained.",
        "",
        "## Method",
        "",
        "- Same overlap CSV, stations, leads, Phase 6 UTC cutoffs, RF config, and F2/alert-rate validation threshold as Phase 8D.",
        "- Keep rows with missing `nwp_convective_inhibition` or `nwp_precipitation`.",
        "- Median of each of those two columns fit on **train only**, then applied to train/val/test.",
        "- No forward/back fill, no time interpolation, targets not filled.",
        "- Other eight NWP fields had zero missingness and were not imputed.",
        "",
        "## Imputation (train medians)",
        "",
        "| lead | nwp_convective_inhibition median | nwp_precipitation median | train imputed CIN | train imputed precip | val imputed CIN | val imputed precip | test imputed CIN | test imputed precip |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for h in HORIZONS:
        p = all_results[h]
        im = p["imputed_value_counts"]
        lines.append(
            f"| {h} | {p['train_medians']['nwp_convective_inhibition']:.6f} | "
            f"{p['train_medians']['nwp_precipitation']:.6f} | "
            f"{im['train']['nwp_convective_inhibition']} | {im['train']['nwp_precipitation']} | "
            f"{im['validation']['nwp_convective_inhibition']} | {im['validation']['nwp_precipitation']} | "
            f"{im['test']['nwp_convective_inhibition']} | {im['test']['nwp_precipitation']} |"
        )
    lines += [
        "",
        "## Row counts vs Phase 8D complete-case",
        "",
        "| lead | 8E train (pos) | 8D train | 8E val (pos) | 8D val | 8E test (pos) | 8D test |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for h in HORIZONS:
        p = all_results[h]
        lines.append(
            f"| {h} | {p['n_train']} ({p['positives_train']}) | {p['phase8d_n_train']} | "
            f"{p['n_validation']} ({p['positives_validation']}) | {p['phase8d_n_validation']} | "
            f"{p['n_test']} ({p['positives_test']}) | {p['phase8d_n_test']} |"
        )
    lines += [
        "",
        "## Compact test comparison vs Phase 8D NWP complete-case",
        "",
        "| Lead | Phase 8D Complete-case PR-AUC | Phase 8E Imputed PR-AUC | Phase 8D Recall | Phase 8E Recall | Phase 8D F1 | Phase 8E F1 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for h in HORIZONS:
        d = all_results[h]["phase8d_complete_case_test"]
        e = all_results[h]["metrics_test"]
        lines.append(
            f"| {h} | {r(d['pr_auc'])} | {r(e['pr_auc'])} | {r(d['recall'])} | {r(e['recall'])} | "
            f"{r(d['f1'])} | {r(e['f1'])} |"
        )
    lines += [
        "",
        "## Phase 8E test details",
        "",
        "| lead | threshold | PR-AUC | ROC-AUC | P | R | F1 | alert rate | TN | FP | FN | TP |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for h in HORIZONS:
        m = all_results[h]["metrics_test"]
        thr = all_results[h]["threshold"]
        lines.append(
            f"| {h} | {thr:.6f} | {r(m['pr_auc'])} | {r(m['roc_auc'])} | {r(m['precision'])} | "
            f"{r(m['recall'])} | {r(m['f1'])} | {r(m['predicted_alert_rate'])} | "
            f"{m['true_negatives']} | {m['false_positives']} | {m['false_negatives']} | {m['true_positives']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "Imputation is not claimed to improve the model. The question is whether Phase 8D's complete-case NWP ranking-skill pattern is reasonably stable when the ~3,460 NWP-null rows are retained via train-only medians. Row counts differ (especially validation, which contained both documented CIN and precip null windows). Differences may reflect extra rows, extra positives, and/or the filled values—not a production recommendation.",
        "",
        "## Integrity",
        "",
        "- Medians fit on train only.",
        "- Thresholds selected on validation only.",
        "- Phase 8D files hash-unchanged.",
        "- No FF/BF/time interpolation; targets not filled.",
        "",
        "## PHASE STATUS",
        "",
        "**READY FOR REVIEW** — stop. Do not proceed to Phase 9 in this task.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    log(f"Wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
