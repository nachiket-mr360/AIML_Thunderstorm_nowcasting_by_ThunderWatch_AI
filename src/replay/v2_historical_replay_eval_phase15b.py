"""Phase 15B: multi-case historical replay evaluation of frozen Model B.

Does not train, retune thresholds, or modify Phase 13A/13B/14A.
Predictions are generated before any target-label lookup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.features.v2_feature_builder import NWP_SOURCE, TIME
from src.replay.v2_historical_replay import (
    BUILDER_COLUMNS as REPLAY_BUILDER_COLUMNS,
    REPLAY_MODE,
    TARGET_COLUMNS,
    V2HistoricalReplay,
)

BASE_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = BASE_DIR / "outputs" / "v2_replay" / "phase15b"
PHASE8D_DIR = BASE_DIR / "outputs" / "v2_nwp_experiment"

STATIONS = ("VOTV", "VECC", "VIDP", "VOCI", "VABB")
SAMPLING_INTERVAL_HOURS = 6
FROZEN_THRESHOLD = 0.065
PERIOD_START = pd.Timestamp("2021-04-01T00:00:00+00:00")
PERIOD_END = pd.Timestamp("2025-12-31T23:00:00+00:00")

PREDICTION_COLUMNS = [
    "station_id",
    "prediction_timestamp_utc",
    "replay_mode",
    "data_status",
    "unavailable_reason",
    "lead_1h_probability",
    "lead_1h_alert",
    "target_1h",
    "lead_2h_probability",
    "lead_2h_alert",
    "target_2h",
    "lead_3h_probability",
    "lead_3h_alert",
    "target_3h",
    "model_version",
    "threshold_1h",
    "threshold_2h",
    "threshold_3h",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def classify_unavailable(reason: str | None) -> str:
    r = (reason or "").lower()
    if "insufficient_history" in r or "timestamp_not_in_history" in r:
        return "missing_history"
    if "nwp" in r:
        return "missing_nwp"
    if any(k in r for k in ("invalid", "duplicate", "non_hourly", "unknown_station", "outside")):
        return "invalid_input"
    return "other"


def select_candidate_timestamps(
    times: pd.Series,
    *,
    interval_hours: int = SAMPLING_INTERVAL_HOURS,
) -> pd.DatetimeIndex:
    """Deterministic UTC hours with hour % interval == 0, inside the overlap period."""
    ts = pd.to_datetime(times, utc=True)
    ts = ts[(ts >= PERIOD_START) & (ts <= PERIOD_END)]
    ts = ts[ts.dt.hour % interval_hours == 0]
    return pd.DatetimeIndex(ts.drop_duplicates().sort_values())


def compute_lead_metrics(
    y: np.ndarray,
    proba: np.ndarray,
    *,
    threshold: float = FROZEN_THRESHOLD,
) -> dict[str, Any]:
    y = np.asarray(y, dtype=int)
    proba = np.asarray(proba, dtype=float)
    n = int(len(y))
    empty = {
        "n": n,
        "positive_count": 0,
        "negative_count": 0,
        "positive_rate": None,
        "PR-AUC": None,
        "ROC-AUC": None,
        "precision": None,
        "recall": None,
        "F1": None,
        "F2": None,
        "alert_rate": None,
        "TP": 0,
        "FP": 0,
        "FN": 0,
        "TN": 0,
        "threshold": float(threshold),
    }
    if n == 0:
        return empty
    pred = (proba >= threshold).astype(int)
    pos = int(np.sum(y == 1))
    neg = int(n - pos)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    two_class = len(np.unique(y)) > 1
    pr_auc = float(average_precision_score(y, proba)) if two_class else None
    roc = float(roc_auc_score(y, proba)) if two_class else None
    prec = float(precision_score(y, pred, zero_division=0))
    rec = float(recall_score(y, pred, zero_division=0))
    f1 = float(fbeta_score(y, pred, beta=1.0, zero_division=0))
    f2 = float(fbeta_score(y, pred, beta=2.0, zero_division=0))
    return {
        "n": n,
        "positive_count": pos,
        "negative_count": neg,
        "positive_rate": float(pos / n),
        "PR-AUC": pr_auc,
        "ROC-AUC": roc,
        "precision": prec,
        "recall": rec,
        "F1": f1,
        "F2": f2,
        "alert_rate": float(np.mean(pred)),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "threshold": float(threshold),
    }


def metrics_from_predictions(df: pd.DataFrame, station: str | None = None) -> dict[str, dict[str, Any]]:
    sub = df.loc[df["data_status"] == "COMPLETE"].copy()
    if station is not None:
        sub = sub.loc[sub["station_id"] == station]
    out: dict[str, dict[str, Any]] = {}
    for lead in (1, 2, 3):
        ycol = f"target_{lead}h"
        pcol = f"lead_{lead}h_probability"
        mask = sub[ycol].notna() & sub[pcol].notna()
        y = sub.loc[mask, ycol].to_numpy(dtype=float)
        p = sub.loc[mask, pcol].to_numpy(dtype=float)
        out[f"{lead}h"] = compute_lead_metrics(y.astype(int), p)
    return out


def _row_from_replay(pred: dict[str, Any]) -> dict[str, Any]:
    return {
        "station_id": pred.get("station_id"),
        "prediction_timestamp_utc": pred.get("prediction_timestamp_utc"),
        "replay_mode": pred.get("replay_mode", REPLAY_MODE),
        "data_status": pred.get("data_status"),
        "unavailable_reason": pred.get("unavailable_reason"),
        "lead_1h_probability": pred.get("lead_1h_probability"),
        "lead_1h_alert": pred.get("lead_1h_alert"),
        "target_1h": pred.get("historical_target_1h"),
        "lead_2h_probability": pred.get("lead_2h_probability"),
        "lead_2h_alert": pred.get("lead_2h_alert"),
        "target_2h": pred.get("historical_target_2h"),
        "lead_3h_probability": pred.get("lead_3h_probability"),
        "lead_3h_alert": pred.get("lead_3h_alert"),
        "target_3h": pred.get("historical_target_3h"),
        "model_version": pred.get("model_version"),
        "threshold_1h": pred.get("threshold_1h"),
        "threshold_2h": pred.get("threshold_2h"),
        "threshold_3h": pred.get("threshold_3h"),
    }


def _hourly_window_ok(times: np.ndarray, i: int) -> tuple[bool, str | None]:
    if i < 24:
        return False, "insufficient_history"
    w = np.asarray(times[i - 24 : i + 1], dtype="datetime64[h]")
    if not np.all(np.diff(w) == np.timedelta64(1, "h")):
        return False, "non_hourly_or_non_monotonic_timestamp"
    return True, None


def evaluate_station(
    replay: V2HistoricalReplay,
    station_df: pd.DataFrame,
    station_id: str,
    candidates: Sequence[pd.Timestamp],
    *,
    feature_builder_hook: Callable[[pd.DataFrame], None] | None = None,
    force_rebuild: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Replay each candidate. Inference runs before target lookup.

    When the overlap table already stores the Phase 13A 83-vector (validated
    against Phase 14A), use that vector after a fail-closed T-24..T window
    check. Otherwise rebuild with Phase 14A. Targets are never in the vector.
    """
    counts = Counter()
    rows: list[dict[str, Any]] = []
    names = list(replay.engine.feature_names)
    has_stored = (not force_rebuild) and all(c in station_df.columns for c in names)
    keep = [c for c in REPLAY_BUILDER_COLUMNS if c in station_df.columns]
    builder_df = station_df[keep].copy()
    leaked = [c for c in TARGET_COLUMNS if c in builder_df.columns]
    if leaked:
        builder_df = builder_df.drop(columns=leaked)
    times = pd.DatetimeIndex(builder_df[TIME])
    times_np = times.to_numpy()
    time_to_i = {pd.Timestamp(t): i for i, t in enumerate(times)}
    tgt = {}
    for lead in (1, 2, 3):
        col = f"target_{lead}h"
        tgt[lead] = station_df[col].to_numpy() if col in station_df.columns else None
    feat_mat = station_df[names].to_numpy(dtype=np.float64) if has_stored else None
    nwp_mat = None
    if has_stored and all(c in station_df.columns for c in NWP_SOURCE):
        nwp_mat = station_df[list(NWP_SOURCE)].to_numpy(dtype=np.float64)

    pending_X: list[np.ndarray] = []
    pending_meta: list[tuple[int, Any, str]] = []

    def _targets_at(i: int) -> dict[str, int | None]:
        replay.targets_looked_up = True
        hist = {
            "historical_target_1h": None,
            "historical_target_2h": None,
            "historical_target_3h": None,
        }
        for lead, key in ((1, "historical_target_1h"), (2, "historical_target_2h"), (3, "historical_target_3h")):
            arr = tgt[lead]
            if arr is None:
                continue
            val = arr[i]
            if val is None or (isinstance(val, float) and not np.isfinite(val)):
                continue
            try:
                hist[key] = int(val)
            except (TypeError, ValueError):
                pass
        return hist

    for t in candidates:
        counts["candidate_timestamps"] += 1
        i = time_to_i.get(pd.Timestamp(t))
        if i is None:
            pred = replay._unavailable(station_id, t, "timestamp_outside_nwp_overlap")
            rows.append(_row_from_replay(pred))
            counts["unavailable_timestamps"] += 1
            counts["unavailable_invalid_input"] += 1
            continue
        lo = max(0, i - 24)
        replay.last_builder_columns = list(builder_df.columns)
        if feature_builder_hook is not None:
            feature_builder_hook(builder_df.iloc[lo : i + 1])
        replay.prediction_before_target_lookup = False
        replay.targets_looked_up = False

        ok_win, win_reason = _hourly_window_ok(times_np, i)
        if has_stored:
            if not ok_win:
                pred = replay._unavailable(station_id, t, win_reason or "insufficient_history")
                rows.append(_row_from_replay(pred))
                counts["unavailable_timestamps"] += 1
                counts[f"unavailable_{classify_unavailable(win_reason)}"] += 1
                continue
            if nwp_mat is None or not np.isfinite(nwp_mat[i]).all():
                pred = replay._unavailable(station_id, t, "missing_required_nwp")
                rows.append(_row_from_replay(pred))
                counts["unavailable_timestamps"] += 1
                counts["unavailable_missing_nwp"] += 1
                continue
            vec = feat_mat[i]
            if not np.isfinite(vec).all():
                pred = replay._unavailable(station_id, t, "missing_or_nonfinite_features")
                rows.append(_row_from_replay(pred))
                counts["unavailable_timestamps"] += 1
                counts["unavailable_invalid_input"] += 1
                continue
            pending_X.append(np.asarray(vec, dtype=np.float32))
            pending_meta.append((i, t))
            continue

        window = builder_df.iloc[lo : i + 1]
        built = replay.builder.build_features(window, timestamp_utc=t, station_id=station_id)
        if not built.ok or built.features is None:
            pred = replay._unavailable(
                station_id, t, built.reason or "feature_build_unavailable", built.data_completeness
            )
            rows.append(_row_from_replay(pred))
            counts["unavailable_timestamps"] += 1
            counts[f"unavailable_{classify_unavailable(pred.get('unavailable_reason'))}"] += 1
            continue
        pred = replay.engine.predict(
            station_id, built.timestamp_utc, built.features, feature_names=built.feature_names
        )
        replay.prediction_before_target_lookup = True
        hist = _targets_at(i) if pred.get("data_status") == "COMPLETE" else {
            "historical_target_1h": None,
            "historical_target_2h": None,
            "historical_target_3h": None,
        }
        merged = {
            "replay_mode": REPLAY_MODE,
            **pred,
            **hist,
            "unavailable_reason": None if pred.get("data_status") == "COMPLETE" else "inference_unavailable",
        }
        if pred.get("data_status") == "COMPLETE":
            counts["successful_replay_timestamps"] += 1
        else:
            counts["unavailable_timestamps"] += 1
            counts["unavailable_other"] += 1
        rows.append(_row_from_replay(merged))

    if pending_X:
        models = replay.engine._load_models()
        replay.engine.models_called += 1
        Xm = np.vstack(pending_X)
        p1 = models["1h"].predict_proba(Xm)[:, 1]
        p2 = models["2h"].predict_proba(Xm)[:, 1]
        p3 = models["3h"].predict_proba(Xm)[:, 1]
        replay.prediction_before_target_lookup = True
        thr1, thr2, thr3 = replay.engine.threshold_1h, replay.engine.threshold_2h, replay.engine.threshold_3h
        for k, (i, t) in enumerate(pending_meta):
            pred = {
                "prediction_timestamp_utc": pd.Timestamp(t).isoformat(),
                "station_id": station_id,
                "lead_1h_probability": float(p1[k]),
                "lead_1h_alert": replay.engine.alert(float(p1[k]), thr1),
                "lead_2h_probability": float(p2[k]),
                "lead_2h_alert": replay.engine.alert(float(p2[k]), thr2),
                "lead_3h_probability": float(p3[k]),
                "lead_3h_alert": replay.engine.alert(float(p3[k]), thr3),
                "model_version": replay.engine.model_version,
                "threshold_1h": thr1,
                "threshold_2h": thr2,
                "threshold_3h": thr3,
                "data_completeness": 1.0,
                "data_status": "COMPLETE",
            }
            hist = _targets_at(i)
            merged = {"replay_mode": REPLAY_MODE, **pred, **hist, "unavailable_reason": None}
            counts["successful_replay_timestamps"] += 1
            rows.append(_row_from_replay(merged))
    return rows, dict(counts)


def _flatten_metrics(block: dict[str, dict[str, Any]], *, station: str | None) -> list[dict[str, Any]]:
    recs = []
    for lead, m in block.items():
        rec = {"station_id": station if station else "POOLED", "lead": lead, **m}
        recs.append(rec)
    return recs


def _phase8d_test_metrics() -> dict[str, dict[str, Any]]:
    out = {}
    for lead in (1, 2, 3):
        path = PHASE8D_DIR / f"metrics_B_atmospheric_nwp_target_{lead}h.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        m = payload["metrics_test"]
        p, r = m["precision"], m["recall"]
        f2 = None
        if (4 * p + r) > 0:
            f2 = float((5 * p * r) / (4 * p + r))
        out[f"{lead}h"] = {
            "phase": "8D",
            "lead": f"{lead}h",
            "n": m["n_samples"],
            "positive_count": m["positives"],
            "PR-AUC": m["pr_auc"],
            "ROC-AUC": m["roc_auc"],
            "precision": m["precision"],
            "recall": m["recall"],
            "F1": m["f1"],
            "F2": f2,
            "alert_rate": m["predicted_alert_rate"],
            "TP": m["true_positives"],
            "FP": m["false_positives"],
            "FN": m["false_negatives"],
            "TN": m["true_negatives"],
            "threshold": payload["threshold"],
            "sample": "Phase 8D Model B held-out test split (complete-case NWP features as stored)",
        }
    return out


def build_comparison(pooled: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    p8 = _phase8d_test_metrics()
    for lead in ("1h", "2h", "3h"):
        a = p8[lead]
        b = pooled[lead]
        rows.append(
            {
                "lead": lead,
                "phase8d_n": a["n"],
                "phase15b_n": b["n"],
                "phase8d_PR-AUC": a["PR-AUC"],
                "phase15b_PR-AUC": b["PR-AUC"],
                "phase8d_ROC-AUC": a["ROC-AUC"],
                "phase15b_ROC-AUC": b["ROC-AUC"],
                "phase8d_precision": a["precision"],
                "phase15b_precision": b["precision"],
                "phase8d_recall": a["recall"],
                "phase15b_recall": b["recall"],
                "phase8d_F1": a["F1"],
                "phase15b_F1": b["F1"],
                "phase8d_F2": a["F2"],
                "phase15b_F2": b["F2"],
                "phase8d_alert_rate": a["alert_rate"],
                "phase15b_alert_rate": b["alert_rate"],
                "phase8d_threshold": a["threshold"],
                "phase15b_threshold": b["threshold"],
                "methodological_note": (
                    "Phase 8D = Model B held-out time-split TEST only. "
                    "Phase 15B = 6-hourly historical replay over 2021-04-01..2025-12-31 "
                    "(includes hours that were train/validation in 8D). "
                    "Fail-closed T-24..T window; frozen 0.065 threshold. "
                    "Not interchangeable benchmarks; no improvement claim."
                ),
            }
        )
    return pd.DataFrame(rows)


def yearly_counts(df: pd.DataFrame) -> dict[str, int]:
    sub = df.loc[df["data_status"] == "COMPLETE"].copy()
    ts = pd.to_datetime(sub["prediction_timestamp_utc"], utc=True)
    years = ts.dt.year.value_counts().sort_index()
    return {str(int(k)): int(v) for k, v in years.items()}


def run_evaluation(
    *,
    interval_hours: int = SAMPLING_INTERVAL_HOURS,
    stations: Sequence[str] = STATIONS,
    write: bool = True,
    replay: V2HistoricalReplay | None = None,
    history: pd.DataFrame | None = None,
    out_dir: Path | None = None,
    feature_builder_hook: Callable[[pd.DataFrame], None] | None = None,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    out_dir = Path(out_dir) if out_dir is not None else OUT_DIR
    replay = replay if replay is not None else V2HistoricalReplay()
    source = history if history is not None else replay.load_dataset()

    all_rows: list[dict[str, Any]] = []
    quality = Counter()
    station_counts: dict[str, int] = {}
    first_ts = None
    last_ts = None

    for sid in stations:
        sdf = replay._station_frame(source, sid)
        sdf = replay._ensure_wind_direction(sdf)
        cands = list(select_candidate_timestamps(sdf[TIME], interval_hours=interval_hours))
        station_counts[sid] = len(cands)
        rows, counts = evaluate_station(
            replay,
            sdf,
            sid,
            cands,
            feature_builder_hook=feature_builder_hook,
            force_rebuild=force_rebuild,
        )
        all_rows.extend(rows)
        for k, v in counts.items():
            quality[k] += v
        if cands:
            first_ts = cands[0] if first_ts is None else min(first_ts, cands[0])
            last_ts = cands[-1] if last_ts is None else max(last_ts, cands[-1])

    pred_df = pd.DataFrame(all_rows)
    for c in PREDICTION_COLUMNS:
        if c not in pred_df.columns:
            pred_df[c] = None
    pred_df = pred_df[PREDICTION_COLUMNS]

    complete = pred_df.loc[pred_df["data_status"] == "COMPLETE"]
    target_missing = {
        "target_1h_missing": int(complete["target_1h"].isna().sum()) if len(complete) else 0,
        "target_2h_missing": int(complete["target_2h"].isna().sum()) if len(complete) else 0,
        "target_3h_missing": int(complete["target_3h"].isna().sum()) if len(complete) else 0,
    }
    pooled = metrics_from_predictions(pred_df)
    by_station_rows = []
    for sid in stations:
        block = metrics_from_predictions(pred_df, station=sid)
        by_station_rows.extend(_flatten_metrics(block, station=sid))
    pooled_rows = _flatten_metrics(pooled, station=None)
    eval_n = {lead: pooled[lead]["n"] for lead in ("1h", "2h", "3h")}

    quality_payload = {
        "phase": "15B",
        "sampling_interval_hours": interval_hours,
        "period_start_utc": PERIOD_START.isoformat(),
        "period_end_utc": PERIOD_END.isoformat(),
        "candidate_timestamps": int(quality.get("candidate_timestamps", 0)),
        "successful_replay_timestamps": int(quality.get("successful_replay_timestamps", 0)),
        "unavailable_timestamps": int(quality.get("unavailable_timestamps", 0)),
        "unavailable_due_to_missing_history": int(quality.get("unavailable_missing_history", 0)),
        "unavailable_due_to_missing_nwp": int(quality.get("unavailable_missing_nwp", 0)),
        "unavailable_due_to_invalid_input": int(quality.get("unavailable_invalid_input", 0)),
        "unavailable_other": int(quality.get("unavailable_other", 0)),
        **target_missing,
        "evaluation_sample_count_per_lead": eval_n,
        "station_candidate_counts": station_counts,
        "first_evaluated_timestamp_utc": None if first_ts is None else pd.Timestamp(first_ts).isoformat(),
        "last_evaluated_timestamp_utc": None if last_ts is None else pd.Timestamp(last_ts).isoformat(),
        "yearly_successful_replay_counts": yearly_counts(pred_df),
        "frozen_threshold": FROZEN_THRESHOLD,
        "replay_mode": REPLAY_MODE,
        "note": "Phase 15B evaluates the frozen V2 Model B through historical replay. It does not retrain, retune, or modify the model.",
        "feature_source": (
            "Phase 14A rebuild when stored 83-vectors are absent; otherwise the "
            "validated overlap 83-vector at T after a fail-closed T-24..T window check "
            "(Phase 14A previously matched stored columns)."
        ),
    }

    comparison = build_comparison(pooled) if PHASE8D_DIR.exists() else pd.DataFrame()

    # Reproducibility: recompute metrics twice from the same table
    m1 = metrics_from_predictions(pred_df)
    m2 = metrics_from_predictions(pred_df)
    repro = {
        "metrics_recomputed_identical": m1 == m2,
        "sampling_interval_hours": interval_hours,
        "deterministic_timestamp_rule": "UTC hour % interval_hours == 0; no random sampling",
        "n_prediction_rows": int(len(pred_df)),
        "n_complete": int((pred_df["data_status"] == "COMPLETE").sum()),
    }

    result = {
        "predictions": pred_df,
        "quality": quality_payload,
        "metrics_pooled": pd.DataFrame(pooled_rows),
        "metrics_by_station": pd.DataFrame(by_station_rows),
        "comparison": comparison,
        "reproducibility": repro,
        "pooled": pooled,
    }

    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(out_dir / "replay_predictions.csv", index=False)
        (out_dir / "replay_data_quality.json").write_text(
            json.dumps(quality_payload, indent=2), encoding="utf-8"
        )
        result["metrics_pooled"].to_csv(out_dir / "metrics_pooled.csv", index=False)
        result["metrics_by_station"].to_csv(out_dir / "metrics_by_station.csv", index=False)
        if not comparison.empty:
            comparison.to_csv(out_dir / "phase8d_vs_15b_comparison.csv", index=False)
        (out_dir / "reproducibility_check.json").write_text(
            json.dumps(repro, indent=2), encoding="utf-8"
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase 15B historical replay evaluation (frozen Model B).")
    p.add_argument("--interval-hours", type=int, default=SAMPLING_INTERVAL_HOURS)
    p.add_argument("--no-write", action="store_true")
    args = p.parse_args(list(argv) if argv is not None else None)
    run_evaluation(interval_hours=args.interval_hours, write=not args.no_write)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
