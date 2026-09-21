# Phase 16A — Historical Replay Demonstration Interface

## 1. Purpose

Phase 16A is a demonstration interface for historical replay. It does not
provide live, near-real-time, or operational forecasting.

Users select one of five supported stations and a historical UTC time `T`,
run the frozen Phase 15A replay engine, and inspect 1h / 2h / 3h risk
probabilities together with post-prediction historical labels.

## 2. Architecture

The Flask app (`app.py`) is a presentation/API layer only.

```
Browser form
    -> POST /replay
    -> run_historical_replay(station_id, timestamp_utc)
    -> src/replay/v2_historical_replay.py  (Phase 15A, frozen)
         -> V2FeatureBuilder (Phase 14A, frozen)
         -> V2InferenceEngine (Phase 13B, frozen)
    -> HTML result (templates/replay.html)
```

Feature engineering, model loading, thresholds, targets, and NWP join are
not duplicated in the web layer.

## 3. Input flow

1. `GET /` renders the replay form.
2. Station selector: `VOTV`, `VECC`, `VIDP`, `VOCI`, `VABB` (ICAO IDs internally).
3. Datetime-local field labelled **Prediction time T (UTC)**.
4. **Run Historical Replay** submits `POST /replay` with `station_id` and `timestamp_utc`.
5. The backend normalizes the datetime to an ISO UTC string and validates the station list before calling the replay engine. The engine still validates the timestamp against the NWP overlap dataset.

## 4. Replay engine integration

The web app calls `run_historical_replay(...)` from
`src/replay/v2_historical_replay.py`. It does not call Open-Meteo, IMD,
MOSDAC, NASA, IITM, or any other external weather API.

## 5. Prediction display

On `data_status = COMPLETE` the page shows:

- prediction timestamp, station, replay mode (`HISTORICAL_REPLAY`)
- model version, data status, data completeness
- three lead cards with **risk probability**, frozen threshold, and ALERT / NO ALERT

Probabilities are not labelled as confidence scores.

## 6. Historical verification

A separate **Historical Verification** section states that observations are
revealed only after the replay prediction and are not model inputs.

For each lead: model alert → historical target, plus MATCH / MISS from
Phase 15A `alert_hit` when present. The page does not compute aggregate metrics.

## 7. Failure handling

- Invalid station or empty timestamp: friendly form validation (HTTP 400).
- Engine `data_status = UNAVAILABLE`: **Replay unavailable** plus a safe reason
  (insufficient causal history, missing NWP, invalid timestamp, outside overlap).
  No fake probabilities.
- Unexpected exceptions: **Replay could not be completed.** Technical detail is logged server-side only.

## 8. Scientific limitations

This UI must not be described as real-time, live, current weather, an
operational or official forecast, or a guaranteed prediction. It does not
imply IMD, radar, satellite, or lightning integration. Those sources are
not part of the current prediction pipeline.

Phase 16A is a demonstration interface for historical replay. It does not
provide live, near-real-time, or operational forecasting.

## 9. How to run locally

From the repository root (with project dependencies installed):

```
python app.py
```

Open:

```
http://127.0.0.1:5000/
```

Example case: station `VOTV`, prediction time `2023-06-15T12:00` UTC.

## Phase 16A.2 command center

`GET /` is the ThunderWatch AI dashboard. Primary action is
`POST /replay/all` with `timestamp_utc`, which runs the frozen replay
for VOTV, VECC, VIDP, VOCI, and VABB, ranks by **1h model probability**,
and focuses the highest available station. `POST /replay` remains for
single-station compatibility.

Default historical time is `2023-06-15T12:00:00Z` (not the wall clock).
Satellite, radar, and ground lightning are shown as **Not connected**.
