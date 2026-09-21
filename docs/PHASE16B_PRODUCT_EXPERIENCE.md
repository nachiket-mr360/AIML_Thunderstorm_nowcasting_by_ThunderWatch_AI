# Phase 16B / 16B.1 — ThunderWatch AI product experience

Opaque cinematic intro overlay (`z-index` 80). Command center is `hidden` until SKIP or timeout. Compatibility HTML is visually clipped (`.tech-compat`) so it never sits on the landing.

Presentation-only cinematic landing and command-center UI over frozen historical replay.

## Run locally

```
python app.py
```

Open `http://127.0.0.1:5000/`

Primary action: **RUN 5-LOCATION REPLAY** → `POST /replay/all`.

`POST /replay` remains for single-station compatibility HTML.

## Integrity

Decorative storm canvas is visual atmosphere only — not radar, satellite, or live weather.

Satellite / radar / lightning cards are BUILDING.

Frozen scientific files were not modified.
