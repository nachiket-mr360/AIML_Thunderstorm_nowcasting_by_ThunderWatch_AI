# Phase 16B.4 — Command-center functional integration

Presentation-layer audit. Map providers unchanged. Frozen science untouched.

## Routes
- `GET /`
- `POST /replay`
- `POST /replay/all`
- `GET /api/map-config` (no crash without key)

## Frontend
- Timestamp from `#hist-time` posted to `/replay/all`
- Highest 1h from backend `focus_station_id`
- Marker click does not re-fetch
- Empty timestamp / HTTP errors shown as REPLAY ERROR without fake probabilities
- Atmospheric fields always listed; missing → `N/A`
