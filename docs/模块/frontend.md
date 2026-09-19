# templates / static / frontend

Phase 9 decision-support dashboard for **one** VOTV point.

## 结构

```
templates/index.html
static/dashboard.js
static/style.css
frontend/verify_frontend_phase9.py
```

Serves from Flask `GET /`. Map is location context only (Esri streets). No radar/satellite/lightning overlay.

`frontend/` holds the Phase 9 verifier, not a separate SPA.
