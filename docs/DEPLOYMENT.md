# ThunderWatch AI V2 — deployment (historical replay)

Production: Gunicorn on `0.0.0.0:$PORT` (Dockerfile). Local: `python app.py` (127.0.0.1:5000).

`GET /health` → `{"status":"ok"}` (no model load, no weather APIs).

`MAPTILER_API_KEY` is optional. The app starts without it.

## Required runtime files (must be in the Docker build context)

| Path | Approx. size | Role |
|------|----------------|------|
| `models/v2/nwp_experiment/B_atmospheric_nwp_target_{1,2,3}h.joblib` | ~47 MB each | Frozen Model B |
| `dataset/multilocation/features_nwp/multilocation_features_nwp_overlap_2021_2025.csv` | ~183 MB | Historical replay table |
| `outputs/v2_inference/phase13a/inference_contract.json` | small | Feature order + threshold 0.065 |
| `dataset/multilocation/feature_engineering_phase3.py` | small | Feature formulas |

These V2 joblibs, the overlap CSV, and the Phase 13A contract are **untracked in git** as of this audit. A Render build from GitHub will fail unless they are committed (or otherwise present in the build context). Local `docker build` from this working tree includes them via `COPY . .`.

Recommend **≥2 GB RAM** on Render: first `/replay/all` loads the 183 MB CSV plus three Random Forest joblibs.

scikit-learn **1.6.1** (do not upgrade).
