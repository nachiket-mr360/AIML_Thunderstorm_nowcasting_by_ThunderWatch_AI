# V2 Pre-Commit Safety Audit

**Date:** 2026-09-19  
**Scope:** Read-only scan of `C:\College\SIH2_v2`  
**Action taken:** This report file only. No deletes, no code/data/model changes, no commit, no push, no remote change.

**Verdict: REVIEW**

No credential-like secrets were found. The first commit is **not blocked** on security. It **should not proceed as `git add .`** until junk, cache, logs, agent files, and Git LFS / large-artifact policy are decided.

---

## 1. Repository identity

| Item | Value |
|------|--------|
| Working directory | `C:\College\SIH2_v2` |
| Intended V2 remote | `https://github.com/nachiket-mr360/AIML_Thunderstorm_nowcasting_v2.git` |
| Git remotes observed | `origin` fetch/push → that URL |
| Branch | `main` |
| Commits | **None yet** (`No commits yet on main`) |
| Tracked vs untracked | Entire tree is untracked (`??`) |
| `.gitignore` | **Absent** |
| `.gitattributes` | Present; `models/*.joblib filter=lfs` |
| V1 (must not modify) | `https://github.com/nachiket-mr360/AI-Based-Thunderstorm-Nowcasting-Atmospheric-Risk-Monitoring` — not this folder |
| Approx. working tree size | **~161 files, ~530 MB** (excluding `.git`) |

This matches the stated V2 setup: new Git repo, V1 history not carried in.

---

## 2. Secret scan result

**Result: no secrets found. Not BLOCK.**

Searched for (non-exhaustive but typical):

- `api_key` / `secret_key` / `access_token` / `password=` / `private_key`
- PEM / OpenSSH private key headers
- AWS `AKIA…`, GitHub `ghp_` / `github_pat`, Slack `xox*`, OpenAI-style `sk-`
- `.env*`, `*.pem`, `*.key`, `*.p12`, `*.pfx`
- Hard-coded `Bearer` / `Authorization` headers

**Hits that are not secrets:**

- `app.py` CORS comment mentioning `credentials` (browser CORS, not a password).
- Public weather URLs: `https://api.open-meteo.com/v1/forecast`, `https://archive-api.open-meteo.com/v1/archive` (key-free).
- Public demo: `https://ai-based-thunderstorm-nowcasting-by.onrender.com/` in `README.md`.
- CDN: jsDelivr Leaflet / Chart.js in `templates/index.html`.
- Env vars used: `HOST`, `PORT`, `FLASK_DEBUG`, `CORS_ALLOWED_ORIGINS`, `SIH2_WEATHER_CSV` — names only, no secret values in repo.

No `.env` files. No credential files. Open-Meteo and IEM usage in this prototype does not embed API keys.

---

## 3. Suspicious / should-not-commit files

| Path | Why exclude |
|------|-------------|
| `flask_err.log` | Runtime log |
| `flask_run.log` | Runtime log |
| `outputs/_phase9_server.log` | Verifier/server log |
| `dataset/.cache.sqlite` | ~14 MB cache DB |
| `__pycache__/` (root, `ml/`, `dataset/`, …) | Bytecode |
| `dataset/_tmp_*.py`, `dataset/_tmp_ver.py` | Scratch probes |
| `dataset/_probe_openmeteo_net_phase3.py` | Network probe scratch |
| `ERROR` | Empty junk |
| `test.txt` | Empty junk |
| `.ohmyagent/` | Local agent config, tile dumps (`*.bin`, `*.png`, `diag_tiles*.ps1`, `settings.json`) |
| `.monkeycode/` if present | Generated wiki/docs from agent (optional; not required for baseline) |

**Incomplete artifacts (commit policy, not secrets):**

- `models/thunderstorm_nowcast_2h.joblib` — **134 bytes**
- `models/thunderstorm_nowcast_3h.joblib` — **134 bytes**

These are not usable estimators. Metadata/eval JSON for 2h/3h still exist. Committing stub joblibs will confuse Git LFS and future clones.

---

## 4. Large files

Approximate sizes (working tree):

| Path | Size |
|------|------|
| `models/thunderstorm_nowcast_1h.joblib` | **137.7 MB** |
| `models/storm_risk_model.joblib` | **123.7 MB** |
| `dataset/raw_metar/votv_metar_iem_raw.csv` | 52.7 MB |
| `dataset/votv_thunderstorm_features_2014_2025.csv` | 35.5 MB |
| `dataset/votv_thunderstorm_nowcast_2014_2025.csv` | 35.4 MB |
| `dataset/weather_data_with_code.csv` | 18.8 MB |
| `dataset/weather_data.csv` | 17.1 MB |
| `dataset/historical_thunderstorm_labels_votv.csv` | 14.6 MB |
| `dataset/.cache.sqlite` | 14.0 MB — **do not commit** |
| `dataset/votv_thunderstorm_synchronized_2014_2025.csv` | 9.1 MB |
| `dataset/raw_openmeteo/votv_openmeteo_hourly_2014_2025.csv` | 8.6 MB |
| Yearly METAR CSVs 2000–2025 | ~0.4–3.5 MB each (sum tens of MB) |
| Yearly Open-Meteo CSVs 2014–2025 | ~0.7 MB each |

JSON metadata under `dataset/` and `outputs/` is small (typically &lt; 25 KB). Evaluation PNGs are ~40 KB.

**GitHub / LFS:** `.gitattributes` already marks `models/*.joblib` for LFS. Raw CSVs are **not** LFS-tracked. A naive first commit of ~500 MB of CSV + two large joblibs needs Git LFS installed and GitHub LFS quota; otherwise push will fail or bloat the repo.

No `.pkl` files found.

---

## 5. `.gitignore` assessment

**There is no `.gitignore`.**

Therefore:

- `git add .` would stage logs, `__pycache__`, SQLite cache, `.ohmyagent/`, empty junk, and all large CSVs.
- Secrets are not currently present, but there is **no future protection** for `.env`, keys, or venvs.

**Recommended ignore (do not apply in this audit; listed only):**

```
.venv/
__pycache__/
*.pyc
.env
.env.*
*.log
*.sqlite
.ohmyagent/
.monkeycode/
ERROR
test.txt
dataset/_tmp_*.py
dataset/.cache.sqlite
```

Plus an explicit decision on:

- whether `dataset/raw_*` and engineered CSVs live in git or object storage / LFS
- whether `storm_risk_model.joblib` is required for V2 baseline
- excluding stub 2h/3h joblibs until real files exist

---

## 6. Files that appear safe to commit (after ignore + LFS policy)

Typical V2 baseline source (no secrets observed):

- `README.md`
- `app.py`
- `Dockerfile`, `.dockerignore`, `.gitattributes`
- `requirements.txt`, `requirements_backend.txt`, `requirements_ml.txt`
- `backend/*.py` (except generated validation JSON if you prefer regenerating)
- `ml/*.py`
- `dataset/*.py` **excluding** `_tmp_*` / `_probe_*`
- `dataset/*.md` and small `*_metadata.json` / `*_stats.json`
- `frontend/verify_frontend_phase9.py`
- `templates/`, `static/`
- `outputs/*.md`, evaluation JSON/CSV/PNG, `outputs/verify_phase6_models.py`
- `models/*_metadata.json`
- `models/thunderstorm_nowcast_1h.joblib` **if** Git LFS is used as `.gitattributes` intends

**Safe only with an explicit size/LFS decision:** engineered and raw CSVs, `storm_risk_model.joblib`.

---

## 7. Status for the first commit

| Check | Status |
|-------|--------|
| Secrets / keys / tokens | **PASS** — none found |
| Credential files | **PASS** — none found |
| `.gitignore` | **REVIEW** — missing; `git add .` is unsafe |
| Logs / cache / scratch / agent dirs | **REVIEW** — present and currently untracked |
| Large joblib + CSV | **REVIEW** — ~260 MB models + hundreds of MB data; LFS only on `models/*.joblib` |
| Stub 2h/3h joblib | **REVIEW** — 134-byte files |
| Remote points at V2 not V1 | **PASS** — `AIML_Thunderstorm_nowcasting_v2` |

### Overall: **REVIEW**

- **Not PASS:** first commit should not be “add everything.”
- **Not BLOCK:** no possible secret/security issue that must be resolved before *any* commit of source files.

**Minimum before a first commit:** add a `.gitignore`, keep logs/cache/`__pycache__`/`.ohmyagent`/scratch out, confirm Git LFS for joblibs, and decide whether datasets go in the same commit.
