# SIH26072 V2 — inventory documentation

Documentation of **what exists in the copied V1 codebase**. Not a V2 design. No source code was changed for this pass.

**快速链接**: [架构](./ARCHITECTURE.md) | [接口](./INTERFACES.md) | [开发者指南](./DEVELOPER_GUIDE.md)

---

## 核心文档

### [架构](./ARCHITECTURE.md)

Single-station 1h nowcast: METAR labels + Open-Meteo features → RF → Flask → dashboard. V2 multimodal/multi-site is **not** in the tree.

### [接口](./INTERFACES.md)

HTTP routes, env vars, Python engine entry points.

### [开发者指南](./DEVELOPER_GUIDE.md)

Run locally/Docker, verifiers, copy-integrity notes (2h/3h joblib stubs).

---

## 模块

| 模块 | 描述 | README |
|------|------|--------|
| `dataset/` | Labels, sync, 28 features, targets | [dataset](./模块/dataset.md) |
| `ml/` | Phase 1 proxy + Phase 6/7 nowcast | [ml](./模块/ml.md) |
| `backend/` | Flask transport | [backend](./模块/backend.md) |
| UI | Dashboard + map | [frontend](./模块/frontend.md) |
| `app.py` | Process entry | (root) |
| `models/`, `outputs/` | Artifacts and phase evidence | ARCHITECTURE |

---

## 核心概念

| 概念 | 描述 |
|------|------|
| [Genuine METAR label](./专有概念/GenuineMETARLabel.md) | Station thunderstorm observation, not weather_code |
| [28 causal features](./专有概念/CausalFeatures.md) | Hours ≤ t only |
| [Nowcast vs proxy](./专有概念/NowcastVsProxy.md) | Two different models/APIs |

---

## Inventory snapshot (this copy)

**Present and used for serving**

- `app.py`, `backend/nowcast_service.py`, `ml/predict_thunderstorm_nowcast.py`
- `templates/` + `static/`
- `models/thunderstorm_nowcast_1h.joblib`, `storm_risk_model.joblib`
- Full METAR/Open-Meteo CSV archives and engineered tables
- `Dockerfile`, `requirements.txt`
- Phase reports in `outputs/PHASE6`–`PHASE9`

**Present but not the served product**

- Phase 1 proxy train/predict and `/api/prediction/proxy`
- 2h/3h **metadata** and evaluation JSON/PNGs
- Verifiers and `dataset/_tmp_*` probes

**Broken / stub in this copy**

- `models/thunderstorm_nowcast_2h.joblib` (134 bytes)
- `models/thunderstorm_nowcast_3h.joblib` (134 bytes)

**Not present**

- Radar, satellite, lightning, NWP ingest
- Multi-location configuration
- Auth, CI, `.env.example`

Human-oriented product description remains in root `README.md`.

---

## 入门

1. [ARCHITECTURE.md](./ARCHITECTURE.md)
2. Root `README.md` (metrics, limitations, planned-not-integrated)
3. [DEVELOPER_GUIDE.md](./DEVELOPER_GUIDE.md)
4. [INTERFACES.md](./INTERFACES.md)

Docs were **not** git-pushed (no remote push without confirmation).
