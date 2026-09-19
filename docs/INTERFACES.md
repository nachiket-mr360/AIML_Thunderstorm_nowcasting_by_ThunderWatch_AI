# Interfaces — SIH26072 V2 (copied V1)

Read-only HTTP API. No authentication, cookies, or write endpoints. CORS: `CORS_ALLOWED_ORIGINS` default `*`.

Flask binds `HOST` (default `127.0.0.1`) and `PORT` (default `5000` for `python app.py`; Docker/gunicorn default `10000`).

## HTTP 端点

| Method | Path | 产品 | 实现 |
|--------|------|------|------|
| GET | `/` | Dashboard HTML | `app.py` `dashboard()` |
| GET | `/api/prediction` | **Primary** 1h thunderstorm nowcast | `nowcast_service.live_nowcast()` |
| GET | `/api/health` | App + model + engine + optional live probe | `health_snapshot()`; `?live=0` skips upstream |
| GET | `/api/prediction/proxy` | Phase 1 precipitation **surrogate** | `app.build_prediction()` |
| GET | `/api/history` | Recent observed hours for chart | Open-Meteo via `app.py` |
| GET | `/api/evaluation` | Phase 1 metrics + importance (read-only) | `outputs/evaluation.json` |
| GET | `/api/artifacts/<filename>` | Whitelisted files under `outputs/` | images/CSV |
| GET | `/api/scenario` | Phase 1 historical test replay | `compute_historical_scenario()` |

Health: **200** only if everything `ok`; otherwise **503**.

Primary nowcast failures: JSON error, `probability` / `predicted_class` / `risk_label` are **null**. Do not treat errors as “no thunderstorm”.

## Primary nowcast payload (success)

Documented in README. Fields include (among others):

- `probability` — uncalibrated positive-class score
- `predicted_class` — 0/1 vs threshold `0.0775`
- `risk_label` — `Thunderstorm alert` / `No thunderstorm alert`
- `threshold`, `lead_time_hours` (1)
- provenance / freshness / feature completeness / disclaimer

Example:

```json
{
  "probability": 0.005,
  "predicted_class": 0,
  "risk_label": "No thunderstorm alert",
  "threshold": 0.0775,
  "lead_time_hours": 1
}
```

## Python 入口（离线 / 引擎）

| 模块 | 关键符号 | 用途 |
|------|----------|------|
| `ml.predict_thunderstorm_nowcast` | `predict_current_thunderstorm_risk`, `load_model_bundle`, `ThunderstormNowcastError` | 实时 1h 引擎 |
| `backend.nowcast_service` | `live_nowcast`, `health_snapshot`, `warm_up` | HTTP 适配 |
| `dataset.feature_engineering_phase4` | (imported by engine) | 28 特征 |
| `ml.predict` / `ml.train_model` | Phase 1 代理 | 与雷暴标签无关 |

## 前端

- `templates/index.html` — 仪表盘
- `static/dashboard.js` — 调用 `/api/prediction`, `/api/health`, `/api/history`
- 地图：Leaflet + Esri World Street Map；单点 VOTV，无雷达/闪电图层

## 环境变量

| 变量 | 默认 | 含义 |
|------|------|------|
| `HOST` | `127.0.0.1` | 开发服务器绑定 |
| `PORT` | `5000` / 容器 `10000` | 端口 |
| `FLASK_DEBUG` | off | `1` 开 reloader |
| `CORS_ALLOWED_ORIGINS` | `*` | 逗号分隔或 `*` |

无 API key。实时预测需要出站 HTTPS 到 Open-Meteo。
