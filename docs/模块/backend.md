# backend/

Phase 8 HTTP service layer. **No** feature math of its own.

## 结构

```
backend/
├── __init__.py
├── nowcast_service.py          # load model, live_nowcast, health, errors
└── verify_backend_phase8.py
```

## 关键函数

`load_nowcast_model`, `warm_up`, `live_nowcast`, `health_snapshot`, `probe_live_provider`, `engine_error_payload`.

Delegates science to `ml/predict_thunderstorm_nowcast.py`. Flask routes in `app.py`.
