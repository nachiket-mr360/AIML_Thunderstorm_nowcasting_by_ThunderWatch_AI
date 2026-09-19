# ml/

Machine-learning package: Phase 1 surrogate **and** Phase 6/7 thunderstorm nowcast.

## 结构

```
ml/
├── __init__.py                         # exports train_model, predict (Phase 1)
├── train_model.py                      # precipitation-proxy RF
├── predict.py                          # score proxy
├── train_thunderstorm_nowcast_phase6.py
├── predict_thunderstorm_nowcast.py     # live 1h engine
└── verify_prediction_engine_phase7.py
```

`__init__.py` disclaimer still describes the **surrogate** era; the real nowcast lives in the Phase 6/7 modules.

## 依赖

`dataset/feature_engineering_phase4.py`, `models/*.joblib`, Open-Meteo at runtime. Used by `backend/nowcast_service.py` and `app.py` (legacy).
