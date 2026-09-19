# Developer guide — SIH26072 V2 copy

## 项目目的

This tree is a **verbatim-style copy of the working V1 scientific baseline**. V2 work has **not** started in application code.

**核心职责 (as copied)**

- Serve 1h VOTV thunderstorm nowcast
- Keep Phase 1 precipitation proxy endpoints without mixing metrics
- Preserve phase verifiers and evidence under `outputs/`

**V2 目标 (意图，未实现)**

- Multimodal, multi-location thunderstorm/lightning decision support

**规则**

- Do not treat this inventory as a license to change V1 science.
- Do not invent radar/satellite/lightning pipelines that are not in the tree.

## 环境搭建

### 前置条件

- Python 3.12 (3.12.10 recorded at train time)
- Outbound HTTPS to Open-Meteo for live `/api/prediction`
- Optional Docker

### 安装与运行

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
# http://127.0.0.1:5000/
```

Docker:

```bash
docker build -t sih26072-nowcast .
docker run --rm -p 10000:10000 -e PORT=10000 sih26072-nowcast
```

Training extras: `requirements_ml.txt` (matplotlib). Historical backend pins: `requirements_backend.txt`.

### 验证脚本（审计，非 serving 路径）

- `dataset/verify_synchronized_dataset_phase3.py`
- `dataset/verify_features_phase4.py`
- `dataset/verify_nowcast_targets_phase5.py`
- `outputs/verify_phase6_models.py`
- `ml/verify_prediction_engine_phase7.py`
- `backend/verify_backend_phase8.py`
- `frontend/verify_frontend_phase9.py`

Some verifiers rewrite JSON under `outputs/` when run.

## 开发工作流

No project ESLint/CI/pre-commit config was found in this copy. Branch strategy is not defined beyond git `main`.

**编码现实**

- Python modules use snake_case files, phase suffixes (`_phase3` … `_phase9`)
- Scientific refusal: no imputation; null predictions on failure
- Two products must stay labeled: nowcast vs precipitation proxy

## 常见任务（仅文档；非要求实现）

### 本地打一次 nowcast

```bash
curl -s http://127.0.0.1:5000/api/prediction
curl -s http://127.0.0.1:5000/api/health
```

### 理解数据流水线顺序

1. METAR download / labels — `dataset/download_votv_metar_iem.py`, `build_thunderstorm_labels.py`
2. Open-Meteo fetch — `fetch_votv_openmeteo_phase3.py`
3. Sync — `build_synchronized_dataset_phase3.py`
4. Features — `feature_engineering_phase4.py`
5. Targets t+1/2/3h — `nowcast_targets_phase5.py`
6. Train — `ml/train_thunderstorm_nowcast_phase6.py`

### 拷贝完整性注意

- `models/thunderstorm_nowcast_2h.joblib` and `_3h.joblib` are **134 bytes** here — not usable estimators.
- Large CSVs and 1h/proxy joblibs **are** present.
- Scratch probes: `dataset/_tmp_*.py`, empty `ERROR`, `test.txt`.

## 编码规范（从现有代码归纳）

- Fail closed on unit mismatch, stale data (>3h), missing history (<6 contiguous hours).
- Do not use Open-Meteo `weather_code` as thunderstorm label.
- Chronological splits only in Phase 6 (70/15/15); no shuffle.
- Dashboard must not plot future hours as observations.
