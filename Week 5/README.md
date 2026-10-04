# Week 5 – Titanic Survival Prediction API

**Advance Soft Logics – AI/ML Internship (Week 5 of 6): Model Deployment**

This project serves the tuned Gradient Boosting model from Week 4 as a REST API built with **FastAPI**.
Clients send passenger details as JSON and get back a survival prediction, a probability and the model version.

## What was built

- `/predict` endpoint with strict input validation and structured JSON responses
- `/health` endpoint that reports whether the model loaded
- Model loaded **once** at startup (not per request)
- Distinct 400 errors for each kind of bad input
- Structured JSON-lines request log (input, output, timestamp)
- Model versioning (`models/titanic_gb_v1.joblib` + `/v1/...` endpoints)
- 26 automated tests with pytest

### Bonus features (all implemented)

| Bonus | Where |
|---|---|
| Docker container | `Dockerfile` |
| HTML front-end form | `frontend/index.html`, served at `/` |
| Batch prediction | `POST /predict/batch` |
| API key protection + rate limiting | `app/security.py` |

## Folder structure

```
asl-internship-aiml-week5-hashir/
├── app/                      # Inference code (the API)
│   ├── main.py               # Routes only
│   ├── config.py             # Settings: paths, model version, limits
│   ├── schemas.py            # Request/response contract (pydantic)
│   ├── errors.py             # Clear 400/401/429/503 error responses
│   ├── model_loader.py       # Loads the joblib model once
│   ├── predictor.py          # Runs the model on validated input
│   ├── prediction_logger.py  # Writes the JSON request log
│   └── security.py           # API key check + rate limiter
├── training/
│   └── train_model.py        # Training-time code (run once, offline)
├── models/                   # Versioned model + metadata
│   ├── titanic_gb_v1.joblib
│   └── titanic_gb_v1.json
├── data/raw/titanic.csv      # Public Titanic dataset (seaborn-data)
├── frontend/index.html       # Bonus HTML form
├── tests/                    # pytest tests
├── logs/                     # predictions.log is created here (git-ignored)
├── screenshots/              # Proof the project works
├── reports/Week5_Report.docx
├── Dockerfile
├── requirements.txt
└── README.md
```

**Training vs inference:** `training/` fits the model; `app/` only loads it and predicts.
The API never refits anything. The saved object is one scikit-learn `Pipeline`
(imputer, scaler, encoder, model), so the exact fitted preprocessing from training is reused for every request.

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Recommended) Retrain so the model matches YOUR scikit-learn version
python -m training.train_model
```

## Run the API

```bash
uvicorn app.main:app --reload
```

- Front-end form: http://127.0.0.1:8000/
- Interactive docs (Swagger): http://127.0.0.1:8000/docs

## Run the tests

```bash
python -m pytest -v
```

## Run with Docker

```bash
docker build -t titanic-api .
docker run -p 8000:8000 titanic-api
```

To turn on the API key, pass it as an environment variable:

```bash
docker run -p 8000:8000 -e API_KEY=mysecret titanic-api
```

## API contract

### Request body (all fields required)

| Field | Type | Allowed values |
|---|---|---|
| `pclass` | integer | 1, 2 or 3 |
| `sex` | string | `"male"` or `"female"` |
| `age` | number | 0 to 100 |
| `sibsp` | integer | 0 to 10 |
| `parch` | integer | 0 to 10 |
| `fare` | number | 0 to 600 |
| `embarked` | string | `"S"`, `"C"` or `"Q"` |

### `POST /predict`

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"pclass":1,"sex":"female","age":29,"sibsp":0,"parch":0,"fare":211.34,"embarked":"S"}'
```

```json
{
  "prediction": 1,
  "label": "survived",
  "probability_survived": 0.9762,
  "confidence": 0.9762,
  "model_name": "titanic-gradient-boosting",
  "model_version": "v1"
}
```

### `POST /predict/batch` (up to 100 records)

```bash
curl -X POST http://127.0.0.1:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"records":[{"pclass":3,"sex":"male","age":30,"sibsp":0,"parch":0,"fare":8.05,"embarked":"S"}]}'
```

### `GET /health`

```bash
curl http://127.0.0.1:8000/health
```

```json
{"status":"ok","model_loaded":true,"model_name":"titanic-gradient-boosting","model_version":"v1"}
```

It returns HTTP 503 with `"model_loaded": false` if the model file could not be loaded.

### `GET /model-info`

Returns the model's metadata: features, best parameters, cross-validation F1 and test metrics.

### Error responses

Bad input returns **HTTP 400** with an `error` category and a `details` list:

| `error` | Meaning | Example |
|---|---|---|
| `missing_field` | A required field is absent | no `age` |
| `wrong_type` | Value has the wrong type | `"age": "old"` or `"pclass": "1"` |
| `invalid_value` | Right type, not allowed | `"age": 250`, `"sex": "unknown"` |
| `unknown_field` | Extra field not in the contract | `"ticket_number": 5` |
| `malformed_json` | Body is not valid JSON | `{this is not json` |
| `multiple_errors` | Several different problems at once | see `details` |

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"pclass":1,"sex":"female","age":"old","sibsp":0,"parch":0,"fare":10,"embarked":"S"}'
```

```json
{
  "error": "wrong_type",
  "message": "The request body is not valid. See details.",
  "details": [{"field": "age", "category": "wrong_type", "problem": "Input should be a valid number"}]
}
```

Other statuses: `401` wrong/missing API key, `429` rate limit reached, `503` model not loaded.

## API key and rate limiting

- **API key:** set the `API_KEY` environment variable. Clients must then send `X-API-Key: <key>`.
  If `API_KEY` is not set, protection is off (so local development is easy). `/health` is always open.
- **Rate limit:** 60 requests per minute per client by default. Change it with `RATE_LIMIT_PER_MINUTE`.

```bash
# Windows PowerShell
$env:API_KEY="mysecret"; uvicorn app.main:app
# macOS / Linux
API_KEY=mysecret uvicorn app.main:app
```

Secrets are never hard-coded; they come from environment variables.

## Logging

Each prediction adds one JSON line to `logs/predictions.log`:

```json
{"timestamp": "2026-10-04T13:55:34+00:00", "endpoint": "/predict", "model_version": "v1", "input": {"pclass": 1, "...": "..."}, "output": {"prediction": 1, "probability_survived": 0.9762}}
```

## Model versioning

- Model files are named by version: `titanic_gb_v1.joblib` with a matching `titanic_gb_v1.json`.
- Every response includes `model_version`.
- To deploy a new model, train it as `v2` (set `MODEL_VERSION=v2`), keep the `v1` file, and
  existing clients keep working because `/v1/...` paths and the response shape do not change.

## Design decisions

- **FastAPI + pydantic** gives validation, automatic docs and clear types with little code.
- **`strict=True`** in the schema rejects `"3"` for a number, so wrong types are caught instead of silently converted.
- **400 instead of FastAPI's default 422**, because the assignment asks for 400 with a clear message.
- **Sync endpoints:** scikit-learn prediction is CPU work, so plain `def` routes (run in a thread pool) are the right fit.
- **In-memory rate limiter** is simple and fine for one server. A real multi-server deployment would need Redis.

## Model results (from training)

| Metric | Value |
|---|---|
| 5-fold CV F1 (train set) | 0.7628 |
| Test accuracy | 0.8101 |
| Test precision | 0.8070 |
| Test recall | 0.6667 |
| Test F1 | 0.7302 |

## Screenshots

See the `screenshots/` folder:

1. `01_health_endpoint.png` – `/health` response
2. `02_predict_success.png` – successful `/predict` call
3. `03_validation_error.png` – a 400 error response
4. `04_frontend_form.png` – HTML front-end with a result
5. `05_pytest_passing.png` – all tests passing

## Known limitations

- The Titanic dataset is small (891 rows), so predictions are illustrative, not production quality.
- The rate limiter resets when the server restarts and is not shared between servers.
- Pickled (`joblib`) models can break across scikit-learn versions. Retrain if you see a version warning.
- `age` is required; the model itself could handle missing ages, but the API contract keeps inputs explicit.
