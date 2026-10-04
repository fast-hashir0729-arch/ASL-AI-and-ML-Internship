"""Tests for /predict, /predict/batch, /health and logging."""

import json


def test_health_reports_model_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_predict_returns_structured_response(client, valid_passenger):
    response = client.post("/predict", json=valid_passenger)
    body = response.json()

    assert response.status_code == 200
    assert body["prediction"] in (0, 1)
    assert 0 <= body["probability_survived"] <= 1
    assert 0.5 <= body["confidence"] <= 1
    assert body["model_name"] == "titanic-gradient-boosting"
    assert body["model_version"] == "v1"


def test_known_input_first_class_woman_survives(client, valid_passenger):
    """Regression test: this answer should not change by accident."""
    body = client.post("/predict", json=valid_passenger).json()
    assert body["prediction"] == 1
    assert body["probability_survived"] > 0.8


def test_known_input_third_class_man_does_not_survive(client):
    passenger = {
        "pclass": 3,
        "sex": "male",
        "age": 30,
        "sibsp": 0,
        "parch": 0,
        "fare": 8.05,
        "embarked": "S",
    }
    body = client.post("/predict", json=passenger).json()
    assert body["prediction"] == 0
    assert body["probability_survived"] < 0.2


def test_same_input_gives_same_output(client, valid_passenger):
    first = client.post("/predict", json=valid_passenger).json()
    second = client.post("/predict", json=valid_passenger).json()
    assert first == second


def test_batch_prediction(client, valid_passenger):
    body = {"records": [valid_passenger, valid_passenger]}
    response = client.post("/predict/batch", json=body)

    assert response.status_code == 200
    assert response.json()["count"] == 2
    assert len(response.json()["predictions"]) == 2


def test_versioned_path_works(client, valid_passenger):
    response = client.post("/v1/predict", json=valid_passenger)
    assert response.status_code == 200


def test_prediction_is_logged(client, valid_passenger, log_file):
    client.post("/predict", json=valid_passenger)

    lines = log_file.read_text().strip().splitlines()
    record = json.loads(lines[-1])
    assert record["input"]["pclass"] == 1
    assert "prediction" in record["output"]
    assert "timestamp" in record


def test_model_info_endpoint(client):
    body = client.get("/model-info").json()
    assert body["model_version"] == "v1"
    assert "test_metrics" in body["metadata"]


def test_home_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_health_and_predict_when_model_missing(client, valid_passenger, monkeypatch):
    monkeypatch.setattr(client.app.state, "model", None)

    assert client.get("/health").status_code == 503
    assert client.post("/predict", json=valid_passenger).status_code == 503
