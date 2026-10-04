"""Tests for the optional API key and the rate limiter."""


def test_api_key_is_required_when_set(client, valid_passenger, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret123")

    response = client.post("/predict", json=valid_passenger)
    assert response.status_code == 401
    assert response.json()["error"] == "unauthorized"


def test_wrong_api_key_is_rejected(client, valid_passenger, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret123")

    response = client.post(
        "/predict", json=valid_passenger, headers={"X-API-Key": "wrong"}
    )
    assert response.status_code == 401


def test_correct_api_key_is_accepted(client, valid_passenger, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret123")

    response = client.post(
        "/predict", json=valid_passenger, headers={"X-API-Key": "secret123"}
    )
    assert response.status_code == 200


def test_health_does_not_need_api_key(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret123")
    assert client.get("/health").status_code == 200


def test_rate_limit_blocks_extra_requests(client, valid_passenger, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "2")

    assert client.post("/predict", json=valid_passenger).status_code == 200
    assert client.post("/predict", json=valid_passenger).status_code == 200

    response = client.post("/predict", json=valid_passenger)
    assert response.status_code == 429
    assert response.json()["error"] == "rate_limited"
