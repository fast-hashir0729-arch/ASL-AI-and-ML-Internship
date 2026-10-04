"""Shared test setup."""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.security import reset_rate_limiter


@pytest.fixture
def log_file(tmp_path, monkeypatch):
    """Send prediction logs to a temporary file so tests never touch real logs."""
    path = tmp_path / "predictions.log"
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(path))
    return path


@pytest.fixture
def client(log_file):
    """A test client. 'with' makes the app load the model at startup, like production."""
    reset_rate_limiter()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_passenger():
    """A correct request body."""
    return {
        "pclass": 1,
        "sex": "female",
        "age": 29,
        "sibsp": 0,
        "parch": 0,
        "fare": 211.34,
        "embarked": "S",
    }
