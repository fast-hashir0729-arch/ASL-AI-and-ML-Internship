"""Tests for bad input. Each kind of mistake gets its own error category."""


def test_missing_field(client, valid_passenger):
    del valid_passenger["age"]
    response = client.post("/predict", json=valid_passenger)
    body = response.json()

    assert response.status_code == 400
    assert body["error"] == "missing_field"
    assert body["details"][0]["field"] == "age"


def test_wrong_type(client, valid_passenger):
    valid_passenger["age"] = "twenty"
    response = client.post("/predict", json=valid_passenger)

    assert response.status_code == 400
    assert response.json()["error"] == "wrong_type"


def test_number_sent_as_text_is_rejected(client, valid_passenger):
    valid_passenger["pclass"] = "1"
    response = client.post("/predict", json=valid_passenger)

    assert response.status_code == 400
    assert response.json()["error"] == "wrong_type"


def test_out_of_range_value(client, valid_passenger):
    valid_passenger["age"] = 250
    response = client.post("/predict", json=valid_passenger)

    assert response.status_code == 400
    assert response.json()["error"] == "invalid_value"


def test_unknown_category_value(client, valid_passenger):
    valid_passenger["sex"] = "unknown"
    response = client.post("/predict", json=valid_passenger)

    assert response.status_code == 400
    assert response.json()["error"] == "invalid_value"


def test_unknown_field(client, valid_passenger):
    valid_passenger["ticket_number"] = 12345
    response = client.post("/predict", json=valid_passenger)

    assert response.status_code == 400
    assert response.json()["error"] == "unknown_field"


def test_malformed_json(client):
    response = client.post(
        "/predict",
        content="{this is not json",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 400
    assert response.json()["error"] == "malformed_json"


def test_several_problems_are_all_reported(client, valid_passenger):
    del valid_passenger["fare"]
    valid_passenger["age"] = "old"
    body = client.post("/predict", json=valid_passenger).json()

    assert body["error"] == "multiple_errors"
    assert len(body["details"]) == 2


def test_empty_batch_is_rejected(client):
    response = client.post("/predict/batch", json={"records": []})
    assert response.status_code == 400


def test_bad_record_inside_batch_points_to_its_position(client, valid_passenger):
    bad_passenger = dict(valid_passenger, age=-5)
    response = client.post(
        "/predict/batch", json={"records": [valid_passenger, bad_passenger]}
    )

    assert response.status_code == 400
    assert response.json()["details"][0]["field"] == "records.1.age"
