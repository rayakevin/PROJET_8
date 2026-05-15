import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.main as main_module

client = TestClient(main_module.app)


@pytest.fixture(autouse=True)
def isolate_monitoring_logs(tmp_path: Path) -> None:
    main_module.monitoring_service.log_path = tmp_path / "api_predictions.jsonl"


def valid_raw_data() -> dict:
    return {
        "application": {
            "AMT_CREDIT": 500000.0,
            "AMT_ANNUITY": 25000.0,
            "AMT_GOODS_PRICE": 450000.0,
            "AMT_INCOME_TOTAL": 180000.0,
            "DAYS_BIRTH": -16000,
            "DAYS_EMPLOYED": -2300,
            "DAYS_ID_PUBLISH": -3200,
            "EXT_SOURCE_1": 0.51,
            "EXT_SOURCE_2": 0.62,
            "EXT_SOURCE_3": 0.31,
            "OWN_CAR_AGE": 5.0,
            "CODE_GENDER": "M",
        },
        "bureau": [
            {
                "CREDIT_ACTIVE": "Active",
                "DAYS_CREDIT": -120,
                "DAYS_CREDIT_ENDDATE": 300,
                "AMT_CREDIT_SUM": 120000.0,
                "AMT_CREDIT_SUM_DEBT": 50000.0,
                "AMT_CREDIT_MAX_OVERDUE": 0.0,
            }
        ],
        "previous_applications": [
            {
                "NAME_CONTRACT_STATUS": "Approved",
                "CNT_PAYMENT": 12.0,
            }
        ],
        "installments_payments": [
            {
                "AMT_INSTALMENT": 10000.0,
                "AMT_PAYMENT": 10000.0,
                "DAYS_INSTALMENT": -40,
                "DAYS_ENTRY_PAYMENT": -39,
            }
        ],
        "credit_card_balance": [
            {
                "CNT_DRAWINGS_ATM_CURRENT": 1.0,
            }
        ],
    }


def test_health_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "credit-scoring-api",
    }


def test_model_info_returns_model_metadata() -> None:
    response = client.get("/model/info")

    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "lightgbm_top30_optimized"
    assert body["feature_count"] == 30
    assert body["threshold"] == 0.5
    assert body["input_contract"] == "raw_business_data"
    assert body["internal_feature_contract"] == "top30_features"


def test_predict_returns_prediction_for_valid_raw_payload() -> None:
    response = client.post(
        "/predict",
        json={
            "client_id": 100001,
            "raw_data": valid_raw_data(),
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["client_id"] == 100001
    assert 0 <= body["score"] <= 1
    assert body["threshold"] == 0.5
    assert body["prediction"] in [0, 1]
    assert body["decision"] in ["low_risk", "high_risk"]
    assert body["model_version"] == "lightgbm_top30_optimized"
    assert body["latency_ms"] >= body["inference_latency_ms"]


def test_predict_writes_monitoring_success_event() -> None:
    response = client.post(
        "/predict",
        json={
            "client_id": 100001,
            "raw_data": valid_raw_data(),
        },
    )

    assert response.status_code == 200

    events = [
        json.loads(line)
        for line in main_module.monitoring_service.log_path.read_text(encoding="utf-8").splitlines()
    ]

    assert len(events) == 1
    assert events[0]["endpoint"] == "/predict"
    assert events[0]["status"] == "success"
    assert events[0]["client_id"] == 100001
    assert len(events[0]["features"]) == 30
    assert 0 <= events[0]["score"] <= 1


def test_predict_rejects_missing_application_block() -> None:
    response = client.post(
        "/predict",
        json={
            "client_id": 100001,
            "raw_data": {},
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "raw_data.application est obligatoire"


def test_predict_writes_monitoring_error_event() -> None:
    response = client.post(
        "/predict",
        json={
            "client_id": 100001,
            "raw_data": {},
        },
    )

    assert response.status_code == 422

    events = [
        json.loads(line)
        for line in main_module.monitoring_service.log_path.read_text(encoding="utf-8").splitlines()
    ]

    assert len(events) == 1
    assert events[0]["endpoint"] == "/predict"
    assert events[0]["status"] == "error"
    assert events[0]["client_id"] == 100001
    assert events[0]["error_type"] == "ValueError"
    assert events[0]["error_message"] == "raw_data.application est obligatoire"


def test_predict_batch_rejects_empty_client_list() -> None:
    response = client.post(
        "/predict/batch",
        json={"clients": []},
    )

    assert response.status_code == 422


def test_predict_batch_returns_predictions_for_valid_raw_payloads() -> None:
    response = client.post(
        "/predict/batch",
        json={
            "clients": [
                {
                    "client_id": 100001,
                    "raw_data": valid_raw_data(),
                },
                {
                    "client_id": 100002,
                    "raw_data": valid_raw_data(),
                },
            ]
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert body["model_version"] == "lightgbm_top30_optimized"
    assert [item["client_id"] for item in body["predictions"]] == [100001, 100002]


def test_predict_batch_writes_one_monitoring_event_per_client() -> None:
    response = client.post(
        "/predict/batch",
        json={
            "clients": [
                {
                    "client_id": 100001,
                    "raw_data": valid_raw_data(),
                },
                {
                    "client_id": 100002,
                    "raw_data": valid_raw_data(),
                },
            ]
        },
    )

    assert response.status_code == 200

    events = [
        json.loads(line)
        for line in main_module.monitoring_service.log_path.read_text(encoding="utf-8").splitlines()
    ]

    assert len(events) == 2
    assert {event["endpoint"] for event in events} == {"/predict/batch"}
    assert {event["status"] for event in events} == {"success"}
    assert [event["client_id"] for event in events] == [100001, 100002]
