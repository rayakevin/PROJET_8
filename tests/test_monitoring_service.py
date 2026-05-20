import json
from pathlib import Path

from app.services.monitoring_service import MonitoringService


def read_jsonl(log_path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_create_request_id_returns_unique_strings(tmp_path: Path) -> None:
    service = MonitoringService(log_path=tmp_path / "api_predictions.jsonl")

    first_request_id = service.create_request_id()
    second_request_id = service.create_request_id()

    assert isinstance(first_request_id, str)
    assert isinstance(second_request_id, str)
    assert first_request_id != second_request_id


def test_log_prediction_success_writes_jsonl_event(tmp_path: Path) -> None:
    log_path = tmp_path / "api_predictions.jsonl"
    service = MonitoringService(log_path=log_path)

    service.log_prediction_success(
        request_id="request-1",
        endpoint="/predict",
        client_id=100001,
        features={
            "amt_credit": 500000.0,
            "missing_feature": float("nan"),
        },
        prediction={
            "model_version": "lightgbm_top30_optimized",
            "score": 0.42,
            "threshold": 0.5,
            "prediction": 0,
            "decision": "low_risk",
            "latency_ms": 12.3,
            "preprocessing_latency_ms": 1.2,
            "inference_latency_ms": 11.1,
        },
    )

    events = read_jsonl(log_path)

    assert len(events) == 1
    assert events[0]["timestamp"]
    assert events[0]["request_id"] == "request-1"
    assert events[0]["endpoint"] == "/predict"
    assert events[0]["status"] == "success"
    assert events[0]["client_id"] == 100001
    assert events[0]["model_version"] == "lightgbm_top30_optimized"
    assert events[0]["features"]["amt_credit"] == 500000.0
    assert events[0]["features"]["missing_feature"] is None
    assert events[0]["score"] == 0.42
    assert events[0]["decision"] == "low_risk"
    assert events[0]["error_type"] is None
    assert events[0]["error_message"] is None


def test_log_prediction_error_writes_jsonl_event(tmp_path: Path) -> None:
    log_path = tmp_path / "api_predictions.jsonl"
    service = MonitoringService(log_path=log_path)

    service.log_prediction_error(
        request_id="request-2",
        endpoint="/predict",
        client_id=100002,
        error=ValueError("raw_data.application est obligatoire"),
        latency_ms=3.4,
    )

    events = read_jsonl(log_path)

    assert len(events) == 1
    assert events[0]["request_id"] == "request-2"
    assert events[0]["endpoint"] == "/predict"
    assert events[0]["status"] == "error"
    assert events[0]["client_id"] == 100002
    assert events[0]["features"] is None
    assert events[0]["score"] is None
    assert events[0]["latency_ms"] == 3.4
    assert events[0]["error_type"] == "ValueError"
    assert events[0]["error_message"] == "raw_data.application est obligatoire"


def test_write_event_appends_multiple_events(tmp_path: Path) -> None:
    log_path = tmp_path / "nested" / "api_predictions.jsonl"
    service = MonitoringService(log_path=log_path)

    service.write_event({"request_id": "request-1", "status": "success"})
    service.write_event({"request_id": "request-2", "status": "error"})

    events = read_jsonl(log_path)

    assert [event["request_id"] for event in events] == ["request-1", "request-2"]
    assert [event["status"] for event in events] == ["success", "error"]


def test_write_events_appends_multiple_jsonl_events(tmp_path: Path) -> None:
    log_path = tmp_path / "api_predictions.jsonl"
    service = MonitoringService(log_path=log_path)

    service.write_events(
        [
            {
                "request_id": "request-1",
                "status": "success",
                "score": 0.1,
            },
            {
                "request_id": "request-2",
                "status": "success",
                "score": float("nan"),
            },
        ]
    )

    events = read_jsonl(log_path)

    assert len(events) == 2
    assert [event["request_id"] for event in events] == ["request-1", "request-2"]
    assert events[0]["score"] == 0.1
    assert events[1]["score"] is None


def test_write_event_converts_infinite_values_to_null(tmp_path: Path) -> None:
    log_path = tmp_path / "api_predictions.jsonl"
    service = MonitoringService(log_path=log_path)

    service.write_event(
        {
            "request_id": "request-3",
            "status": "success",
            "values": [float("inf"), float("-inf")],
        }
    )

    events = read_jsonl(log_path)

    assert events[0]["values"] == [None, None]
