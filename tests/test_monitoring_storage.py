import json

from scripts.import_monitoring_logs_to_postgres import load_jsonl, parse_timestamp


def test_load_jsonl_reads_monitoring_events(tmp_path) -> None:
    log_path = tmp_path / "api_predictions.jsonl"
    event = {
        "timestamp": "2026-05-15T10:00:00+00:00",
        "request_id": "request-1",
        "status": "success",
        "features": {"amt_credit": 100000.0},
    }
    log_path.write_text(json.dumps(event, ensure_ascii=False) + "\n", encoding="utf-8")

    events = load_jsonl(log_path)

    assert events == [event]


def test_parse_timestamp_accepts_z_suffix() -> None:
    timestamp = parse_timestamp("2026-05-15T10:00:00Z")

    assert timestamp.isoformat() == "2026-05-15T10:00:00+00:00"
