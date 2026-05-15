from scripts.analyze_monitoring_logs import extract_current_features, summarize_operational_metrics


def test_summarize_operational_metrics_counts_success_errors_and_latency() -> None:
    events = [
        {
            "status": "success",
            "latency_ms": 100.0,
            "score": 0.2,
            "decision": "low_risk",
        },
        {
            "status": "success",
            "latency_ms": 200.0,
            "score": 0.8,
            "decision": "high_risk",
        },
        {
            "status": "error",
            "latency_ms": 10.0,
            "error_type": "ValueError",
        },
    ]

    summary = summarize_operational_metrics(events)

    assert summary["total_events"] == 3
    assert summary["success_count"] == 2
    assert summary["error_count"] == 1
    assert summary["error_rate"] == 0.3333
    assert summary["latency_ms"]["mean"] == 103.333
    assert summary["scores"]["mean"] == 0.5
    assert summary["decisions"] == {"low_risk": 1, "high_risk": 1}
    assert summary["errors"] == {"ValueError": 1}


def test_extract_current_features_keeps_schema_order_and_ignores_errors() -> None:
    feature_names = ["amt_credit", "ext_source_1"]
    events = [
        {
            "status": "success",
            "features": {
                "ext_source_1": 0.5,
                "amt_credit": 100000.0,
                "ignored_feature": 1.0,
            },
        },
        {
            "status": "error",
            "features": None,
        },
    ]

    frame = extract_current_features(events, feature_names)

    assert frame.shape == (1, 2)
    assert frame.columns.tolist() == feature_names
    assert frame.iloc[0].to_dict() == {
        "amt_credit": 100000.0,
        "ext_source_1": 0.5,
    }
