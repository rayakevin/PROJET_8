from scripts.benchmark_api_performance import percentile, summarize_values


def test_percentile_interpolates_values() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.95) == 3.85


def test_summarize_values_returns_core_metrics() -> None:
    summary = summarize_values([1.0, 2.0, 3.0])

    assert summary == {
        "mean": 2.0,
        "median": 2.0,
        "p95": 2.9,
        "max": 3.0,
    }


def test_summarize_values_handles_empty_list() -> None:
    summary = summarize_values([])

    assert summary == {
        "mean": None,
        "median": None,
        "p95": None,
        "max": None,
    }
