import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg
from evidently import Report
from evidently.presets import DataDriftPreset
from psycopg.rows import dict_row

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_PATH = PROJECT_ROOT / "logs" / "api_predictions.jsonl"
DEFAULT_REFERENCE_PATH = PROJECT_ROOT / "monitoring" / "reference" / "top30_reference.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "monitoring"
FEATURE_SCHEMA_PATH = PROJECT_ROOT / "model" / "schema" / "top30_feature_schema.json"
DEFAULT_DATABASE_URL = "postgresql://projet8:projet8@localhost:5433/projet8_monitoring"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse les logs de monitoring et produit un rapport de drift."
    )
    parser.add_argument(
        "--source",
        choices=["jsonl", "postgres"],
        default="jsonl",
        help="Source des événements de monitoring.",
    )
    parser.add_argument(
        "--logs",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help="Chemin du fichier JSONL de logs API si --source=jsonl.",
    )
    parser.add_argument(
        "--database-url",
        default=os.getenv("MONITORING_DATABASE_URL", DEFAULT_DATABASE_URL),
        help="URL PostgreSQL si --source=postgres.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Nombre maximal d'événements PostgreSQL à charger, du plus récent au plus ancien.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE_PATH,
        help="Chemin du parquet de référence TOP30.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Dossier de sortie des rapports.",
    )
    return parser.parse_args()


def load_feature_names() -> list[str]:
    schema = json.loads(FEATURE_SCHEMA_PATH.read_text(encoding="utf-8"))
    return [item["feature"] for item in schema["features"]]


def load_jsonl(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.exists():
        raise FileNotFoundError(f"Fichier de logs introuvable : {log_path}")

    events = []
    for line_number, line in enumerate(log_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue

        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Ligne JSON invalide dans {log_path}, ligne {line_number}") from error

        if not isinstance(event, dict):
            raise ValueError(f"La ligne {line_number} ne contient pas un objet JSON")

        events.append(event)

    if not events:
        raise ValueError(f"Aucun événement exploitable dans {log_path}")

    return events


def load_postgres_events(database_url: str, limit: int | None = None) -> list[dict[str, Any]]:
    query = """
        SELECT
            timestamp,
            request_id,
            endpoint,
            status,
            client_id,
            model_version,
            features,
            score,
            threshold,
            prediction,
            decision,
            latency_ms,
            preprocessing_latency_ms,
            inference_latency_ms,
            error_type,
            error_message
        FROM prediction_logs
        ORDER BY timestamp DESC
    """
    params: dict[str, Any] = {}

    if limit is not None:
        query += " LIMIT %(limit)s"
        params["limit"] = limit

    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()

    events: list[dict[str, Any]] = []
    for row in rows:
        event = dict(row)
        timestamp = event.get("timestamp")
        if timestamp is not None:
            event["timestamp"] = timestamp.isoformat()
        events.append(event)

    if not events:
        raise ValueError("Aucun événement exploitable dans PostgreSQL")

    return events


def percentile(values: list[float], q: float) -> float | None:
    clean_values = pd.Series(values, dtype="float64").dropna()
    if clean_values.empty:
        return None

    return round(float(clean_values.quantile(q)), 3)


def mean(values: list[float]) -> float | None:
    clean_values = pd.Series(values, dtype="float64").dropna()
    if clean_values.empty:
        return None

    return round(float(clean_values.mean()), 3)


def summarize_operational_metrics(events: list[dict[str, Any]]) -> dict[str, Any]:
    success_events = [event for event in events if event.get("status") == "success"]
    error_events = [event for event in events if event.get("status") == "error"]
    latencies = [event.get("latency_ms") for event in events if event.get("latency_ms") is not None]
    scores = [event.get("score") for event in success_events if event.get("score") is not None]
    decisions = Counter(event.get("decision") for event in success_events)
    error_types = Counter(event.get("error_type") for event in error_events)

    total_events = len(events)
    error_count = len(error_events)

    return {
        "total_events": total_events,
        "success_count": len(success_events),
        "error_count": error_count,
        "error_rate": round(error_count / total_events, 4) if total_events else None,
        "latency_ms": {
            "mean": mean(latencies),
            "median": percentile(latencies, 0.5),
            "p95": percentile(latencies, 0.95),
            "max": round(max(latencies), 3) if latencies else None,
        },
        "scores": {
            "mean": mean(scores),
            "median": percentile(scores, 0.5),
            "p95": percentile(scores, 0.95),
        },
        "decisions": {key: value for key, value in decisions.items() if key is not None},
        "errors": {key: value for key, value in error_types.items() if key is not None},
    }


def extract_current_features(
    events: list[dict[str, Any]],
    feature_names: list[str],
) -> pd.DataFrame:
    rows = [
        event["features"]
        for event in events
        if event.get("status") == "success" and isinstance(event.get("features"), dict)
    ]

    if not rows:
        return pd.DataFrame(columns=feature_names)

    frame = pd.DataFrame(rows)
    missing_features = [feature for feature in feature_names if feature not in frame.columns]
    for feature in missing_features:
        frame[feature] = pd.NA

    frame = frame[feature_names]
    for feature in feature_names:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")

    return frame


def load_reference(reference_path: Path, feature_names: list[str]) -> pd.DataFrame:
    if not reference_path.exists():
        raise FileNotFoundError(f"Référence de drift introuvable : {reference_path}")

    reference = pd.read_parquet(reference_path)
    missing_features = [feature for feature in feature_names if feature not in reference.columns]
    if missing_features:
        raise ValueError(f"Features absentes de la référence : {', '.join(missing_features)}")

    return reference[feature_names]


def extract_drift_summary(snapshot: Any) -> dict[str, Any]:
    snapshot_dict = snapshot.dict()
    metrics = snapshot_dict.get("metrics", [])

    drifted_columns = None
    drifted_share = None
    column_metrics = []

    for metric in metrics:
        metric_name = metric.get("metric_name", "")
        config = metric.get("config", {})
        value = metric.get("value")

        if metric_name.startswith("DriftedColumnsCount") and isinstance(value, dict):
            drifted_columns = value.get("count")
            drifted_share = value.get("share")
            continue

        if metric_name.startswith("ValueDrift"):
            column_metrics.append(
                {
                    "column": config.get("column"),
                    "method": config.get("method"),
                    "threshold": config.get("threshold"),
                    "value": value,
                }
            )

    return {
        "drift_detected": bool(drifted_columns and drifted_columns > 0),
        "drifted_columns_count": drifted_columns,
        "drifted_columns_share": drifted_share,
        "columns": column_metrics,
    }


def run_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "data_drift_report.html"
    json_path = output_dir / "data_drift_report.json"

    if current.empty:
        return {
            "status": "skipped",
            "reason": "Aucune prédiction réussie disponible pour le drift.",
            "html_report_path": None,
            "json_report_path": None,
            "skipped_features": list(reference.columns),
        }

    usable_features = [
        feature
        for feature in current.columns
        if current[feature].notna().any() and reference[feature].notna().any()
    ]
    skipped_features = [feature for feature in current.columns if feature not in usable_features]

    if not usable_features:
        return {
            "status": "skipped",
            "reason": "Aucune feature non vide disponible pour le drift.",
            "html_report_path": None,
            "json_report_path": None,
            "skipped_features": skipped_features,
        }

    snapshot = Report([DataDriftPreset()]).run(
        reference_data=reference[usable_features],
        current_data=current[usable_features],
    )
    snapshot.save_html(str(html_path))
    snapshot.save_json(str(json_path))

    return {
        "status": "computed",
        "html_report_path": str(html_path),
        "json_report_path": str(json_path),
        "features_used": usable_features,
        "skipped_features": skipped_features,
        **extract_drift_summary(snapshot),
    }


def write_summary(summary: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "monitoring_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary_path


def main() -> None:
    args = parse_args()
    feature_names = load_feature_names()
    if args.source == "postgres":
        events = load_postgres_events(args.database_url, args.limit)
        source_description = args.database_url
    else:
        events = load_jsonl(args.logs)
        source_description = str(args.logs)

    current_features = extract_current_features(events, feature_names)
    reference = load_reference(args.reference, feature_names)
    operational_summary = summarize_operational_metrics(events)
    drift_summary = run_drift_report(reference, current_features, args.output_dir)

    summary = {
        "source": args.source,
        "source_path": source_description,
        "logs_path": str(args.logs) if args.source == "jsonl" else None,
        "database_url": args.database_url if args.source == "postgres" else None,
        "reference_path": str(args.reference),
        "current_feature_rows": len(current_features),
        "feature_count": len(feature_names),
        "operational": operational_summary,
        "drift": drift_summary,
    }
    summary_path = write_summary(summary, args.output_dir)

    print(f"Synthèse générée : {summary_path}")
    if drift_summary["status"] == "computed":
        print(f"Rapport HTML généré : {drift_summary['html_report_path']}")
    else:
        print(f"Rapport de drift non généré : {drift_summary['reason']}")


if __name__ == "__main__":
    main()
