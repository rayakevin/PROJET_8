import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_PATH = PROJECT_ROOT / "logs" / "api_predictions.jsonl"
DEFAULT_DATABASE_URL = "postgresql://projet8:projet8@localhost:5433/projet8_monitoring"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Importe les logs JSONL de l'API dans PostgreSQL.")
    parser.add_argument(
        "--logs",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help="Chemin du fichier JSONL à importer.",
    )
    parser.add_argument(
        "--database-url",
        default=os.getenv("MONITORING_DATABASE_URL", DEFAULT_DATABASE_URL),
        help="URL de connexion PostgreSQL.",
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Vide la table prediction_logs avant l'import.",
    )
    return parser.parse_args()


def load_jsonl(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.exists():
        raise FileNotFoundError(f"Fichier de logs introuvable : {log_path}")

    events: list[dict[str, Any]] = []
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

    return events


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def insert_events(
    events: list[dict[str, Any]],
    database_url: str,
    *,
    truncate: bool = False,
) -> int:
    if not events:
        return 0

    query = """
        INSERT INTO prediction_logs (
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
        )
        VALUES (
            %(timestamp)s,
            %(request_id)s,
            %(endpoint)s,
            %(status)s,
            %(client_id)s,
            %(model_version)s,
            %(features)s,
            %(score)s,
            %(threshold)s,
            %(prediction)s,
            %(decision)s,
            %(latency_ms)s,
            %(preprocessing_latency_ms)s,
            %(inference_latency_ms)s,
            %(error_type)s,
            %(error_message)s
        )
        ON CONFLICT (request_id, client_id, status) DO NOTHING
    """

    rows = [
        {
            "timestamp": parse_timestamp(event["timestamp"]),
            "request_id": event.get("request_id"),
            "endpoint": event.get("endpoint"),
            "status": event.get("status"),
            "client_id": event.get("client_id"),
            "model_version": event.get("model_version"),
            "features": Jsonb(event.get("features")) if event.get("features") is not None else None,
            "score": event.get("score"),
            "threshold": event.get("threshold"),
            "prediction": event.get("prediction"),
            "decision": event.get("decision"),
            "latency_ms": event.get("latency_ms"),
            "preprocessing_latency_ms": event.get("preprocessing_latency_ms"),
            "inference_latency_ms": event.get("inference_latency_ms"),
            "error_type": event.get("error_type"),
            "error_message": event.get("error_message"),
        }
        for event in events
    ]

    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor:
            if truncate:
                cursor.execute("TRUNCATE TABLE prediction_logs RESTART IDENTITY")

            cursor.executemany(query, rows)
            imported_count = cursor.rowcount

        connection.commit()

    return imported_count


def main() -> None:
    args = parse_args()
    events = load_jsonl(args.logs)
    imported_count = insert_events(events, args.database_url, truncate=args.truncate)
    print(f"Événements lus : {len(events)}")
    print(f"Événements importés : {imported_count}")


if __name__ == "__main__":
    main()
