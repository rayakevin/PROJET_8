import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg
import streamlit as st
from psycopg.rows import dict_row

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_PATH = PROJECT_ROOT / "logs" / "api_predictions.jsonl"
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "reports" / "monitoring" / "monitoring_summary.json"
DEFAULT_DRIFT_HTML_PATH = PROJECT_ROOT / "reports" / "monitoring" / "data_drift_report.html"
DEFAULT_DATABASE_URL = os.getenv(
    "MONITORING_DATABASE_URL",
    "postgresql://projet8:projet8@localhost:5433/projet8_monitoring",
)


def load_jsonl_events(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []

    events: list[dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))

    return events


def load_postgres_events(database_url: str, limit: int) -> list[dict[str, Any]]:
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
        LIMIT %(limit)s
    """

    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, {"limit": limit})
            rows = cursor.fetchall()

    events: list[dict[str, Any]] = []
    for row in rows:
        event = dict(row)
        if event.get("timestamp") is not None:
            event["timestamp"] = event["timestamp"].isoformat()
        events.append(event)

    return events


def load_summary(summary_path: Path) -> dict[str, Any] | None:
    if not summary_path.exists():
        return None

    return json.loads(summary_path.read_text(encoding="utf-8"))


def events_to_frame(events: list[dict[str, Any]]) -> pd.DataFrame:
    rows = [
        {
            "timestamp": event.get("timestamp"),
            "endpoint": event.get("endpoint"),
            "status": event.get("status"),
            "client_id": event.get("client_id"),
            "score": event.get("score"),
            "decision": event.get("decision"),
            "latency_ms": event.get("latency_ms"),
            "preprocessing_latency_ms": event.get("preprocessing_latency_ms"),
            "inference_latency_ms": event.get("inference_latency_ms"),
            "error_type": event.get("error_type"),
        }
        for event in events
    ]

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        frame = frame.sort_values("timestamp")

    return frame


def display_operational_metrics(summary: dict[str, Any]) -> None:
    operational = summary.get("operational", {})
    latency = operational.get("latency_ms", {})
    scores = operational.get("scores", {})

    columns = st.columns(5)
    columns[0].metric("Événements", operational.get("total_events", 0))
    columns[1].metric("Succès", operational.get("success_count", 0))
    columns[2].metric("Erreurs", operational.get("error_count", 0))
    columns[3].metric("Taux d'erreur", f"{operational.get('error_rate', 0):.2%}")
    columns[4].metric("Latence p95", f"{latency.get('p95', 0)} ms")

    columns = st.columns(3)
    columns[0].metric("Latence moyenne", f"{latency.get('mean', 0)} ms")
    columns[1].metric("Score moyen", f"{scores.get('mean', 0)}")
    columns[2].metric("Score p95", f"{scores.get('p95', 0)}")

    decisions = operational.get("decisions", {})
    errors = operational.get("errors", {})

    if decisions:
        st.subheader("Décisions")
        st.bar_chart(pd.Series(decisions, name="count"))

    if errors:
        st.subheader("Types d'erreurs")
        st.bar_chart(pd.Series(errors, name="count"))


def display_drift_summary(summary: dict[str, Any]) -> None:
    drift = summary.get("drift", {})
    status = drift.get("status")

    if status != "computed":
        st.warning(drift.get("reason", "Rapport de drift non calculé."))
        return

    columns = st.columns(3)
    columns[0].metric("Drift détecté", "Oui" if drift.get("drift_detected") else "Non")
    columns[1].metric("Variables driftées", drift.get("drifted_columns_count"))
    columns[2].metric("Part driftée", f"{drift.get('drifted_columns_share', 0):.2%}")

    column_metrics = pd.DataFrame(drift.get("columns", []))
    if not column_metrics.empty:
        st.dataframe(column_metrics, use_container_width=True, hide_index=True)


def display_event_charts(frame: pd.DataFrame) -> None:
    if frame.empty:
        st.info("Aucun événement disponible pour les graphiques.")
        return

    st.subheader("Événements récents")
    st.dataframe(frame.tail(100), use_container_width=True, hide_index=True)

    latency_frame = frame.dropna(subset=["timestamp", "latency_ms"])
    if not latency_frame.empty:
        st.subheader("Latence par appel")
        st.line_chart(latency_frame, x="timestamp", y="latency_ms")

    score_frame = frame.dropna(subset=["score"])
    if not score_frame.empty:
        st.subheader("Distribution des scores")
        st.bar_chart(score_frame["score"].value_counts(bins=10).sort_index())

    status_counts = frame["status"].value_counts()
    if not status_counts.empty:
        st.subheader("Répartition des statuts")
        st.bar_chart(status_counts)


st.set_page_config(
    page_title="Monitoring API scoring",
    page_icon=None,
    layout="wide",
)

st.title("Monitoring API scoring")

with st.sidebar:
    st.header("Sources")
    source = st.radio(
        "Source des événements",
        ["Rapport généré", "JSONL local", "PostgreSQL local"],
    )
    log_path = Path(st.text_input("Fichier JSONL", value=str(DEFAULT_LOG_PATH)))
    summary_path = Path(st.text_input("Synthèse JSON", value=str(DEFAULT_SUMMARY_PATH)))
    drift_html_path = Path(
        st.text_input("Rapport Evidently HTML", value=str(DEFAULT_DRIFT_HTML_PATH))
    )
    database_url = st.text_input("URL PostgreSQL", value=DEFAULT_DATABASE_URL)
    postgres_limit = st.number_input("Nombre d'événements PostgreSQL", 10, 10000, 1000, 10)

summary = load_summary(summary_path)

tab_summary, tab_events, tab_drift = st.tabs(["Synthèse", "Événements", "Drift"])

with tab_summary:
    if summary:
        display_operational_metrics(summary)
    else:
        st.info("Lancez d'abord le script d'analyse pour générer la synthèse de monitoring.")

with tab_events:
    if source == "PostgreSQL local":
        try:
            events = load_postgres_events(database_url, int(postgres_limit))
        except Exception as error:
            st.error(f"Impossible de lire PostgreSQL : {error}")
            events = []
    else:
        events = load_jsonl_events(log_path)

    display_event_charts(events_to_frame(events))

with tab_drift:
    if summary:
        display_drift_summary(summary)

    if drift_html_path.exists():
        st.link_button("Ouvrir le rapport Evidently", str(drift_html_path))
    else:
        st.info("Le rapport HTML Evidently n'est pas encore généré.")
