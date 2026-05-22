"""Mesure la baseline historique du modèle importé.

Cette expérience correspond à l'étape 2.1 bis, expérience 1.
Elle évalue le modèle P6 importé sans le modifier afin de disposer d'un point
de comparaison avant la simplification du nombre de variables.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "model" / "artifacts" / "mlflow_model"
FEATURE_SCHEMA_PATH = PROJECT_ROOT / "model" / "schema" / "feature_schema.json"
MODEL_METADATA_PATH = PROJECT_ROOT / "model" / "schema" / "model_metadata.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "docs" / "experiments" / "baseline_experiment_1.md"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "docs" / "experiments" / "baseline_experiment_1_metrics.json"
DEFAULT_DATASET_PATH = (
    PROJECT_ROOT / "data" / "reference" / "application_train_modeling_sample.parquet"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_column_name(name: str) -> str:
    normalized = name.lower().strip()
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized)
    return normalized.strip("_")


def load_feature_names() -> list[str]:
    schema = load_json(FEATURE_SCHEMA_PATH)
    return [feature["name"] for feature in schema["features"]]


def resolve_default_dataset() -> Path:
    return DEFAULT_DATASET_PATH


def format_dataset_reference(dataset_path: Path) -> str:
    try:
        return dataset_path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(dataset_path)


def load_modeling_frame(
    dataset_path: Path,
    feature_names: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Jeu de données introuvable : {dataset_path}")

    frame = pd.read_parquet(dataset_path)
    raw_shape = frame.shape
    frame.columns = [normalize_column_name(column) for column in frame.columns]

    required_columns = ["sk_id_curr", "target", *feature_names]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        preview = ", ".join(missing_columns[:20])
        raise ValueError(f"Colonnes manquantes après normalisation : {preview}")

    useful_columns = ["sk_id_curr", "target", *feature_names]
    aligned_frame = frame.loc[:, useful_columns].copy()
    diagnostics = {
        "raw_rows": int(raw_shape[0]),
        "raw_columns": int(raw_shape[1]),
        "aligned_rows": int(aligned_frame.shape[0]),
        "aligned_columns": int(aligned_frame.shape[1]),
    }
    return aligned_frame, diagnostics


def sample_frame(frame: pd.DataFrame, sample_size: int, random_state: int) -> pd.DataFrame:
    if sample_size <= 0 or sample_size >= len(frame):
        return frame

    sampled_frame, _ = train_test_split(
        frame,
        train_size=sample_size,
        random_state=random_state,
        stratify=frame["target"],
    )
    return sampled_frame.sort_index()


def compute_business_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    false_negative_cost: float,
    false_positive_cost: float,
) -> dict[str, float | int]:
    false_negatives = int(((y_true == 1) & (y_pred == 0)).sum())
    false_positives = int(((y_true == 0) & (y_pred == 1)).sum())
    total_cost = false_negatives * false_negative_cost + false_positives * false_positive_cost
    return {
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "business_cost": float(total_cost),
        "business_cost_per_obs": float(total_cost / len(y_true)),
    }


def compute_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    false_negative_cost: float,
    false_positive_cost: float,
) -> dict[str, float | int]:
    y_pred = (scores >= threshold).astype(int)
    beta = float(np.sqrt(false_negative_cost / false_positive_cost))

    metrics: dict[str, float | int] = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "business_fbeta": float(fbeta_score(y_true, y_pred, beta=beta, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
    }
    metrics.update(
        compute_business_cost(
            y_true,
            y_pred,
            false_negative_cost=false_negative_cost,
            false_positive_cost=false_positive_cost,
        )
    )
    return metrics


def measure_single_row_latency(
    model: Any,
    features: pd.DataFrame,
    row_count: int,
) -> dict[str, float]:
    latencies_ms: list[float] = []
    limit = min(row_count, len(features))

    for index in range(limit):
        start = perf_counter()
        model.predict_proba(features.iloc[[index]])[:, 1]
        latencies_ms.append((perf_counter() - start) * 1000)

    latency_array = np.asarray(latencies_ms, dtype=float)
    return {
        "single_row_count": int(limit),
        "single_row_latency_mean_ms": float(latency_array.mean()),
        "single_row_latency_p50_ms": float(np.percentile(latency_array, 50)),
        "single_row_latency_p95_ms": float(np.percentile(latency_array, 95)),
    }


def format_float(value: Any, digits: int = 4) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build_markdown_report(results: dict[str, Any]) -> str:
    metrics = results["metrics"]
    latency = results["latency"]
    historical = results["historical_holdout_metrics"]

    metric_rows = [
        ("ROC-AUC", metrics["roc_auc"], historical.get("roc_auc")),
        ("PR-AUC", metrics["pr_auc"], historical.get("pr_auc")),
        ("F-bêta métier", metrics["business_fbeta"], historical.get("business_fbeta")),
        (
            "Coût métier moyen",
            metrics["business_cost_per_obs"],
            historical.get("business_cost_per_obs"),
        ),
        ("Rappel", metrics["recall"], historical.get("recall")),
        ("Précision", metrics["precision"], historical.get("precision")),
    ]

    metric_table = "\n".join(
        "| "
        + " | ".join(
            [
                label,
                format_float(measured),
                format_float(reference) if reference is not None else "n/a",
            ]
        )
        + " |"
        for label, measured, reference in metric_rows
    )

    latency_per_row = format_float(latency["batch_latency_per_row_ms"], 6)

    return f"""# Expérience 1 - Baseline historique

## Objectif

Cette expérience mesure le comportement du modèle historique importé depuis le projet P6.
Elle sert de point de comparaison avant la réduction du nombre de variables.

## Données utilisées

- Source : `{results["dataset_path"]}`
- Lignes disponibles : `{results["data"]["raw_rows"]}`
- Lignes évaluées : `{results["data"]["evaluated_rows"]}`
- Variables modèle : `{results["model"]["feature_count"]}`
- Taux de défaut dans l'échantillon : `{format_float(results["data"]["target_rate"])}`.

Les colonnes du parquet préparé sont normalisées pour correspondre au schéma du modèle :
minuscules, remplacement des espaces et caractères spéciaux par des underscores, puis alignement
sur `model/schema/feature_schema.json`.

## Modèle évalué

- Nom : `{results["model"]["name"]}`
- Famille : `{results["model"]["family"]}`
- Version : `{results["model"]["version"]}`
- Seuil métier : `{results["model"]["threshold"]}`
- Artefact : `{results["model"]["artifact_path"]}`

## Résultats

Les mesures recalculées ci-dessous sont réalisées sur un échantillon du parquet préparé
`application_train_modeling_sample.parquet`. Elles servent de contrôle technique du modèle
importé et de mesure de latence. La colonne holdout P6 reste la référence de performance à
utiliser pour comparer les futurs modèles simplifiés.

| Métrique | Contrôle local sur échantillon | Référence holdout P6 |
| --- | ---: | ---: |
{metric_table}

## Latence

| Mesure | Valeur |
| --- | ---: |
| Temps de chargement du modèle | `{format_float(latency["model_load_time_ms"])} ms` |
| Temps d'inférence batch | `{format_float(latency["batch_inference_time_ms"])} ms` |
| Latence moyenne par client en batch | `{latency_per_row} ms` |
| Nombre de tests unitaires mono-client | `{latency["single_row_count"]}` |
| Latence mono-client moyenne | `{format_float(latency["single_row_latency_mean_ms"])} ms` |
| Latence mono-client p50 | `{format_float(latency["single_row_latency_p50_ms"])} ms` |
| Latence mono-client p95 | `{format_float(latency["single_row_latency_p95_ms"])} ms` |

## Lecture

Le modèle historique est techniquement exploitable, mais son contrat d'entrée reste trop lourd
pour une API de production : `{results["model"]["feature_count"]}` variables sont nécessaires
pour produire un score. Cette expérience confirme donc que le modèle doit rester une baseline
de comparaison pendant la construction d'un modèle simplifié.

## Environnement

- Date d'exécution UTC : `{results["run"]["executed_at_utc"]}`
- Python : `{results["run"]["python_version"]}`
- Système : `{results["run"]["platform"]}`
- Taille d'échantillon demandée : `{results["run"]["sample_size"]}`
- Graine aléatoire : `{results["run"]["random_state"]}`
"""


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    metadata = load_json(MODEL_METADATA_PATH)
    feature_names = load_feature_names()
    dataset_path = args.dataset or resolve_default_dataset()

    frame, diagnostics = load_modeling_frame(dataset_path, feature_names)
    evaluated_frame = sample_frame(frame, args.sample_size, args.random_state)

    y_true = evaluated_frame["target"].to_numpy(dtype=int)
    features = evaluated_frame.loc[:, feature_names]

    load_start = perf_counter()
    model = mlflow.sklearn.load_model(str(MODEL_DIR))
    model_load_time_ms = (perf_counter() - load_start) * 1000

    warmup_count = min(args.warmup_rows, len(features))
    if warmup_count > 0:
        model.predict_proba(features.iloc[:warmup_count])[:, 1]

    batch_start = perf_counter()
    scores = model.predict_proba(features)[:, 1]
    batch_inference_time_ms = (perf_counter() - batch_start) * 1000

    threshold = float(metadata["decision_threshold_business"])
    false_negative_cost = float(metadata["business_cost"]["false_negative_cost"])
    false_positive_cost = float(metadata["business_cost"]["false_positive_cost"])

    metrics = compute_metrics(
        y_true,
        scores,
        threshold=threshold,
        false_negative_cost=false_negative_cost,
        false_positive_cost=false_positive_cost,
    )
    single_latency = measure_single_row_latency(model, features, args.single_latency_rows)

    latency = {
        "model_load_time_ms": float(model_load_time_ms),
        "batch_inference_time_ms": float(batch_inference_time_ms),
        "batch_latency_per_row_ms": float(batch_inference_time_ms / len(features)),
        **single_latency,
    }

    return {
        "run": {
            "executed_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "sample_size": int(args.sample_size),
            "random_state": int(args.random_state),
        },
        "dataset_path": format_dataset_reference(dataset_path),
        "model": {
            "name": metadata["selected_model_name"],
            "family": metadata["selected_model_family"],
            "version": metadata["registered_model_version"],
            "threshold": threshold,
            "feature_count": len(feature_names),
            "artifact_path": str(MODEL_DIR.relative_to(PROJECT_ROOT)),
        },
        "data": {
            **diagnostics,
            "evaluated_rows": int(len(evaluated_frame)),
            "target_rate": float(np.mean(y_true)),
        },
        "metrics": metrics,
        "latency": latency,
        "historical_holdout_metrics": metadata["holdout_metrics"]["tuned_business_threshold_0_45"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mesure la baseline historique du modèle P6.")
    parser.add_argument("--dataset", type=Path, default=None, help="Chemin du parquet préparé.")
    parser.add_argument("--sample-size", type=int, default=5000, help="Nombre de lignes évaluées.")
    parser.add_argument("--random-state", type=int, default=42, help="Graine aléatoire.")
    parser.add_argument(
        "--warmup-rows",
        type=int,
        default=20,
        help="Lignes utilisées pour le warm-up.",
    )
    parser.add_argument(
        "--single-latency-rows",
        type=int,
        default=100,
        help="Nombre d'appels mono-client utilisés pour mesurer la latence.",
    )
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_experiment(args)

    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(build_markdown_report(results), encoding="utf-8")
    args.metrics_output.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Rapport écrit : {args.report_output}")
    print(f"Métriques écrites : {args.metrics_output}")


if __name__ == "__main__":
    main()
