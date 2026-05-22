"""Expérience 2 : sélection de variables par importance LightGBM.

Le script réentraîne plusieurs modèles LightGBM simplifiés à partir des variables
les plus importantes du modèle historique P6, puis compare leurs performances
et leur latence à la baseline holdout documentée.
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

import lightgbm as lgb
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
DEFAULT_REPORT_PATH = PROJECT_ROOT / "docs" / "experiments" / "feature_selection_experiment_2.md"
DEFAULT_METRICS_PATH = (
    PROJECT_ROOT / "docs" / "experiments" / "feature_selection_experiment_2_metrics.json"
)
DEFAULT_FEATURES_PATH = (
    PROJECT_ROOT / "docs" / "experiments" / "feature_selection_experiment_2_features.csv"
)
DEFAULT_SUMMARY_PATH = (
    PROJECT_ROOT / "docs" / "experiments" / "feature_selection_experiment_2_summary.csv"
)
DEFAULT_DATASET_PATH = (
    PROJECT_ROOT / "data" / "reference" / "application_train_modeling_sample.parquet"
)
DEFAULT_IMPORTANCE_PATH = (
    PROJECT_ROOT / "data" / "reference" / "lightgbm_bonus_native_importance.csv"
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


def resolve_default_dataset(metadata: dict[str, Any] | None = None) -> Path:
    return DEFAULT_DATASET_PATH


def resolve_default_importance(metadata: dict[str, Any] | None = None) -> Path:
    return DEFAULT_IMPORTANCE_PATH


def format_project_reference(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def format_legacy_reference(path: Path, metadata: dict[str, Any] | None = None) -> str:
    return format_project_reference(path)


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


def load_ranked_features(importance_path: Path, available_features: set[str]) -> pd.DataFrame:
    if not importance_path.exists():
        raise FileNotFoundError(f"Fichier d'importance introuvable : {importance_path}")

    importance = pd.read_csv(importance_path)
    importance["feature"] = importance["feature"].map(normalize_column_name)
    importance = importance[importance["feature"].isin(available_features)].copy()
    importance = importance.sort_values("importance", ascending=False).reset_index(drop=True)
    importance["rank"] = np.arange(1, len(importance) + 1)
    return importance


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


def build_model_params(random_state: int) -> dict[str, Any]:
    historical_model = mlflow.sklearn.load_model(str(MODEL_DIR))
    historical_params = historical_model.get_params()
    return {
        "objective": "binary",
        "learning_rate": historical_params.get("learning_rate", 0.03),
        "n_estimators": historical_params.get("n_estimators", 298),
        "num_leaves": historical_params.get("num_leaves", 63),
        "min_child_samples": historical_params.get("min_child_samples", 100),
        "reg_lambda": historical_params.get("reg_lambda", 0.0),
        "subsample": historical_params.get("subsample", 0.8),
        "colsample_bytree": historical_params.get("colsample_bytree", 0.8),
        "class_weight": historical_params.get("class_weight", "balanced"),
        "random_state": random_state,
        "n_jobs": -1,
        "verbosity": -1,
    }


def train_and_evaluate_candidate(
    feature_count: int,
    selected_features: list[str],
    x_train: pd.DataFrame,
    x_holdout: pd.DataFrame,
    y_train: pd.Series,
    y_holdout: pd.Series,
    model_params: dict[str, Any],
    threshold: float,
    false_negative_cost: float,
    false_positive_cost: float,
    single_latency_rows: int,
) -> dict[str, Any]:
    model = lgb.LGBMClassifier(**model_params)
    x_train_selected = x_train.loc[:, selected_features]
    x_holdout_selected = x_holdout.loc[:, selected_features]

    train_start = perf_counter()
    model.fit(x_train_selected, y_train)
    training_time_ms = (perf_counter() - train_start) * 1000

    warmup_count = min(20, len(x_holdout_selected))
    if warmup_count > 0:
        model.predict_proba(x_holdout_selected.iloc[:warmup_count])[:, 1]

    batch_start = perf_counter()
    scores = model.predict_proba(x_holdout_selected)[:, 1]
    batch_inference_time_ms = (perf_counter() - batch_start) * 1000

    metrics = compute_metrics(
        y_holdout.to_numpy(dtype=int),
        scores,
        threshold=threshold,
        false_negative_cost=false_negative_cost,
        false_positive_cost=false_positive_cost,
    )
    latency = {
        "training_time_ms": float(training_time_ms),
        "batch_inference_time_ms": float(batch_inference_time_ms),
        "batch_latency_per_row_ms": float(batch_inference_time_ms / len(x_holdout_selected)),
        **measure_single_row_latency(model, x_holdout_selected, single_latency_rows),
    }

    return {
        "feature_count": int(feature_count),
        "features": selected_features,
        "metrics": metrics,
        "latency": latency,
    }


def format_float(value: Any, digits: int = 4) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build_summary_frame(results: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for candidate in results["candidates"]:
        metrics = candidate["metrics"]
        latency = candidate["latency"]
        rows.append(
            {
                "feature_count": candidate["feature_count"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "business_fbeta": metrics["business_fbeta"],
                "business_cost_per_obs": metrics["business_cost_per_obs"],
                "recall": metrics["recall"],
                "precision": metrics["precision"],
                "single_row_latency_p95_ms": latency["single_row_latency_p95_ms"],
                "batch_latency_per_row_ms": latency["batch_latency_per_row_ms"],
                "training_time_ms": latency["training_time_ms"],
            }
        )
    return pd.DataFrame(rows)


def build_features_frame(results: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for candidate in results["candidates"]:
        for rank, feature in enumerate(candidate["features"], start=1):
            rows.append(
                {
                    "feature_count": candidate["feature_count"],
                    "rank": rank,
                    "feature": feature,
                }
            )
    return pd.DataFrame(rows)


def build_markdown_report(results: dict[str, Any]) -> str:
    baseline = results["baseline_holdout_metrics"]
    summary = build_summary_frame(results)
    best_pr_auc = summary.sort_values("pr_auc", ascending=False).iloc[0]
    best_compromise = (
        summary[summary["feature_count"] <= 50]
        .sort_values(
            ["pr_auc", "business_fbeta"],
            ascending=False,
        )
        .iloc[0]
    )

    metric_rows = []
    for _, row in summary.iterrows():
        metric_rows.append(
            "| "
            + " | ".join(
                [
                    str(int(row["feature_count"])),
                    format_float(row["roc_auc"]),
                    format_float(row["pr_auc"]),
                    format_float(row["business_fbeta"]),
                    format_float(row["business_cost_per_obs"]),
                    format_float(row["recall"]),
                    format_float(row["precision"]),
                    format_float(row["single_row_latency_p95_ms"]),
                ]
            )
            + " |"
        )

    top_features = results["candidates"][0]["features"]
    top_features_lines = "\n".join(f"- `{feature}`" for feature in top_features)
    result_table_header = (
        "| Variables | ROC-AUC | PR-AUC | F-bêta métier | Coût métier moyen | "
        "Rappel | Précision | Latence p95 |\n"
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )

    return f"""# Expérience 2 - Sélection par importance LightGBM

## Objectif

Cette expérience évalue plusieurs modèles LightGBM simplifiés entraînés avec les variables
les plus importantes du modèle historique.

## Protocole

- Données : `{results["dataset_path"]}`
- Importance utilisée : `{results["importance_path"]}`
- Split : holdout stratifié de `20 %`, `random_state=42`
- Seuil métier appliqué : `{results["threshold"]}`
- Coûts métier : faux négatif `{results["business_cost"]["false_negative_cost"]}`,
  faux positif `{results["business_cost"]["false_positive_cost"]}`
- Tailles testées : `{", ".join(str(size) for size in results["feature_counts"])}`

Le seuil métier n'est pas réoptimisé dans cette expérience. Il reste fixé à `0.45` pour comparer
les modèles réduits avec la baseline historique dans un cadre simple et stable.

## Référence baseline P6

| Métrique | Baseline historique holdout |
| --- | ---: |
| ROC-AUC | `{format_float(baseline["roc_auc"])}` |
| PR-AUC | `{format_float(baseline["pr_auc"])}` |
| F-bêta métier | `{format_float(baseline["business_fbeta"])}` |
| Coût métier moyen | `{format_float(baseline["business_cost_per_obs"])}` |
| Rappel | `{format_float(baseline["recall"])}` |
| Précision | `{format_float(baseline["precision"])}` |

## Résultats des modèles simplifiés

{result_table_header}
{chr(10).join(metric_rows)}

## Lecture provisoire

- Meilleur score PR-AUC : modèle top `{int(best_pr_auc["feature_count"])}` variables.
- Meilleur compromis sous `50` variables : modèle top
  `{int(best_compromise["feature_count"])}` variables.
- La réduction du contrat d'entrée est forte dès le top `20`, mais le choix final devra intégrer
  la performance, la stabilité, la facilité de production des variables et l'explicabilité.

## Top 20 variables natives

{top_features_lines}

## Livrables générés

- métriques détaillées : `docs/experiments/feature_selection_experiment_2_metrics.json` ;
- synthèse tabulaire : `docs/experiments/feature_selection_experiment_2_summary.csv` ;
- listes de variables : `docs/experiments/feature_selection_experiment_2_features.csv`.

## Environnement

- Date d'exécution UTC : `{results["run"]["executed_at_utc"]}`
- Python : `{results["run"]["python_version"]}`
- Système : `{results["run"]["platform"]}`
"""


def parse_feature_counts(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    metadata = load_json(MODEL_METADATA_PATH)
    feature_names = load_feature_names()
    dataset_path = args.dataset or resolve_default_dataset()
    importance_path = args.importance or resolve_default_importance()

    frame, diagnostics = load_modeling_frame(dataset_path, feature_names)
    ranked_features = load_ranked_features(importance_path, set(feature_names))

    x = frame.loc[:, feature_names]
    y = frame["target"].astype(int)
    x_train, x_holdout, y_train, y_holdout = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    threshold = float(metadata["decision_threshold_business"])
    false_negative_cost = float(metadata["business_cost"]["false_negative_cost"])
    false_positive_cost = float(metadata["business_cost"]["false_positive_cost"])
    model_params = build_model_params(args.random_state)

    candidates = []
    for feature_count in args.feature_counts:
        selected_features = ranked_features.head(feature_count)["feature"].tolist()
        candidates.append(
            train_and_evaluate_candidate(
                feature_count=feature_count,
                selected_features=selected_features,
                x_train=x_train,
                x_holdout=x_holdout,
                y_train=y_train,
                y_holdout=y_holdout,
                model_params=model_params,
                threshold=threshold,
                false_negative_cost=false_negative_cost,
                false_positive_cost=false_positive_cost,
                single_latency_rows=args.single_latency_rows,
            )
        )

    return {
        "run": {
            "executed_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "random_state": int(args.random_state),
            "test_size": float(args.test_size),
        },
        "dataset_path": format_project_reference(dataset_path),
        "importance_path": format_project_reference(importance_path),
        "data": {
            **diagnostics,
            "train_rows": int(len(x_train)),
            "holdout_rows": int(len(x_holdout)),
            "train_target_rate": float(y_train.mean()),
            "holdout_target_rate": float(y_holdout.mean()),
        },
        "feature_counts": args.feature_counts,
        "threshold": threshold,
        "business_cost": {
            "false_negative_cost": false_negative_cost,
            "false_positive_cost": false_positive_cost,
        },
        "model_params": model_params,
        "baseline_holdout_metrics": metadata["holdout_metrics"]["tuned_business_threshold_0_45"],
        "candidates": candidates,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare des modèles réduits par importance LightGBM.",
    )
    parser.add_argument("--dataset", type=Path, default=None, help="Chemin du parquet préparé.")
    parser.add_argument("--importance", type=Path, default=None, help="CSV d'importance native.")
    parser.add_argument("--feature-counts", type=parse_feature_counts, default="20,30,50,100")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--single-latency-rows", type=int, default=100)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--features-output", type=Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_experiment(args)

    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.features_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)

    args.report_output.write_text(build_markdown_report(results), encoding="utf-8")
    args.metrics_output.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    build_summary_frame(results).to_csv(args.summary_output, index=False)
    build_features_frame(results).to_csv(args.features_output, index=False)

    print(f"Rapport écrit : {args.report_output}")
    print(f"Métriques écrites : {args.metrics_output}")
    print(f"Synthèse écrite : {args.summary_output}")
    print(f"Variables écrites : {args.features_output}")


if __name__ == "__main__":
    main()
