"""Expérience 2.1 : optimisation du modèle TOP30.

Le script part des 30 variables les plus importantes selon le modèle historique,
optimise légèrement les hyperparamètres LightGBM, règle le seuil métier sur une
validation interne, puis sauvegarde un artefact MLflow et un schéma d'entrée
réduit pour la future API.
"""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import shutil
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import lightgbm as lgb
import mlflow.sklearn
import numpy as np
import pandas as pd
from run_feature_selection_experiment import (
    MODEL_METADATA_PATH,
    PROJECT_ROOT,
    build_model_params,
    compute_metrics,
    format_float,
    format_legacy_reference,
    load_feature_names,
    load_json,
    load_modeling_frame,
    load_ranked_features,
    measure_single_row_latency,
    resolve_default_dataset,
    resolve_default_importance,
)
from sklearn.model_selection import train_test_split

DEFAULT_REPORT_PATH = PROJECT_ROOT / "docs" / "experiments" / "top30_optimization_experiment_2_1.md"
DEFAULT_METRICS_PATH = (
    PROJECT_ROOT / "docs" / "experiments" / "top30_optimization_experiment_2_1_metrics.json"
)
DEFAULT_SEARCH_PATH = (
    PROJECT_ROOT / "docs" / "experiments" / "top30_optimization_experiment_2_1_search.csv"
)
DEFAULT_MODEL_OUTPUT = PROJECT_ROOT / "model" / "artifacts" / "mlflow_model_top30_optimized"
DEFAULT_SCHEMA_OUTPUT = PROJECT_ROOT / "model" / "schema" / "top30_feature_schema.json"
DEFAULT_METADATA_OUTPUT = PROJECT_ROOT / "model" / "schema" / "top30_model_metadata.json"


def iter_parameter_grid() -> list[dict[str, Any]]:
    grid = {
        "learning_rate": [0.02, 0.03],
        "num_leaves": [31, 63],
        "min_child_samples": [50, 100],
        "reg_lambda": [0.0, 1.0],
    }
    keys = list(grid)
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*grid.values())]


def threshold_grid(start: float, stop: float, step: float) -> np.ndarray:
    return np.round(np.arange(start, stop + step / 2, step), 6)


def tune_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    thresholds: np.ndarray,
    false_negative_cost: float,
    false_positive_cost: float,
) -> dict[str, Any]:
    rows = []
    for threshold in thresholds:
        metrics = compute_metrics(
            y_true,
            scores,
            threshold=float(threshold),
            false_negative_cost=false_negative_cost,
            false_positive_cost=false_positive_cost,
        )
        rows.append(metrics)

    threshold_results = pd.DataFrame(rows).sort_values(
        ["business_cost_per_obs", "business_fbeta", "pr_auc"],
        ascending=[True, False, False],
    )
    best = threshold_results.iloc[0].to_dict()
    return {
        "best_threshold": float(best["threshold"]),
        "best_metrics": best,
        "threshold_results": threshold_results,
    }


def train_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    base_params: dict[str, Any],
    overrides: dict[str, Any],
) -> tuple[lgb.LGBMClassifier, float]:
    params = {**base_params, **overrides}
    model = lgb.LGBMClassifier(**params)
    start = perf_counter()
    model.fit(x_train, y_train)
    return model, (perf_counter() - start) * 1000


def evaluate_model(
    model: lgb.LGBMClassifier,
    x_eval: pd.DataFrame,
    y_eval: pd.Series,
    threshold: float,
    false_negative_cost: float,
    false_positive_cost: float,
    single_latency_rows: int,
) -> dict[str, Any]:
    warmup_count = min(20, len(x_eval))
    if warmup_count > 0:
        model.predict_proba(x_eval.iloc[:warmup_count])[:, 1]

    start = perf_counter()
    scores = model.predict_proba(x_eval)[:, 1]
    batch_inference_time_ms = (perf_counter() - start) * 1000

    metrics = compute_metrics(
        y_eval.to_numpy(dtype=int),
        scores,
        threshold=threshold,
        false_negative_cost=false_negative_cost,
        false_positive_cost=false_positive_cost,
    )
    latency = {
        "batch_inference_time_ms": float(batch_inference_time_ms),
        "batch_latency_per_row_ms": float(batch_inference_time_ms / len(x_eval)),
        **measure_single_row_latency(model, x_eval, single_latency_rows),
    }
    return {"scores": scores, "metrics": metrics, "latency": latency}


def run_search(
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
    base_params: dict[str, Any],
    thresholds: np.ndarray,
    false_negative_cost: float,
    false_positive_cost: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    best_payload: dict[str, Any] | None = None

    for index, params in enumerate(iter_parameter_grid(), start=1):
        model, training_time_ms = train_model(x_train, y_train, base_params, params)
        valid_scores = model.predict_proba(x_valid)[:, 1]
        threshold_payload = tune_threshold(
            y_valid.to_numpy(dtype=int),
            valid_scores,
            thresholds=thresholds,
            false_negative_cost=false_negative_cost,
            false_positive_cost=false_positive_cost,
        )
        best_metrics = threshold_payload["best_metrics"]
        row = {
            "search_rank": index,
            **params,
            "training_time_ms": training_time_ms,
            "selected_threshold": threshold_payload["best_threshold"],
            "valid_roc_auc": best_metrics["roc_auc"],
            "valid_pr_auc": best_metrics["pr_auc"],
            "valid_business_fbeta": best_metrics["business_fbeta"],
            "valid_business_cost_per_obs": best_metrics["business_cost_per_obs"],
            "valid_recall": best_metrics["recall"],
            "valid_precision": best_metrics["precision"],
        }
        rows.append(row)

        if best_payload is None:
            best_payload = {"params": params, "row": row}
            continue

        current_key = (
            row["valid_business_cost_per_obs"],
            -row["valid_business_fbeta"],
            -row["valid_pr_auc"],
        )
        best_key = (
            best_payload["row"]["valid_business_cost_per_obs"],
            -best_payload["row"]["valid_business_fbeta"],
            -best_payload["row"]["valid_pr_auc"],
        )
        if current_key < best_key:
            best_payload = {"params": params, "row": row}

    search_results = pd.DataFrame(rows).sort_values(
        ["valid_business_cost_per_obs", "valid_business_fbeta", "valid_pr_auc"],
        ascending=[True, False, False],
    )
    if best_payload is None:
        raise RuntimeError("Aucun modèle n'a été entraîné pendant la recherche.")
    return search_results, best_payload


def save_model_artifacts(
    model: lgb.LGBMClassifier,
    model_output: Path,
    schema_output: Path,
    metadata_output: Path,
    selected_features: list[str],
    ranked_features: pd.DataFrame,
    results: dict[str, Any],
) -> None:
    if model_output.exists():
        shutil.rmtree(model_output)
    mlflow.sklearn.save_model(model, path=str(model_output))

    feature_details = (
        ranked_features.set_index("feature")
        .loc[selected_features, ["rank", "importance", "importance_type"]]
        .reset_index()
    )
    schema = {
        "source": "experience_2_1_top30_optimization",
        "feature_count": len(selected_features),
        "features": feature_details.to_dict(orient="records"),
        "preprocessing": {
            "column_name_normalization": "lowercase_and_special_characters_to_underscores",
            "missing_value_policy": "LightGBM native handling",
        },
    }
    metadata = {
        "model_name": "lightgbm_top30_optimized",
        "model_family": "LightGBM",
        "source_experiment": "2.1",
        "selection_objective": "minimize_business_cost_per_observation",
        "artifact_path": str(model_output.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "schema_path": str(schema_output.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "selected_threshold": results["selected_threshold"],
        "threshold_profiles": results["threshold_profiles"],
        "selected_params": results["selected_params"],
        "holdout_metrics": results["optimized_holdout"]["metrics"],
        "feature_count": len(selected_features),
        "created_at_utc": results["run"]["executed_at_utc"],
    }

    schema_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    schema_output.write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")
    metadata_output.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def build_markdown_report(results: dict[str, Any]) -> str:
    baseline = results["baseline_holdout_metrics"]
    raw = results["raw_top30_holdout"]["metrics"]
    optimized_business = results["optimized_business_threshold_holdout"]["metrics"]
    optimized = results["optimized_holdout"]["metrics"]
    latency = results["optimized_holdout"]["latency"]
    selected_params = results["selected_params"]
    selected_features_lines = "\n".join(f"- `{feature}`" for feature in results["features"])
    latency_per_row = format_float(latency["batch_latency_per_row_ms"], 6)

    baseline_row = " | ".join(
        [
            "Baseline P6",
            "733",
            "0.45",
            format_float(baseline["roc_auc"]),
            format_float(baseline["pr_auc"]),
            format_float(baseline["business_fbeta"]),
            format_float(baseline["business_cost_per_obs"]),
            format_float(baseline["recall"]),
            format_float(baseline["precision"]),
        ]
    )
    raw_row = " | ".join(
        [
            "TOP30 brut",
            "30",
            "0.45",
            format_float(raw["roc_auc"]),
            format_float(raw["pr_auc"]),
            format_float(raw["business_fbeta"]),
            format_float(raw["business_cost_per_obs"]),
            format_float(raw["recall"]),
            format_float(raw["precision"]),
        ]
    )
    optimized_row = " | ".join(
        [
            "TOP30 optimisé - seuil coût",
            "30",
            format_float(results["selected_threshold"], 2),
            format_float(optimized["roc_auc"]),
            format_float(optimized["pr_auc"]),
            format_float(optimized["business_fbeta"]),
            format_float(optimized["business_cost_per_obs"]),
            format_float(optimized["recall"]),
            format_float(optimized["precision"]),
        ]
    )
    optimized_business_row = " | ".join(
        [
            "TOP30 optimisé - seuil rappel",
            "30",
            "0.45",
            format_float(optimized_business["roc_auc"]),
            format_float(optimized_business["pr_auc"]),
            format_float(optimized_business["business_fbeta"]),
            format_float(optimized_business["business_cost_per_obs"]),
            format_float(optimized_business["recall"]),
            format_float(optimized_business["precision"]),
        ]
    )

    return f"""# Expérience 2.1 - Optimisation du TOP30

## Objectif

Cette expérience reprend le modèle simplifié à `30` variables, puis optimise légèrement
les hyperparamètres LightGBM et le seuil de décision métier.

## Protocole

- Données : `{results["dataset_path"]}`
- Importance utilisée : `{results["importance_path"]}`
- Split final : holdout stratifié de `20 %`, `random_state=42`
- Validation interne : `20 %` du jeu d'entraînement, stratifiée
- Nombre de combinaisons testées : `{results["search"]["candidate_count"]}`
- Grille de seuils : `{results["threshold_grid"]["start"]}` à `{results["threshold_grid"]["stop"]}`
  par pas de `{results["threshold_grid"]["step"]}`
- Critère de choix : coût métier moyen minimal sur la validation interne.

## Résultats holdout

| Modèle | Variables | Seuil | ROC-AUC | PR-AUC | F-bêta métier | Coût moyen | Rappel | Précision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| {baseline_row} |
| {raw_row} |
| {optimized_business_row} |
| {optimized_row} |

## Hyperparamètres retenus

```json
{json.dumps(selected_params, indent=2, ensure_ascii=False)}
```

## Latence du modèle optimisé

| Mesure | Valeur |
| --- | ---: |
| Temps d'entraînement final | `{format_float(results["optimized_training_time_ms"])} ms` |
| Temps d'inférence batch holdout | `{format_float(latency["batch_inference_time_ms"])} ms` |
| Latence moyenne par client en batch | `{latency_per_row} ms` |
| Latence mono-client moyenne | `{format_float(latency["single_row_latency_mean_ms"])} ms` |
| Latence mono-client p95 | `{format_float(latency["single_row_latency_p95_ms"])} ms` |

## Variables retenues

{selected_features_lines}

## Lecture

Le modèle TOP30 optimisé devient le candidat principal pour l'API. Il réduit le contrat
d'entrée de `733` à `30` variables, tout en conservant une performance proche de la baseline.

Le critère métier prioritaire du projet est la réduction du coût moyen.
Le seuil par défaut retenu pour le candidat API est donc
`{format_float(results["selected_threshold"], 2)}`.

Deux profils de seuil restent documentés :

- seuil `{format_float(results["selected_threshold"], 2)}` : profil par défaut orienté coût moyen ;
- seuil `0.45` : profil alternatif orienté rappel et F-bêta métier.

Le seuil `{format_float(results["selected_threshold"], 2)}` est issu de la validation interne,
pas du holdout final.

## Artefacts générés

- modèle MLflow : `model/artifacts/mlflow_model_top30_optimized` ;
- schéma d'entrée : `model/schema/top30_feature_schema.json` ;
- métadonnées : `model/schema/top30_model_metadata.json` ;
- recherche hyperparamètres : `docs/experiments/top30_optimization_experiment_2_1_search.csv` ;
- métriques détaillées : `docs/experiments/top30_optimization_experiment_2_1_metrics.json`.

## Environnement

- Date d'exécution UTC : `{results["run"]["executed_at_utc"]}`
- Python : `{results["run"]["python_version"]}`
- Système : `{results["run"]["platform"]}`
"""


def run_experiment(args: argparse.Namespace) -> tuple[dict[str, Any], pd.DataFrame]:
    metadata = load_json(MODEL_METADATA_PATH)
    feature_names = load_feature_names()
    dataset_path = args.dataset or resolve_default_dataset(metadata)
    importance_path = args.importance or resolve_default_importance(metadata)
    frame, diagnostics = load_modeling_frame(dataset_path, feature_names)
    ranked_features = load_ranked_features(importance_path, set(feature_names))
    selected_features = ranked_features.head(args.feature_count)["feature"].tolist()

    x = frame.loc[:, selected_features]
    y = frame["target"].astype(int)
    x_model, x_holdout, y_model, y_holdout = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_model,
        y_model,
        test_size=args.validation_size,
        random_state=args.random_state,
        stratify=y_model,
    )

    false_negative_cost = float(metadata["business_cost"]["false_negative_cost"])
    false_positive_cost = float(metadata["business_cost"]["false_positive_cost"])
    default_threshold = float(metadata["decision_threshold_business"])
    base_params = build_model_params(args.random_state)
    thresholds = threshold_grid(args.threshold_start, args.threshold_stop, args.threshold_step)

    raw_model, raw_training_time_ms = train_model(x_model, y_model, base_params, {})
    raw_holdout = evaluate_model(
        raw_model,
        x_holdout,
        y_holdout,
        threshold=default_threshold,
        false_negative_cost=false_negative_cost,
        false_positive_cost=false_positive_cost,
        single_latency_rows=args.single_latency_rows,
    )

    search_results, best_payload = run_search(
        x_train,
        x_valid,
        y_train,
        y_valid,
        base_params,
        thresholds=thresholds,
        false_negative_cost=false_negative_cost,
        false_positive_cost=false_positive_cost,
    )
    selected_params = best_payload["params"]
    selected_threshold = float(best_payload["row"]["selected_threshold"])

    final_model, optimized_training_time_ms = train_model(
        x_model,
        y_model,
        base_params,
        selected_params,
    )
    optimized_holdout = evaluate_model(
        final_model,
        x_holdout,
        y_holdout,
        threshold=selected_threshold,
        false_negative_cost=false_negative_cost,
        false_positive_cost=false_positive_cost,
        single_latency_rows=args.single_latency_rows,
    )
    optimized_business_threshold_holdout = evaluate_model(
        final_model,
        x_holdout,
        y_holdout,
        threshold=default_threshold,
        false_negative_cost=false_negative_cost,
        false_positive_cost=false_positive_cost,
        single_latency_rows=args.single_latency_rows,
    )

    results = {
        "run": {
            "executed_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "random_state": int(args.random_state),
        },
        "dataset_path": format_legacy_reference(dataset_path, metadata),
        "importance_path": format_legacy_reference(importance_path, metadata),
        "data": {
            **diagnostics,
            "model_rows": int(len(x_model)),
            "holdout_rows": int(len(x_holdout)),
            "validation_rows": int(len(x_valid)),
            "holdout_target_rate": float(y_holdout.mean()),
        },
        "feature_count": int(args.feature_count),
        "features": selected_features,
        "threshold_grid": {
            "start": float(args.threshold_start),
            "stop": float(args.threshold_stop),
            "step": float(args.threshold_step),
        },
        "business_cost": {
            "false_negative_cost": false_negative_cost,
            "false_positive_cost": false_positive_cost,
        },
        "baseline_holdout_metrics": metadata["holdout_metrics"]["tuned_business_threshold_0_45"],
        "raw_top30_training_time_ms": float(raw_training_time_ms),
        "raw_top30_holdout": {
            "threshold": default_threshold,
            "metrics": raw_holdout["metrics"],
            "latency": raw_holdout["latency"],
        },
        "search": {
            "candidate_count": int(len(search_results)),
            "best_validation_row": best_payload["row"],
        },
        "selected_params": selected_params,
        "selected_threshold": selected_threshold,
        "threshold_profiles": {
            "recall_priority": default_threshold,
            "cost_priority": selected_threshold,
        },
        "optimized_training_time_ms": float(optimized_training_time_ms),
        "optimized_business_threshold_holdout": {
            "threshold": default_threshold,
            "metrics": optimized_business_threshold_holdout["metrics"],
            "latency": optimized_business_threshold_holdout["latency"],
        },
        "optimized_holdout": {
            "threshold": selected_threshold,
            "metrics": optimized_holdout["metrics"],
            "latency": optimized_holdout["latency"],
        },
    }

    save_model_artifacts(
        final_model,
        args.model_output,
        args.schema_output,
        args.metadata_output,
        selected_features,
        ranked_features,
        results,
    )
    return results, search_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimise le modèle LightGBM TOP30.")
    parser.add_argument("--dataset", type=Path, default=None, help="Chemin du parquet préparé.")
    parser.add_argument("--importance", type=Path, default=None, help="CSV d'importance native.")
    parser.add_argument("--feature-count", type=int, default=30)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--validation-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--threshold-start", type=float, default=0.10)
    parser.add_argument("--threshold-stop", type=float, default=0.90)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--single-latency-rows", type=int, default=100)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--search-output", type=Path, default=DEFAULT_SEARCH_PATH)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_OUTPUT)
    parser.add_argument("--schema-output", type=Path, default=DEFAULT_SCHEMA_OUTPUT)
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_METADATA_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results, search_results = run_experiment(args)

    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.search_output.parent.mkdir(parents=True, exist_ok=True)

    args.report_output.write_text(build_markdown_report(results), encoding="utf-8")
    args.metrics_output.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    search_results.to_csv(args.search_output, index=False)

    print(f"Rapport écrit : {args.report_output}")
    print(f"Métriques écrites : {args.metrics_output}")
    print(f"Recherche écrite : {args.search_output}")
    print(f"Modèle sauvegardé : {args.model_output}")
    print(f"Schéma sauvegardé : {args.schema_output}")


if __name__ == "__main__":
    main()
