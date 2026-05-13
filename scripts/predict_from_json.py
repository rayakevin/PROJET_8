"""Exécute une prédiction locale à partir d'un fichier JSON de variables.

Format d'entrée attendu :

{
  "client_id": 100001,
  "features": {
    "amt_credit": 500000.0,
    "...": "..."
  }
}

Le fichier d'entrée doit contenir les noms de variables prêts pour le modèle,
listés dans `model/schema/feature_schema.json`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import mlflow.pyfunc
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "model" / "artifacts" / "mlflow_model"
FEATURE_SCHEMA_PATH = PROJECT_ROOT / "model" / "schema" / "feature_schema.json"
MODEL_METADATA_PATH = PROJECT_ROOT / "model" / "schema" / "model_metadata.json"


def load_feature_names() -> list[str]:
    schema = json.loads(FEATURE_SCHEMA_PATH.read_text(encoding="utf-8"))
    return [feature["name"] for feature in schema["features"]]


def load_threshold() -> float:
    metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    return float(metadata["decision_threshold_business"])


def build_frame(features: dict, feature_names: list[str]) -> pd.DataFrame:
    missing_features = [name for name in feature_names if name not in features]
    if missing_features:
        preview = ", ".join(missing_features[:10])
        raise ValueError(f"Variables modèle manquantes : {preview}")
    return pd.DataFrame([{name: features[name] for name in feature_names}])


def predict(payload_path: Path) -> dict:
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    feature_names = load_feature_names()
    threshold = load_threshold()
    frame = build_frame(payload["features"], feature_names)

    model = mlflow.pyfunc.load_model(str(MODEL_DIR))
    start = perf_counter()
    prediction_output = model.predict(frame)
    latency_ms = (perf_counter() - start) * 1000

    score = float(prediction_output[0])
    prediction = int(score >= threshold)
    return {
        "client_id": payload.get("client_id"),
        "score": score,
        "threshold": threshold,
        "prediction": prediction,
        "decision": "high_risk" if prediction else "low_risk",
        "latency_ms": round(latency_ms, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=Path, help="Chemin du fichier JSON de variables.")
    args = parser.parse_args()
    print(json.dumps(predict(args.payload), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
