import json
import math
from pathlib import Path
from time import perf_counter

import mlflow.sklearn
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "model" / "artifacts" / "mlflow_model_top30_optimized"
FEATURE_SCHEMA_PATH = PROJECT_ROOT / "model" / "schema" / "top30_feature_schema.json"
MODEL_METADATA_PATH = PROJECT_ROOT / "model" / "schema" / "top30_model_metadata.json"


class InferenceService:
    def __init__(self) -> None:
        self.model = None
        self.feature_names = self._load_feature_names()
        self.metadata = self._load_metadata()
        self.threshold = float(self.metadata["selected_threshold"])

    def _load_feature_names(self) -> list[str]:
        schema = json.loads(FEATURE_SCHEMA_PATH.read_text(encoding="utf-8"))
        return [item["feature"] for item in schema["features"]]

    def _load_metadata(self) -> dict:
        return json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))

    def load_model(self):
        if self.model is None:
            self.model = mlflow.sklearn.load_model(str(MODEL_DIR))
        return self.model

    def validate_features(self, features: dict) -> dict[str, float]:
        if not isinstance(features, dict):
            raise ValueError("features doit être un objet JSON")

        missing = [name for name in self.feature_names if name not in features]
        if missing:
            raise ValueError(f"Variables modèle manquantes : {', '.join(missing[:10])}")

        validated = {}

        for name in self.feature_names:
            raw_value = features[name]

            if raw_value is None:
                value = float("nan")
            else:
                try:
                    value = float(raw_value)
                except (TypeError, ValueError):
                    raise ValueError(f"La variable {name} doit être numérique") from None

            if math.isinf(value):
                raise ValueError(f"La variable {name} ne peut pas être infinie")

            validated[name] = value

        def is_present(value: float) -> bool:
            return not math.isnan(value)

        if is_present(validated["amt_credit"]) and validated["amt_credit"] < 0:
            raise ValueError("La variable amt_credit ne peut pas être négative")

        if is_present(validated["amt_annuity"]) and validated["amt_annuity"] < 0:
            raise ValueError("La variable amt_annuity ne peut pas être négative")

        if is_present(validated["days_birth"]) and validated["days_birth"] > 0:
            raise ValueError("La variable days_birth doit être négative")

        if is_present(validated["amt_goods_price"]) and validated["amt_goods_price"] < 0:
            raise ValueError("La variable amt_goods_price ne peut pas être négative")

        if is_present(validated["days_employed"]) and validated["days_employed"] > 0:
            raise ValueError("La variable days_employed doit être négative")

        if (
            is_present(validated["annuity_to_income_ratio"])
            and validated["annuity_to_income_ratio"] < 0
        ):
            raise ValueError("La variable annuity_to_income_ratio ne peut pas être négative")

        if (
            is_present(validated["credit_to_annuity_ratio"])
            and validated["credit_to_annuity_ratio"] < 0
        ):
            raise ValueError("La variable credit_to_annuity_ratio ne peut pas être négative")

        if is_present(validated["payment_rate"]) and validated["payment_rate"] < 0:
            raise ValueError("La variable payment_rate ne peut pas être négative")

        return validated

    def build_frame(self, features: dict[str, float]) -> pd.DataFrame:
        return pd.DataFrame([{name: features[name] for name in self.feature_names}])

    def warm_up(self) -> None:
        features = {name: 0.0 for name in self.feature_names}
        features.update(
            {
                "amt_credit": 100000.0,
                "amt_annuity": 10000.0,
                "amt_goods_price": 100000.0,
                "days_birth": -12000.0,
                "days_employed": -1000.0,
                "annuity_to_income_ratio": 0.1,
                "credit_to_annuity_ratio": 10.0,
                "payment_rate": 0.1,
            }
        )
        validated_features = self.validate_features(features)
        frame = self.build_frame(validated_features)
        self.load_model().predict_proba(frame)

    def build_batch_frame(self, clients: list[dict]) -> tuple[pd.DataFrame, list[int | None]]:
        rows = []
        client_ids = []

        for client in clients:
            validated_features = self.validate_features(client["features"])
            rows.append({name: validated_features[name] for name in self.feature_names})
            client_ids.append(client.get("client_id"))

        return pd.DataFrame(rows), client_ids

    def predict(self, features: dict, client_id: int | None = None) -> dict:
        validated_features = self.validate_features(features)
        frame = self.build_frame(validated_features)
        model = self.load_model()

        start = perf_counter()
        score = float(model.predict_proba(frame)[:, 1][0])
        latency_ms = (perf_counter() - start) * 1000

        prediction = int(score >= self.threshold)

        return {
            "client_id": client_id,
            "score": score,
            "threshold": self.threshold,
            "prediction": prediction,
            "decision": "high_risk" if prediction else "low_risk",
            "latency_ms": round(latency_ms, 3),
            "model_version": self.metadata["model_name"],
        }

    def predict_batch(self, clients: list[dict]) -> dict:
        if not clients:
            raise ValueError("Le lot de clients ne peut pas être vide")

        start = perf_counter()
        frame, client_ids = self.build_batch_frame(clients)
        model = self.load_model()
        scores = model.predict_proba(frame)[:, 1]
        latency_ms = (perf_counter() - start) * 1000

        predictions = []
        for client_id, score in zip(client_ids, scores, strict=True):
            score = float(score)
            prediction = int(score >= self.threshold)
            predictions.append(
                {
                    "client_id": client_id,
                    "score": score,
                    "threshold": self.threshold,
                    "prediction": prediction,
                    "decision": "high_risk" if prediction else "low_risk",
                    "model_version": self.metadata["model_name"],
                }
            )

        return {
            "predictions": predictions,
            "count": len(predictions),
            "latency_ms": round(latency_ms, 3),
            "model_version": self.metadata["model_name"],
        }
