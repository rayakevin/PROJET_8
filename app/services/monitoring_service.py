import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_PATH = PROJECT_ROOT / "logs" / "api_predictions.jsonl"


class MonitoringService:
    def __init__(self, log_path: Path = DEFAULT_LOG_PATH) -> None:
        self.log_path = log_path

    def create_request_id(self) -> str:
        return str(uuid4())

    def write_event(self, event: dict[str, Any]) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        event_with_timestamp = {
            "timestamp": datetime.now(UTC).isoformat(),
            **event,
        }
        clean_event = self._json_safe(event_with_timestamp)

        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(clean_event, ensure_ascii=False) + "\n")

    def log_prediction_success(
        self,
        *,
        request_id: str,
        endpoint: str,
        client_id: int | None,
        features: dict[str, float],
        prediction: dict[str, Any],
    ) -> None:
        self.write_event(
            {
                "request_id": request_id,
                "endpoint": endpoint,
                "status": "success",
                "client_id": client_id,
                "model_version": prediction.get("model_version"),
                "features": features,
                "score": prediction.get("score"),
                "threshold": prediction.get("threshold"),
                "prediction": prediction.get("prediction"),
                "decision": prediction.get("decision"),
                "latency_ms": prediction.get("latency_ms"),
                "preprocessing_latency_ms": prediction.get("preprocessing_latency_ms"),
                "inference_latency_ms": prediction.get("inference_latency_ms"),
                "error_type": None,
                "error_message": None,
            }
        )

    def log_prediction_error(
        self,
        *,
        request_id: str,
        endpoint: str,
        client_id: int | None,
        error: Exception,
        latency_ms: float | None = None,
    ) -> None:
        self.write_event(
            {
                "request_id": request_id,
                "endpoint": endpoint,
                "status": "error",
                "client_id": client_id,
                "model_version": None,
                "features": None,
                "score": None,
                "threshold": None,
                "prediction": None,
                "decision": None,
                "latency_ms": latency_ms,
                "preprocessing_latency_ms": None,
                "inference_latency_ms": None,
                "error_type": error.__class__.__name__,
                "error_message": str(error),
            }
        )

    def _json_safe(self, value: Any) -> Any:
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return None

            return value

        if isinstance(value, dict):
            return {key: self._json_safe(item) for key, item in value.items()}

        if isinstance(value, list):
            return [self._json_safe(item) for item in value]

        return value
