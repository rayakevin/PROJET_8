from time import perf_counter
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.services.inference_service import InferenceService
from app.services.preprocessing_service import PreprocessingService

app = FastAPI(
    title="Credit Scoring API",
    description="API de scoring crédit basée sur un modèle LightGBM TOP30 optimisé.",
    version="0.1.0",
)

preprocessing_service = PreprocessingService()
inference_service = InferenceService()


class PredictRequest(BaseModel):
    client_id: int | None = None
    raw_data: dict[str, Any]


class BatchClientRequest(BaseModel):
    client_id: int | None = None
    raw_data: dict[str, Any]


class BatchPredictRequest(BaseModel):
    clients: list[BatchClientRequest] = Field(..., min_length=1)


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "credit-scoring-api",
    }


@app.get("/model/info")
def model_info() -> dict[str, Any]:
    return {
        "model_name": inference_service.metadata["model_name"],
        "model_family": inference_service.metadata.get("model_family"),
        "feature_count": len(inference_service.feature_names),
        "threshold": inference_service.threshold,
        "input_contract": "raw_business_data",
        "internal_feature_contract": "top30_features",
    }


@app.post("/predict")
def predict(request: PredictRequest) -> dict[str, Any]:
    try:
        start = perf_counter()

        preprocessing_start = perf_counter()
        features = preprocessing_service.transform(request.raw_data)
        preprocessing_latency_ms = (perf_counter() - preprocessing_start) * 1000

        inference_start = perf_counter()
        prediction = inference_service.predict(features, client_id=request.client_id)
        inference_latency_ms = (perf_counter() - inference_start) * 1000

        total_latency_ms = (perf_counter() - start) * 1000

        prediction["latency_ms"] = round(total_latency_ms, 3)
        prediction["preprocessing_latency_ms"] = round(preprocessing_latency_ms, 3)
        prediction["inference_latency_ms"] = round(inference_latency_ms, 3)

        return prediction

    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/predict/batch")
def predict_batch(request: BatchPredictRequest) -> dict[str, Any]:
    try:
        start = perf_counter()
        predictions = []

        preprocessing_latency_ms = 0.0
        inference_latency_ms = 0.0

        for client in request.clients:
            preprocessing_start = perf_counter()
            features = preprocessing_service.transform(client.raw_data)
            preprocessing_latency_ms += (perf_counter() - preprocessing_start) * 1000

            inference_start = perf_counter()
            prediction = inference_service.predict(features, client_id=client.client_id)
            inference_latency_ms += (perf_counter() - inference_start) * 1000

            predictions.append(prediction)

        total_latency_ms = (perf_counter() - start) * 1000

        return {
            "predictions": predictions,
            "count": len(predictions),
            "latency_ms": round(total_latency_ms, 3),
            "preprocessing_latency_ms": round(preprocessing_latency_ms, 3),
            "inference_latency_ms": round(inference_latency_ms, 3),
            "model_version": inference_service.metadata["model_name"],
        }

    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
