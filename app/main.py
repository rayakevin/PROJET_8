from time import perf_counter
from typing import Annotated, Any

from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app.services.inference_service import InferenceService
from app.services.monitoring_service import MonitoringService
from app.services.preprocessing_service import PreprocessingService

app = FastAPI(
    title="Credit Scoring API",
    description="API de scoring crédit basée sur un modèle LightGBM TOP30 optimisé.",
    version="0.1.0",
)

preprocessing_service = PreprocessingService()
inference_service = InferenceService()
monitoring_service = MonitoringService()


RAW_DATA_EXAMPLE: dict[str, Any] = {
    "application": {
        "AMT_CREDIT": 500000.0,
        "AMT_ANNUITY": 25000.0,
        "AMT_GOODS_PRICE": 450000.0,
        "AMT_INCOME_TOTAL": 180000.0,
        "DAYS_BIRTH": -16000,
        "DAYS_EMPLOYED": -2300,
        "DAYS_ID_PUBLISH": -3200,
        "EXT_SOURCE_1": 0.51,
        "EXT_SOURCE_2": 0.62,
        "EXT_SOURCE_3": 0.31,
        "OWN_CAR_AGE": 5.0,
        "CODE_GENDER": "M",
    },
    "bureau": [
        {
            "CREDIT_ACTIVE": "Active",
            "DAYS_CREDIT": -120,
            "DAYS_CREDIT_ENDDATE": 300,
            "AMT_CREDIT_SUM": 120000.0,
            "AMT_CREDIT_SUM_DEBT": 50000.0,
            "AMT_CREDIT_MAX_OVERDUE": 0.0,
        }
    ],
    "previous_applications": [
        {
            "NAME_CONTRACT_STATUS": "Approved",
            "CNT_PAYMENT": 12.0,
        }
    ],
    "installments_payments": [
        {
            "AMT_INSTALMENT": 10000.0,
            "AMT_PAYMENT": 10000.0,
            "DAYS_INSTALMENT": -40,
            "DAYS_ENTRY_PAYMENT": -39,
        }
    ],
    "credit_card_balance": [
        {
            "CNT_DRAWINGS_ATM_CURRENT": 1.0,
        }
    ],
}

SECOND_RAW_DATA_EXAMPLE: dict[str, Any] = {
    **RAW_DATA_EXAMPLE,
    "application": {
        **RAW_DATA_EXAMPLE["application"],
        "AMT_CREDIT": 320000.0,
        "AMT_ANNUITY": 18000.0,
        "AMT_GOODS_PRICE": 300000.0,
        "AMT_INCOME_TOTAL": 135000.0,
        "DAYS_BIRTH": -18500,
        "DAYS_EMPLOYED": -4200,
        "EXT_SOURCE_1": 0.42,
        "EXT_SOURCE_2": 0.57,
        "EXT_SOURCE_3": 0.48,
        "CODE_GENDER": "F",
    },
}

PREDICT_REQUEST_EXAMPLE: dict[str, Any] = {
    "client_id": 100001,
    "raw_data": RAW_DATA_EXAMPLE,
}

BATCH_REQUEST_EXAMPLE: dict[str, Any] = {
    "clients": [
        PREDICT_REQUEST_EXAMPLE,
        {
            "client_id": 100002,
            "raw_data": SECOND_RAW_DATA_EXAMPLE,
        },
    ]
}


class PredictRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": PREDICT_REQUEST_EXAMPLE})

    client_id: int | None = Field(
        default=None,
        description="Identifiant optionnel du client.",
    )
    raw_data: dict[str, Any] = Field(
        ...,
        description="Données brutes métier utilisées pour calculer les features TOP30.",
    )


class BatchClientRequest(BaseModel):
    client_id: int | None = Field(
        default=None,
        description="Identifiant optionnel du client.",
    )
    raw_data: dict[str, Any] = Field(
        ...,
        description="Données brutes métier d'un client du lot.",
    )


class BatchPredictRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": BATCH_REQUEST_EXAMPLE})

    clients: list[BatchClientRequest] = Field(
        ...,
        min_length=1,
        description="Liste non vide de clients à scorer.",
    )


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
def predict(
    request: Annotated[
        PredictRequest,
        Body(
            openapi_examples={
                "client_valide": {
                    "summary": "Client brut valide",
                    "description": "Exemple complet de payload brut pour un client.",
                    "value": PREDICT_REQUEST_EXAMPLE,
                }
            }
        ),
    ],
) -> dict[str, Any]:
    request_id = monitoring_service.create_request_id()
    start = perf_counter()

    try:
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

        monitoring_service.log_prediction_success(
            request_id=request_id,
            endpoint="/predict",
            client_id=request.client_id,
            features=features,
            prediction=prediction,
        )

        return prediction

    except ValueError as error:
        latency_ms = (perf_counter() - start) * 1000
        monitoring_service.log_prediction_error(
            request_id=request_id,
            endpoint="/predict",
            client_id=request.client_id,
            error=error,
            latency_ms=round(latency_ms, 3),
        )
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/predict/batch")
def predict_batch(
    request: Annotated[
        BatchPredictRequest,
        Body(
            openapi_examples={
                "lot_valide": {
                    "summary": "Lot de clients valide",
                    "description": "Exemple de batch avec deux clients bruts.",
                    "value": BATCH_REQUEST_EXAMPLE,
                }
            }
        ),
    ],
) -> dict[str, Any]:
    request_id = monitoring_service.create_request_id()
    start = perf_counter()
    current_client_id = None

    try:
        predictions = []

        preprocessing_latency_ms = 0.0
        inference_latency_ms = 0.0

        for client in request.clients:
            current_client_id = client.client_id
            preprocessing_start = perf_counter()
            features = preprocessing_service.transform(client.raw_data)
            client_preprocessing_latency_ms = (perf_counter() - preprocessing_start) * 1000
            preprocessing_latency_ms += client_preprocessing_latency_ms

            inference_start = perf_counter()
            prediction = inference_service.predict(features, client_id=client.client_id)
            client_inference_latency_ms = (perf_counter() - inference_start) * 1000
            inference_latency_ms += client_inference_latency_ms

            prediction["latency_ms"] = round(
                client_preprocessing_latency_ms + client_inference_latency_ms,
                3,
            )
            prediction["preprocessing_latency_ms"] = round(client_preprocessing_latency_ms, 3)
            prediction["inference_latency_ms"] = round(client_inference_latency_ms, 3)

            monitoring_service.log_prediction_success(
                request_id=request_id,
                endpoint="/predict/batch",
                client_id=client.client_id,
                features=features,
                prediction=prediction,
            )

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
        latency_ms = (perf_counter() - start) * 1000
        monitoring_service.log_prediction_error(
            request_id=request_id,
            endpoint="/predict/batch",
            client_id=current_client_id,
            error=error,
            latency_ms=round(latency_ms, 3),
        )
        raise HTTPException(status_code=422, detail=str(error)) from error
