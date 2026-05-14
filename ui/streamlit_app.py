import json
import os
from typing import Any

import pandas as pd
import requests
import streamlit as st

DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL", "https://rayakevin-projet-8.hf.space")
REQUEST_TIMEOUT_SECONDS = 60


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

SINGLE_PAYLOAD_EXAMPLE: dict[str, Any] = {
    "client_id": 100001,
    "raw_data": RAW_DATA_EXAMPLE,
}

BATCH_PAYLOAD_EXAMPLE: dict[str, Any] = {
    "clients": [
        SINGLE_PAYLOAD_EXAMPLE,
        {
            "client_id": 100002,
            "raw_data": {
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
            },
        },
    ]
}


def pretty_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def parse_json_payload(raw_payload: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError as error:
        st.error(f"JSON invalide : {error}")
        return None

    if not isinstance(payload, dict):
        st.error("Le payload doit être un objet JSON.")
        return None

    return payload


def normalize_api_url(api_base_url: str) -> str:
    return api_base_url.strip().rstrip("/")


def request_api(
    api_base_url: str,
    endpoint: str,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    url = f"{normalize_api_url(api_base_url)}{endpoint}"

    try:
        if method == "POST":
            response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
        else:
            response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.RequestException as error:
        st.error(f"Erreur de connexion à l'API : {error}")
        return None

    if not response.ok:
        st.error(f"Erreur API {response.status_code}")
        st.code(response.text, language="json")
        return None

    return response.json()


def display_prediction(prediction: dict[str, Any]) -> None:
    score = float(prediction["score"])
    decision = prediction["decision"]

    columns = st.columns(4)
    columns[0].metric("Score de défaut", f"{score:.3f}")
    columns[1].metric("Décision", "Risque élevé" if decision == "high_risk" else "Risque faible")
    columns[2].metric("Seuil", f"{prediction['threshold']:.2f}")
    columns[3].metric("Latence totale", f"{prediction['latency_ms']:.1f} ms")

    st.progress(min(max(score, 0.0), 1.0))

    details = {
        "client_id": prediction.get("client_id"),
        "prediction": prediction.get("prediction"),
        "model_version": prediction.get("model_version"),
        "preprocessing_latency_ms": prediction.get("preprocessing_latency_ms"),
        "inference_latency_ms": prediction.get("inference_latency_ms"),
    }
    st.json(details)


def display_batch_predictions(response_body: dict[str, Any]) -> None:
    predictions = response_body.get("predictions", [])
    if not predictions:
        st.warning("Aucune prédiction retournée.")
        return

    frame = pd.DataFrame(predictions)
    st.metric("Clients scorés", response_body["count"])
    st.dataframe(
        frame[
            [
                "client_id",
                "score",
                "threshold",
                "prediction",
                "decision",
                "latency_ms",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    chart_frame = frame[["client_id", "score"]].copy()
    chart_frame["client_id"] = chart_frame["client_id"].astype(str)
    st.bar_chart(chart_frame, x="client_id", y="score")

    st.json(
        {
            "latency_ms": response_body.get("latency_ms"),
            "preprocessing_latency_ms": response_body.get("preprocessing_latency_ms"),
            "inference_latency_ms": response_body.get("inference_latency_ms"),
            "model_version": response_body.get("model_version"),
        }
    )


st.set_page_config(
    page_title="Scoring crédit",
    page_icon=None,
    layout="wide",
)

st.title("Scoring crédit")

with st.sidebar:
    st.header("API")
    api_base_url = st.text_input(
        "URL de l'API",
        value=DEFAULT_API_BASE_URL,
        help="URL du Space FastAPI ou de l'API locale.",
    )

    if st.button("Tester /health", use_container_width=True):
        health = request_api(api_base_url, "/health")
        if health:
            st.success("API disponible")
            st.json(health)

    if st.button("Afficher /model/info", use_container_width=True):
        model_info = request_api(api_base_url, "/model/info")
        if model_info:
            st.json(model_info)

tab_predict, tab_batch, tab_about = st.tabs(
    ["Prédiction client", "Prédiction batch", "Informations"]
)

with tab_predict:
    st.subheader("Prédiction pour un client")
    single_payload_text = st.text_area(
        "Payload JSON",
        value=pretty_json(SINGLE_PAYLOAD_EXAMPLE),
        height=420,
        key="single_payload",
    )

    if st.button("Lancer la prédiction", type="primary", use_container_width=True):
        payload = parse_json_payload(single_payload_text)
        if payload:
            prediction = request_api(api_base_url, "/predict", method="POST", payload=payload)
            if prediction:
                display_prediction(prediction)

with tab_batch:
    st.subheader("Prédiction pour un lot de clients")
    batch_payload_text = st.text_area(
        "Payload JSON batch",
        value=pretty_json(BATCH_PAYLOAD_EXAMPLE),
        height=420,
        key="batch_payload",
    )

    if st.button("Lancer le batch", type="primary", use_container_width=True):
        payload = parse_json_payload(batch_payload_text)
        if payload:
            response_body = request_api(
                api_base_url,
                "/predict/batch",
                method="POST",
                payload=payload,
            )
            if response_body:
                display_batch_predictions(response_body)

with tab_about:
    st.subheader("Contrat d'utilisation")
    st.write(
        "Cette interface appelle l'API FastAPI déployée séparément sur Hugging Face Spaces. "
        "Les payloads attendus sont des données brutes métier proches des tables historiques "
        "Home Credit. L'API réalise ensuite le preprocessing, l'inférence et retourne le score."
    )
    st.code(
        "GET /health\nGET /model/info\nPOST /predict\nPOST /predict/batch",
        language="text",
    )
