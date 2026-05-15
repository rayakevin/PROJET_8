import pytest

from app.services.inference_service import InferenceService


def valid_features() -> dict[str, float]:
    return {
        "ext_sources_mean": 0.48,
        "credit_to_annuity_ratio": 20.0,
        "days_birth": -16000.0,
        "ext_source_1": 0.51,
        "ext_source_2": 0.62,
        "payment_rate": 0.05,
        "credit_to_goods_ratio": 1.11,
        "amt_annuity": 25000.0,
        "days_employed": -2300.0,
        "approved_cnt_payment_mean": 12.0,
        "ext_source_3": 0.31,
        "days_employed_perc": 0.14375,
        "prev_cnt_payment_mean": 12.0,
        "new_active_debt_ratio": 0.4167,
        "annuity_to_income_ratio": 0.1389,
        "instal_amt_payment_sum": 10000.0,
        "new_late_payment_ratio": 0.0,
        "own_car_age": 5.0,
        "buro_amt_credit_max_overdue_mean": 0.0,
        "code_gender": 1.0,
        "amt_goods_price": 450000.0,
        "days_id_publish": -3200.0,
        "active_days_credit_max": -120.0,
        "active_days_credit_enddate_max": 300.0,
        "amt_credit": 500000.0,
        "instal_days_entry_payment_max": -39.0,
        "new_bureau_debt_ratio": 0.4167,
        "instal_payment_diff_mean": 0.0,
        "instal_dpd_mean": 1.0,
        "cc_cnt_drawings_atm_current_mean": 1.0,
    }


def test_vectorized_batch_inference_matches_individual_prediction() -> None:
    service = InferenceService()
    features = valid_features()

    individual = service.predict(features, client_id=1)
    batch = service.predict_batch(
        [
            {
                "client_id": 1,
                "features": features,
            }
        ]
    )
    batch_prediction = batch["predictions"][0]

    assert batch_prediction["client_id"] == individual["client_id"]
    assert batch_prediction["score"] == pytest.approx(individual["score"])
    assert batch_prediction["prediction"] == individual["prediction"]
    assert batch_prediction["decision"] == individual["decision"]
