import json
import math
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_SCHEMA_PATH = PROJECT_ROOT / "model" / "schema" / "top30_feature_schema.json"

MISSING_VALUE = float("nan")
EMPLOYMENT_DAYS_ANOMALY = 365243


class PreprocessingService:
    """Transforme les donnees brutes metier en features TOP30 du modele."""

    def __init__(self) -> None:
        self.feature_names = self._load_feature_names()

    def _load_feature_names(self) -> list[str]:
        schema = json.loads(FEATURE_SCHEMA_PATH.read_text(encoding="utf-8"))
        return [item["feature"] for item in schema["features"]]

    def transform(self, raw_data: dict[str, Any]) -> dict[str, float]:
        if not isinstance(raw_data, dict):
            raise ValueError("raw_data doit etre un objet JSON")

        application = self._get_application(raw_data)
        bureau = self._get_records(raw_data, "bureau")
        previous_applications = self._get_records(raw_data, "previous_applications")
        installments = self._get_records(raw_data, "installments_payments")
        credit_card_balance = self._get_records(raw_data, "credit_card_balance")

        features = {
            **self._build_application_features(application),
            **self._build_previous_application_features(previous_applications),
            **self._build_bureau_features(bureau),
            **self._build_installment_features(installments),
            **self._build_credit_card_features(credit_card_balance),
        }

        missing = [name for name in self.feature_names if name not in features]
        if missing:
            raise ValueError(f"Features TOP30 non construites : {', '.join(missing)}")

        return {name: features[name] for name in self.feature_names}

    def transform_batch(self, clients: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not isinstance(clients, list) or not clients:
            raise ValueError("Le lot de clients ne peut pas etre vide")

        transformed_clients = []
        for client in clients:
            if not isinstance(client, dict):
                raise ValueError("Chaque client du lot doit etre un objet JSON")

            transformed_clients.append(
                {
                    "client_id": client.get("client_id"),
                    "features": self.transform(client.get("raw_data")),
                }
            )

        return transformed_clients

    def _get_application(self, raw_data: dict[str, Any]) -> dict[str, Any]:
        application = raw_data.get("application")
        if not isinstance(application, dict) or not application:
            raise ValueError("raw_data.application est obligatoire")
        return application

    def _get_records(self, raw_data: dict[str, Any], block_name: str) -> list[dict[str, Any]]:
        records = raw_data.get(block_name, [])
        if records is None:
            return []

        if not isinstance(records, list):
            raise ValueError(f"Le bloc {block_name} doit etre une liste")

        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"Le bloc {block_name}[{index}] doit etre un objet JSON")

        return records

    def _build_application_features(self, application: dict[str, Any]) -> dict[str, float]:
        amt_credit = self._number(application, "AMT_CREDIT")
        amt_annuity = self._number(application, "AMT_ANNUITY")
        amt_goods_price = self._number(application, "AMT_GOODS_PRICE")
        amt_income_total = self._number(application, "AMT_INCOME_TOTAL")
        days_birth = self._number(application, "DAYS_BIRTH")
        days_employed = self._normalize_days_employed(self._number(application, "DAYS_EMPLOYED"))
        ext_source_1 = self._number(application, "EXT_SOURCE_1")
        ext_source_2 = self._number(application, "EXT_SOURCE_2")
        ext_source_3 = self._number(application, "EXT_SOURCE_3")

        return {
            "ext_sources_mean": self._mean([ext_source_1, ext_source_2, ext_source_3]),
            "credit_to_annuity_ratio": self._safe_divide(amt_credit, amt_annuity),
            "days_birth": days_birth,
            "ext_source_1": ext_source_1,
            "ext_source_2": ext_source_2,
            "payment_rate": self._safe_divide(amt_annuity, amt_credit),
            "credit_to_goods_ratio": self._safe_divide(amt_credit, amt_goods_price),
            "amt_annuity": amt_annuity,
            "days_employed": days_employed,
            "ext_source_3": ext_source_3,
            "days_employed_perc": self._safe_divide(days_employed, days_birth),
            "annuity_to_income_ratio": self._safe_divide(amt_annuity, amt_income_total),
            "own_car_age": self._number(application, "OWN_CAR_AGE"),
            "code_gender": self._encode_gender(application),
            "amt_goods_price": amt_goods_price,
            "days_id_publish": self._number(application, "DAYS_ID_PUBLISH"),
            "amt_credit": amt_credit,
        }

    def _build_previous_application_features(
        self, previous_applications: list[dict[str, Any]]
    ) -> dict[str, float]:
        cnt_payments = [self._number(record, "CNT_PAYMENT") for record in previous_applications]
        approved_cnt_payments = [
            self._number(record, "CNT_PAYMENT")
            for record in previous_applications
            if self._matches(record, "NAME_CONTRACT_STATUS", "Approved")
        ]

        return {
            "approved_cnt_payment_mean": self._mean(approved_cnt_payments),
            "prev_cnt_payment_mean": self._mean(cnt_payments),
        }

    def _build_bureau_features(self, bureau: list[dict[str, Any]]) -> dict[str, float]:
        active_bureau = [
            record for record in bureau if self._matches(record, "CREDIT_ACTIVE", "Active")
        ]

        bureau_debt_sum = self._sum(
            [self._number(record, "AMT_CREDIT_SUM_DEBT") for record in bureau]
        )
        bureau_credit_sum = self._sum([self._number(record, "AMT_CREDIT_SUM") for record in bureau])
        active_debt_sum = self._sum(
            [self._number(record, "AMT_CREDIT_SUM_DEBT") for record in active_bureau]
        )
        active_credit_sum = self._sum(
            [self._number(record, "AMT_CREDIT_SUM") for record in active_bureau]
        )

        return {
            "new_active_debt_ratio": self._safe_divide(active_debt_sum, active_credit_sum),
            "buro_amt_credit_max_overdue_mean": self._mean(
                [self._number(record, "AMT_CREDIT_MAX_OVERDUE") for record in bureau]
            ),
            "active_days_credit_max": self._max(
                [self._number(record, "DAYS_CREDIT") for record in active_bureau]
            ),
            "active_days_credit_enddate_max": self._max(
                [self._number(record, "DAYS_CREDIT_ENDDATE") for record in active_bureau]
            ),
            "new_bureau_debt_ratio": self._safe_divide(bureau_debt_sum, bureau_credit_sum),
        }

    def _build_installment_features(self, installments: list[dict[str, Any]]) -> dict[str, float]:
        payments = [self._number(record, "AMT_PAYMENT") for record in installments]
        entry_payment_days = [self._number(record, "DAYS_ENTRY_PAYMENT") for record in installments]
        payment_differences = [self._payment_difference(record) for record in installments]
        days_past_due = [self._days_past_due(record) for record in installments]

        return {
            "instal_amt_payment_sum": self._sum(payments),
            "new_late_payment_ratio": self._safe_divide(
                self._sum(days_past_due), len(installments)
            ),
            "instal_days_entry_payment_max": self._max(entry_payment_days),
            "instal_payment_diff_mean": self._mean(payment_differences),
            "instal_dpd_mean": self._mean(days_past_due),
        }

    def _build_credit_card_features(
        self, credit_card_balance: list[dict[str, Any]]
    ) -> dict[str, float]:
        return {
            "cc_cnt_drawings_atm_current_mean": self._mean(
                [self._number(record, "CNT_DRAWINGS_ATM_CURRENT") for record in credit_card_balance]
            )
        }

    def _number(self, record: dict[str, Any], field_name: str) -> float:
        value = self._get_value(record, field_name)
        if value is None:
            return MISSING_VALUE

        if isinstance(value, str) and not value.strip():
            return MISSING_VALUE

        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"Le champ {field_name} doit etre numerique") from None

        if math.isinf(number):
            raise ValueError(f"Le champ {field_name} ne peut pas etre infini")

        return number

    def _get_value(self, record: dict[str, Any], field_name: str) -> Any:
        if field_name in record:
            return record[field_name]

        normalized_field_name = field_name.lower()
        for key, value in record.items():
            if isinstance(key, str) and key.lower() == normalized_field_name:
                return value

        return None

    def _encode_gender(self, application: dict[str, Any]) -> float:
        value = self._get_value(application, "CODE_GENDER")
        if value is None:
            return MISSING_VALUE

        normalized_value = str(value).strip().upper()
        if normalized_value == "F":
            return 0.0
        if normalized_value == "M":
            return 1.0
        if normalized_value == "XNA" or normalized_value == "":
            return MISSING_VALUE

        raise ValueError("Le champ CODE_GENDER doit valoir F, M ou XNA")

    def _matches(self, record: dict[str, Any], field_name: str, expected_value: str) -> bool:
        value = self._get_value(record, field_name)
        if value is None:
            return False

        return str(value).strip().lower() == expected_value.lower()

    def _payment_difference(self, record: dict[str, Any]) -> float:
        installment_amount = self._number(record, "AMT_INSTALMENT")
        payment_amount = self._number(record, "AMT_PAYMENT")
        if self._is_missing(installment_amount) or self._is_missing(payment_amount):
            return MISSING_VALUE

        return installment_amount - payment_amount

    def _days_past_due(self, record: dict[str, Any]) -> float:
        entry_payment_day = self._number(record, "DAYS_ENTRY_PAYMENT")
        installment_day = self._number(record, "DAYS_INSTALMENT")
        if self._is_missing(entry_payment_day) or self._is_missing(installment_day):
            return MISSING_VALUE

        return max(entry_payment_day - installment_day, 0.0)

    def _normalize_days_employed(self, days_employed: float) -> float:
        if self._is_missing(days_employed):
            return MISSING_VALUE

        if days_employed == EMPLOYMENT_DAYS_ANOMALY:
            return MISSING_VALUE

        return days_employed

    def _safe_divide(self, numerator: float, denominator: float | int) -> float:
        if isinstance(denominator, int):
            denominator = float(denominator)

        if self._is_missing(numerator) or self._is_missing(denominator) or denominator == 0:
            return MISSING_VALUE

        return numerator / denominator

    def _mean(self, values: list[float]) -> float:
        clean_values = [value for value in values if not self._is_missing(value)]
        if not clean_values:
            return MISSING_VALUE

        return sum(clean_values) / len(clean_values)

    def _sum(self, values: list[float]) -> float:
        clean_values = [value for value in values if not self._is_missing(value)]
        if not clean_values:
            return MISSING_VALUE

        return sum(clean_values)

    def _max(self, values: list[float]) -> float:
        clean_values = [value for value in values if not self._is_missing(value)]
        if not clean_values:
            return MISSING_VALUE

        return max(clean_values)

    def _is_missing(self, value: float) -> bool:
        return isinstance(value, float) and math.isnan(value)
