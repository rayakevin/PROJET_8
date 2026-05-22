# Expérience 2.1 - Optimisation du TOP30

## Objectif

Cette expérience reprend le modèle simplifié à `30` variables, puis optimise légèrement
les hyperparamètres LightGBM et le seuil de décision métier.

## Protocole

- Données : `data/reference/application_train_modeling_sample.parquet`
- Importance utilisée : `data/reference/lightgbm_bonus_native_importance.csv`
- Split final : holdout stratifié de `20 %`, `random_state=42`
- Validation interne : `20 %` du jeu d'entraînement, stratifiée
- Nombre de combinaisons testées : `16`
- Grille de seuils : `0.1` à `0.9`
  par pas de `0.01`
- Critère de choix : coût métier moyen minimal sur la validation interne.

## Résultats holdout

| Modèle | Variables | Seuil | ROC-AUC | PR-AUC | F-bêta métier | Coût moyen | Rappel | Précision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline P6 | 733 | 0.45 | 0.7878 | 0.2874 | 0.5688 | 0.4912 | 0.7297 | 0.1775 |
| TOP30 brut | 30 | 0.45 | 0.7787 | 0.2713 | 0.5640 | 0.5063 | 0.7376 | 0.1682 |
| TOP30 optimisé - seuil rappel | 30 | 0.45 | 0.7796 | 0.2725 | 0.5667 | 0.5027 | 0.7404 | 0.1694 |
| TOP30 optimisé - seuil coût | 30 | 0.50 | 0.7796 | 0.2725 | 0.5484 | 0.4987 | 0.6820 | 0.1853 |

## Hyperparamètres retenus

```json
{
  "learning_rate": 0.03,
  "num_leaves": 63,
  "min_child_samples": 50,
  "reg_lambda": 1.0
}
```

## Latence du modèle optimisé

| Mesure | Valeur |
| --- | ---: |
| Temps d'entraînement final | `1814.6307 ms` |
| Temps d'inférence batch holdout | `84.6050 ms` |
| Latence moyenne par client en batch | `0.001376 ms` |
| Latence mono-client moyenne | `0.5509 ms` |
| Latence mono-client p95 | `0.5733 ms` |

## Variables retenues

- `ext_sources_mean`
- `credit_to_annuity_ratio`
- `days_birth`
- `ext_source_1`
- `ext_source_2`
- `payment_rate`
- `credit_to_goods_ratio`
- `amt_annuity`
- `days_employed`
- `approved_cnt_payment_mean`
- `ext_source_3`
- `days_employed_perc`
- `prev_cnt_payment_mean`
- `new_active_debt_ratio`
- `annuity_to_income_ratio`
- `instal_amt_payment_sum`
- `new_late_payment_ratio`
- `own_car_age`
- `buro_amt_credit_max_overdue_mean`
- `code_gender`
- `amt_goods_price`
- `days_id_publish`
- `active_days_credit_max`
- `active_days_credit_enddate_max`
- `amt_credit`
- `instal_days_entry_payment_max`
- `new_bureau_debt_ratio`
- `instal_payment_diff_mean`
- `instal_dpd_mean`
- `cc_cnt_drawings_atm_current_mean`

## Lecture

Le modèle TOP30 optimisé devient le candidat principal pour l'API. Il réduit le contrat
d'entrée de `733` à `30` variables, tout en conservant une performance proche de la baseline.

Le critère métier prioritaire du projet est la réduction du coût moyen.
Le seuil par défaut retenu pour le candidat API est donc
`0.50`.

Deux profils de seuil restent documentés :

- seuil `0.50` : profil par défaut orienté coût moyen ;
- seuil `0.45` : profil alternatif orienté rappel et F-bêta métier.

Le seuil `0.50` est issu de la validation interne,
pas du holdout final.

## Artefacts générés

- modèle MLflow : `model/artifacts/mlflow_model_top30_optimized` ;
- schéma d'entrée : `model/schema/top30_feature_schema.json` ;
- métadonnées : `model/schema/top30_model_metadata.json` ;
- recherche hyperparamètres : `docs/experiments/top30_optimization_experiment_2_1_search.csv` ;
- métriques détaillées : `docs/experiments/top30_optimization_experiment_2_1_metrics.json`.

## Environnement

- Date d'exécution UTC : `2026-05-13T11:25:18+00:00`
- Python : `3.12.13`
- Système : `Windows-11-10.0.26200-SP0`
