# Expérience 2 - Sélection par importance LightGBM

## Objectif

Cette expérience évalue plusieurs modèles LightGBM simplifiés entraînés avec les variables
les plus importantes du modèle historique.

## Protocole

- Données : `<ancien projet local>/data/processed/application_train_full.parquet`
- Importance utilisée : `<ancien projet local>/data/processed/reports/lightgbm_bonus_native_importance.csv`
- Split : holdout stratifié de `20 %`, `random_state=42`
- Seuil métier appliqué : `0.45`
- Coûts métier : faux négatif `10.0`,
  faux positif `1.0`
- Tailles testées : `20, 30, 50, 100`

Le seuil métier n'est pas réoptimisé dans cette expérience. Il reste fixé à `0.45` pour comparer
les modèles réduits avec la baseline historique dans un cadre simple et stable.

## Référence baseline P6

| Métrique | Baseline historique holdout |
| --- | ---: |
| ROC-AUC | `0.7878` |
| PR-AUC | `0.2874` |
| F-bêta métier | `0.5688` |
| Coût métier moyen | `0.4912` |
| Rappel | `0.7297` |
| Précision | `0.1775` |

## Résultats des modèles simplifiés

| Variables | ROC-AUC | PR-AUC | F-bêta métier | Coût métier moyen | Rappel | Précision | Latence p95 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 0.7738 | 0.2623 | 0.5587 | 0.5162 | 0.7360 | 0.1639 | 0.5466 |
| 30 | 0.7787 | 0.2713 | 0.5640 | 0.5063 | 0.7376 | 0.1682 | 0.5864 |
| 50 | 0.7828 | 0.2749 | 0.5658 | 0.5006 | 0.7345 | 0.1716 | 0.5897 |
| 100 | 0.7865 | 0.2813 | 0.5696 | 0.4935 | 0.7355 | 0.1750 | 0.6783 |

## Lecture provisoire

- Meilleur score PR-AUC : modèle top `100` variables.
- Meilleur compromis sous `50` variables : modèle top
  `50` variables.
- La réduction du contrat d'entrée est forte dès le top `20`. Le choix final a intégré la
  performance, la stabilité, la facilité de production des variables et l'explicabilité.

## Top 20 variables natives

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

## Livrables générés

- métriques détaillées : `docs/experiments/feature_selection_experiment_2_metrics.json` ;
- synthèse tabulaire : `docs/experiments/feature_selection_experiment_2_summary.csv` ;
- listes de variables : `docs/experiments/feature_selection_experiment_2_features.csv`.

## Environnement

- Date d'exécution UTC : `2026-05-13T10:52:43+00:00`
- Python : `3.12.13`
- Système : `Windows-11-10.0.26200-SP0`
