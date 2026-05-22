# Expérience 1 - Baseline historique

## Objectif

Cette expérience mesure le comportement du modèle historique importé depuis le projet P6.
Elle sert de point de comparaison avant la réduction du nombre de variables.

## Données utilisées

- Source : `data/reference/application_train_modeling_sample.parquet`
- Lignes disponibles : `307507`
- Lignes évaluées : `5000`
- Variables modèle : `733`
- Taux de défaut dans l'échantillon : `0.0808`.

Les colonnes du parquet préparé sont normalisées pour correspondre au schéma du modèle :
minuscules, remplacement des espaces et caractères spéciaux par des underscores, puis alignement
sur `model/schema/feature_schema.json`.

## Modèle évalué

- Nom : `lightgbm_bonus`
- Famille : `Boosting externe`
- Version : `1`
- Seuil métier : `0.45`
- Artefact : `model\artifacts\mlflow_model`

## Résultats

Les mesures recalculées ci-dessous sont réalisées sur un échantillon du parquet préparé
`application_train_modeling_sample.parquet`. Elles servent de contrôle technique du modèle importé et
de mesure de latence. La colonne holdout P6 reste la référence de performance à utiliser pour
comparer les futurs modèles simplifiés.

| Métrique | Contrôle local sur échantillon | Référence holdout P6 |
| --- | ---: | ---: |
| ROC-AUC | 0.8700 | 0.7878 |
| PR-AUC | 0.3853 | 0.2874 |
| F-bêta métier | 0.6670 | 0.5688 |
| Coût métier moyen | 0.3790 | 0.4912 |
| Rappel | 0.8540 | 0.7297 |
| Précision | 0.2091 | 0.1775 |

## Latence

| Mesure | Valeur |
| --- | ---: |
| Temps de chargement du modèle | `293.3848 ms` |
| Temps d'inférence batch | `34.0965 ms` |
| Latence moyenne par client en batch | `0.006819 ms` |
| Nombre de tests unitaires mono-client | `100` |
| Latence mono-client moyenne | `1.7382 ms` |
| Latence mono-client p50 | `1.8577 ms` |
| Latence mono-client p95 | `2.3000 ms` |

## Lecture

Le modèle historique est techniquement exploitable, mais son contrat d'entrée reste trop lourd
pour une API de production : `733` variables sont nécessaires
pour produire un score. Cette expérience confirme donc que le modèle doit rester une baseline
de comparaison pendant la construction d'un modèle simplifié.

## Environnement

- Date d'exécution UTC : `2026-05-13T10:45:15+00:00`
- Python : `3.12.13`
- Système : `Windows-11-10.0.26200-SP0`
- Taille d'échantillon demandée : `5000`
- Graine aléatoire : `42`
