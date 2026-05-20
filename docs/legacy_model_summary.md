# Modèles issus du projet P6

Le projet a importé l'artefact MLflow final du P6 afin de conserver une baseline technique
reproductible.

## Baseline historique

- modèle : `lightgbm_bonus` ;
- famille : LightGBM ;
- registre MLflow P6 : `P6_HOME_CREDIT_DEFAULT_RISK_MODEL` ;
- version du registre P6 : `1` ;
- exécution MLflow : `03_optimization_lightgbm_bonus` ;
- artefact importé : `model/artifacts/mlflow_model/` ;
- contrat d'entrée : 733 variables.

Cet artefact reste utile pour tracer le point de départ du projet, mais il n'a pas été retenu pour
l'API de production. Son contrat d'entrée était trop large pour une exposition réaliste.

## Modèle exposé par l'API

Le modèle utilisé par l'API est le modèle simplifié et optimisé :

- modèle : `lightgbm_top30_optimized` ;
- famille : LightGBM ;
- artefact MLflow : `model/artifacts/mlflow_model_top30_optimized/` ;
- schéma : `model/schema/top30_feature_schema.json` ;
- métadonnées : `model/schema/top30_model_metadata.json` ;
- contrat interne : 30 features.

L'API ne demande pas directement ces 30 features au consommateur. Elle reçoit des données brutes
métier, puis le service de preprocessing reconstruit les features nécessaires.

## Seuil de décision

Le seuil retenu dans l'artefact TOP30 est `0.5`. Le choix a privilégié la réduction du coût métier
moyen dans les expériences de simplification, tout en conservant un modèle plus simple à exposer,
tester, monitorer et optimiser.
