# Modèle retenu depuis le projet P6

Le modèle retenu pour PROJET_8 est le modèle MLflow final enregistré dans le projet précédent :

- modèle : `lightgbm_bonus` ;
- famille : boosting externe LightGBM ;
- registre MLflow P6 : `P6_HOME_CREDIT_DEFAULT_RISK_MODEL` ;
- version du registre MLflow P6 : `1` ;
- exécution finale MLflow : `03_optimization_lightgbm_bonus` ;
- artefact importé : `model/artifacts/mlflow_model/`.

## Pourquoi importer l'artefact MLflow ?

L'artefact MLflow contient le modèle sérialisé, le fichier `MLmodel` et les dépendances nécessaires.
Cela permettra à l'API de charger le modèle avec `mlflow.pyfunc.load_model(...)` sans dépendre de l'ancienne base `mlflow.db`.

## Seuils documentés

- seuil par défaut : `0.5` ;
- seuil métier retenu dans le notebook 03 : `0.45`.

Point de vigilance : le seuil métier améliore le rappel et le F-bêta métier sur le jeu de validation final, mais le coût métier exporté est légèrement moins bon que la référence au seuil `0.5`. Ce compromis devra être explicité avant usage en production.
