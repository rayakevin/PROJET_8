# Modèle

Ce dossier contient les éléments liés au modèle de scoring retenu.

- `artifacts/mlflow_model/` : artefact MLflow importé depuis le projet P6.
- `schema/` : schéma des variables, métadonnées du modèle et rapport de qualité.

L'API devra charger l'artefact local avec `mlflow.pyfunc.load_model(...)`, sans dépendre de l'ancienne base de suivi `mlflow.db`.
