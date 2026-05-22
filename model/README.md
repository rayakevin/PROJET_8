# Modèle

Ce dossier contient les éléments liés au modèle de scoring retenu.

- `artifacts/mlflow_model_top30_optimized/` : artefact MLflow chargé par l'API.
- `artifacts/mlflow_model/` : artefact baseline conservé pour les scripts d'expérimentation.
- `schema/` : schémas des variables, métadonnées du modèle et rapport de qualité.

L'API charge uniquement l'artefact TOP30 local. Elle ne dépend ni de l'ancien dépôt Projet 6 ni de
l'ancienne base de suivi `mlflow.db`.
