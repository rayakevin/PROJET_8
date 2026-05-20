# Synthèse de simplification du modèle

Le modèle historique importé depuis le P6 attendait 733 variables. Ce contrat était trop complexe
pour une API de scoring exploitable : il dépendait fortement de l'ancien pipeline de feature
engineering, compliquait les tests et rendait le payload d'entrée peu réaliste.

Le projet a donc ajouté une étape de simplification avant le développement de l'API.

## Objectif retenu

L'objectif était de réduire fortement le nombre de variables sans perdre la logique métier du modèle.
Le modèle final devait :

- conserver une performance acceptable par rapport à la baseline historique ;
- réduire le contrat interne du modèle ;
- utiliser des variables disponibles ou reconstructibles à partir de données brutes ;
- rester compatible avec MLflow, FastAPI, Docker et le CI/CD ;
- produire des logs exploitables pour le monitoring et l'analyse de drift.

## Expériences réalisées

### Expérience 1 : baseline historique

Le modèle importé avec 733 variables a été mesuré pour servir de point de comparaison.

Livrables :

- `scripts/run_baseline_experiment.py` ;
- `docs/experiments/baseline_experiment_1.md` ;
- `docs/experiments/baseline_experiment_1_metrics.json`.

### Expérience 2 : sélection par importance LightGBM

Plusieurs modèles ont été entraînés à partir des variables les plus importantes du modèle
historique. Les jeux testés ont porté sur des sous-ensembles de tailles différentes, notamment
top 20, top 30 et top 50.

Livrables :

- `scripts/run_feature_selection_experiment.py` ;
- `docs/experiments/feature_selection_experiment_2.md` ;
- `docs/experiments/feature_selection_experiment_2_metrics.json` ;
- `docs/experiments/feature_selection_experiment_2_summary.csv` ;
- `docs/experiments/feature_selection_experiment_2_features.csv`.

### Expérience 2.1 : optimisation du TOP30

Le TOP30 a été retenu comme meilleur compromis entre simplicité et performance. Une optimisation
légère des hyperparamètres LightGBM et du seuil de décision a ensuite été réalisée.

Travaux effectués :

- recherche courte d'hyperparamètres LightGBM ;
- séparation entre validation interne et holdout final ;
- comparaison orientée coût métier moyen ;
- export d'un artefact MLflow dédié ;
- génération du schéma des 30 features.

Livrables :

- `scripts/run_top30_optimization_experiment.py` ;
- `docs/experiments/top30_optimization_experiment_2_1.md` ;
- `docs/experiments/top30_optimization_experiment_2_1_metrics.json` ;
- `docs/experiments/top30_optimization_experiment_2_1_search.csv` ;
- `model/artifacts/mlflow_model_top30_optimized/` ;
- `model/schema/top30_feature_schema.json` ;
- `model/schema/top30_model_metadata.json`.

## Décision finale

Le modèle exposé par l'API est le modèle `lightgbm_top30_optimized`.

Ce choix a été retenu car il :

- réduit le contrat modèle de 733 à 30 features ;
- conserve un niveau de performance compatible avec l'objectif de la mission ;
- facilite la validation des entrées et les tests automatisés ;
- rend le monitoring de drift plus lisible ;
- limite le coût d'inférence ;
- reste simple à déployer sur Hugging Face Spaces.

L'API reçoit des données brutes métier et non les 30 features directement. Cette décision évite de
déléguer le feature engineering au consommateur de l'API et rapproche le contrat d'entrée d'un usage
applicatif réel.
