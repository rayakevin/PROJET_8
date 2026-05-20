# Notebooks

## Livrable Mission 8

- `analyse_data_drift.ipynb` : notebook d'analyse du data drift. Il charge les événements de
  monitoring, compare les features observées à la référence TOP30 et génère une synthèse Evidently.

## Notebooks historiques

Le dossier `legacy/` contient les trois notebooks importés depuis le projet P6 précédent :

- `01_PREPARATION_DONNEES.ipynb`
- `02_MODELISATION_BASELINES_MLFLOW.ipynb`
- `03_OPTIMISATION_SEUIL_EXPLICABILITE.ipynb`

Ils sont conservés pour la traçabilité de l'analyse et de la modélisation.
L'API de production ne dépend pas de l'exécution de ces notebooks.
