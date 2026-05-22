# Livrables - Mission 8

Ce dossier centralise les livrables demandés pour la mission "Déployez et monitorez votre modèle de
scoring". Les fichiers techniques restent à leur emplacement naturel dans le dépôt pour éviter les
copies incohérentes ; ce README sert de table d'orientation.

## Synthèse des livrables

| Livrable demandé | Emplacement |
| --- | --- |
| Historique des versions | `livrables_projet8/historique_versions/historique_git.txt` |
| Stratégie de branches | `livrables_projet8/historique_versions/strategie_branches.md` |
| Scripts API | `app/main.py`, `app/services/` |
| Dockerfile | `Dockerfile` |
| Scripts de tests automatisés | `tests/` |
| Pipeline CI/CD YAML | `.github/workflows/ci-cd.yml` |
| Analyse du Data Drift au format notebook | `notebooks/analyse_data_drift.ipynb` |
| Screenshots de la solution de stockage des données de production | `livrables_projet8/screenshots/` |
| Preuves d'exécution qualité/tests | `livrables_projet8/preuves_execution/` |

## Historique des versions

L'historique de version est documenté à deux niveaux :

- versioning logiciel : branches, commits et Pull Requests GitHub ;
- versioning MLOps : artefacts modèle, schémas et rapports d'expérience.

Fichiers utiles :

- `livrables_projet8/historique_versions/historique_git.txt` : export de l'historique Git ;
- `livrables_projet8/historique_versions/strategie_branches.md` : stratégie de branches ;
- `docs/experiments/` : rapports d'expérimentation modèle ;
- `model/schema/model_metadata.json` et `model/schema/top30_model_metadata.json` : métadonnées modèle.

## API

L'API est développée avec FastAPI.

Fichiers principaux :

- `app/main.py` : endpoints API ;
- `app/services/preprocessing_service.py` : transformation des données brutes ;
- `app/services/inference_service.py` : chargement modèle et prédiction ;
- `app/services/monitoring_service.py` : journalisation des appels.

Routes principales :

- `GET /health` ;
- `GET /model/info` ;
- `POST /predict` ;
- `POST /predict/batch`.

## Docker

Le Dockerfile principal est :

```text
Dockerfile
```

Il reconstruit l'environnement API avec Python, `uv`, le code FastAPI, le modèle TOP30 et les schémas
nécessaires à l'inférence.

## Tests automatisés

Les tests sont dans :

```text
tests/
```

Les preuves d'exécution sont exportées dans :

```text
livrables_projet8/preuves_execution/
```

Commandes utilisées :

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

## CI/CD

Le pipeline GitHub Actions est défini ici :

```text
.github/workflows/ci-cd.yml
```

Il couvre :

- qualité et formatage Ruff ;
- tests Pytest ;
- build Docker ;
- smoke tests Docker ;
- déploiement API Hugging Face sur `main` ;
- déploiement UI Streamlit Hugging Face sur `main`.

## Data Drift

Le notebook demandé est :

```text
notebooks/analyse_data_drift.ipynb
```

Le pipeline automatisable associé est porté par :

```text
scripts/analyze_monitoring_logs.py
monitoring/reference/top30_reference.parquet
```

## Stockage des données de production

Les captures demandées sont dans :

```text
livrables_projet8/screenshots/
```

La solution de stockage de démonstration repose sur :

- écriture automatique des appels API dans `logs/api_predictions.jsonl` ;
- import batch dans PostgreSQL local ;
- table `prediction_logs` définie par `monitoring/db/schema.sql` ;
- dashboard Streamlit de monitoring dans `dashboard/monitoring_dashboard.py`.

## Démo live

Les scripts de démo sont dans :

```text
demo_live/
```

Commandes :

```powershell
powershell -ExecutionPolicy Bypass -File demo_live\01_preparer_et_lancer_api.ps1
powershell -ExecutionPolicy Bypass -File demo_live\02_declencher_pipeline_cicd.ps1
```
