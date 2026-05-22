# PROJET_8 - API MLOps de scoring crédit

Ce dépôt contient la mise en production du modèle de scoring crédit issu du projet P6 Home Credit.
Le projet couvre l'exposition du modèle via API, la conteneurisation, le CI/CD, une interface
Streamlit, le monitoring local, la détection de drift et l'optimisation des performances.

## Démarrage rapide pour une démo live

Cette section permet de lancer le projet rapidement depuis un poste Windows, même avec peu
d'expérience technique.

Prérequis à installer ou ouvrir avant de commencer :

- Python `3.12` ;
- `uv` ;
- Docker Desktop lancé.

Depuis PowerShell :

```powershell
git clone https://github.com/rayakevin/PROJET_8.git
cd PROJET_8
```

Préparer l'environnement et lancer l'API :

```powershell
powershell -ExecutionPolicy Bypass -File demo_live\01_preparer_et_lancer_api.ps1
```

Ce script installe les dépendances, vérifie le modèle, démarre PostgreSQL pour le monitoring, lance
l'API FastAPI, génère quelques événements de monitoring, prépare l'analyse, puis démarre les deux
interfaces Streamlit.

Tester le projet dans le navigateur :

```text
Swagger API : http://127.0.0.1:8000/docs
Interface Streamlit API : http://127.0.0.1:8501
Dashboard monitoring : http://127.0.0.1:8502
```

Déclencher le pipeline CI/CD GitHub Actions :

```powershell
powershell -ExecutionPolicy Bypass -File demo_live\02_declencher_pipeline_cicd.ps1
```

Ce script crée une branche `feature/demo-ci-cd-trigger-*`, fait une modification minimale contrôlée,
commit, push et affiche le lien vers GitHub Actions. Le déploiement Hugging Face ne part que lors
d'un push sur `main`.

## Vue d'ensemble

L'application expose un modèle LightGBM optimisé sur 30 features. L'API reçoit des données brutes
métier proches des tables utilisateurs, les transforme en features modèle, calcule un score de risque
et journalise les appels pour le monitoring.

```text
Client ou Streamlit
  -> API FastAPI
  -> preprocessing des données brutes
  -> modèle LightGBM TOP30 chargé via MLflow
  -> réponse de scoring
  -> logs JSONL
  -> PostgreSQL local
  -> analyse Evidently
  -> dashboard Streamlit de monitoring
```

Le monitoring est volontairement local. Il permet de rejouer des appels, d'importer les logs dans
PostgreSQL et d'analyser les anomalies sans maintenir une base cloud.

## Stack technique

| Besoin | Choix retenu | Justification |
| --- | --- | --- |
| Gestion Python | `uv`, Python 3.12 | Environnement reproductible et rapide à installer |
| API | FastAPI | Swagger natif, validation Pydantic, simplicité Docker |
| Modèle | LightGBM + MLflow | Modèle tabulaire performant et artefact traçable |
| Interface utilisateur | Streamlit | UI légère pour tester le scoring |
| Tests et qualité | Pytest, Ruff | Contrôles rapides intégrés au CI |
| Conteneurisation | Docker | Déploiement identique en local et sur Hugging Face Spaces |
| CI/CD | GitHub Actions | Tests, build Docker et déploiements automatisés |
| Déploiement | Hugging Face Spaces | Solution simple pour exposer l'API et l'UI |
| Monitoring | JSONL + PostgreSQL local | Stockage exploitable sans infrastructure lourde |
| Drift | Evidently | Rapports automatiques de comparaison de distributions |

Fluentd, Logstash, Elasticsearch, Kibana et Grafana n'ont pas été retenus. Ils sont adaptés à des
architectures multi-services ou à fort volume, alors que ce projet contient une API principale et un
PoC de monitoring local.

## Architecture du dépôt

```text
app/                         API FastAPI et services applicatifs
dashboard/                   dashboard local de monitoring
docs/                        documentation projet et rapports
model/                       artefacts MLflow et schémas modèle
data/reference/              échantillon de données et importance LightGBM versionnés
monitoring/                  schéma PostgreSQL et référence locale générée
notebooks/                   notebook d'analyse du drift
scripts/                     scripts d'import, benchmark, monitoring et vérification
tests/                       tests unitaires et d'intégration
ui/                          interface Streamlit de scoring
.github/workflows/ci-cd.yml  pipeline CI/CD
Dockerfile                   image de l'API
Dockerfile.ui                image de l'interface Streamlit
docker-compose.monitoring.yml PostgreSQL local de monitoring
pyproject.toml               configuration Python, dépendances et outils
```

## Installation locale

Le dépôt est installable sans dépendance à un ancien dossier local du projet P6. Les artefacts
minimums nécessaires au fonctionnement et aux contrôles sont versionnés :

- modèle API TOP30 : `model/artifacts/mlflow_model_top30_optimized/` ;
- schémas modèle : `model/schema/` ;
- échantillon de modélisation : `data/reference/application_train_modeling_sample.parquet` ;
- importance native LightGBM : `data/reference/lightgbm_bonus_native_importance.csv` ;
- référence de drift TOP30 : `monitoring/reference/top30_reference.parquet`.

Prérequis :

- Python `3.12` ;
- `uv` ;
- Docker, uniquement pour lancer PostgreSQL local ou construire les images.

Depuis un clone propre :

```powershell
git clone <url-du-repo>
cd PROJET_8
uv sync --extra dev
```

Activation manuelle de l'environnement virtuel :

```powershell
.venv\Scripts\Activate.ps1
```

Vérification du chargement du modèle :

```powershell
uv run python scripts/check_model_load.py
```

Contrôles qualité :

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

## API de scoring

Routes exposées :

- `GET /health` : état de l'API ;
- `GET /model/info` : métadonnées du modèle chargé ;
- `POST /predict` : scoring d'un client ;
- `POST /predict/batch` : scoring d'un lot de clients.

Lancement local :

```powershell
uv run uvicorn app.main:app --reload
```

Swagger local :

```text
http://127.0.0.1:8000/docs
```

Les exemples Swagger sont préremplis avec des payloads valides. L'API accepte des blocs bruts
`application`, `bureau`, `previous_applications`, `installments_payments` et `credit_card_balance`.
Le service de preprocessing reconstruit ensuite les 30 features attendues par le modèle.

Build et lancement Docker :

```powershell
docker build -t credit-scoring-api .
docker run --rm --name credit-scoring-api -p 8000:8000 credit-scoring-api
```

## Interface Streamlit

L'interface de scoring se trouve dans `ui/streamlit_app.py`.

```powershell
uv run streamlit run ui/streamlit_app.py
```

Par défaut, elle appelle l'API Hugging Face :

```text
https://rayakevin-projet-8.hf.space
```

L'URL peut être changée dans la barre latérale ou via la variable d'environnement `API_BASE_URL`.

## Monitoring local et drift

L'API écrit des événements structurés dans :

```text
logs/api_predictions.jsonl
```

Chaque événement contient les inputs modèle, le score, la décision, les temps de preprocessing et
d'inférence, ainsi que les erreurs éventuelles.

Lancement de PostgreSQL local :

```powershell
docker compose -f docker-compose.monitoring.yml up -d
```

Import des logs :

```powershell
uv run python scripts/import_monitoring_logs_to_postgres.py --truncate
```

La référence de drift TOP30 est déjà versionnée dans `monitoring/reference/`. Pour la régénérer
depuis l'échantillon versionné :

```powershell
uv run python scripts/build_monitoring_reference.py
```

Analyse automatique :

```powershell
uv run python scripts/analyze_monitoring_logs.py --source postgres
```

Dashboard de monitoring :

```powershell
uv run streamlit run dashboard/monitoring_dashboard.py
```

Les rapports générés localement sont stockés dans `reports/monitoring/` et ne sont pas versionnés.

## Optimisation des performances

Le projet mesure les performances avec `scripts/benchmark_api_performance.py`. Les optimisations
retenues sont :

- chargement et warm-up du modèle au démarrage de l'API ;
- inférence batch vectorisée avec un seul appel `predict_proba` par lot ;
- écriture groupée des logs JSONL pour les batchs.

Sur un batch local de 1 000 clients, le débit est passé d'environ `793` à `4 658` clients par seconde,
sans changement des décisions observées.

Exemple de benchmark :

```powershell
uv run python scripts/benchmark_api_performance.py `
  --endpoint batch `
  --payload logs/batch_1000_random_clients.json `
  --repeats 3 `
  --profile `
  --label batch_vectorized_logs `
  --reset-monitoring-log
```

## CI/CD et déploiement Hugging Face

Le workflow `.github/workflows/ci-cd.yml` exécute :

- Ruff ;
- Pytest ;
- build Docker ;
- smoke tests Docker ;
- déploiement de l'API sur Hugging Face Spaces après merge sur `main` ;
- déploiement de l'interface Streamlit sur un deuxième Space.

Variables et secrets GitHub nécessaires :

- secret `HF_TOKEN` : token Hugging Face avec droit d'écriture ;
- variable `HF_USERNAME` : compte ou organisation Hugging Face ;
- variable `HF_SPACE_NAME` : Space API ;
- variable `HF_UI_SPACE_NAME` : Space Streamlit.

Les Spaces utilisent le SDK `Docker`. Le workflow envoie un dépôt minimal vers Hugging Face avec
`huggingface_hub` et `hf_xet`, ce qui permet de stocker les artefacts binaires du modèle.

## Tests

Commandes locales :

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Les tests couvrent notamment le preprocessing, l'inférence, les endpoints FastAPI, le monitoring, les
scripts de benchmark et les cas d'erreur.

## Documentation

Documents principaux :

- `docs/project_structure.md` : architecture du dépôt ;
- `docs/model_simplification_plan.md` : synthèse de simplification vers le modèle TOP30 ;
- `docs/api_contract.md` : contrat d'API ;
- `docs/monitoring_plan.md` : architecture de monitoring ;
- `docs/drift_analysis_report.md` : rapport d'analyse de drift ;
- `docs/optimization_plan.md` : protocole d'optimisation ;
- `docs/optimization_report.md` : résultats d'optimisation.

## Conventions Git

Branches permanentes :

- `main` : branche stable et livrable public ;
- `develop` : branche d'intégration.

Branches temporaires :

- `feature/<sujet>` : nouvelle fonctionnalité ;
- `fix/<sujet>` : correction ;
- `docs/<sujet>` : documentation ;
- `ci/<sujet>` : CI/CD ;
- `perf/<sujet>` : optimisation ;
- `test/<sujet>` : tests.

Format de commit :

```text
type: description courte
```

Types utilisés : `feat`, `fix`, `docs`, `test`, `ci`, `chore`, `refactor`, `perf`.

Les PR décrivent l'objectif, les principaux fichiers modifiés, les tests exécutés et les impacts sur
l'API, le modèle, Docker, le monitoring ou le déploiement.

Test CI/CD
