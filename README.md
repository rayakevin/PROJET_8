# PROJET_8 - Déploiement et monitoring du modèle de scoring

Ce dépôt est le nouveau projet de mise en production du modèle de scoring Home Credit issu du projet précédent.

Le dépôt contient les livrables des premières étapes de la mission :

- code source applicatif ;
- tests ;
- notebooks d'analyse ;
- scripts ;
- emplacement des artefacts modèle ;
- documentation ;
- API FastAPI conteneurisée ;
- CI/CD GitHub Actions ;
- interface Streamlit ;
- monitoring local avec PostgreSQL, Evidently et Streamlit.

L'artefact MLflow du modèle retenu est importé dans `model/artifacts/mlflow_model/`.
Les notebooks d'analyse du projet précédent sont conservés dans `notebooks/legacy/`.
Les premiers scripts d'inférence locale sont disponibles dans `scripts/`.

## Projet source

Ancien projet de modélisation :

```text
C:\Users\kevin\Desktop\FORMATION AI\01_FORMATION AI ENGINEER\02_PROJETS\06_PROJET 06\P6\P6_MLOps_1-2
```

## Objectifs Mission 8

- Exposer le modèle via une API.
- Ajouter des tests automatisés.
- Conteneuriser l'application avec Docker.
- Mettre en place un pipeline CI/CD.
- Stocker les appels de production simulés.
- Analyser le drift et les performances.
- Documenter le lancement et l'interprétation du monitoring.

## Environnement

Le projet utilise Python `3.12` et `uv`.

Initialisation locale :

```powershell
uv sync --extra dev
```

Activation manuelle de l'environnement virtuel :

```powershell
.venv\Scripts\Activate.ps1
```

Vérification du modèle MLflow importé :

```powershell
uv run python scripts/check_model_load.py
```

## API, Docker et CI/CD

L'API FastAPI expose le modèle de scoring via les routes suivantes :

- `GET /health` : vérification de disponibilité de l'API ;
- `GET /model/info` : métadonnées du modèle chargé ;
- `POST /predict` : prédiction pour un client à partir de données brutes ;
- `POST /predict/batch` : prédictions pour un lot de clients.

Lancement local de l'API :

```powershell
uv run uvicorn app.main:app --reload
```

Build et lancement Docker :

```powershell
docker build -t credit-scoring-api .
docker run --rm --name credit-scoring-api -p 8000:8000 credit-scoring-api
```

Tests locaux :

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Le workflow GitHub Actions `.github/workflows/ci-cd.yml` exécute :

- les contrôles Ruff ;
- les tests Pytest ;
- le build de l'image Docker ;
- un smoke test Docker sur `/health`, `/model/info` et `/predict/batch` ;
- le déploiement vers Hugging Face Spaces après merge sur `main` ;
- le déploiement d'une interface Streamlit dans un deuxième Space Hugging Face.

### Déploiement Hugging Face Spaces

Le déploiement continu nécessite un Space Hugging Face créé une première fois avec le SDK
`Docker`.

Configuration GitHub à ajouter dans `Settings > Secrets and variables > Actions` :

- secret `HF_TOKEN` : token Hugging Face avec droit d'écriture sur le Space ;
- variable `HF_USERNAME` : nom du compte ou de l'organisation Hugging Face ;
- variable `HF_SPACE_NAME` : nom du Space API cible ;
- variable `HF_UI_SPACE_NAME` : nom du Space Streamlit cible.

Le job de déploiement ne se lance que sur un `push` vers `main`, donc après fusion d'une PR.
Le workflow prépare un dépôt Space minimal contenant `Dockerfile`, `pyproject.toml`, `uv.lock`,
`app/` et `model/`, puis l'envoie vers Hugging Face avec `huggingface_hub` et `hf_xet`.
Ce mode d'envoi permet de stocker correctement les artefacts binaires du modèle.

Space cible :

```text
https://huggingface.co/spaces/<HF_USERNAME>/<HF_SPACE_NAME>
```

### Interface Streamlit

L'interface utilisateur Streamlit se trouve dans `ui/streamlit_app.py`.
Elle est déployée dans un deuxième Space Hugging Face via le même workflow GitHub Actions.
Ce Space utilise aussi le SDK `Docker`, car l'API Hugging Face de création de Space accepte
les valeurs `gradio`, `docker` et `static`. Le conteneur UI lance Streamlit sur le port `8501`.

Lancement local :

```powershell
uv run streamlit run ui/streamlit_app.py
```

Par défaut, l'interface appelle l'API déployée sur :

```text
https://rayakevin-projet-8.hf.space
```

Pour changer l'API cible, définir la variable d'environnement `API_BASE_URL` dans le Space
Streamlit ou modifier l'URL depuis la barre latérale de l'interface.

## Monitoring et drift

L'API produit des logs structurés au format JSONL dans :

```text
logs/api_predictions.jsonl
```

Le stockage PostgreSQL local se lance avec :

```powershell
docker compose -f docker-compose.monitoring.yml up -d
```

Import des logs JSONL vers PostgreSQL :

```powershell
uv run python scripts/import_monitoring_logs_to_postgres.py --truncate
```

Construction de la référence de drift :

```powershell
uv run python scripts/build_monitoring_reference.py
```

Analyse automatique depuis le JSONL :

```powershell
uv run python scripts/analyze_monitoring_logs.py --source jsonl
```

Analyse automatique depuis PostgreSQL :

```powershell
uv run python scripts/analyze_monitoring_logs.py --source postgres
```

Dashboard local de monitoring :

```powershell
uv run streamlit run dashboard/monitoring_dashboard.py
```

Documentation détaillée :

- `docs/monitoring_plan.md` ;
- `docs/drift_analysis_report.md`.

## Conventions de branches

Branches permanentes :

- `main` : branche stable, livrable public, protégée après configuration GitHub.
- `develop` : branche d'intégration.

Branches temporaires :

- `feature/<sujet>` : nouvelle fonctionnalité.
- `fix/<sujet>` : correction d'anomalie.
- `docs/<sujet>` : documentation.
- `ci/<sujet>` : pipeline CI/CD.
- `chore/<sujet>` : maintenance sans changement fonctionnel.
- `perf/<sujet>` : optimisation de performance.
- `test/<sujet>` : ajout ou correction de tests.

Exemples :

```text
feature/import-modele
feature/api-scoring
ci/github-actions
docs/soutenance
perf/latence-inference
```

## Conventions de commits

Format :

```text
type: description courte à l'impératif ou à l'infinitif
```

Types autorisés :

- `feat` : ajout fonctionnel.
- `fix` : correction.
- `docs` : documentation.
- `test` : tests.
- `ci` : intégration ou déploiement continu.
- `chore` : maintenance, configuration, nettoyage.
- `refactor` : changement interne sans modification de comportement.
- `perf` : amélioration de performance.

Exemples :

```text
feat: importer les métadonnées du modèle historique
test: ajouter les contrôles de chargement du modèle
ci: ajouter le workflow de tests GitHub Actions
docs: documenter la stratégie de monitoring
```

## Conventions de pull request

Chaque PR doit viser `develop`, sauf livraison finale ou correction urgente vers `main`.

Titre recommandé :

```text
[type] description courte
```

Checklist minimale :

- Décrire l'objectif de la PR.
- Lister les fichiers ou modules principaux modifiés.
- Indiquer les tests exécutés ou expliquer pourquoi ils ne le sont pas encore.
- Mentionner les impacts sur les artefacts modèle, les données ou le monitoring.
- Joindre des captures ou journaux d'exécution si la PR touche l'API, le tableau de bord, Docker ou la CI/CD.

Règle de fusion :

- `feature/*` vers `develop`.
- `develop` vers `main` lors d'un jalon stable.
