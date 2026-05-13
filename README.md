# PROJET_8 - Déploiement et monitoring du modèle de scoring

Ce dépôt est le nouveau projet de mise en production du modèle de scoring Home Credit issu du projet précédent.

Pour l'instant, le dépôt contient la structure attendue pour l'étape 1 de la mission :

- code source applicatif ;
- tests ;
- notebooks d'analyse ;
- scripts ;
- emplacement des artefacts modèle ;
- documentation ;
- Dockerfile temporaire ;
- fichiers de dépendances.

L'artefact MLflow du modèle retenu est importé dans `model/artifacts/mlflow_model/`.
Les notebooks d'analyse du projet précédent sont conservés dans `notebooks/legacy/`.
Les premiers scripts d'inférence locale sont disponibles dans `scripts/`.
L'API sera ajoutée dans un commit dédié afin de conserver un historique clair et pertinent.

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
