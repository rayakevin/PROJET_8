# Plan de monitoring et de détection de dérive

## Objectif

L'objectif de l'étape 3 est de mettre en place un prototype complet de monitoring pour l'API de
scoring crédit.

Le dispositif doit permettre de :

- collecter les appels API ;
- stocker les inputs modèle, outputs et temps d'exécution ;
- analyser automatiquement les erreurs, latences et distributions observées ;
- détecter une dérive des données ;
- visualiser les indicateurs de suivi.

L'approche retenue est un PoC local compatible avec le déploiement Hugging Face Spaces.

## Architecture retenue

```text
API FastAPI
  -> preprocessing raw data vers features TOP30
  -> inférence modèle
  -> logging structuré JSONL
  -> import optionnel vers PostgreSQL
  -> analyse automatique Python + Evidently
  -> rapports JSON / HTML
  -> dashboard Streamlit local
```

Cette architecture couvre les attendus de l'étape tout en restant raisonnable pour un projet de
formation. Elle évite d'ajouter une stack lourde de centralisation de logs alors que le projet ne
contient qu'une API principale.

## Choix de stack

| Besoin | Outil retenu | Justification |
| --- | --- | --- |
| Logging applicatif | Service Python de journalisation structurée en JSONL | Format structuré, simple à tester, exploitable localement |
| Stockage | PostgreSQL local via Docker Compose | Base relationnelle robuste, requêtable, suffisante pour un PoC |
| Analyse automatique | Python, Pandas | Calcul des métriques opérationnelles |
| Détection de drift | Evidently | Bibliothèque spécialisée adaptée à la comparaison de distributions |
| Visualisation | Streamlit | Dashboard léger, cohérent avec le reste du projet |

## Outils non retenus

Fluentd et Logstash n'ont pas été retenus dans cette version du PoC.

Ces outils sont utiles pour ingérer, transformer et router des logs provenant de plusieurs services
ou serveurs. Dans ce projet, le volume est limité, l'architecture ne comporte qu'une API principale,
et les logs sont déjà produits dans un format structuré directement exploitable.

Elasticsearch, Kibana, Grafana ou une stack complète de logs n'ont pas non plus été mis en place.
Ils représenteraient une complexité supérieure au besoin actuel. Le rôle de visualisation est couvert
par Streamlit, et le stockage structuré est assuré par PostgreSQL.

## Données journalisées

Chaque appel aux routes de prédiction produit un événement structuré.

Chemin local par défaut :

```text
logs/api_predictions.jsonl
```

Champs communs :

| Champ | Description |
| --- | --- |
| `timestamp` | Date et heure UTC de l'événement |
| `request_id` | Identifiant unique de la requête |
| `endpoint` | Route appelée, par exemple `/predict` ou `/predict/batch` |
| `status` | `success` ou `error` |
| `client_id` | Identifiant client si fourni |
| `model_version` | Version du modèle utilisé |
| `latency_ms` | Latence totale |
| `preprocessing_latency_ms` | Temps de preprocessing |
| `inference_latency_ms` | Temps d'inférence |
| `error_type` | Type d'erreur si l'appel échoue |
| `error_message` | Message d'erreur si l'appel échoue |

Champs propres aux prédictions réussies :

| Champ | Description |
| --- | --- |
| `features` | Dictionnaire des 30 features consommées par le modèle |
| `score` | Score de risque retourné par le modèle |
| `threshold` | Seuil de décision |
| `prediction` | Classe prédite, `0` ou `1` |
| `decision` | Décision lisible, `low_risk` ou `high_risk` |

Les données brutes complètes ne sont pas conservées par défaut. Le monitoring stocke les features
réellement utilisées par le modèle, ce qui limite l'exposition de données sensibles tout en permettant
l'analyse de drift.

## Stockage PostgreSQL

Le stockage PostgreSQL est lancé localement avec :

```powershell
docker compose -f docker-compose.monitoring.yml up -d
```

La base écoute sur le port local `5433`.

URL par défaut :

```text
postgresql://projet8:projet8@localhost:5433/projet8_monitoring
```

Le schéma SQL est versionné dans :

```text
monitoring/db/schema.sql
```

La table principale est :

```text
prediction_logs
```

Elle contient les événements de prédiction, les métriques de latence, les outputs modèle et les
features TOP30 au format JSONB.

## Import des logs vers PostgreSQL

Les logs JSONL peuvent être importés avec :

```powershell
uv run python scripts/import_monitoring_logs_to_postgres.py
```

Pour vider la table avant import :

```powershell
uv run python scripts/import_monitoring_logs_to_postgres.py --truncate
```

Pour utiliser une autre base :

```powershell
uv run python scripts/import_monitoring_logs_to_postgres.py `
  --database-url "postgresql://user:password@host:5432/database"
```

La variable d'environnement `MONITORING_DATABASE_URL` peut aussi être utilisée.

## Référence de drift

La détection de dérive nécessite une distribution de référence.

La référence est construite depuis l'ancien dataset préparé du projet P6 :

```text
D:\FORMATION AI\01_FORMATION AI ENGINEER\02_PROJETS\06_PROJET 06\P6\P6_MLOps_1-2\data\processed\application_train_full.parquet
```

Commande :

```powershell
uv run python scripts/build_monitoring_reference.py
```

Sortie locale :

```text
monitoring/reference/top30_reference.parquet
```

Ce fichier n'est pas versionné dans Git, car il est généré localement.

## Analyse automatique

Analyse depuis le fichier JSONL :

```powershell
uv run python scripts/analyze_monitoring_logs.py --source jsonl
```

Analyse depuis PostgreSQL :

```powershell
uv run python scripts/analyze_monitoring_logs.py --source postgres
```

Sorties générées :

```text
reports/monitoring/monitoring_summary.json
reports/monitoring/data_drift_report.html
reports/monitoring/data_drift_report.json
```

Le script calcule notamment :

- nombre total d'événements ;
- nombre de succès ;
- nombre d'erreurs ;
- taux d'erreur ;
- latence moyenne, médiane, p95 et maximale ;
- score moyen, médian et p95 ;
- répartition des décisions ;
- répartition des types d'erreurs ;
- détection de drift globale et par variable.

## Détection de drift

Le drift est analysé avec Evidently :

```python
Report([DataDriftPreset()])
```

La comparaison porte sur :

- les 30 features de référence ;
- les 30 features observées dans les logs de production ou de simulation.

Evidently choisit les tests statistiques ou distances adaptés aux types de variables. Le rapport
indique les colonnes driftées, la méthode appliquée, le seuil et la valeur observée.

## Dashboard Streamlit

Le dashboard local de monitoring se lance avec :

```powershell
uv run streamlit run dashboard/monitoring_dashboard.py
```

Il permet de visualiser :

- la synthèse opérationnelle ;
- les événements récents ;
- l'évolution des latences ;
- la distribution des scores ;
- la répartition des statuts ;
- la synthèse de drift ;
- le chemin du rapport HTML Evidently.

Le dashboard peut lire les événements depuis le JSONL local ou depuis PostgreSQL.

## Points de vigilance

La détection de drift nécessite un volume suffisant d'événements. Sur quelques appels seulement, les
résultats peuvent être instables ou peu représentatifs.

Pour une lecture fiable, il est préférable d'analyser :

- au moins 100 prédictions pour un premier diagnostic ;
- plusieurs centaines ou milliers de prédictions pour une analyse plus robuste ;
- une fenêtre temporelle stable, par exemple une journée de production simulée.

Le stockage local PostgreSQL reste un PoC. En production réelle, il faudrait ajouter une stratégie de
rétention, une gestion des accès, des sauvegardes, et éventuellement un système d'alerting.

## Livrables

L'étape 3 contient désormais :

- un service de logging structuré intégré à l'API ;
- un format de logs JSONL ;
- un stockage PostgreSQL local ;
- un schéma SQL versionné ;
- un script d'import JSONL vers PostgreSQL ;
- un script de construction de référence ;
- un script d'analyse automatique JSONL ou PostgreSQL ;
- des rapports Evidently JSON et HTML ;
- un dashboard Streamlit local ;
- une documentation des choix, limites et points de vigilance.
