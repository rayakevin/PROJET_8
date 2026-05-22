# Architecture de monitoring et de détection de drift

Le monitoring mis en place couvre les logs d'appels API, les inputs modèle, les outputs, les temps
d'exécution, les erreurs et la dérive des données.

## Architecture retenue

```text
API FastAPI
  -> logs structurés JSONL
  -> import PostgreSQL local
  -> analyse Pandas + Evidently
  -> rapports JSON / HTML
  -> dashboard Streamlit local
```

Cette architecture est un PoC local complet. Elle répond aux attendus de la mission sans imposer une
base cloud ou une stack de logs lourde à maintenir.

## Choix de stack

| Besoin | Outil retenu | Rôle |
| --- | --- | --- |
| Journalisation | service Python interne | Écriture d'événements JSONL structurés |
| Stockage | PostgreSQL local via Docker Compose | Requêtage et conservation des événements |
| Analyse | Pandas | Métriques opérationnelles |
| Drift | Evidently | Comparaison automatique des distributions |
| Visualisation | Streamlit | Dashboard local de suivi |

Fluentd et Logstash n'ont pas été utilisés. Ils servent surtout à collecter, transformer et router
des logs entre plusieurs services. Ici, l'API produit déjà un format JSONL propre et directement
exploitable. Elasticsearch, Kibana et Grafana ont aussi été écartés pour éviter une complexité
disproportionnée dans un PoC local.

## Données journalisées

Chemin par défaut :

```text
logs/api_predictions.jsonl
```

Champs principaux :

- `timestamp` ;
- `request_id` ;
- `endpoint` ;
- `status` ;
- `client_id` ;
- `model_version` ;
- `features` ;
- `score` ;
- `threshold` ;
- `prediction` ;
- `decision` ;
- `latency_ms` ;
- `preprocessing_latency_ms` ;
- `inference_latency_ms` ;
- `error_type` et `error_message`.

Les données brutes complètes ne sont pas conservées. Le monitoring stocke les 30 features réellement
utilisées par le modèle, ce qui suffit pour l'analyse de drift tout en limitant l'exposition de
données sensibles.

## PostgreSQL local

Lancement :

```powershell
docker compose -f docker-compose.monitoring.yml up -d
```

URL par défaut :

```text
postgresql://projet8:projet8@localhost:5433/projet8_monitoring
```

Le schéma SQL est versionné dans `monitoring/db/schema.sql`. La table principale est
`prediction_logs`. Elle contient les événements, les métriques de latence, les outputs modèle et les
features TOP30 en `JSONB`.

Import des logs :

```powershell
uv run python scripts/import_monitoring_logs_to_postgres.py --truncate
```

La variable `MONITORING_DATABASE_URL` permet de cibler une autre base.

## Référence de drift

La référence TOP30 est versionnée dans le dépôt pour que l'installation soit autonome :

```text
monitoring/reference/top30_reference.parquet
```

Elle peut être régénérée depuis l'échantillon de modélisation versionné :

```powershell
uv run python scripts/build_monitoring_reference.py
```

Source par défaut du script :

```text
data/reference/application_train_modeling_sample.parquet
```

Pour recalculer la référence depuis le dataset P6 complet, il reste possible de fournir explicitement
un autre parquet avec l'option `--source`.

## Analyse automatique

Analyse depuis PostgreSQL :

```powershell
uv run python scripts/analyze_monitoring_logs.py --source postgres
```

Analyse depuis le JSONL :

```powershell
uv run python scripts/analyze_monitoring_logs.py --source jsonl
```

Sorties générées :

```text
reports/monitoring/monitoring_summary.json
reports/monitoring/data_drift_report.html
reports/monitoring/data_drift_report.json
```

Le script calcule les volumes, le taux d'erreur, les latences moyenne/médiane/p95/max, les scores,
la répartition des décisions et la synthèse de drift.

## Détection de drift

Le projet utilise le preset Evidently :

```python
Report([DataDriftPreset()])
```

Evidently choisit automatiquement la méthode de comparaison selon le type de chaque variable. Dans
les rapports générés sur ce projet, les variables numériques sont principalement évaluées avec la
distance de Wasserstein normalisée, tandis que `code_gender` est évaluée avec la distance de
Jensen-Shannon.

Le rapport Evidently indique pour chaque colonne :

- la méthode appliquée ;
- le seuil ;
- la valeur observée ;
- le statut drifté ou non.

Ce choix évite de configurer manuellement 30 tests différents et reste auditable grâce au rapport
détaillé.

## Dashboard

Lancement :

```powershell
uv run streamlit run dashboard/monitoring_dashboard.py
```

Le dashboard affiche :

- synthèse opérationnelle ;
- événements récents depuis JSONL ou PostgreSQL ;
- distribution des scores ;
- latences ;
- statuts ;
- synthèse du drift ;
- rapport Evidently HTML intégré.

## Points de vigilance

Le drift doit être interprété sur un volume suffisant. Quelques appels isolés produisent des signaux
instables. Les analyses locales ont été réalisées sur des lots simulés de plusieurs centaines à
plusieurs milliers de clients.

Pour une production réelle, il resterait à ajouter une politique de rétention, une gestion fine des
accès, des sauvegardes, un alerting et un stockage cloud managé.
