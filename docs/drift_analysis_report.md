# Rapport de monitoring et d'analyse de dérive

## Objectif

Ce rapport présente la solution mise en place pour suivre les données générées par l'API de scoring
crédit et détecter automatiquement les premiers signaux d'anomalies opérationnelles ou de dérive des
données.

La solution couvre trois niveaux :

- collecte structurée des appels API ;
- stockage local exploitable dans PostgreSQL ;
- analyse automatique avec métriques opérationnelles et détection de drift.

## Données collectées

Chaque appel à une route de prédiction produit un événement structuré.

Les événements sont d'abord écrits au format JSONL :

```text
logs/api_predictions.jsonl
```

Chaque ligne correspond à un événement JSON indépendant.

Les champs principaux journalisés sont :

- `timestamp` : date et heure UTC de l'événement ;
- `request_id` : identifiant unique de la requête ;
- `endpoint` : route appelée ;
- `status` : `success` ou `error` ;
- `client_id` : identifiant client si fourni ;
- `features` : features TOP30 calculées par le preprocessing ;
- `score` : score de risque retourné par le modèle ;
- `threshold` : seuil de décision appliqué ;
- `prediction` : classe prédite ;
- `decision` : décision lisible ;
- `latency_ms` : latence totale ;
- `preprocessing_latency_ms` : temps de preprocessing ;
- `inference_latency_ms` : temps d'inférence ;
- `error_type` et `error_message` en cas d'échec.

Les données brutes complètes ne sont pas stockées par défaut. Ce choix réduit l'exposition de données
sensibles tout en gardant les variables nécessaires à l'analyse de dérive.

## Stockage

Le PoC utilise PostgreSQL comme stockage structuré local.

Lancement :

```powershell
docker compose -f docker-compose.monitoring.yml up -d
```

Import des logs :

```powershell
uv run python scripts/import_monitoring_logs_to_postgres.py --truncate
```

Le schéma est défini dans :

```text
monitoring/db/schema.sql
```

La table `prediction_logs` contient les inputs modèle, les outputs, les métriques de latence et les
erreurs éventuelles. Les features sont stockées en `JSONB`.

## Référence utilisée pour le drift

La détection de dérive nécessite une distribution de référence.

Dans ce projet, la référence est construite à partir de l'ancien dataset préparé du projet P6 :

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

Par défaut, la référence contient un échantillon reproductible de `10 000` lignes. Ce fichier est
généré localement et n'est pas versionné dans Git.

## Analyse automatique

L'analyse peut lire les événements depuis le fichier JSONL :

```powershell
uv run python scripts/analyze_monitoring_logs.py --source jsonl
```

Ou depuis PostgreSQL :

```powershell
uv run python scripts/analyze_monitoring_logs.py --source postgres
```

Le script utilise :

- les logs de prédiction ;
- la référence TOP30 ;
- Pandas pour les métriques opérationnelles ;
- Evidently pour le drift.

Il produit :

```text
reports/monitoring/monitoring_summary.json
reports/monitoring/data_drift_report.html
reports/monitoring/data_drift_report.json
```

## Indicateurs opérationnels

Le script calcule notamment :

- nombre total d'événements ;
- nombre de succès ;
- nombre d'erreurs ;
- taux d'erreur ;
- latence moyenne ;
- latence médiane ;
- latence p95 ;
- latence maximale ;
- score moyen ;
- score médian ;
- score p95 ;
- répartition des décisions `low_risk` et `high_risk` ;
- répartition des types d'erreurs.

Ces métriques permettent d'identifier des problèmes opérationnels comme une hausse du taux d'erreur,
une latence anormale ou une modification forte de la distribution des décisions.

## Méthode de détection de dérive

La détection de dérive est réalisée avec le preset Evidently :

```python
Report([DataDriftPreset()])
```

Le script compare les features TOP30 de référence aux features TOP30 observées dans les logs.

Selon les caractéristiques des variables, Evidently applique des tests statistiques ou des distances
de distribution. Les méthodes peuvent inclure :

- distance de Wasserstein normalisée pour des variables numériques ;
- distance de Jensen-Shannon pour certaines variables discrètes ;
- tests basés sur des p-values selon les cas.

Pour chaque variable, le rapport contient :

- le nom de la colonne ;
- la méthode utilisée ;
- le seuil ;
- la valeur mesurée ;
- l'indication de dérive ou non.

Une variable est considérée comme driftée lorsque la distance ou le test dépasse le seuil retenu par
Evidently.

## Visualisation

Un dashboard Streamlit local est disponible :

```powershell
uv run streamlit run dashboard/monitoring_dashboard.py
```

Il permet de visualiser :

- la synthèse opérationnelle ;
- les événements récents ;
- les latences ;
- la distribution des scores ;
- la répartition des statuts ;
- la synthèse du drift ;
- le chemin du rapport Evidently HTML.

## Choix d'architecture

Les outils proposés dans le cahier des charges sont interprétés comme une liste d'options possibles.
Le projet retient une stack cohérente et proportionnée :

- bibliothèques Python pour produire et analyser les logs ;
- PostgreSQL pour stocker les événements ;
- Evidently pour la dérive ;
- Streamlit pour la visualisation.

Fluentd, Logstash, Elasticsearch, Kibana et Grafana ne sont pas utilisés dans ce PoC. Leur ajout
serait pertinent dans une architecture multi-services ou à fort volume, mais il augmenterait la
complexité sans bénéfice déterminant pour le besoin actuel.

## Points de vigilance

La détection de drift dépend fortement du volume de données observées. Sur un très petit nombre
d'appels, les résultats peuvent être instables ou trop sensibles.

Pour une analyse plus fiable, il est recommandé de comparer la référence à une fenêtre d'observation
suffisamment large :

- au moins 100 prédictions pour un premier diagnostic ;
- idéalement plusieurs centaines ou plusieurs milliers de prédictions ;
- ou une fenêtre temporelle stable, par exemple une journée de production simulée.

Les résultats doivent être interprétés comme des signaux d'alerte, pas comme une preuve automatique
de dégradation du modèle.

## Limites du PoC

Le stockage PostgreSQL est local et reste volontairement simple.

Limites identifiées :

- pas d'alerting automatique ;
- pas de politique de rétention automatisée ;
- pas de sauvegarde de base ;
- pas de gestion fine des droits ;
- pas de monitoring d'infrastructure temps réel.

Ces limites sont acceptables pour l'étape 3, mais devront être traitées pour une production réelle.

## Conclusion

La solution mise en place couvre les attendus de l'étape 3 :

- logging structuré des appels API ;
- stockage des inputs modèle, outputs et temps d'exécution ;
- stockage PostgreSQL local ;
- analyse automatique des logs ;
- détection de problèmes opérationnels ;
- détection de dérive via Evidently ;
- dashboard Streamlit ;
- documentation des choix, limites et points de vigilance.

Ce dispositif fournit une base exploitable pour l'étape suivante, dédiée à l'analyse des performances
et aux optimisations post-déploiement.
