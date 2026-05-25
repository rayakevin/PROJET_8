# Rapport de couverture des tests

Ce dossier contient le rapport de couverture généré pour les tests automatisés du projet.

## Commande exécutée

```powershell
uv run coverage erase
uv run coverage run `
  --source=app,scripts `
  --omit="scripts/run_*","scripts/check_model_load.py","scripts/predict_from_json.py","scripts/preprocess_raw_payload.py","scripts/build_monitoring_reference.py" `
  -m pytest
uv run coverage report -m
uv run coverage html -d docs/livrables/coverage/htmlcov
uv run coverage xml -o docs/livrables/coverage/coverage.xml
uv run coverage json -o docs/livrables/coverage/coverage.json
```

## Périmètre mesuré

Le rapport couvre :

- `app/` : API FastAPI et services de preprocessing, inférence et monitoring ;
- `scripts/analyze_monitoring_logs.py` : analyse des logs et drift ;
- `scripts/benchmark_api_performance.py` : benchmark de performance ;
- `scripts/import_monitoring_logs_to_postgres.py` : import des logs vers PostgreSQL.

Les scripts d'expérimentation modèle et les scripts utilitaires manuels ont été exclus, car ils ne
font pas partie du runtime applicatif déployé.

## Résultat global

```text
25 tests passés
2 warnings Evidently / NumPy connus
Couverture globale du périmètre mesuré : 62 %
Couverture du code applicatif app/ : 84 %
```

## Détail par fichier

| Fichier | Couverture |
| --- | ---: |
| `app/main.py` | 94 % |
| `app/services/inference_service.py` | 80 % |
| `app/services/monitoring_service.py` | 98 % |
| `app/services/preprocessing_service.py` | 77 % |
| `scripts/analyze_monitoring_logs.py` | 34 % |
| `scripts/benchmark_api_performance.py` | 33 % |
| `scripts/import_monitoring_logs_to_postgres.py` | 49 % |

## Lecture du résultat

La couverture applicative est correcte pour les endpoints, le chargement modèle, le preprocessing,
l'inférence et le monitoring. La couverture globale baisse lorsque les scripts opérationnels sont
inclus, car une partie de leur code correspond à des chemins CLI, à des accès fichiers ou à des accès
PostgreSQL qui ne sont pas tous exécutés par la suite de tests automatisés.

Les tests couvrent les cas principaux attendus pour la mission :

- endpoints API ;
- payload valide ;
- payload invalide ;
- écriture des logs de monitoring ;
- inférence unitaire et batch ;
- écriture groupée des événements ;
- agrégation des métriques de monitoring ;
- fonctions de benchmark.

## Artefacts générés

- `htmlcov/index.html` : rapport HTML navigable ;
- `coverage.xml` : rapport XML exploitable par des outils CI ;
- `coverage.json` : rapport JSON exploitable par script.
