# Protocole d'optimisation des performances

L'étape d'optimisation a utilisé les métriques de monitoring et un benchmark local reproductible pour
identifier les goulots d'étranglement de l'API.

## Métriques suivies

Les métriques principales sont :

- latence totale API ;
- latence vue par le client local ;
- débit en clients par seconde ;
- latence moyenne, médiane, p95 et maximale par client ;
- temps de preprocessing ;
- temps d'inférence ;
- stabilité des scores et des décisions.

L'utilisation GPU n'a pas été retenue comme métrique principale. Le modèle est un LightGBM tabulaire
léger, exécuté efficacement en CPU.

## Benchmark

Le script utilisé est :

```text
scripts/benchmark_api_performance.py
```

Il mesure deux scénarios :

- `--endpoint single` : appels successifs à `/predict` ;
- `--endpoint batch` : un appel à `/predict/batch` contenant plusieurs clients.

Exemple :

```powershell
uv run python scripts/benchmark_api_performance.py `
  --endpoint batch `
  --payload logs/batch_1000_random_clients.json `
  --repeats 3 `
  --profile `
  --label batch_vectorized_logs `
  --reset-monitoring-log
```

Le payload local `logs/batch_1000_random_clients.json` contient 1 000 clients tirés du dataset
initial. Il n'est pas versionné car il dérive des données du projet P6.

## Goulots identifiés

Les mesures ont mis en évidence trois coûts principaux :

- le premier appel chargeait le modèle MLflow et le pickle LightGBM ;
- le batch initial appelait le modèle client par client ;
- le logging batch ouvrait et écrivait le fichier JSONL à chaque client.

Le preprocessing reste mesuré, mais il n'a pas été le goulot principal sur les tests réalisés.

## Optimisations retenues

### Warm-up au démarrage

Le modèle est chargé au démarrage de FastAPI avec une prédiction factice. Le premier appel utilisateur
ne subit donc plus le coût d'initialisation MLflow.

### Inférence batch vectorisée

Le batch construit un seul `DataFrame` contenant les features TOP30 de tous les clients, puis appelle
`predict_proba` une seule fois.

```text
N clients -> N preprocessings -> 1 DataFrame -> 1 prédiction vectorisée
```

### Écriture groupée des logs

Les événements de monitoring d'un batch sont préparés en mémoire et écrits en une seule ouverture de
fichier.

```text
N événements -> préparation en mémoire -> 1 écriture JSONL groupée
```

Le format des logs reste inchangé : une ligne JSON par client.

## Pistes explorées ou écartées

ONNX Runtime a été testé hors commit. La conversion fonctionnait et les décisions restaient
identiques, mais l'inférence ONNX était plus lente que LightGBM natif sur ce modèle. L'option n'a pas
été intégrée.

Rust n'a pas été testé car l'environnement local ne disposait pas de `rustc` ni de `cargo`, et la
réécriture aurait ajouté une complexité importante pour un gain incertain. Le goulot principal
identifié se situait dans l'orchestration Python et les I/O, pas dans un calcul métier isolé.

Le GPU n'a pas été retenu pour des raisons de coût, de maintenance et de faible pertinence sur ce
modèle tabulaire léger.

## Validation

Les optimisations ont été retenues uniquement après vérification :

- mêmes décisions avant et après optimisation ;
- scores équivalents à tolérance numérique près ;
- tests Pytest passants ;
- contrôles Ruff passants ;
- amélioration mesurée sur benchmark local.
