# Rapport d'optimisation des performances

## Objectif

Cette étape analyse les performances de l'API de scoring après mise en place du déploiement et du
monitoring. L'objectif est d'identifier les goulots d'étranglement, de tester des optimisations, puis
de démontrer l'amélioration obtenue sans régression fonctionnelle.

## Protocole

Le benchmark est réalisé localement sur un batch de `1 000` candidats issus du dataset initial.

Payload utilisé :

```text
logs/batch_1000_random_clients.json
```

Ce fichier est généré localement et n'est pas versionné dans Git.

Le script de benchmark permet de mesurer deux modes :

- `--endpoint batch` : un appel à `/predict/batch` contenant 1 000 clients ;
- `--endpoint single` : 1 000 appels successifs à `/predict`.

Exemple pour le batch :

```powershell
uv run python scripts/benchmark_api_performance.py `
  --endpoint batch `
  --payload logs/batch_1000_random_clients.json `
  --repeats 3 `
  --profile `
  --label batch_vectorized_after_warmup `
  --reset-monitoring-log
```

Exemple pour l'endpoint unitaire :

```powershell
uv run python scripts/benchmark_api_performance.py `
  --endpoint single `
  --payload logs/batch_1000_random_clients.json `
  --repeats 3 `
  --profile `
  --label single_endpoint_after_warmup `
  --reset-monitoring-log
```

Le script mesure :

- la latence totale ;
- la latence vue par le client local ;
- le débit en clients par seconde ;
- la latence moyenne, médiane, p95 et maximale par client ;
- le temps de preprocessing ;
- le temps d'inférence ;
- la distribution des scores ;
- la répartition des décisions.

Un profil `cProfile` est généré afin d'identifier les fonctions les plus coûteuses.

## Baseline batch

La baseline batch correspond au traitement initial, dans lequel `/predict/batch` appelait le modèle
client par client.

Résultats sur le troisième run, après échauffement :

| Métrique | Baseline |
| --- | ---: |
| Clients | 1 000 |
| Latence batch API | 1 172,451 ms |
| Latence murale locale | 1 260,325 ms |
| Débit | 793,446 clients/s |
| Latence moyenne par client | 0,818 ms |
| Latence p95 par client | 0,972 ms |
| Inférence moyenne par client | 0,732 ms |
| Inférence p95 par client | 0,821 ms |
| `low_risk` | 686 |
| `high_risk` | 314 |
| Score moyen | 0,387 |

Le premier run était plus lent en raison du chargement du modèle :

| Métrique | Premier run baseline |
| --- | ---: |
| Latence batch API | 4 350,205 ms |
| Latence maximale par client | 1 939,614 ms |

Le profil `cProfile` confirme que le premier run est fortement influencé par le chargement MLflow et
pickle du modèle. Une fois le modèle chargé, le coût principal reste la répétition de nombreux appels
d'inférence individuels.

## Baseline endpoint unitaire

L'endpoint `/predict` a aussi été mesuré avec 1 000 appels successifs.

Avant warm-up, sur le premier run :

| Métrique | Avant warm-up |
| --- | ---: |
| Latence murale totale | 13 085,774 ms |
| Débit | 76,419 clients/s |
| Latence moyenne par client | 7,763 ms |
| Latence p95 par client | 7,185 ms |
| Latence maximale par client | 1 840,029 ms |
| Inférence moyenne par client | 7,236 ms |
| Inférence maximale par client | 1 838,920 ms |

Cette mesure confirme que le premier appel unitaire subit le coût de chargement du modèle.

## Goulots identifiés

Deux goulots principaux ont été observés :

- le traitement batch initial multipliait les constructions de `DataFrame` et les appels
  `predict_proba` ;
- le premier appel utilisateur payait le coût de chargement du modèle MLflow et du modèle pickle.

Le preprocessing reste mesuré, mais il n'apparaît pas comme le goulot principal.

## Optimisation 1 : inférence batch vectorisée

L'optimisation intégrée consiste à vectoriser l'inférence batch :

```text
N clients -> validation -> 1 DataFrame batch -> 1 appel predict_proba
```

Le preprocessing reste réalisé client par client, car les payloads raw sont composés de blocs métier
hétérogènes. En revanche, une fois les features TOP30 construites, l'inférence est exécutée en une
seule fois sur un `DataFrame` contenant toutes les lignes du batch.

Fichiers modifiés :

- `app/services/inference_service.py` ;
- `app/main.py`.

Résultats batch sur le troisième run :

| Métrique | Baseline | Optimisé | Évolution |
| --- | ---: | ---: | ---: |
| Latence batch API | 1 172,451 ms | 647,807 ms | -44,75 % |
| Latence murale locale | 1 260,325 ms | 746,862 ms | -40,74 % |
| Débit | 793,446 clients/s | 1 338,935 clients/s | +68,75 % |
| Latence moyenne par client | 0,818 ms | 0,108 ms | -86,80 % |
| Latence p95 par client | 0,972 ms | 0,293 ms | -69,86 % |
| Inférence moyenne par client | 0,732 ms | 0,016 ms | -97,81 % |
| Inférence p95 par client | 0,821 ms | 0,016 ms | -98,05 % |

Les décisions restent identiques :

| Décision | Baseline | Optimisé |
| --- | ---: | ---: |
| `low_risk` | 686 | 686 |
| `high_risk` | 314 | 314 |

Le score moyen reste identique :

```text
0,387
```

## Optimisation 2 : warm-up du modèle

Un warm-up est exécuté au démarrage de l'application FastAPI. Il charge le modèle et exécute une
prédiction factice, afin que le premier utilisateur ne supporte pas le coût d'initialisation.

Résultats sur le premier run de 1 000 appels à `/predict` :

| Métrique | Avant warm-up | Après warm-up | Évolution |
| --- | ---: | ---: | ---: |
| Latence murale totale | 13 085,774 ms | 4 168,628 ms | -68,14 % |
| Débit | 76,419 clients/s | 239,887 clients/s | +213,91 % |
| Latence moyenne par client | 7,763 ms | 2,081 ms | -73,19 % |
| Latence p95 par client | 7,185 ms | 2,976 ms | -58,58 % |
| Latence maximale par client | 1 840,029 ms | 8,595 ms | -99,53 % |
| Inférence moyenne par client | 7,236 ms | 1,621 ms | -77,60 % |
| Inférence maximale par client | 1 838,920 ms | 7,443 ms | -99,60 % |

Les décisions restent identiques :

| Décision | Avant warm-up | Après warm-up |
| --- | ---: | ---: |
| `low_risk` | 686 | 686 |
| `high_risk` | 314 | 314 |

Le score moyen reste identique :

```text
0,387
```

## Non-régression

Un test automatisé compare l'inférence individuelle et l'inférence batch vectorisée sur les mêmes
features.

Contrôles réalisés :

- même `client_id` ;
- même score, à tolérance numérique près ;
- même classe prédite ;
- même décision.

Test ajouté :

```text
tests/test_inference_service.py
```

La suite de tests complète passe après optimisation.

## Choix final

Deux optimisations sont retenues :

- inférence batch vectorisée ;
- warm-up du modèle au démarrage de l'API.

Elles présentent un bon compromis :

- gain important sur la latence batch ;
- gain important sur le débit ;
- réduction forte du coût du premier appel unitaire ;
- pas de dépendance supplémentaire ;
- compatibilité avec LightGBM et MLflow ;
- pas de modification du contrat API ;
- pas de régression sur les décisions observées.

## Pistes non retenues

### ONNX Runtime

ONNX Runtime n'est pas intégré à ce stade. Pour ce modèle LightGBM, l'export ONNX ajouterait une
dépendance, une étape de conversion et un besoin de validation fine des scores. Le gain n'est pas
nécessaire pour atteindre une amélioration démontrable dans cette étape.

### GPU

Le GPU n'est pas retenu. Le modèle est tabulaire, léger, et exécuté efficacement en CPU. Le coût de
déploiement et de maintenance d'une ressource GPU ne serait pas justifié.

### Changement de format d'artefact

Le chargement direct du pickle n'est pas retenu dans cette version. MLflow apporte une traçabilité
utile pour le projet et reste compatible avec le pipeline CI/CD actuel.

## Configuration finale

Configuration conservée :

- Python 3.12 ;
- FastAPI ;
- LightGBM ;
- MLflow pour l'artefact modèle ;
- Pandas pour la construction du `DataFrame` ;
- Docker ;
- Hugging Face Spaces ;
- GitHub Actions.

Optimisations intégrées :

- inférence batch vectorisée dans le service d'inférence ;
- warm-up du modèle au démarrage de l'API.

Cette configuration améliore la performance sans augmenter la complexité d'exploitation.
