# Rapport d'optimisation des performances

Cette étape a mesuré les performances de l'API déployable, identifié les goulots d'étranglement et
intégré les optimisations utiles sans changer le contrat API ni les décisions du modèle.

## Protocole

Les benchmarks ont été réalisés localement avec `scripts/benchmark_api_performance.py` sur un payload
de 1 000 clients issus du dataset initial.

Scénarios mesurés :

- `/predict` : 1 000 appels unitaires successifs ;
- `/predict/batch` : un appel batch contenant 1 000 clients.

Les mesures suivies sont la latence totale, la latence par client, le p95, le débit, le temps de
preprocessing, le temps d'inférence, les scores et la répartition des décisions.

## Baseline

Le batch initial traitait les clients un par un. Sur le troisième run, après chargement du modèle, les
résultats étaient :

| Métrique | Baseline batch |
| --- | ---: |
| Clients | 1 000 |
| Latence API | 1 172,451 ms |
| Latence murale locale | 1 260,325 ms |
| Débit | 793,446 clients/s |
| Latence moyenne par client | 0,818 ms |
| Latence p95 par client | 0,972 ms |
| Inférence moyenne par client | 0,732 ms |

Le premier appel était beaucoup plus lent, car il chargeait le modèle MLflow et le modèle LightGBM
sérialisé. Sur l'endpoint unitaire, le premier run atteignait `13 085,774 ms` pour 1 000 appels, avec
une latence maximale par client de `1 840,029 ms`.

## Goulots identifiés

Les profils `cProfile` et les métriques de monitoring ont mis en évidence trois points :

- le chargement du modèle était payé par le premier utilisateur ;
- `/predict/batch` répétait les appels `predict_proba` client par client ;
- le logging batch ouvrait le fichier JSONL pour chaque événement.

## Optimisation 1 : warm-up du modèle

Le modèle est chargé au démarrage de FastAPI avec une prédiction factice.

Résultat sur 1 000 appels unitaires :

| Métrique | Avant warm-up | Après warm-up | Évolution |
| --- | ---: | ---: | ---: |
| Latence murale totale | 13 085,774 ms | 4 168,628 ms | -68,14 % |
| Débit | 76,419 clients/s | 239,887 clients/s | +213,91 % |
| Latence moyenne par client | 7,763 ms | 2,081 ms | -73,19 % |
| Latence p95 par client | 7,185 ms | 2,976 ms | -58,58 % |
| Latence maximale par client | 1 840,029 ms | 8,595 ms | -99,53 % |

Les décisions sont restées identiques : 686 `low_risk` et 314 `high_risk`.

## Optimisation 2 : inférence batch vectorisée

Le batch construit maintenant un seul `DataFrame` de features et appelle `predict_proba` une seule
fois.

```text
N clients -> N preprocessings -> 1 DataFrame -> 1 appel modèle
```

Résultat sur `/predict/batch` :

| Métrique | Baseline | Batch vectorisé | Évolution |
| --- | ---: | ---: | ---: |
| Latence API | 1 172,451 ms | 647,807 ms | -44,75 % |
| Latence murale locale | 1 260,325 ms | 746,862 ms | -40,74 % |
| Débit | 793,446 clients/s | 1 338,935 clients/s | +68,75 % |
| Latence moyenne par client | 0,818 ms | 0,108 ms | -86,80 % |
| Latence p95 par client | 0,972 ms | 0,293 ms | -69,86 % |
| Inférence moyenne par client | 0,732 ms | 0,016 ms | -97,81 % |

Les décisions sont restées identiques.

## Optimisation 3 : écriture groupée des logs

Après vectorisation, le coût d'écriture des événements de monitoring devenait visible sur les gros
batchs. Le service prépare maintenant les événements en mémoire et les écrit en une seule ouverture
de fichier.

Le format reste inchangé : une ligne JSON par client.

Résultat sur `/predict/batch` :

| Métrique | Batch vectorisé | Batch vectorisé + logs groupés | Évolution |
| --- | ---: | ---: | ---: |
| Latence API | 647,807 ms | 104,735 ms | -83,83 % |
| Latence murale locale | 746,862 ms | 214,667 ms | -71,26 % |
| Débit | 1 338,935 clients/s | 4 658,378 clients/s | +247,92 % |
| Latence moyenne par client | 0,108 ms | 0,084 ms | -22,22 % |
| Latence p95 par client | 0,293 ms | 0,212 ms | -27,65 % |

Les décisions sont restées identiques : 686 `low_risk` et 314 `high_risk`.

## Non-régression

Les contrôles automatisés comparent l'inférence individuelle et l'inférence batch vectorisée :

- même `client_id` ;
- score équivalent à tolérance numérique près ;
- même prédiction ;
- même décision.

Les tests de monitoring vérifient aussi l'écriture groupée des événements JSONL.

## Pistes non retenues

ONNX Runtime a été testé hors commit. La conversion fonctionnait, les décisions étaient identiques,
mais l'inférence était plus lente que LightGBM natif sur ce modèle. L'option n'a pas été conservée.

Le GPU n'a pas été retenu, car le modèle est tabulaire, léger et déjà rapide en CPU.

Le chargement direct du pickle n'a pas été retenu afin de conserver la traçabilité MLflow et la
compatibilité avec le pipeline actuel.

Rust n'a pas été intégré. L'environnement local ne contenait pas l'outillage Rust et les gains les
plus importants ont été obtenus par vectorisation et réduction des I/O.

## Configuration finale

Configuration conservée :

- Python 3.12 ;
- FastAPI ;
- LightGBM ;
- MLflow ;
- Pandas ;
- Docker ;
- Hugging Face Spaces ;
- GitHub Actions.

Optimisations intégrées :

- warm-up du modèle au démarrage ;
- inférence batch vectorisée ;
- écriture groupée des logs batch.

La configuration finale améliore fortement la latence et le débit sans ajout de dépendance, sans
changement du contrat API et sans régression observée sur les décisions.
