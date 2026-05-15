# Plan d'optimisation des performances

## Objectif

L'objectif de l'étape 4 est d'analyser les performances de l'API de scoring après déploiement et
monitoring, puis de tester des optimisations mesurables sans introduire de régression fonctionnelle
ou métier.

La démarche retenue est volontairement progressive :

```text
mesurer -> profiler -> identifier -> optimiser -> comparer -> documenter
```

Avant toute optimisation, une baseline doit être figée avec un protocole reproductible.

## Métriques suivies

Les métriques principales à améliorer ou surveiller sont :

| Métrique | Description | Objectif |
| --- | --- | --- |
| `latency_ms` | Temps total API par client | Réduire la latence moyenne et p95 |
| `preprocessing_latency_ms` | Temps de transformation raw data vers features TOP30 | Vérifier que le preprocessing n'est pas le goulot |
| `inference_latency_ms` | Temps d'appel au service d'inférence | Réduire le coût modèle |
| `batch_latency_ms` | Temps total d'un appel `/predict/batch` | Améliorer le débit batch |
| `latency_p95` | Latence p95 | Priorité sur l'expérience utilisateur |
| `latency_max` | Latence maximale observée | Identifier le coût de premier appel ou les anomalies |
| `throughput_clients_per_second` | Nombre de clients scorés par seconde | Mesurer la capacité batch |
| `error_rate` | Taux d'erreur API | Ne pas dégrader la fiabilité |

Les métriques secondaires sont :

- consommation CPU ;
- consommation mémoire ;
- temps de chargement du modèle ;
- stabilité des scores ;
- nombre de prédictions modifiées.

L'utilisation GPU n'est pas retenue comme métrique cible principale, car le modèle est un LightGBM
tabulaire exécuté en CPU dans un conteneur Docker simple. Un GPU ne serait pas justifié dans cette
configuration.

## Baseline actuelle

La baseline doit être construite sur un batch simulé de clients issus du dataset initial.

Protocole recommandé :

1. sélectionner un lot aléatoire de candidats ;
2. appeler `/predict/batch` ;
3. collecter les logs JSONL ;
4. importer les logs dans PostgreSQL ;
5. générer l'analyse de monitoring ;
6. profiler le traitement avec `cProfile`.

Dernière simulation locale disponible :

| Indicateur | Valeur |
| --- | ---: |
| Clients simulés | 1 000 |
| Succès API | 1 000 |
| Erreurs API | 0 |
| Taux d'erreur | 0 % |
| Latence moyenne | 3,973 ms |
| Latence p95 | 2,713 ms |
| Latence maximale | 2 118,237 ms |
| Score moyen | 0,387 |
| Prédictions `low_risk` | 686 |
| Prédictions `high_risk` | 314 |
| Drift détecté | Oui |
| Variables driftées | 1 sur 30 |

La latence maximale est fortement influencée par le premier appel. Ce point devra être confirmé par
profiling et pourra justifier une stratégie de warm-up.

## Profiling

Le profiling doit permettre d'identifier la part du temps passée dans :

- le preprocessing ;
- la validation des features ;
- la construction des `DataFrame` Pandas ;
- l'appel au modèle LightGBM ;
- le logging ;
- l'overhead FastAPI / TestClient lors des simulations locales.

Outil retenu :

```text
cProfile
```

Le profiling est adapté ici car il permet d'identifier rapidement les fonctions Python les plus
coûteuses sans modifier l'architecture applicative.

## Hypothèses d'optimisation

### 1. Warm-up du modèle

Le premier appel peut être plus lent à cause du chargement, de l'initialisation ou de caches internes.

Optimisation envisagée :

- exécuter une prédiction factice au démarrage de l'API ;
- mesurer l'impact sur la latence maximale du premier appel utilisateur.

Critère d'acceptation :

- réduction de la latence maximale ou du premier appel ;
- aucune modification des scores.

### 2. Inférence batch vectorisée

Le traitement batch actuel appelle l'inférence client par client. Cette approche est simple, mais elle
peut multiplier les constructions de `DataFrame` et les appels au modèle.

Optimisation envisagée :

- préprocesser les clients ;
- construire un seul `DataFrame` batch ;
- appeler le modèle une seule fois avec toutes les lignes ;
- reconstruire les réponses client à partir des scores.

Critère d'acceptation :

- réduction du temps total batch ;
- réduction de l'inférence moyenne ;
- scores identiques ou écarts négligeables ;
- décisions identiques.

### 3. Chargement direct du modèle

Le modèle est actuellement chargé via l'artefact MLflow. Cette approche est traçable et compatible
avec le projet, mais elle peut ajouter un overhead.

Optimisation envisageable :

- comparer le chargement MLflow actuel avec un chargement direct du modèle pickle si l'artefact le
  permet ;
- conserver MLflow si le gain est faible ou si la compatibilité est moins bonne.

Critère d'acceptation :

- gain mesurable ;
- pas de perte de traçabilité critique ;
- compatibilité Docker et CI/CD.

### 4. ONNX Runtime

ONNX Runtime est cité dans les ressources, mais il doit rester une piste expérimentale.

Pour un modèle LightGBM tabulaire, l'export ONNX peut être possible mais ajoute :

- une dépendance supplémentaire ;
- un risque de compatibilité ;
- une étape de conversion ;
- un besoin de comparaison fine des scores.

Cette piste ne sera intégrée que si elle apporte un gain clair et reproductible sans régression.

## Non-régression

Chaque optimisation doit être validée sur le même batch de référence.

Contrôles attendus :

- même nombre de prédictions ;
- taux d'erreur inchangé ;
- écart maximal de score documenté ;
- nombre de décisions différentes documenté ;
- métriques métier inchangées ou explicitement justifiées.

Une optimisation ne doit pas être retenue si elle améliore la latence mais modifie significativement
les décisions ou la qualité métier.

## Livrables attendus

L'étape 4 doit produire :

- un script de benchmark reproductible ;
- un profil `cProfile` exploitable ;
- une baseline documentée ;
- une ou plusieurs optimisations testées ;
- un tableau comparatif avant / après ;
- une justification de la configuration finale ;
- un rapport détaillé dans `docs/optimization_report.md`.
