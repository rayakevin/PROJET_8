# Plan de simplification du modèle

Ce document décrit l'étape de simplification du modèle issu du P6.
Elle est ajoutée avant l'implémentation de l'API afin d'éviter d'exposer en production un modèle nécessitant plusieurs centaines de variables en entrée.

## Constat

Le modèle importé depuis l'ancien projet est utilisable comme baseline technique, mais son contrat d'entrée actuel n'est pas adapté à une API de production.

Le schéma importé contient `733` variables attendues par le modèle.
Un tel volume de variables pose plusieurs problèmes :

- contrat d'API difficile à comprendre et à maintenir ;
- forte dépendance à l'ancien pipeline de feature engineering ;
- risque élevé d'erreurs d'entrée côté consommateur de l'API ;
- tests plus complexes à écrire et à maintenir ;
- faible lisibilité métier ;
- difficulté à expliquer les décisions du modèle ;
- déploiement moins robuste dans un environnement léger comme Hugging Face Spaces.

Le modèle historique doit donc être considéré comme une baseline, et non comme le modèle cible de production.

## Objectif

L'objectif de cette étape est de construire un modèle simplifié, compatible avec une exposition API réaliste.

Le modèle cible devra :

- conserver une performance acceptable par rapport à la baseline historique ;
- réduire fortement le nombre de variables nécessaires ;
- utiliser des variables compréhensibles ou facilement reconstructibles ;
- limiter la dépendance au pipeline historique ;
- rester compatible avec MLflow, FastAPI, Docker et le pipeline CI/CD ;
- permettre une journalisation exploitable pour le monitoring et l'optimisation.

Une cible raisonnable sera testée autour de plusieurs tailles de jeux de variables :

- `20` variables ;
- `30` variables ;
- `50` variables ;
- `100` variables, uniquement si les jeux plus courts entraînent une perte de performance trop importante.


## Stratégie retenue

La stratégie principale consiste à sélectionner un sous-ensemble de variables pertinentes, puis à réentraîner un modèle plus simple.

Approches à comparer :

- importance des variables du modèle LightGBM historique ;
- importance par permutation ;
- analyse SHAP si le temps de calcul reste raisonnable ;
- suppression des variables très corrélées ;
- suppression des variables difficiles à produire en production ;
- validation métier des variables retenues.

Le modèle simplifié pourra rester basé sur LightGBM si ce choix reste performant et stable.
Un modèle plus simple, comme une régression logistique régularisée, pourra aussi être testé comme référence interprétable.

## Données de travail

Les données brutes ne doivent pas être versionnées dans le dépôt Git.

Pour cette étape, les données seront lues depuis l'ancien projet local afin de :

- retrouver les jeux d'entraînement et de test disponibles ;
- générer des exemples de clients réalistes ;
- comparer les scores de la baseline avec ceux du modèle simplifié ;
- préparer des fichiers de tests légers et non sensibles si nécessaire.

Si des échantillons sont ajoutés au dépôt pour les tests, ils devront être :

- anonymisés ou déjà dépourvus de données directement sensibles ;
- très légers ;
- strictement limités aux besoins de test ;
- documentés.

## Expériences prévues

### Expérience 1 : baseline historique

Objectif : mesurer les performances du modèle importé avec ses `733` variables.

Résultats attendus :

- score de référence ;
- temps d'inférence moyen ;
- taille du contrat d'entrée ;
- contraintes identifiées.

Livrables associés :

- script reproductible : `scripts/run_baseline_experiment.py` ;
- rapport généré : `docs/experiments/baseline_experiment_1.md` ;
- métriques détaillées : `docs/experiments/baseline_experiment_1_metrics.json`.

### Expérience 2 : sélection par importance LightGBM

Objectif : entraîner plusieurs modèles à partir des variables les plus importantes du modèle historique.

Jeux à tester :

- top `20` variables ;
- top `30` variables ;
- top `50` variables ;
- top `100` variables si nécessaire.

Résultats attendus :

- comparaison des performances ;
- comparaison des temps d'inférence ;
- liste des variables retenues ;
- impact sur le contrat API.

Livrables associés :

- script reproductible : `scripts/run_feature_selection_experiment.py` ;
- rapport généré : `docs/experiments/feature_selection_experiment_2.md` ;
- métriques détaillées : `docs/experiments/feature_selection_experiment_2_metrics.json` ;
- synthèse tabulaire : `docs/experiments/feature_selection_experiment_2_summary.csv` ;
- listes de variables : `docs/experiments/feature_selection_experiment_2_features.csv`.

### Expérience 2.1 : optimisation du TOP30

Objectif : retenir le modèle TOP30 comme candidat principal, puis améliorer son compromis
performance / simplicité avec une optimisation légère.

Travaux réalisés :

- optimisation d'une grille courte d'hyperparamètres LightGBM ;
- séparation d'une validation interne pour choisir les hyperparamètres et le seuil ;
- conservation d'un holdout final intact pour la comparaison ;
- génération d'un artefact MLflow dédié au modèle TOP30 optimisé ;
- génération d'un schéma d'entrée à `30` variables.

Livrables associés :

- script reproductible : `scripts/run_top30_optimization_experiment.py` ;
- rapport généré : `docs/experiments/top30_optimization_experiment_2_1.md` ;
- métriques détaillées : `docs/experiments/top30_optimization_experiment_2_1_metrics.json` ;
- recherche hyperparamètres : `docs/experiments/top30_optimization_experiment_2_1_search.csv` ;
- modèle MLflow : `model/artifacts/mlflow_model_top30_optimized` ;
- schéma d'entrée : `model/schema/top30_feature_schema.json` ;
- métadonnées modèle : `model/schema/top30_model_metadata.json`.

### Expérience 3 : sélection enrichie par SHAP ou permutation

Objectif : vérifier que la sélection par importance native du modèle ne retient pas uniquement des variables utiles au modèle historique, mais aussi des variables robustes et explicables.

Résultats attendus :

- classement alternatif des variables ;
- comparaison avec le classement LightGBM ;
- choix final argumenté.

## Critères de décision

Le modèle retenu devra être choisi selon plusieurs critères, pas uniquement selon le score statistique.

Critères principaux :

- performance proche de la baseline ;
- nombre de variables compatible avec une API réaliste ;
- variables disponibles ou reconstructibles en production ;
- stabilité des prédictions ;
- temps d'inférence ;
- simplicité du pipeline de préparation ;
- interprétabilité ;
- facilité de test ;
- compatibilité avec le déploiement Hugging Face Spaces.

Le critère métier prioritaire du projet est la réduction du coût moyen.
Un modèle légèrement moins performant sur le rappel ou le F-bêta métier pourra donc être préféré
si son coût moyen est meilleur et si l'écart de performance globale reste acceptable.

## Impact sur le contrat API

Le contrat API actuel décrit le modèle importé, avec `733` variables.
Après cette étape, il devra être mis à jour pour refléter le modèle réellement retenu.

La route `POST /predict` devra idéalement recevoir un nombre limité de variables métier.

Exemple de cible :

```json
{
  "client_id": 100001,
  "features": {
    "ext_source_2": 0.54,
    "ext_source_3": 0.31,
    "days_birth": -16000,
    "amt_credit": 500000.0,
    "amt_income_total": 180000.0
  }
}
```

Cet exemple est uniquement indicatif.
La liste finale des variables sera déterminée par les expériences de simplification.

## Livrables attendus

Cette étape devra produire :

- un notebook ou script d'analyse de simplification ;
- une liste des variables candidates ;
- une comparaison des modèles testés ;
- un modèle simplifié sauvegardé avec MLflow ;
- un schéma d'entrée mis à jour ;
- une mise à jour du contrat d'API ;
- une documentation courte justifiant le choix final.

## Décision attendue avant développement API

Avant de développer l'API FastAPI, il faudra valider :

- le nombre cible de variables ;
- la liste des variables retenues ;
- le modèle simplifié à exposer ;
- le format définitif des entrées `POST /predict` et `POST /predict/batch` ;
- les exemples de clients utilisés dans les tests.
