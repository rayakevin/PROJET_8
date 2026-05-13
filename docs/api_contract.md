# Contrat d'API de scoring

Ce document décrit le contrat cible de l'API de scoring du projet.
Il sert de référence pour l'implémentation FastAPI, les tests automatisés, la conteneurisation Docker et le pipeline CI/CD.

## Objectif

L'API doit prédire le risque de défaut d'un client à partir de données brutes issues des bases utilisateurs.

Le modèle cible reste le modèle TOP30 optimisé, issu de l'expérience `2.1`.
En revanche, le contrat public de l'API ne doit pas demander à l'utilisateur de fournir directement les 30 features transformées.

L'architecture cible est donc :

1. réception de données brutes métier ;
2. validation du payload brut ;
3. transformation interne des données brutes en 30 variables modèle ;
4. appel du modèle TOP30 optimisé ;
5. retour du score, de la décision et des informations de latence.

Contraintes principales :

- charger le modèle une seule fois au démarrage de l'application ou lors du premier appel ;
- réutiliser le modèle chargé pour toutes les requêtes suivantes ;
- valider les données brutes avant la transformation ;
- produire les 30 features dans l'ordre attendu par le modèle ;
- retourner une réponse explicite et exploitable ;
- documenter automatiquement les routes via Swagger, fourni par FastAPI ;
- journaliser les informations utiles au monitoring et à l'optimisation post-déploiement.

## Choix technique

Framework retenu : `FastAPI`.

Justification :

- validation stricte des entrées avec Pydantic ;
- documentation Swagger automatique ;
- gestion claire des codes HTTP ;
- intégration simple avec `pytest`, Docker et GitHub Actions ;
- meilleure adéquation à une API de production qu'une interface purement démonstrative.

## Modèle cible

Le modèle cible est stocké dans :

```text
model/artifacts/mlflow_model_top30_optimized/
```

Le chargement cible se fera avec :

```python
mlflow.sklearn.load_model("model/artifacts/mlflow_model_top30_optimized")
```

Le chargement `mlflow.sklearn` est retenu car l'API doit retourner un score de probabilité via `predict_proba`.

Le schéma interne des variables modèle est stocké dans :

```text
model/schema/top30_feature_schema.json
```

Les métadonnées du modèle sont stockées dans :

```text
model/schema/top30_model_metadata.json
```

Métadonnées principales :

```json
{
  "model_name": "lightgbm_top30_optimized",
  "model_family": "LightGBM",
  "feature_count": 30,
  "selected_threshold": 0.5,
  "selection_objective": "minimize_business_cost_per_observation"
}
```

Le seuil par défaut de l'API est `0.50`, car le critère métier prioritaire du projet est la réduction du coût moyen.
Le seuil `0.45` reste documenté comme profil alternatif orienté rappel.

## Contrat externe et contrat interne

Deux contrats doivent être distingués.

Contrat externe :

- utilisé par les consommateurs de l'API ;
- reçoit des données brutes proches des tables métier ;
- ne demande pas de fournir les features TOP30 calculées.

Contrat interne :

- utilisé entre la couche de préparation et le service d'inférence ;
- contient les 30 features numériques attendues par le modèle ;
- correspond à `model/schema/top30_feature_schema.json`.

Le fichier `app/services/inference_service.py` correspond au contrat interne.
Une couche de préparation devra être ajoutée avant l'API ou dans un service dédié pour convertir les données brutes en features TOP30.

## Données brutes attendues

La route de prédiction attend un objet `raw_data` composé de blocs proches des tables historiques Home Credit.

Bloc obligatoire :

- `application` : données principales de la demande client.

Blocs optionnels mais utiles :

- `bureau` : crédits déclarés dans les bureaux de crédit ;
- `previous_applications` : précédentes demandes de crédit ;
- `installments_payments` : historique de paiements ;
- `credit_card_balance` : historique carte de crédit.

Si un bloc optionnel est absent ou vide, les features agrégées associées seront produites avec des valeurs manquantes.
LightGBM sait gérer les valeurs manquantes, mais l'absence de source devra être journalisée.

## Mapping brut vers TOP30

Les 30 variables modèle seront reconstruites à partir des données brutes selon le mapping cible suivant.

| Feature TOP30 | Source brute principale | Règle cible |
| --- | --- | --- |
| `ext_sources_mean` | `application` | moyenne de `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3` |
| `credit_to_annuity_ratio` | `application` | `AMT_CREDIT / AMT_ANNUITY` |
| `days_birth` | `application` | `DAYS_BIRTH` |
| `ext_source_1` | `application` | `EXT_SOURCE_1` |
| `ext_source_2` | `application` | `EXT_SOURCE_2` |
| `payment_rate` | `application` | `AMT_ANNUITY / AMT_CREDIT` |
| `credit_to_goods_ratio` | `application` | `AMT_CREDIT / AMT_GOODS_PRICE` |
| `amt_annuity` | `application` | `AMT_ANNUITY` |
| `days_employed` | `application` | `DAYS_EMPLOYED` |
| `approved_cnt_payment_mean` | `previous_applications` | moyenne de `CNT_PAYMENT` sur demandes approuvées |
| `ext_source_3` | `application` | `EXT_SOURCE_3` |
| `days_employed_perc` | `application` | `DAYS_EMPLOYED / DAYS_BIRTH` |
| `prev_cnt_payment_mean` | `previous_applications` | moyenne de `CNT_PAYMENT` |
| `new_active_debt_ratio` | `bureau` | ratio de dette active, à confirmer avec le pipeline P6 |
| `annuity_to_income_ratio` | `application` | `AMT_ANNUITY / AMT_INCOME_TOTAL` |
| `instal_amt_payment_sum` | `installments_payments` | somme de `AMT_PAYMENT` |
| `new_late_payment_ratio` | `installments_payments` | ratio de paiements en retard |
| `own_car_age` | `application` | `OWN_CAR_AGE` |
| `buro_amt_credit_max_overdue_mean` | `bureau` | moyenne de `AMT_CREDIT_MAX_OVERDUE` |
| `code_gender` | `application` | encodage numérique de `CODE_GENDER` |
| `amt_goods_price` | `application` | `AMT_GOODS_PRICE` |
| `days_id_publish` | `application` | `DAYS_ID_PUBLISH` |
| `active_days_credit_max` | `bureau` | maximum de `DAYS_CREDIT` sur crédits actifs |
| `active_days_credit_enddate_max` | `bureau` | maximum de `DAYS_CREDIT_ENDDATE` sur crédits actifs |
| `amt_credit` | `application` | `AMT_CREDIT` |
| `instal_days_entry_payment_max` | `installments_payments` | maximum de `DAYS_ENTRY_PAYMENT` |
| `new_bureau_debt_ratio` | `bureau` | ratio de dette bureau, à confirmer avec le pipeline P6 |
| `instal_payment_diff_mean` | `installments_payments` | moyenne de `AMT_INSTALMENT - AMT_PAYMENT` |
| `instal_dpd_mean` | `installments_payments` | moyenne des jours de retard calculés |
| `cc_cnt_drawings_atm_current_mean` | `credit_card_balance` | moyenne de `CNT_DRAWINGS_ATM_CURRENT` |

Les règles marquées “à confirmer avec le pipeline P6” devront être vérifiées dans les notebooks ou fonctions de préparation historiques avant l'implémentation.

## Routes prévues

### `GET /health`

Vérifie que l'API répond.

Réponse attendue :

```json
{
  "status": "ok",
  "service": "credit-scoring-api"
}
```

Codes HTTP :

- `200` : API disponible.

### `GET /model/info`

Retourne les informations principales du modèle chargé ou disponible.

Réponse attendue :

```json
{
  "model_name": "lightgbm_top30_optimized",
  "model_family": "LightGBM",
  "model_version": "top30_optimized",
  "model_uri": "model/artifacts/mlflow_model_top30_optimized",
  "schema_uri": "model/schema/top30_feature_schema.json",
  "feature_count": 30,
  "input_contract": "raw_business_data",
  "internal_feature_contract": "top30_features",
  "threshold": 0.5,
  "alternative_thresholds": {
    "recall_priority": 0.45,
    "cost_priority": 0.5
  },
  "selection_objective": "minimize_business_cost_per_observation"
}
```

Codes HTTP :

- `200` : métadonnées disponibles ;
- `503` : artefact modèle ou métadonnées indisponibles.

### `POST /predict`

Retourne une prédiction pour un client à partir de données brutes.

Requête attendue :

```json
{
  "client_id": 100001,
  "raw_data": {
    "application": {
      "AMT_CREDIT": 500000.0,
      "AMT_ANNUITY": 25000.0,
      "AMT_GOODS_PRICE": 450000.0,
      "AMT_INCOME_TOTAL": 180000.0,
      "DAYS_BIRTH": -16000,
      "DAYS_EMPLOYED": -2300,
      "DAYS_ID_PUBLISH": -3200,
      "EXT_SOURCE_1": 0.51,
      "EXT_SOURCE_2": 0.62,
      "EXT_SOURCE_3": 0.31,
      "OWN_CAR_AGE": 5.0,
      "CODE_GENDER": "M"
    },
    "bureau": [
      {
        "CREDIT_ACTIVE": "Active",
        "DAYS_CREDIT": -120,
        "DAYS_CREDIT_ENDDATE": 300,
        "AMT_CREDIT_SUM": 120000.0,
        "AMT_CREDIT_SUM_DEBT": 50000.0,
        "AMT_CREDIT_MAX_OVERDUE": 0.0
      }
    ],
    "previous_applications": [
      {
        "NAME_CONTRACT_STATUS": "Approved",
        "CNT_PAYMENT": 12.0
      }
    ],
    "installments_payments": [
      {
        "AMT_INSTALMENT": 10000.0,
        "AMT_PAYMENT": 10000.0,
        "DAYS_INSTALMENT": -40,
        "DAYS_ENTRY_PAYMENT": -39
      }
    ],
    "credit_card_balance": [
      {
        "CNT_DRAWINGS_ATM_CURRENT": 1.0
      }
    ]
  }
}
```

Remarques :

- `client_id` est optionnel ;
- `raw_data` est obligatoire ;
- `raw_data.application` est obligatoire ;
- les blocs historiques secondaires peuvent être absents ou vides ;
- les noms de champs bruts suivent volontairement les noms Home Credit en majuscules ;
- le backend normalisera ensuite les données et produira les 30 features internes.

Réponse attendue :

```json
{
  "client_id": 100001,
  "score": 0.72,
  "threshold": 0.5,
  "prediction": 1,
  "decision": "high_risk",
  "latency_ms": 5.4,
  "preprocessing_latency_ms": 2.1,
  "inference_latency_ms": 3.3,
  "model_version": "top30_optimized"
}
```

Interprétation :

- `score` : probabilité estimée de défaut ;
- `threshold` : seuil de décision métier appliqué ;
- `prediction` : classe prédite, `1` pour risque élevé, `0` pour risque faible ;
- `decision` : libellé lisible de la décision ;
- `latency_ms` : durée totale de traitement en millisecondes ;
- `preprocessing_latency_ms` : durée de transformation des données brutes en features TOP30 ;
- `inference_latency_ms` : durée d'appel modèle ;
- `model_version` : version documentée du modèle.

Codes HTTP :

- `200` : prédiction réalisée ;
- `400` : JSON invalide ou mal formé ;
- `422` : données valides au format JSON mais non conformes au contrat métier ;
- `503` : modèle indisponible ou impossible à charger.

### `POST /predict/batch`

Retourne des prédictions pour plusieurs clients dans une seule requête.

Cette route est prévue dès la première version de l'API afin de faciliter les tests d'intégration, les tests de performance et les futurs scénarios de monitoring.

Structure de requête :

```json
{
  "clients": [
    {
      "client_id": 100001,
      "raw_data": {
        "application": {
          "AMT_CREDIT": 500000.0,
          "AMT_ANNUITY": 25000.0,
          "AMT_GOODS_PRICE": 450000.0,
          "AMT_INCOME_TOTAL": 180000.0,
          "DAYS_BIRTH": -16000,
          "DAYS_EMPLOYED": -2300,
          "DAYS_ID_PUBLISH": -3200,
          "EXT_SOURCE_1": 0.51,
          "EXT_SOURCE_2": 0.62,
          "EXT_SOURCE_3": 0.31,
          "OWN_CAR_AGE": 5.0,
          "CODE_GENDER": "M"
        },
        "bureau": [],
        "previous_applications": [],
        "installments_payments": [],
        "credit_card_balance": []
      }
    }
  ]
}
```

Réponse attendue :

```json
{
  "predictions": [
    {
      "client_id": 100001,
      "score": 0.72,
      "threshold": 0.5,
      "prediction": 1,
      "decision": "high_risk"
    }
  ],
  "count": 1,
  "latency_ms": 6.8,
  "preprocessing_latency_ms": 3.2,
  "inference_latency_ms": 3.6,
  "model_version": "top30_optimized"
}
```

Remarques :

- chaque entrée du tableau `clients` suit le même contrat que la route `POST /predict` ;
- chaque client doit fournir un bloc `raw_data.application` ;
- la route doit refuser une liste vide ;
- une taille maximale de batch pourra être fixée pour éviter une surcharge mémoire.

Codes HTTP :

- `200` : prédictions réalisées ;
- `400` : JSON invalide ou mal formé ;
- `422` : lot vide, client invalide, variables brutes manquantes ou valeurs non conformes ;
- `503` : modèle indisponible ou impossible à charger.

## Validation des entrées brutes

Validations minimales à prévoir :

- présence du champ `raw_data` ;
- présence du bloc `raw_data.application` ;
- `application` doit être un objet JSON non vide ;
- les blocs secondaires doivent être des listes lorsqu'ils sont fournis ;
- les champs numériques doivent être convertibles en nombres ou être explicitement absents ;
- les valeurs infinies doivent être refusées ;
- certaines valeurs métier incohérentes doivent être refusées.

Exemples de valeurs incohérentes :

- `AMT_CREDIT < 0` ;
- `AMT_ANNUITY < 0` ;
- `AMT_GOODS_PRICE < 0` ;
- `AMT_INCOME_TOTAL <= 0` ;
- `DAYS_BIRTH > 0`, car les variables temporelles Home Credit sont exprimées en jours négatifs avant la date de demande ;
- `DAYS_EMPLOYED > 0`, pour la même raison.

## Préparation interne des features

Un service de préparation devra être ajouté avant l'appel au service d'inférence.

Responsabilités :

- lire le payload brut ;
- calculer les ratios et agrégats nécessaires ;
- produire un dictionnaire de 30 features internes ;
- conserver les noms et l'ordre du schéma TOP30 ;
- transmettre ce dictionnaire à `InferenceService.predict`.

Le service d'inférence existant reste utile : il est responsable du contrat interne et de l'appel modèle.
La nouvelle couche à créer sera responsable du contrat externe brut.

## Gestion des erreurs

Format recommandé des erreurs :

```json
{
  "detail": "Message d'erreur explicite"
}
```

Exemples :

- `raw_data est obligatoire` ;
- `raw_data.application est obligatoire` ;
- `Le champ application.AMT_CREDIT doit être numérique` ;
- `Le champ bureau doit être une liste` ;
- `Impossible de construire la feature credit_to_annuity_ratio : AMT_ANNUITY est nul ou manquant` ;
- `Le modèle est indisponible`.

## Chargement du modèle

Le modèle ne doit pas être chargé à chaque requête.

Stratégie cible :

1. créer un service applicatif responsable du modèle ;
2. charger l'artefact MLflow TOP30 une seule fois ;
3. conserver l'objet modèle en mémoire ;
4. réutiliser cet objet pour chaque appel à `/predict` et `/predict/batch`.

Cette stratégie réduit la latence, évite une surcharge mémoire et améliore la scalabilité.

## Tests attendus

Tests unitaires :

- lecture des métadonnées du modèle TOP30 ;
- lecture du schéma de variables TOP30 ;
- validation d'un payload brut complet ;
- rejet d'un payload sans `raw_data` ;
- rejet d'un payload sans `application` ;
- transformation d'un payload brut en 30 features ;
- rejet d'une valeur brute non numérique ;
- rejet d'une valeur métier incohérente ;
- conversion d'un score en décision avec le seuil `0.50`.

Tests d'intégration API :

- `GET /health` retourne `200` ;
- `GET /model/info` retourne les informations du modèle TOP30 ;
- `POST /predict` retourne une prédiction pour une entrée brute valide ;
- `POST /predict/batch` retourne plusieurs prédictions pour un lot valide ;
- `POST /predict` retourne une erreur pour une entrée vide ;
- `POST /predict` retourne une erreur pour un bloc `application` manquant ;
- `POST /predict/batch` retourne une erreur pour un lot vide ;
- le modèle n'est pas rechargé à chaque requête.

Les exemples de clients utilisés dans les tests seront construits à partir d'individus tirés au hasard dans l'ancien projet.
Ils devront être présentés sous forme de payloads bruts, pas sous forme de features TOP30 pré-calculées.

## Journalisation de production

La journalisation doit rester simple, lisible et exploitable pour les étapes de monitoring et d'optimisation.

Éléments à journaliser :

- route appelée ;
- statut HTTP retourné ;
- identifiant client, lorsqu'il est fourni ;
- présence ou absence des blocs secondaires ;
- nombre de clients traités pour les appels batch ;
- temps total de traitement ;
- temps de préparation des features ;
- temps d'inférence modèle ;
- version du modèle ;
- seuil appliqué ;
- message d'erreur en cas d'échec.

Ces informations serviront de base à l'étape 4 pour analyser les performances réelles ou simulées en production : latence, temps d'inférence, stabilité des réponses et identification des goulots d'étranglement.
Le format cible sera compatible avec une sortie standard de conteneur Docker et pourra être exploité dans Hugging Face Spaces ou dans un environnement de monitoring plus complet.

## Préparation Docker et CI/CD

Le contrat d'API doit rester compatible avec :

- exécution locale via `uv run uvicorn ...` ;
- exécution Docker sur le port `8000` ;
- tests automatisés avec `pytest` ;
- build Docker dans GitHub Actions ;
- déploiement sur Hugging Face Spaces après réussite des tests.

## Points à décider plus tard

- taille maximale autorisée pour un appel `POST /predict/batch` ;
- niveau de tolérance si certains blocs historiques secondaires sont absents ;
- format exact des fichiers de logs si une solution de monitoring plus structurée est ajoutée ;
- ajout éventuel d'une route de diagnostic ou de métriques, par exemple `/metrics`.
