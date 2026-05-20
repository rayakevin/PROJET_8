# Contrat de l'API de scoring

L'API expose le modèle `lightgbm_top30_optimized` via FastAPI. Elle reçoit des données brutes métier,
les transforme en features TOP30, puis retourne un score de risque et une décision.

## Principe retenu

Le consommateur de l'API n'envoie pas les 30 features internes du modèle. Il envoie des blocs proches
des données sources :

- `application` : informations principales du client et de la demande ;
- `bureau` : historiques de crédits externes ;
- `previous_applications` : anciennes demandes ;
- `installments_payments` : paiements d'échéances ;
- `credit_card_balance` : données de carte de crédit.

Le service `PreprocessingService` calcule ensuite les 30 features attendues par le modèle. Ce choix
centralise le feature engineering côté API et évite de rendre le contrat externe dépendant du schéma
interne du modèle.

## Modèle utilisé

- artefact MLflow : `model/artifacts/mlflow_model_top30_optimized/` ;
- schéma interne : `model/schema/top30_feature_schema.json` ;
- métadonnées : `model/schema/top30_model_metadata.json` ;
- seuil de décision : `0.5`.

Le modèle est chargé une seule fois par `InferenceService`. Un warm-up est exécuté au démarrage de
l'API pour éviter que le premier appel utilisateur supporte le coût de chargement MLflow.

## Endpoints

### `GET /health`

Vérifie que l'API répond.

Réponse :

```json
{
  "status": "ok",
  "service": "credit-scoring-api"
}
```

### `GET /model/info`

Retourne les informations principales du modèle chargé.

Réponse type :

```json
{
  "model_name": "lightgbm_top30_optimized",
  "model_family": "lightgbm",
  "feature_count": 30,
  "threshold": 0.5,
  "input_contract": "raw_business_data",
  "internal_feature_contract": "top30_features"
}
```

### `POST /predict`

Score un client.

Payload :

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
    "bureau": [],
    "previous_applications": [],
    "installments_payments": [],
    "credit_card_balance": []
  }
}
```

Réponse :

```json
{
  "client_id": 100001,
  "score": 0.24,
  "prediction": 0,
  "decision": "low_risk",
  "threshold": 0.5,
  "model_version": "lightgbm_top30_optimized",
  "latency_ms": 2.1,
  "preprocessing_latency_ms": 0.4,
  "inference_latency_ms": 1.7
}
```

### `POST /predict/batch`

Score un lot de clients.

Payload :

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
        }
      }
    }
  ]
}
```

Réponse :

```json
{
  "predictions": [
    {
      "client_id": 100001,
      "score": 0.24,
      "prediction": 0,
      "decision": "low_risk",
      "threshold": 0.5,
      "model_version": "lightgbm_top30_optimized",
      "latency_ms": 0.11,
      "preprocessing_latency_ms": 0.09,
      "inference_latency_ms": 0.02
    }
  ],
  "count": 1,
  "latency_ms": 1.2,
  "preprocessing_latency_ms": 0.09,
  "inference_latency_ms": 0.02,
  "model_version": "lightgbm_top30_optimized"
}
```

Le batch utilise une inférence vectorisée : les features sont construites client par client, puis
envoyées au modèle dans un seul `DataFrame`.

## Blocs d'entrée

Le bloc `application` est obligatoire. Les autres blocs sont optionnels et peuvent être absents,
vides ou égaux à `null`. Dans ce cas, les agrégats associés sont produits avec des valeurs manquantes,
que LightGBM sait gérer.

Champs principaux utilisés dans `application` :

- `AMT_CREDIT` ;
- `AMT_ANNUITY` ;
- `AMT_GOODS_PRICE` ;
- `AMT_INCOME_TOTAL` ;
- `DAYS_BIRTH` ;
- `DAYS_EMPLOYED` ;
- `DAYS_ID_PUBLISH` ;
- `EXT_SOURCE_1` ;
- `EXT_SOURCE_2` ;
- `EXT_SOURCE_3` ;
- `OWN_CAR_AGE` ;
- `CODE_GENDER`.

Champs utilisés dans les blocs historiques :

- `bureau` : `CREDIT_ACTIVE`, `DAYS_CREDIT`, `DAYS_CREDIT_ENDDATE`, `AMT_CREDIT_SUM`,
  `AMT_CREDIT_SUM_DEBT`, `AMT_CREDIT_MAX_OVERDUE` ;
- `previous_applications` : `NAME_CONTRACT_STATUS`, `CNT_PAYMENT` ;
- `installments_payments` : `AMT_INSTALMENT`, `AMT_PAYMENT`, `DAYS_INSTALMENT`,
  `DAYS_ENTRY_PAYMENT` ;
- `credit_card_balance` : `CNT_DRAWINGS_ATM_CURRENT`.

## Validation et erreurs

L'API retourne `422` lorsque le payload ne peut pas être transformé en features modèle.

Cas couverts :

- `raw_data` absent ou non JSON ;
- bloc `application` absent ;
- bloc historique fourni avec un type différent d'une liste ;
- champ numérique non convertible ;
- valeur infinie ;
- genre différent de `F`, `M`, `XNA` ou valeur manquante.

Les valeurs manquantes ordinaires restent acceptées. Elles sont converties en `NaN` et transmises au
modèle.

## Monitoring

Chaque appel à `/predict` et chaque client de `/predict/batch` produit un événement JSONL. Les logs
contiennent :

- identifiant de requête ;
- endpoint ;
- statut ;
- identifiant client ;
- features TOP30 ;
- score, prédiction et décision ;
- latence totale ;
- temps de preprocessing ;
- temps d'inférence ;
- informations d'erreur si nécessaire.

Le batch écrit les événements en une seule opération groupée afin de réduire le coût d'I/O sans
changer le format des logs.

## Documentation interactive

FastAPI expose Swagger sur `/docs`. Les endpoints de prédiction contiennent des exemples valides
préremplis, ce qui permet de tester l'API directement avec `Try it out`.
