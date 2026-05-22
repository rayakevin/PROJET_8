# Arborescence du projet

Le dépôt est organisé pour séparer clairement l'API, l'interface, le monitoring, les artefacts modèle,
les scripts et la documentation.

```text
PROJET_8/
├── .github/workflows/       # pipeline CI/CD GitHub Actions
├── app/                     # API FastAPI et services de preprocessing, inférence, monitoring
├── dashboard/               # dashboard Streamlit local de monitoring
├── docs/                    # documentation projet et rapports d'expérience
├── data/reference/          # données légères versionnées pour installer et rejouer les contrôles
├── model/                   # artefacts MLflow, schémas et métadonnées modèle
├── monitoring/              # schéma PostgreSQL et référence de drift TOP30
├── notebooks/               # notebook d'analyse du drift
├── scripts/                 # scripts de contrôle, benchmark, import et analyse
├── tests/                   # tests unitaires et d'intégration
├── ui/                      # interface Streamlit de scoring
├── Dockerfile               # image Docker de l'API
├── Dockerfile.ui            # image Docker de l'interface Streamlit
├── docker-compose.monitoring.yml
├── pyproject.toml
└── uv.lock
```

Les données brutes complètes, les rapports générés, les logs, la base MLflow historique, les anciens
notebooks de construction et les caches d'outils ne sont pas versionnés. Le dépôt embarque uniquement
les artefacts légers nécessaires à une installation autonome : l'échantillon
`data/reference/application_train_modeling_sample.parquet`, l'importance
`data/reference/lightgbm_bonus_native_importance.csv` et la référence de drift
`monitoring/reference/top30_reference.parquet`.

## Logique applicative

Le flux applicatif principal est le suivant :

```text
payload brut
  -> PreprocessingService
  -> features TOP30
  -> InferenceService
  -> score, prédiction, décision
  -> MonitoringService
  -> logs JSONL
```

Le monitoring local ajoute ensuite :

```text
logs JSONL
  -> import PostgreSQL
  -> analyse Pandas et Evidently
  -> rapports JSON / HTML
  -> dashboard Streamlit
```
