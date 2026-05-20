# Arborescence du projet

Le dépôt est organisé pour séparer clairement l'API, l'interface, le monitoring, les artefacts modèle,
les scripts et la documentation.

```text
PROJET_8/
├── .github/workflows/       # pipeline CI/CD GitHub Actions
├── app/                     # API FastAPI et services de preprocessing, inférence, monitoring
├── dashboard/               # dashboard Streamlit local de monitoring
├── docs/                    # documentation projet et rapports d'expérience
├── model/                   # artefacts MLflow, schémas et métadonnées modèle
├── monitoring/              # schéma PostgreSQL et référence de drift générée localement
├── notebooks/legacy/        # notebooks conservés de l'ancien projet P6
├── scripts/                 # scripts de contrôle, benchmark, import et analyse
├── tests/                   # tests unitaires et d'intégration
├── ui/                      # interface Streamlit de scoring
├── Dockerfile               # image Docker de l'API
├── Dockerfile.ui            # image Docker de l'interface Streamlit
├── docker-compose.monitoring.yml
├── pyproject.toml
└── uv.lock
```

Les données brutes, les rapports générés, la référence de drift locale, les logs, la base MLflow
historique et les caches d'outils ne sont pas versionnés. Cette règle évite de publier des fichiers
lourds, sensibles ou dépendants de l'environnement local.

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
