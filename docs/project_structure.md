# Arborescence projet

```text
PROJET_8/
├── .github/workflows/      # futurs workflows CI/CD
├── app/                    # future API FastAPI
├── src/scoring/            # logique métier réutilisable
├── tests/                  # tests unitaires et intégration
├── notebooks/              # analyses, drift, optimisation
├── scripts/                # scripts d'import, inférence, monitoring
├── model/                  # artefacts modèle et schémas
├── data/                   # jeux de référence, exemples, journaux locaux
├── dashboard/              # futur tableau de bord de monitoring
├── docs/                   # documentation projet
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
└── pyproject.toml
```

Cette structure est volontairement vide à l'initialisation. Les fichiers métier seront ajoutés par commits explicites.
