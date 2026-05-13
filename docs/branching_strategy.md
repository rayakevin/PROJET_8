# Stratégie de branches

Branches proposées :

- `main` : branche stable et livrable public.
- `develop` : branche d'intégration.
- `feature/model-import` : import contrôlé du modèle et des artefacts utiles.
- `feature/api` : API de scoring.
- `feature/tests-ci` : tests automatisés et pipeline CI.
- `feature/docker` : conteneurisation.
- `feature/monitoring` : stockage production, drift et dashboard.
- `feature/performance` : analyse et optimisation des temps d'inférence.

Flux recommandé :

```text
feature/* -> develop -> main
```
