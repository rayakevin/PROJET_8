# Stratégie de branches

Branches utilisées :

- `main` : branche stable et livrable public.
- `develop` : branche d'intégration.
- `feature/model-import` : import contrôlé du modèle et des artefacts utiles.
- `feature/api` : API de scoring.
- `feature/tests-ci` : tests automatisés et pipeline CI.
- `feature/docker` : conteneurisation.
- `feature/monitoring` : stockage local, drift et tableau de bord.
- `feature/performance-optimization` : analyse et optimisation des temps d'inférence.

Flux retenu :

```text
feature/* -> develop -> main
```

Les noms de branches restent courts et explicites. Les descriptions, les PR et la documentation
associée sont rédigées en français.
