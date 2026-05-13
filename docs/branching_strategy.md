# Stratégie de branches

Branches proposées :

- `main` : branche stable et livrable public.
- `develop` : branche d'intégration.
- `feature/model-import` : import contrôlé du modèle et des artefacts utiles.
- `feature/api` : API de scoring.
- `feature/tests-ci` : tests automatisés et pipeline CI.
- `feature/docker` : conteneurisation.
- `feature/monitoring` : stockage de production, drift et tableau de bord.
- `feature/performance` : analyse et optimisation des temps d'inférence.

Flux recommandé :

```text
feature/* -> develop -> main
```

Les noms de branches restent volontairement courts et explicites. Les descriptions, les PR et la documentation associée doivent être rédigées en français.
