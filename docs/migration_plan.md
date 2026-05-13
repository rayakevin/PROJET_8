# Plan d'import depuis le projet précédent

Règles :

- Ne pas commiter les données brutes volumineuses.
- Ne pas dépendre de chemins absolus issus de l'ancien projet.
- Ne pas utiliser `mlflow.db` comme dépendance runtime de l'API.
- Exporter uniquement les artefacts nécessaires au service de prédiction.
- Ajouter chaque étape par commit explicite.

Plan de commits envisagé :

1. `chore: initialize empty project structure`
2. `docs: document branching and migration strategy`
3. `feat: import legacy model metadata`
4. `feat: add model artifact export script`
5. `feat: add inference skeleton`
6. `feat: add scoring api`
7. `test: add api and inference tests`
8. `feat: add docker packaging`
9. `ci: add github actions pipeline`
10. `feat: add production logging and monitoring`
11. `perf: add inference benchmark and optimization report`
