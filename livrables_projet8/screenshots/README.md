# Captures - Solution de stockage des données de production

Ce dossier contient les captures demandées pour illustrer la solution de stockage mise en place dans
l'étape 3.

- `01_conteneur_postgresql.png` : conteneur PostgreSQL local lancé avec Docker Compose.
- `02_schema_prediction_logs.png` : schéma SQL de la table `prediction_logs`.
- `03_evenements_stockes_postgresql.png` : synthèse des événements stockés en base et exemples de
  lignes.
- `04_artefacts_monitoring_generes.png` : logs JSONL et rapports de monitoring générés.
- `05_DB_pgAdmin.png` : visualisation de la base PostgreSQL dans pgAdmin.

La solution repose sur un double niveau de stockage :

1. logs structurés JSONL produits par l'API ;
2. import dans PostgreSQL local pour requêtage, analyse et dashboard.
