"""Vérifie que le modèle MLflow importé peut être chargé.

Ce script appartient à l'étape d'import du modèle. Il ne sert pas encore le modèle ;
il vérifie seulement que l'artefact MLflow versionné est suffisamment autonome
pour la future API.
"""

from pathlib import Path

import mlflow.pyfunc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "model" / "artifacts" / "mlflow_model"


def main() -> None:
    model = mlflow.pyfunc.load_model(str(MODEL_DIR))
    print("Modèle chargé avec succès")
    print(f"URI du modèle : {MODEL_DIR}")
    print(f"UUID du modèle : {model.metadata.model_uuid}")
    print(f"Formats disponibles : {', '.join(sorted(model.metadata.flavors))}")


if __name__ == "__main__":
    main()
