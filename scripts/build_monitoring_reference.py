import argparse
import json
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_PATH = (
    PROJECT_ROOT / "data" / "reference" / "application_train_modeling_sample.parquet"
)
FEATURE_SCHEMA_PATH = PROJECT_ROOT / "model" / "schema" / "top30_feature_schema.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "monitoring" / "reference" / "top30_reference.parquet"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "monitoring" / "reference" / "top30_reference_metadata.json"


def format_project_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construit la référence de drift à partir du dataset préparé versionné."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE_PATH,
        help="Chemin du parquet préparé.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Chemin du parquet de référence à générer.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Chemin du fichier JSON de métadonnées.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10000,
        help="Nombre maximal de lignes à conserver dans la référence.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Graine d'échantillonnage reproductible.",
    )
    return parser.parse_args()


def normalize_column_name(column_name: str) -> str:
    normalized = column_name.lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized)
    return normalized.strip("_")


def load_feature_names() -> list[str]:
    schema = json.loads(FEATURE_SCHEMA_PATH.read_text(encoding="utf-8"))
    return [item["feature"] for item in schema["features"]]


def build_reference(
    source_path: Path,
    feature_names: list[str],
    sample_size: int,
    random_state: int,
) -> pd.DataFrame:
    if not source_path.exists():
        raise FileNotFoundError(f"Dataset préparé introuvable : {source_path}")

    source = pd.read_parquet(source_path)
    source = source.rename(
        columns={column: normalize_column_name(column) for column in source.columns}
    )

    missing_features = [feature for feature in feature_names if feature not in source.columns]
    if missing_features:
        raise ValueError(f"Features absentes de la source : {', '.join(missing_features)}")

    reference = source[feature_names].copy()

    if sample_size > 0 and len(reference) > sample_size:
        reference = reference.sample(n=sample_size, random_state=random_state)

    return reference.reset_index(drop=True)


def write_metadata(
    metadata_path: Path,
    source_path: Path,
    output_path: Path,
    reference: pd.DataFrame,
    feature_names: list[str],
    sample_size: int,
    random_state: int,
) -> None:
    metadata = {
        "source_path": format_project_path(source_path),
        "output_path": format_project_path(output_path),
        "feature_count": len(feature_names),
        "row_count": len(reference),
        "sample_size": sample_size,
        "random_state": random_state,
        "features": feature_names,
    }

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    feature_names = load_feature_names()
    reference = build_reference(
        source_path=args.source,
        feature_names=feature_names,
        sample_size=args.sample_size,
        random_state=args.random_state,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    reference.to_parquet(args.output, index=False)
    write_metadata(
        metadata_path=args.metadata_output,
        source_path=args.source,
        output_path=args.output,
        reference=reference,
        feature_names=feature_names,
        sample_size=args.sample_size,
        random_state=args.random_state,
    )

    print(f"Référence générée : {args.output}")
    print(f"Métadonnées générées : {args.metadata_output}")
    print(f"Lignes : {len(reference)}")
    print(f"Features : {len(feature_names)}")


if __name__ == "__main__":
    main()
