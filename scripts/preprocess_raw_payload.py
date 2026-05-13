import argparse
import json
import math
from pathlib import Path
from typing import Any

from app.services.preprocessing_service import PreprocessingService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transforme un payload brut en features TOP30.")
    parser.add_argument(
        "input",
        type=Path,
        help="Chemin du fichier JSON contenant raw_data ou un payload de prediction.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Chemin optionnel du fichier JSON de sortie.",
    )
    return parser.parse_args()


def replace_nan_with_none(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None

    if isinstance(value, dict):
        return {key: replace_nan_with_none(item) for key, item in value.items()}

    if isinstance(value, list):
        return [replace_nan_with_none(item) for item in value]

    return value


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    raw_data = payload.get("raw_data", payload)

    features = PreprocessingService().transform(raw_data)
    output_payload = {"features": replace_nan_with_none(features)}
    output_text = json.dumps(output_payload, ensure_ascii=False, indent=2)

    if args.output:
        args.output.write_text(output_text + "\n", encoding="utf-8")
    else:
        print(output_text)


if __name__ == "__main__":
    main()
