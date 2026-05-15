import argparse
import cProfile
import json
import pstats
import sys
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app  # noqa: E402

DEFAULT_PAYLOAD_PATH = PROJECT_ROOT / "logs" / "batch_1000_random_clients.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "optimization"
DEFAULT_MONITORING_LOG_PATH = PROJECT_ROOT / "logs" / "api_predictions.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark local de l'API FastAPI sur un payload batch ou des appels simples."
    )
    parser.add_argument(
        "--endpoint",
        choices=["batch", "single"],
        default="batch",
        help="Endpoint à mesurer.",
    )
    parser.add_argument(
        "--payload",
        type=Path,
        default=DEFAULT_PAYLOAD_PATH,
        help="Payload JSON contenant une clé clients compatible avec /predict/batch.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Dossier de sortie des métriques et profils.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Nombre de répétitions du benchmark.",
    )
    parser.add_argument(
        "--label",
        default="baseline",
        help="Libellé utilisé dans les noms de fichiers de sortie.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Active cProfile sur la première répétition.",
    )
    parser.add_argument(
        "--reset-monitoring-log",
        action="store_true",
        help="Supprime le fichier de logs de monitoring avant le benchmark.",
    )
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None

    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower
    return round(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight, 3)


def summarize_values(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
        }

    return {
        "mean": round(mean(values), 3),
        "median": round(median(values), 3),
        "p95": percentile(values, 0.95),
        "max": round(max(values), 3),
    }


def load_payload(payload_path: Path) -> dict[str, Any]:
    if not payload_path.exists():
        raise FileNotFoundError(f"Payload introuvable : {payload_path}")

    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("clients"), list):
        raise ValueError("Le payload doit contenir une clé clients de type liste")

    if not payload["clients"]:
        raise ValueError("Le payload ne contient aucun client")

    return payload


def run_batch_request(client: TestClient, payload: dict[str, Any]) -> dict[str, Any]:
    start = perf_counter()
    response = client.post("/predict/batch", json=payload)
    wall_latency_ms = (perf_counter() - start) * 1000

    if response.status_code != 200:
        raise RuntimeError(f"Erreur API {response.status_code} : {response.text[:1000]}")

    body = response.json()
    body["wall_latency_ms"] = round(wall_latency_ms, 3)
    return body


def run_single_requests(client: TestClient, payload: dict[str, Any]) -> dict[str, Any]:
    predictions = []
    start = perf_counter()

    for client_payload in payload["clients"]:
        response = client.post("/predict", json=client_payload)
        if response.status_code != 200:
            raise RuntimeError(f"Erreur API {response.status_code} : {response.text[:1000]}")

        predictions.append(response.json())

    wall_latency_ms = (perf_counter() - start) * 1000
    preprocessing_latency_ms = sum(
        prediction["preprocessing_latency_ms"] for prediction in predictions
    )
    inference_latency_ms = sum(prediction["inference_latency_ms"] for prediction in predictions)

    return {
        "predictions": predictions,
        "count": len(predictions),
        "latency_ms": round(wall_latency_ms, 3),
        "wall_latency_ms": round(wall_latency_ms, 3),
        "preprocessing_latency_ms": round(preprocessing_latency_ms, 3),
        "inference_latency_ms": round(inference_latency_ms, 3),
        "model_version": predictions[0]["model_version"] if predictions else None,
    }


def run_request(client: TestClient, payload: dict[str, Any], endpoint: str) -> dict[str, Any]:
    if endpoint == "single":
        return run_single_requests(client, payload)

    return run_batch_request(client, payload)


def summarize_response(response_body: dict[str, Any]) -> dict[str, Any]:
    predictions = response_body["predictions"]
    latency_values = [prediction["latency_ms"] for prediction in predictions]
    preprocessing_values = [prediction["preprocessing_latency_ms"] for prediction in predictions]
    inference_values = [prediction["inference_latency_ms"] for prediction in predictions]
    scores = [prediction["score"] for prediction in predictions]
    high_risk_count = sum(1 for prediction in predictions if prediction["decision"] == "high_risk")
    low_risk_count = sum(1 for prediction in predictions if prediction["decision"] == "low_risk")

    return {
        "client_count": response_body["count"],
        "batch_latency_ms": response_body["latency_ms"],
        "wall_latency_ms": response_body["wall_latency_ms"],
        "throughput_clients_per_second": round(
            response_body["count"] / (response_body["wall_latency_ms"] / 1000),
            3,
        ),
        "latency_ms": summarize_values(latency_values),
        "preprocessing_latency_ms": summarize_values(preprocessing_values),
        "inference_latency_ms": summarize_values(inference_values),
        "scores": summarize_values(scores),
        "decisions": {
            "low_risk": low_risk_count,
            "high_risk": high_risk_count,
        },
    }


def write_profile(profile: cProfile.Profile, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as profile_file:
        stats = pstats.Stats(profile, stream=profile_file)
        stats.strip_dirs().sort_stats("cumtime").print_stats(40)


def run_benchmark(
    payload: dict[str, Any],
    *,
    endpoint: str,
    repeats: int,
    profile_enabled: bool,
    output_dir: Path,
    label: str,
) -> dict[str, Any]:
    runs = []

    with TestClient(app) as client:
        for repeat_index in range(repeats):
            if profile_enabled and repeat_index == 0:
                profile = cProfile.Profile()
                profile.enable()
                response_body = run_request(client, payload, endpoint)
                profile.disable()
                write_profile(profile, output_dir / f"profile_{label}.txt")
            else:
                response_body = run_request(client, payload, endpoint)

            runs.append(
                {
                    "repeat": repeat_index + 1,
                    **summarize_response(response_body),
                }
            )

    return {
        "endpoint": endpoint,
        "payload_client_count": len(payload["clients"]),
        "repeats": repeats,
        "runs": runs,
    }


def write_summary(summary: dict[str, Any], output_dir: Path, label: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{label}_benchmark.json"
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    args = parse_args()
    if args.reset_monitoring_log and DEFAULT_MONITORING_LOG_PATH.exists():
        DEFAULT_MONITORING_LOG_PATH.unlink()

    payload = load_payload(args.payload)
    summary = run_benchmark(
        payload,
        endpoint=args.endpoint,
        repeats=args.repeats,
        profile_enabled=args.profile,
        output_dir=args.output_dir,
        label=args.label,
    )
    output_path = write_summary(summary, args.output_dir, args.label)

    print(f"Benchmark généré : {output_path}")
    if args.profile:
        print(f"Profil cProfile généré : {args.output_dir / f'profile_{args.label}.txt'}")


if __name__ == "__main__":
    main()
