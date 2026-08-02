#!/usr/bin/env python3
"""Measure warm document and facial pipeline latency."""

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Callable

from PIL import Image

from identity_analysis.api import (
    compare_document_portrait,
    facial_models,
    liveness_model,
    recognition_model,
)
from identity_analysis.document_classifier import classify_document
from identity_analysis.facial_identity import compare_face_templates
from identity_analysis.pipeline import (
    process_document,
    process_document_pages,
    process_document_pair,
    warm_up,
)
from identity_analysis.quality import analyze_quality


def measure(operation: Callable[[], object], iterations: int) -> dict[str, float]:
    operation()
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        operation()
        samples.append((time.perf_counter() - started) * 1000)
    ordered = sorted(samples)
    percentile_index = min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))
    return {
        "iterations": iterations,
        "minimumMs": round(ordered[0], 3),
        "medianMs": round(statistics.median(ordered), 3),
        "p95Ms": round(ordered[percentile_index], 3),
        "maximumMs": round(ordered[-1], 3),
    }


def benchmark(root: Path, iterations: int) -> dict:
    assets = root / "assets"
    samples = root / "examples/samples"
    document_path = samples / "synthetic_id_front.jpg"
    reverse_path = samples / "synthetic_id_back.jpg"
    legacy_front_path = samples / "synthetic_id_front.jpg"
    legacy_reverse_path = samples / "synthetic_id_back.jpg"
    selfie_path = root / "examples/samples/synthetic_selfie.jpg"
    warm_up(assets)
    with Image.open(document_path) as source:
        document_image = source.convert("RGB")
    documents = {
        "classification": measure(
            lambda: classify_document(assets, document_image, 10),
            iterations,
        ),
        "frontAutomatic": measure(
            lambda: process_document(document_path, assets),
            iterations,
        ),
        "frontExact": measure(
            lambda: process_document(
                document_path,
                assets,
                "auto_research",
                448926514,
            ),
            iterations,
        ),
        "reverseAutomatic": measure(
            lambda: process_document(reverse_path, assets),
            iterations,
        ),
        "pairAutomatic": measure(
            lambda: process_document_pair(document_path, reverse_path, assets),
            iterations,
        ),
        "pairExplicitProfile": measure(
            lambda: process_document_pair(
                document_path,
                reverse_path,
                assets,
                "mex_ine",
            ),
            iterations,
        ),
        "pairExact": measure(
            lambda: process_document_pair(
                document_path,
                reverse_path,
                assets,
                "auto_research",
                448926514,
                448927101,
            ),
            iterations,
        ),
        "pagesAutomatic": measure(
            lambda: process_document_pages(
                [legacy_front_path, legacy_reverse_path, legacy_reverse_path],
                assets,
            ),
            iterations,
        ),
        "quality": measure(
            lambda: analyze_quality(assets, document_image),
            iterations,
        ),
    }

    detector, landmarks = facial_models()
    liveness = liveness_model()
    recognition = recognition_model()
    with Image.open(selfie_path) as source:
        selfie = source.convert("RGB")
    detection = detector.detect(selfie)[0]
    template = recognition.embedding_for_detection(selfie, detection)["vector"]
    facial = {
        "detection": measure(lambda: detector.detect(selfie), iterations),
        "landmarks": measure(
            lambda: landmarks.infer(selfie, detection), iterations
        ),
        "passiveLiveness": measure(
            lambda: liveness.infer(selfie, detection), iterations
        ),
        "embedding": measure(lambda: recognition.embedding(selfie), iterations),
        "comparison": measure(
            lambda: recognition.compare(selfie, selfie),
            iterations,
        ),
        "templateComparison": measure(
            lambda: compare_face_templates(template, template),
            iterations,
        ),
    }
    document_result = process_document(
        document_path,
        assets,
        "auto_research",
        448926514,
    )
    document_payload = document_path.read_bytes()
    verification = {
        "portraitComparison": measure(
            lambda: compare_document_portrait(
                document_result,
                document_payload,
                selfie,
                recognition,
                0.67,
            ),
            iterations,
        ),
        "portraitComparisonWithLiveness": measure(
            lambda: compare_document_portrait(
                document_result,
                document_payload,
                selfie,
                recognition,
                0.67,
                liveness,
            ),
            iterations,
        ),
    }
    return {
        "documents": documents,
        "facial": facial,
        "verification": verification,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if args.iterations < 2:
        parser.error("--iterations must be at least 2")
    print(json.dumps(benchmark(args.root.resolve(), args.iterations), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
