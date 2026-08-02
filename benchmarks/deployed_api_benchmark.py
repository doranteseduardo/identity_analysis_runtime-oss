#!/usr/bin/env python3
"""Measure deployed API latency and transactions per minute."""

from __future__ import annotations

import argparse
import base64
import json
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from uuid import uuid4


@dataclass(frozen=True)
class Operation:
    name: str
    group: str
    request: Callable[[], urllib.request.Request]


def multipart(
    fields: list[tuple[str, str]],
    files: list[tuple[str, Path]],
) -> tuple[bytes, str]:
    boundary = f"identity-analysis-{uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields:
        chunks.extend(
            (
                f"--{boundary}\r\n".encode(),
                (
                    f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                ).encode(),
                value.encode(),
                b"\r\n",
            )
        )
    for name, path in files:
        chunks.extend(
            (
                f"--{boundary}\r\n".encode(),
                (
                    f'Content-Disposition: form-data; name="{name}"; '
                    f'filename="{path.name}"\r\n'
                ).encode(),
                b"Content-Type: image/jpeg\r\n\r\n",
                path.read_bytes(),
                b"\r\n",
            )
        )
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def get_request(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, method="GET")


def multipart_request(
    url: str,
    fields: list[tuple[str, str]],
    files: list[tuple[str, Path]],
) -> urllib.request.Request:
    body, content_type = multipart(fields, files)
    return urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": content_type},
        method="POST",
    )


def json_request(url: str, value: dict) -> urllib.request.Request:
    return urllib.request.Request(
        url,
        data=json.dumps(value).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )


def send(
    operation: Operation,
    timeout: float,
) -> tuple[int, float, float | None, str | None]:
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(operation.request(), timeout=timeout) as response:
            response.read()
            elapsed_ms = (time.perf_counter() - started) * 1000
            server_value = response.headers.get("X-Process-Time-Ms")
            return (
                response.status,
                elapsed_ms,
                float(server_value) if server_value is not None else None,
                None,
            )
    except urllib.error.HTTPError as error:
        body = error.read().decode(errors="replace")[:500]
        return (
            error.code,
            (time.perf_counter() - started) * 1000,
            None,
            body,
        )
    except Exception as error:
        return (
            0,
            (time.perf_counter() - started) * 1000,
            None,
            str(error),
        )


def percentile(values: list[float], fraction: float) -> float:
    return values[round(fraction * (len(values) - 1))]


def benchmark_operation(
    operation: Operation,
    iterations: int,
    concurrency: int,
    warmup: int,
    timeout: float,
) -> dict:
    for _ in range(warmup):
        send(operation, timeout)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        results = list(
            executor.map(
                lambda _: send(operation, timeout),
                range(iterations),
            )
        )
    wall_seconds = time.perf_counter() - started
    client_times = sorted(result[1] for result in results)
    server_times = sorted(
        result[2] for result in results if result[2] is not None
    )
    successes = sum(200 <= result[0] < 300 for result in results)
    report = {
        "name": operation.name,
        "group": operation.group,
        "concurrency": concurrency,
        "requests": iterations,
        "successes": successes,
        "errors": iterations - successes,
        "statusCodes": sorted({result[0] for result in results}),
        "wallSeconds": round(wall_seconds, 3),
        "transactionsPerMinute": round(successes / wall_seconds * 60, 2),
        "clientLatencyMs": {
            "median": round(statistics.median(client_times), 2),
            "p95": round(percentile(client_times, 0.95), 2),
            "maximum": round(client_times[-1], 2),
        },
    }
    if server_times:
        report["serverLatencyMs"] = {
            "median": round(statistics.median(server_times), 2),
            "p95": round(percentile(server_times, 0.95), 2),
            "maximum": round(server_times[-1], 2),
        }
    messages = sorted({result[3] for result in results if result[3]})
    if messages:
        report["errorMessages"] = messages
    return report


def extract_template(
    base_url: str,
    selfie: Path,
    timeout: float,
) -> str:
    request = multipart_request(
        f"{base_url}/v1/face/template",
        [],
        [("image", selfie)],
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    return payload["data"]["templateBase64"]


def operations(
    base_url: str,
    samples: Path,
    document_identifier: int,
    template: str,
) -> list[Operation]:
    front = samples / "synthetic_id_front.jpg"
    back = samples / "synthetic_id_back.jpg"
    legacy_front = samples / "synthetic_id_front.jpg"
    legacy_back = samples / "synthetic_id_back.jpg"
    selfie = samples / "synthetic_selfie.jpg"
    encoded_identifier = urllib.parse.quote(str(document_identifier))

    def multipart_operation(
        name: str,
        group: str,
        path: str,
        fields: list[tuple[str, str]],
        files: list[tuple[str, Path]],
    ) -> Operation:
        return Operation(
            name,
            group,
            lambda: multipart_request(
                f"{base_url}{path}",
                fields,
                files,
            ),
        )

    return [
        Operation("health", "metadata", lambda: get_request(f"{base_url}/health")),
        Operation(
            "capabilities",
            "metadata",
            lambda: get_request(f"{base_url}/v1/capabilities"),
        ),
        Operation(
            "catalogSearch",
            "metadata",
            lambda: get_request(
                f"{base_url}/v1/document/catalog?q=passport&limit=10"
            ),
        ),
        Operation(
            "catalogFacets",
            "metadata",
            lambda: get_request(f"{base_url}/v1/document/catalog/facets"),
        ),
        Operation(
            "catalogDetail",
            "metadata",
            lambda: get_request(
                f"{base_url}/v1/document/catalog/{encoded_identifier}"
            ),
        ),
        Operation(
            "layoutEvidence",
            "metadata",
            lambda: get_request(
                f"{base_url}/v1/document/layout/{encoded_identifier}/evidence"
            ),
        ),
        multipart_operation(
            "documentClassification",
            "document",
            "/v1/document/classify",
            [("topK", "5")],
            [("image", front)],
        ),
        multipart_operation(
            "layoutOcr",
            "document",
            f"/v1/document/layout/{encoded_identifier}/ocr",
            [],
            [("image", front)],
        ),
        multipart_operation(
            "referenceMetrics",
            "document",
            "/v1/document/reference-metrics",
            [("documentIdentifier", str(document_identifier))],
            [("image", front)],
        ),
        multipart_operation(
            "ocrSingle",
            "document",
            "/v1/ocr",
            [("profile", "auto_research")],
            [("image", front)],
        ),
        multipart_operation(
            "ocrPair",
            "document",
            "/v1/ocr/pair",
            [("profile", "auto_research")],
            [("frontImage", front), ("backImage", back)],
        ),
        multipart_operation(
            "ocrPages",
            "document",
            "/v1/ocr/pages",
            [("profile", "auto_research")],
            [
                ("images", legacy_front),
                ("images", legacy_back),
                ("images", legacy_back),
            ],
        ),
        multipart_operation(
            "faceAnalyze",
            "facial",
            "/v1/face/analyze",
            [],
            [("image", selfie)],
        ),
        multipart_operation(
            "documentPortraitCompare",
            "verification",
            "/v1/document/portrait/compare",
            [("profile", "mex_ine"), ("threshold", "0.67")],
            [("documentImage", front), ("selfieImage", selfie)],
        ),
        multipart_operation(
            "faceLiveness",
            "facial",
            "/v1/face/liveness",
            [("threshold", "0.37"), ("spoofThreshold", "0.25")],
            [("image", selfie)],
        ),
        multipart_operation(
            "faceCompare",
            "facial",
            "/v1/face/compare",
            [("threshold", "0.67"), ("includeTemplates", "false")],
            [("firstImage", selfie), ("secondImage", selfie)],
        ),
        multipart_operation(
            "faceTemplate",
            "facial",
            "/v1/face/template",
            [],
            [("image", selfie)],
        ),
        Operation(
            "faceTemplateCompare",
            "facial",
            lambda: json_request(
                f"{base_url}/v1/face/template/compare",
                {
                    "template1Base64": template,
                    "template2Base64": template,
                    "threshold": 0.67,
                },
            ),
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=Path("examples/samples"),
    )
    parser.add_argument("--document-identifier", type=int, default=448926514)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1])
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0:
        parser.error("iterations must be positive and warmup cannot be negative")
    if any(value < 1 for value in args.concurrency):
        parser.error("concurrency values must be positive")

    base_url = args.base_url.rstrip("/")
    samples = args.samples.resolve()
    template = extract_template(base_url, samples / "synthetic_selfie.jpg", args.timeout)
    selected_operations = operations(
        base_url,
        samples,
        args.document_identifier,
        template,
    )
    started = time.time()
    reports = []
    for concurrency in args.concurrency:
        for operation in selected_operations:
            report = benchmark_operation(
                operation,
                args.iterations,
                concurrency,
                args.warmup,
                args.timeout,
            )
            reports.append(report)
            print(
                f"{operation.name:26} c={concurrency} "
                f"{report['transactionsPerMinute']:8.2f} TPM "
                f"errors={report['errors']}",
                flush=True,
            )
    result = {
        "baseUrl": base_url,
        "startedAtEpoch": started,
        "iterationsPerOperation": args.iterations,
        "warmupPerOperation": args.warmup,
        "concurrencyLevels": args.concurrency,
        "documentIdentifier": args.document_identifier,
        "reports": reports,
    }
    encoded = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    else:
        print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
