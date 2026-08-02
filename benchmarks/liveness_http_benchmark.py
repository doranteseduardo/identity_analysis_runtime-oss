#!/usr/bin/env python3
"""Benchmark passive-liveness HTTP latency and throughput."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4


def multipart_image(path: Path) -> tuple[bytes, str]:
    boundary = f"identity-analysis-{uuid4().hex}"
    payload = path.read_bytes()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{path.name}"\r\n'
        "Content-Type: image/jpeg\r\n\r\n"
    ).encode("ascii") + payload + f"\r\n--{boundary}--\r\n".encode("ascii")
    return body, f"multipart/form-data; boundary={boundary}"


def percentile(values: list[float], fraction: float) -> float:
    index = round(fraction * (len(values) - 1))
    return values[index]


def benchmark_level(
    url: str,
    body: bytes,
    content_type: str,
    requests: int,
    concurrency: int,
    timeout: float,
) -> dict:
    def send() -> tuple[int, float, float | None]:
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": content_type},
            method="POST",
        )
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read()
            elapsed_ms = (time.perf_counter() - started) * 1000
            server_header = response.headers.get("X-Process-Time-Ms")
            return (
                response.status,
                elapsed_ms,
                float(server_header) if server_header is not None else None,
            )

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        results = list(executor.map(lambda _: send(), range(requests)))
    wall_seconds = time.perf_counter() - started
    client_times = sorted(result[1] for result in results)
    server_times = sorted(
        result[2] for result in results if result[2] is not None
    )
    report = {
        "concurrency": concurrency,
        "requests": requests,
        "http200": sum(result[0] == 200 for result in results),
        "wallSeconds": round(wall_seconds, 3),
        "throughputRps": round(requests / wall_seconds, 3),
        "clientLatencyMs": {
            "median": round(statistics.median(client_times), 3),
            "p95": round(percentile(client_times, 0.95), 3),
            "maximum": round(client_times[-1], 3),
        },
    }
    if server_times:
        report["serverLatencyMs"] = {
            "median": round(statistics.median(server_times), 3),
            "p95": round(percentile(server_times, 0.95), 3),
            "maximum": round(server_times[-1], 3),
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        default="http://localhost:8000/v1/face/liveness",
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=40)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    if args.requests < 1 or args.warmup < 0:
        parser.error("requests must be positive and warmup cannot be negative")
    if any(value < 1 for value in args.concurrency):
        parser.error("concurrency values must be positive")
    body, content_type = multipart_image(args.image.resolve())
    for _ in range(args.warmup):
        benchmark_level(args.url, body, content_type, 1, 1, args.timeout)
    report = {
        "url": args.url,
        "image": str(args.image.resolve()),
        "results": [
            benchmark_level(
                args.url,
                body,
                content_type,
                args.requests,
                concurrency,
                args.timeout,
            )
            for concurrency in args.concurrency
        ],
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
