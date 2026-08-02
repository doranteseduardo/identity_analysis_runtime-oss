# Performance Measurements

This document records measured runtime behavior. Values are tied to the stated
hardware, container configuration, samples, warm-up state, and request counts.
They are not service-level objectives and should not be extrapolated to other
environments.

## Benchmark Tools

The repository includes two benchmark entry points:

- `benchmarks/runtime_benchmark.py` measures pipelines and model stages in the
  local Python runtime.
- `benchmarks/deployed_api_benchmark.py` measures the complete HTTP contract,
  including request parsing, pipeline execution, serialization, and transfer.

The liveness endpoint can also be measured independently with
`benchmarks/liveness_http_benchmark.py`.

Every HTTP response includes:

- `Server-Timing: app;dur=<milliseconds>`
- `X-Process-Time-Ms: <milliseconds>`

These headers report server-side processing time through the API middleware.
Client wall time additionally includes upload, response transfer, client-side
parsing, and rendering.

## Local Runtime Measurements

### Environment

The measurements in this section were collected on the local Apple Silicon
development machine after model warm-up. Unless otherwise noted, ten iterations
were used for document and facial component measurements.

```bash
PYTHONPATH=src .venv/bin/python benchmarks/runtime_benchmark.py \
  --iterations 20
```

### Document Front

A complete warm document-front request measured approximately 164 ms median.
The profiled request made 42 ONNX Runtime calls, including 38 line-recognition
calls.

| Component | Approximate cumulative time |
|---|---:|
| ONNX Runtime calls | 66 ms |
| Perspective transformation | 41 ms |
| Image resize operations | 26 ms |
| Image decoding | 9 ms |
| Image statistics and filters | 11 ms |
| CTC decoding | 3 ms |
| Field mapping and MRZ parsing | Less than 2 ms |

### Facial Components

| Component | Warm median |
|---|---:|
| Face detection | 16.6 ms |
| 68-point landmarks | 4.1 ms |
| Seven-model passive liveness | 73.6 ms |
| Full embedding extraction | 69.6 ms |

Liveness preprocessing measured approximately 12.4 ms. Seven sequential model
calls measured approximately 58 ms. Parallel execution measured 64.3 ms median
with higher variance in the same environment.

Repeated local inference stabilized at eight active Python threads, including
the main thread. A 12-request test with four callers completed without exceeding
the bounded liveness worker count.

ONNX Runtime sessions were configured with one intra-op and one inter-op thread.
A 4-vCPU Docker test that executed liveness followed by pair OCR returned HTTP
200 for both requests, retained 16 process threads, and emitted no ONNX Runtime
thread-creation errors.

### Calibrated Liveness

A 20-iteration local run measured the seven-model inference at 84.86 ms median
and 85.47 ms p95. The complete labeled-dataset flow averaged 150.87 ms per
image, including decoding, detection, landmarks, quality checks, and
conditional presentation-attack detection.

The current Docker image was also measured with one Uvicorn worker, a 4-vCPU
limit, and 40 warm requests per concurrency level:

| Concurrency | Client median | Client p95 | Throughput |
|---:|---:|---:|---:|
| 1 | 213.99 ms | 225.56 ms | 4.67 req/s |
| 2 | 248.93 ms | 274.47 ms | 7.94 req/s |
| 4 | 376.00 ms | 428.78 ms | 10.47 req/s |
| 8 | 720.66 ms | 860.92 ms | 10.63 req/s |

All 160 requests returned HTTP 200. From concurrency four to eight, measured
throughput increased by 1.5% and median latency increased from 376.00 ms to
720.66 ms.

After the test, the process retained 23 threads and approximately 1.59 GiB RSS.
A subsequent automatic INE pair request returned HTTP 200 in 630 ms and emitted
no ONNX Runtime thread-creation errors.

### Cross-Pipeline Run

The following values came from a two-iteration warm verification run:

| Pipeline or stage | Warm median |
|---|---:|
| Document classification | 24 ms |
| Modern document front, automatic | 464 ms |
| Modern document front, exact layout | 210 ms |
| Modern document reverse, automatic | 449 ms |
| Modern front/back pair, automatic | 499 ms |
| Three-page legacy collection, automatic | 396 ms |
| Document quality and capture PAD | 116 ms |
| Face detection | 24 ms |
| Facial landmarks | 6 ms |
| Passive facial liveness | 95 ms |
| Face embedding including detection | 116 ms |
| Two-image face comparison | 208 ms |
| Template-only comparison | 0.01 ms |
| Document portrait to selfie | 235 ms |
| Document portrait to selfie with liveness | 361 ms |

### Measured Changes

The following before-and-after values were collected while validating pipeline
changes:

| Change | Before | After | Preserved result |
|---|---:|---:|---|
| Concurrent document-quality stages | 247 ms | 116 ms | Same output semantics |
| Concurrent two-image embedding extraction | 247 ms | 156 ms | Same comparison path |
| Modern voter-card route after incompatible PDF417 suppression | 2.18–3.19 s | 0.32 s | Same document family |
| Legacy INE adaptive field extraction | 0.65 s, 529 OCR calls | 0.59 s, 443 OCR calls | Identical fields; confidence delta below 0.001 |
| Legacy INE automatic versus exact layout | 0.39 s | 0.16 s | Same selected layout |
| Modern INE pair routing and QR reuse | 7.2 s | 0.59 s | Same match, TD1 number, barcodes, and normalized QR evidence |

A separate 61-pair private operational corpus measured 1.55 seconds mean
latency after front/back role-aware routing and strict Mexican TD1-first reverse
processing. The requested front profile is never applied to the back. Three
reverses passed all four ICAO TD1 checks and produced `matched`; 58 remained
`review`. Reverse-side QR evidence did not contain enough holder identity data
for a positive or negative identity comparison. `review` therefore represents
insufficient comparable evidence rather than an OCR failure. The earlier
QR-first role-aware route measured 1.13 seconds mean latency and returned 61
`review`, so the additional 0.42 seconds is the observed cost of strict
rotation-aware MRZ corroboration on this corpus.

The conservative name and CURP geometric searches increased a 61-image
single-side evaluation from 66.03 to 85.26 seconds on the local development
machine, or approximately 0.32 seconds additional processing per image. The
same run improved exact first surname from 44 to 47, second surname from 43 to
47, given names from 42 to 46, and CURP from 48 to 50. These measurements are
corpus-specific and do not replace the container throughput benchmark.

Additional measured routing comparisons:

| Workload | Automatic | Explicit or exact |
|---|---:|---:|
| Modern INE front | 0.31 s | 0.18 s with `profile=mex_ine` |
| Modern INE reverse | 0.64 s | 0.11 s explicit |
| Legacy INE pair | 1.89 s | 1.40 s with exact identifiers |
| Modern INE pair, five-iteration warm run | 334 ms | 320 ms with `mex_ine` |
| Three-page INE collection | 2.84 s | 2.32 s with exact identifiers |

On the modern pair run where whole-image QR decoding succeeded immediately,
automatic routing measured approximately 0.59 s and exact front/reverse
identifiers measured approximately 0.70 s.

Selective extraction on the legacy layout required:

| Requested fields | OCR calls |
|---|---:|
| All fields | 443 |
| Name and birth date | 97 |
| Birth date only | 3 |

Birth-date-only end-to-end processing measured approximately 0.13 s in that
run.

## Full HTTP Throughput: 4 vCPU

### Environment

The Docker image was measured over localhost with:

- Linux `arm64`;
- a hard 4-vCPU limit;
- one `identity-api` worker;
- no explicit memory limit;
- one warm-up request per endpoint and concurrency level;
- 5 measured requests at concurrency 1;
- 6 measured requests at concurrency 2;
- 8 measured requests at concurrency 4.

All measured requests returned HTTP 200.

TPM is successful responses divided by measured wall time and multiplied by 60.

```bash
docker build -t identity-analysis:tpm-benchmark .

docker run --rm -d \
  --name identity-analysis-tpm \
  --cpus=4 \
  -p 8001:8000 \
  identity-analysis:tpm-benchmark

python3 benchmarks/deployed_api_benchmark.py \
  --base-url http://127.0.0.1:8001 \
  --iterations 8 \
  --warmup 1 \
  --concurrency 4 \
  --timeout 90 \
  --output output/benchmarks/api-tpm-4vcpu-c4.json
```

### Results

| Endpoint | TPM c=1 | TPM c=2 | TPM c=4 | c=4 median | c=4 p95 |
|---|---:|---:|---:|---:|---:|
| `GET /health` | 40,237.17 | 49,245.64 | 54,889.03 | 3.19 ms | 5.19 ms |
| `GET /v1/capabilities` | 40,892.83 | 58,429.70 | 53,885.15 | 3.89 ms | 4.39 ms |
| `GET /v1/document/catalog` | 4,591.79 | 4,917.67 | 4,677.68 | 49.68 ms | 75.92 ms |
| `GET /v1/document/catalog/facets` | 23,339.04 | 28,718.92 | 26,564.41 | 6.67 ms | 14.05 ms |
| `GET /v1/document/catalog/{id}` | 42,164.69 | 46,689.58 | 47,885.87 | 4.11 ms | 6.17 ms |
| `GET /v1/document/layout/{id}/evidence` | 38,776.81 | 46,254.91 | 49,302.83 | 4.21 ms | 6.18 ms |
| `POST /v1/document/classify` | 1,125.81 | 1,377.98 | 1,533.96 | 139.65 ms | 231.78 ms |
| `POST /v1/document/layout/{id}/ocr` | 123.17 | 244.21 | 426.38 | 537.51 ms | 587.19 ms |
| `POST /v1/document/reference-metrics` | 661.44 | 568.18 | 512.15 | 435.53 ms | 798.25 ms |
| `POST /v1/ocr` | 39.05 | 74.90 | 118.12 | 1,914.45 ms | 2,160.88 ms |
| `POST /v1/ocr/pair` | 38.49 | 68.89 | 82.08 | 2,831.41 ms | 3,106.54 ms |
| `POST /v1/ocr/pages` | 45.04 | 70.50 | 90.32 | 2,625.81 ms | 2,757.03 ms |
| `POST /v1/face/analyze` | 431.73 | 798.93 | 1,467.15 | 146.36 ms | 176.45 ms |
| `POST /v1/document/portrait/compare` | 43.14 | 85.58 | 125.56 | 1,833.32 ms | 2,002.25 ms |
| `POST /v1/face/liveness` | 202.32 | 307.84 | 403.70 | 535.78 ms | 648.28 ms |
| `POST /v1/face/compare` | 113.43 | 217.47 | 218.32 | 1,080.75 ms | 1,161.62 ms |
| `POST /v1/face/template` | 115.88 | 230.60 | 429.52 | 529.17 ms | 593.56 ms |
| `POST /v1/face/template/compare` | 42,528.81 | 49,166.89 | 77,402.70 | 2.55 ms | 3.34 ms |

After all model families had loaded, the process retained 19 PIDs and
approximately 4.18 GiB resident memory. The container logs contained no
thread-creation or inference errors.

## Measurement Limits

The recorded values do not include a production ingress proxy, external TLS
termination, production network latency, a production traffic mix, or a
production memory limit. The local runtime and HTTP measurements were collected
on different execution environments and should not be directly compared as
equivalent runs.
