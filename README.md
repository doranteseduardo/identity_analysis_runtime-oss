<div align="center">

# Identity Analysis Runtime

**Portable document intelligence, facial analysis, and identity verification**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-CPU-005CED?style=flat-square&logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat-square)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/doranteseduardo/identity_analysis_runtime-oss/ci.yml?branch=main&style=flat-square&label=CI)](https://github.com/doranteseduardo/identity_analysis_runtime-oss/actions/workflows/ci.yml)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](CONTRIBUTING.md)

A self-contained runtime for identity-document OCR, document classification,
machine-readable data, capture-risk signals, face analysis, passive facial
liveness, face comparison, and document-portrait verification. It exposes a
stable REST API, reusable Python pipelines, command-line tools, and an explicit
machine-readable capability boundary.

**Open source, Apache-2.0 licensed.** This repository ships no vendor SDK
material and no model weights — see [`docs/models.md`](docs/models.md) for
how to bring your own. Issues and pull requests are welcome; see
[`CONTRIBUTING.md`](CONTRIBUTING.md).

</div>

---

## About This Release

This repository is a **pruned, public release of a broader internal research
and development effort**, not the original research tree in full.

- **No third-party model weights or catalogs.** This repository includes no
  ONNX model weights, no document-classification catalog, and no other
  vendor-derived material of any kind. The runtime is built entirely around a
  bring-your-own-models architecture — you supply the ONNX models and,
  optionally, the document catalog (see [`docs/models.md`](docs/models.md)).
- **No real personal data.** Sample documents and photos used to develop and
  test the pipeline have been replaced with generated, clearly-fake synthetic
  fixtures (see [`NOTICE`](NOTICE) and `tools/generate_synthetic_samples.py`).
  No real person, document, or credential appears anywhere in this repository
  or its history.

What's left is the runtime, pipeline, and API code itself — a complete,
working system that runs entirely on models and data you supply.

---

## What Is Inside

### Document Intelligence

Classifies document templates, rectifies photographed documents, reads visual
fields, parses ICAO MRZ formats, decodes supported barcodes, and returns a
compact structured response. Exact-layout routing can skip classification and
recognize selected fields only.

> **This repository ships no model weights.** Everything here is runtime code.
> You supply your own ONNX models, and optionally your own document-classification
> catalog, in a directory that `IDENTITY_ANALYSIS_ASSETS` points at. See
> [`docs/models.md`](docs/models.md) for the expected layout and for which
> features degrade gracefully without a given model family.

Document classification and exact-layout field recognition are driven by a
catalog you provide: a classifier model plus declarative layouts describing each
document's field, graphic, barcode, and security regions. The schema is
documented in [`docs/models.md`](docs/models.md), and a small worked example
lives in `tests/fixtures/catalog`. Line recognition is locale-routed, so
per-script models are picked up automatically when present.

### Document Collections

Front/back and multi-page pipelines run pages concurrently, verify compatible
identity fields, isolate mismatches, guide related-page barcode decoding, and
fuse only validated evidence. Related layouts can be ordered canonically when
their page markers are unambiguous. Original request positions remain
traceable. Mexican voter-card pairs prioritize four-check TD1 MRZ identity
evidence on the reverse; QR remains supplementary.

### Facial Analysis

Provides face detection, 68 landmarks, geometric head pose, quality warnings,
seven-model passive liveness, 512-value embeddings, one-to-one comparison, and
portable face templates.

### Identity Verification

Extracts the declared portrait from a recognized document, compares it with a
selfie, optionally evaluates selfie liveness, and combines portrait identity,
selfie liveness, and document capture risk into a conservative verification
decision.

### Evaluation Tooling

Evaluates existing scores, persisted templates, or image pairs. The offline
evaluator reports score distributions, ROC, AUC, EER, confusion matrices, and
candidate thresholds for target false-accept rates.

---

## Supported Operations

| Operation | Interface | Status |
|---|---|---|
| Service health | `GET /health` | Available |
| Capability manifest | `GET /v1/capabilities` | Available |
| Document catalog | `GET /v1/document/catalog` | Available |
| Document classification | `POST /v1/document/classify` | Available |
| Exact-layout OCR | `POST /v1/document/layout/{id}/ocr` | Available |
| Layout evidence | `GET /v1/document/layout/{id}/evidence` | Available |
| Reference metrics | `POST /v1/document/reference-metrics` | Experimental metrics |
| Single-document OCR | `POST /v1/ocr` | Available |
| Front/back fusion | `POST /v1/ocr/pair` | Available |
| Multi-page fusion | `POST /v1/ocr/pages` | Available |
| Face analysis | `POST /v1/face/analyze` | Available |
| Passive facial liveness | `POST /v1/face/liveness` | Exploratory calibration; holdout validation required |
| Face comparison | `POST /v1/face/compare` | Calibration required |
| Face templates | `POST /v1/face/template` | Available |
| Template comparison | `POST /v1/face/template/compare` | Available |
| Document portrait vs. selfie | `POST /v1/document/portrait/compare` | Available |
| Biometric evaluation | `identity-face-eval` | Offline CLI |

`GET /v1/capabilities` is the authoritative source for runtime coverage,
endpoints, operating assumptions, and unavailable capabilities.

---

## Stack

| Layer | Technology |
|---|---|
| HTTP service | FastAPI · Uvicorn |
| Inference | ONNX Runtime CPU |
| Image processing | Pillow · NumPy · ImageMagick for HEIC conversion |
| Machine-readable data | ZXing-C++ |
| Packaging | Python 3.10+ · setuptools |
| Deployment | Multi-stage Docker image |
| Tests | pytest · HTTPX |

---

## Getting Started

### Docker

Build and run the production image:

```bash
docker build -t identity-analysis .
docker run --rm -p 8000:8000 -v "$PWD/assets:/app/assets:ro" identity-analysis
```

Check the service:

```bash
curl http://localhost:8000/v1/capabilities
curl http://localhost:8000/health
```

The image carries no model weights: mount your own asset directory at
`/app/assets` (see [`docs/models.md`](docs/models.md)). It runs as an
unprivileged user and needs no Compose file. Without models the service still
starts and `/v1/capabilities` reports every feature as unavailable.

### Local Development

**Prerequisites:** Python 3.10+, ImageMagick, and a C/C++ build toolchain when a
prebuilt ZXing wheel is unavailable.

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

Tests that need real inference are skipped unless you point the runtime at your
own models; see [`docs/models.md`](docs/models.md) and
[`CONTRIBUTING.md`](CONTRIBUTING.md).

```bash
IDENTITY_ANALYSIS_ASSETS=/path/to/assets .venv/bin/pytest
```

Start the API:

```bash
IDENTITY_ANALYSIS_ASSETS=/path/to/assets .venv/bin/identity-api
```

---

## API Examples

All image endpoints accept semantic multipart image fields or JSON base64.
Base64 may be raw or
use a Data URL prefix.

### Document OCR

```bash
curl -X POST http://localhost:8000/v1/ocr \
  -F 'image=@examples/samples/synthetic_id_front.jpg' \
  -F 'includeImages=true' \
  -F 'analyzePortraits=true'
```

JSON:

```json
{
  "imageBase64": "/9j/4AAQSk...",
  "filename": "document.jpg",
  "profile": "auto_research",
  "includeImages": false
}
```

Exact-layout routing:

```bash
curl -X POST \
  http://localhost:8000/v1/document/layout/900000001/ocr \
  -F 'image=@examples/samples/synthetic_id_front.jpg' \
  -F 'fields=surname' \
  -F 'fields=givenNames'
```

### Document Pair

```bash
curl -X POST http://localhost:8000/v1/ocr/pair \
  -F 'frontImage=@examples/samples/synthetic_id_front.jpg' \
  -F 'backImage=@examples/samples/synthetic_id_back.jpg'
```

### Face Comparison

```bash
curl -X POST http://localhost:8000/v1/face/compare \
  -F 'firstImage=@examples/samples/synthetic_selfie.jpg' \
  -F 'secondImage=@examples/samples/synthetic_selfie.jpg' \
  -F 'threshold=0.67'
```

### Document Portrait Verification

```bash
curl -X POST http://localhost:8000/v1/document/portrait/compare \
  -F 'documentImage=@examples/samples/synthetic_id_front.jpg' \
  -F 'selfieImage=@examples/samples/synthetic_selfie.jpg' \
  -F 'analyzeLiveness=true'
```

For complete request schemas and response contracts, see the
[API guide and curl reference](docs/product/api.md) and
[response reference](docs/product/ocr-response.md).

---

## Command-Line Tools

```bash
# Document processing
identity-document document.jpg
identity-document passport.jpg --profile icao_td3 --output result.json
identity-document mexican-passport.jpg --profile mex_passport --output result.json

# Asset manifests and validation
identity-assets manifest /path/to/assets
identity-assets validate /path/to/assets

# Face detection and landmarks
identity-face selfie.jpg \
  --detector assets/facial/detector/face_detector.onnx \
  --landmarks assets/facial/landmarks/landmarks_quality.onnx

# Biometric threshold evaluation
identity-face-eval evaluation.csv \
  --assets assets \
  --target-far 0.001 \
  --output report.json
```

The Python package also exposes `process_document`, `process_document_pair`,
and `process_document_pages`.

---

## Architecture

```text
Client
  │
  ▼
FastAPI transport and validation
  ├── Document pipelines
  │   ├── template classification
  │   ├── perspective rectification
  │   ├── declarative visual OCR
  │   ├── MRZ and barcode parsing
  │   ├── capture-risk analysis
  │   └── side/page identity-validated fusion
  ├── Facial pipelines
  │   ├── detection and landmarks
  │   ├── head pose and quality
  │   ├── passive liveness
  │   └── embeddings and comparison
  ├── Identity verification policy
  └── Stable response serializers
```

```text
src/identity_analysis/
├── api.py                    # HTTP transport and endpoint orchestration
├── pipeline.py               # document parsing, routing, and fusion
├── document_classifier.py    # template classification and catalog
├── visual_layouts.py         # declarative fields, regions, and page roles
├── ocr.py                    # locale-routed line OCR and CTC decoding
├── barcodes.py               # machine-readable payload decoding
├── rectification.py          # document geometry correction
├── quality.py                # image quality and capture-risk signals
├── face_engines.py           # face detection, landmarks, pose, quality
├── facial_identity.py        # liveness, embeddings, templates, policy
├── biometric_evaluation.py   # ROC, EER, and threshold evaluation
├── responses.py              # compact public response contracts
├── capabilities.py           # machine-readable capability boundary
└── assets.py                 # asset manifests and integrity validation

tools/
├── generate_synthetic_samples.py   # renders examples/samples
└── generate_test_catalog.py        # builds tests/fixtures/catalog
```

The asset directory you supply is described in
[`docs/models.md`](docs/models.md); nothing under it is part of this
repository.

Model sessions and immutable catalogs are cached per process. Pair and
multi-page document work is bounded and concurrent. Product serializers prevent
internal tensors and duplicate intermediate results from crossing the REST
boundary.

See the complete [pipeline architecture](docs/product/architecture.md).

---

## Decision Semantics

- Structural validation reports available machine-readable or field checks; it
  does not prove that a document is legally valid.
- Document capture risk identifies recapture indicators, not full document
  authenticity.
- Passive facial liveness is a still-image signal, not an active challenge.
- Face-comparison thresholds are application policy and require
  population-specific calibration.
- Document portrait verification returns `pass` only when portrait identity,
  selfie liveness, and document capture signals all pass.
- `review` and `not_available` are first-class outcomes and are never coerced
  into a successful decision.

---

## Configuration

| Environment variable | Default | Purpose |
|---|---:|---|
| `IDENTITY_ANALYSIS_ASSETS` | `/app/assets` in Docker | Runtime asset directory |
| `IDENTITY_ANALYSIS_HOST` | `0.0.0.0` | API bind address |
| `IDENTITY_ANALYSIS_PORT` | `8000` | API port |
| `IDENTITY_ANALYSIS_MAX_UPLOAD_BYTES` | `20971520` | Maximum decoded upload size |
| `IDENTITY_ANALYSIS_MAX_DOCUMENT_PAGES` | `10` | Multi-page request limit |
| `IDENTITY_ANALYSIS_WARMUP` | `true` | Preload core inference sessions |

Operational deployment guidance is available in
[`docs/product/deployment.md`](docs/product/deployment.md).

---

## Validation

```bash
.venv/bin/pytest
docker build -t identity-analysis .
```

The test suite covers parsers, CTC decoding, layout interpretation, masks,
classification, image rectification, barcodes, paired and multi-page fusion,
response compaction, API file/base64 input, face geometry, passive liveness,
recognition, templates, verification policy, and biometric evaluation.

Representative performance and optimization guidance is documented in
[`docs/product/performance.md`](docs/product/performance.md).

---

## Documentation

| Audience | Start here |
|---|---|
| Integrators | [`docs/product/api.md`](docs/product/api.md) |
| Document contributors | [`docs/product/adding-documents.md`](docs/product/adding-documents.md) |
| Response consumers | [`docs/product/ocr-response.md`](docs/product/ocr-response.md) |
| Operators | [`docs/product/deployment.md`](docs/product/deployment.md) |
| Architects | [`docs/product/architecture.md`](docs/product/architecture.md) |
| Biometric evaluators | [`docs/product/biometric-evaluation.md`](docs/product/biometric-evaluation.md) |
| Model suppliers | [`docs/models.md`](docs/models.md) |
| Contributors | [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| Complete index | [`docs/README.md`](docs/README.md) |

Product documentation describes supported behavior and stable contracts.
`GET /v1/capabilities` is authoritative for what the running instance can
actually do with the models it has been given.

---

## Capability Boundary

Implemented portable capabilities include document OCR and classification,
declarative layouts, MRZ and supported barcode parsing, document collections,
capture-risk signals, facial analysis, passive facial liveness, facial
recognition, templates, document-to-selfie verification, and biometric
evaluation.

Remaining work is validation rather than another portable runtime pipeline:

- representative per-edition and multilingual OCR fixtures;
- calibrated document and facial presentation-attack benchmarks;
- population-specific biometric threshold validation;
- optional operational observability and further performance tuning.

Capabilities that require unavailable protected model material or unavailable
calibration contracts remain explicitly disabled rather than approximated.

---

## License

Licensed under the [Apache License, Version 2.0](LICENSE). See
[`NOTICE`](NOTICE) for third-party and sample-data attribution — this
repository ships no vendor SDK material, model weights, or real personal
data; see [`docs/models.md`](docs/models.md) for what you need to supply.

---

<div align="center">
  <sub>Portable identity analysis for controlled, consent-based applications</sub>
</div>
