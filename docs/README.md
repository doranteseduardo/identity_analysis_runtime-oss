# Documentation

Start with the section that matches your task.

## Product Documentation

Product guides define supported behavior, stable interfaces, deployment, and
operational expectations.

| Guide | Purpose |
|---|---|
| [`product/README.md`](product/README.md) | Product documentation overview and capability boundary |
| [`product/api.md`](product/api.md) | REST contract and copy-ready curl examples |
| [`product/adding-documents.md`](product/adding-documents.md) | Add catalog layouts, document profiles, or machine-readable parsers |
| [`product/ocr-response.md`](product/ocr-response.md) | Structured response field reference |
| [`product/architecture.md`](product/architecture.md) | Pipeline architecture, concurrency, model lifecycle, and extension points |
| [`product/deployment.md`](product/deployment.md) | Docker, local execution, health checks, and production operation |
| [`product/performance.md`](product/performance.md) | Component performance, HTTP TPM, bottlenecks, and optimization guidance |
| [`product/biometric-evaluation.md`](product/biometric-evaluation.md) | ROC, EER, and threshold-selection workflow |

## Models

This repository contains no model weights. [`models.md`](models.md) specifies
the directory layout the runtime expects, the `manifest.json` schema, and which
features are unavailable without a given model family.

## Source of Truth

- `GET /v1/capabilities` is authoritative for machine-readable runtime
  coverage.
- Product guides are authoritative for public contracts.
- Tests are authoritative for regression behavior.
