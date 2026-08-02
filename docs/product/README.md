# Product Documentation

Stable behavior, integration contracts, deployment guidance, and operational
expectations for Identity Analysis Runtime.

[Back to the complete documentation index](../README.md)

## Choose a Guide

| If you need to... | Read |
|---|---|
| Integrate or exercise the REST API | [`api.md`](api.md) |
| Add a document or edition | [`adding-documents.md`](adding-documents.md) |
| Consume structured responses | [`ocr-response.md`](ocr-response.md) |
| Understand pipeline boundaries | [`architecture.md`](architecture.md) |
| Build or operate the container | [`deployment.md`](deployment.md) |
| Review latency, TPM, or optimize the runtime | [`performance.md`](performance.md) |
| Calibrate face-comparison thresholds | [`biometric-evaluation.md`](biometric-evaluation.md) |

## Contract Map

| Contract | Source of truth |
|---|---|
| Enabled runtime capabilities | `GET /v1/capabilities` |
| Endpoint inputs and errors | [`api.md`](api.md) |
| Response fields and decisions | [`ocr-response.md`](ocr-response.md) |
| Environment and process behavior | [`deployment.md`](deployment.md) |
| Pipeline composition and model lifecycle | [`architecture.md`](architecture.md) |

## Product Scope

The runtime provides:

- document classification and catalog search;
- single-image, front/back, and multi-page document processing;
- declarative visual OCR with locale-routed line models;
- ICAO MRZ and supported barcode parsing;
- document quality and capture-risk signals;
- face detection, landmarks, pose, and quality;
- passive facial liveness;
- facial templates and one-to-one comparison;
- document-portrait to selfie verification;
- offline biometric threshold evaluation.

## Decision Boundary

- Structural validation describes available checks, not legal validity.
- Capture-risk analysis is not full document authenticity.
- Passive liveness is not an active challenge.
- Face-comparison thresholds require deployment-specific calibration.
- An unavailable signal remains `not_available`; an inconclusive signal remains
  `review`.

`GET /v1/capabilities` is authoritative when application behavior depends on
whether an engine, endpoint, or decision is available.

## Language Coverage

Catalog-driven OCR routes fields by locale metadata. Packaged character families
cover Latin, Greek, Cyrillic, Hebrew, Armenian, and Georgian, including several
regional Latin variants. Unsupported locales fall back to baseline Latin OCR.
Model availability does not replace representative per-language validation.
