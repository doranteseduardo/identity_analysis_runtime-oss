# Deployment Guide

## Docker

```bash
docker build -t identity-analysis .
docker run --rm -p 8000:8000 identity-analysis
```

The container runs as a non-root user and includes a health check against
`/health`. The multi-stage build supports both AMD64 and ARM64. When a native
dependency has no wheel for the selected platform, it is compiled in an
isolated builder stage; compilers and build tools are not copied into the final
runtime image.

## Local Runtime

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
identity-api
```

## Configuration

| Variable | Default | Description |
|---|---:|---|
| `IDENTITY_ANALYSIS_ASSETS` | `/app/assets` in Docker | Asset root |
| `IDENTITY_ANALYSIS_HOST` | `0.0.0.0` | Bind address |
| `IDENTITY_ANALYSIS_PORT` | `8000` | HTTP port |
| `IDENTITY_ANALYSIS_MAX_UPLOAD_BYTES` | `20971520` | Decoded request limit |
| `IDENTITY_ANALYSIS_MAX_DOCUMENT_PAGES` | `10` | Maximum images in one multi-page request |
| `IDENTITY_ANALYSIS_WARMUP` | `true` | Preload core models |
| `IDENTITY_ANALYSIS_ONNX_INTRA_OP_THREADS` | `1` | Worker threads inside each ONNX operator |
| `IDENTITY_ANALYSIS_ONNX_INTER_OP_THREADS` | `1` | Parallel ONNX graph branches per session |

Passive liveness keeps a bounded seven-worker executor per service process.
Additional liveness requests queue on those workers rather than creating a new
thread set per request. Account for this fixed pool when choosing the number of
server processes; every process has its own models, caches, and liveness
workers.

All ONNX sessions default to one intra-op and one inter-op thread. The runtime
already parallelizes document sides, pages, quality branches, facial embeddings,
and liveness models, so unrestricted per-session pools multiply thread usage
without respecting those outer bounds. Increase either value only after
concurrency testing on the deployment CPU.

## Production Checklist

- Pin the image digest and dependency versions.
- Mount or include a validated asset bundle.
- Keep the service behind TLS and authenticated application endpoints.
- Apply request-rate and body-size limits at the edge.
- Avoid logging images, base64 payloads, embeddings, or identity fields.
- Define retention and deletion policies for biometric and document data.
- Calibrate liveness and comparison thresholds for the target population and
  capture environment.
- Configure the default liveness review band with
  `IDENTITY_ANALYSIS_LIVENESS_THRESHOLD` (`0.37`) and
  `IDENTITY_ANALYSIS_LIVENESS_SPOOF_THRESHOLD` (`0.25`). The spoof boundary
  must not exceed the real boundary.
- Monitor latency, error rates, score drift, and manual-review outcomes.
