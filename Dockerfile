FROM python:3.10-slim AS dependency-builder

RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt ./
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt


FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    IDENTITY_ANALYSIS_ASSETS=/app/assets \
    IDENTITY_ANALYSIS_WARMUP=true \
    IDENTITY_ANALYSIS_ONNX_INTRA_OP_THREADS=1 \
    IDENTITY_ANALYSIS_ONNX_INTER_OP_THREADS=1 \
    IDENTITY_ANALYSIS_PORT=8000

RUN apt-get update \
    && apt-get install --no-install-recommends -y imagemagick \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
COPY --from=dependency-builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .

# No model weights are distributed with this image. Mount your own asset
# directory at /app/assets (see docs/models.md).
VOLUME ["/app/assets"]

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import json,urllib.request; json.load(urllib.request.urlopen('http://127.0.0.1:8000/v1/capabilities', timeout=3))" || exit 1

CMD ["identity-api"]
