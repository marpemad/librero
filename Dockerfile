# syntax=docker/dockerfile:1.6
# ───────────────────────────────────────────────────────────────────────
# Librero — Telegram bot que sintetiza contenido en notas Obsidian.
# Imagen multiarch (amd64, arm64) — compatible con Mac, Linux y Umbrel.
# ───────────────────────────────────────────────────────────────────────
FROM python:3.13-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=Europe/Madrid

# Dependencias del sistema:
#   ffmpeg          → yt-dlp + faster-whisper (audio/video)
#   tesseract-ocr   → OCR de imágenes
#   tesseract-ocr-spa → idioma español para OCR
#   tini            → init mínimo para señales limpias (SIGTERM, SIGINT)
#   curl            → healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        tesseract-ocr \
        tesseract-ocr-spa \
        tesseract-ocr-eng \
        tini \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Capa cacheada de dependencias Python — solo se reconstruye si requirements.txt cambia
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Código fuente
COPY main.py ./
COPY src ./src

# Directorios de datos (se mapean como volúmenes en docker-compose)
RUN mkdir -p /data/vault /data/config /data/cache /data/temp \
    && chmod -R 755 /data

# Variables por defecto — apuntan a los volúmenes persistentes
ENV TEMP_DIR=/data/temp \
    CACHE_DB_PATH=/data/cache/cache.db \
    VAULT_INDEX_DB_PATH=/data/cache/vault_index.db \
    GCAL_TOKEN_PATH=/data/config/gcal_token.json \
    GCAL_NOTIFY_DB_PATH=/data/cache/gcal_notify.db \
    OBSIDIAN_VAULT_PATH=/data/vault

# Tini como PID 1 para reenviar SIGTERM/SIGINT al proceso Python
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-u", "main.py"]
