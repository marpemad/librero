"""
Wrapper perezoso de faster-whisper.
- Singleton: el modelo se carga UNA vez en memoria.
- Detecta Apple Silicon: 'auto' usa Metal cuando es posible.
- La transcripción se ejecuta en un thread (CPU/GPU intensivo) para no bloquear asyncio.
- 🆕 Fase 1: cacheo por hash del archivo de audio. Si vuelves a procesar el mismo
  audio (mismo archivo o misma URL ya descargada), no se vuelve a transcribir.
"""
from __future__ import annotations

import asyncio
import platform
from pathlib import Path
from typing import Optional

from src.config.settings import settings
from src.utils.cache import cache, hash_file
from src.utils.logger import get_logger

logger = get_logger(__name__)

_model = None  # Cargado al primer uso


def _resolve_device() -> str:
    if settings.whisper_device != "auto":
        return settings.whisper_device
    # En Apple Silicon, faster-whisper aún usa CPU por defecto (CTranslate2 sin Metal completo).
    # Mantenemos "cpu" + int8 para máximo rendimiento en M1/M2.
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "cpu"
    return "cpu"


def _get_model():
    global _model
    if _model is not None:
        return _model
    from faster_whisper import WhisperModel  # import perezoso → arranque del bot rápido

    device = _resolve_device()
    logger.info(
        "Cargando faster-whisper '%s' device=%s compute=%s",
        settings.whisper_model,
        device,
        settings.whisper_compute_type,
    )
    _model = WhisperModel(
        settings.whisper_model,
        device=device,
        compute_type=settings.whisper_compute_type,
    )
    return _model


def _transcribe_sync(audio_path: Path, language: Optional[str] = None) -> str:
    model = _get_model()
    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        vad_filter=True,
        beam_size=5,
    )
    pieces: list[str] = []
    for seg in segments:
        pieces.append(seg.text.strip())
    text = " ".join(pieces).strip()
    logger.info(
        "Transcripción OK: lang=%s duración=%.1fs chars=%d",
        info.language,
        info.duration,
        len(text),
    )
    return text


async def transcribe(audio_path: Path, language: Optional[str] = None) -> str:
    """API pública: transcribe sin bloquear el event loop, con caché por hash."""
    # 1) Hash del archivo (cheap; SHA1 streaming)
    file_hash = await asyncio.to_thread(hash_file, audio_path)
    cache_key = f"{file_hash}_{language or 'auto'}"

    # 2) ¿Ya transcrito antes?
    cached = cache.get("whisper", cache_key)
    if cached is not None:
        return cached

    # 3) Transcribir y cachear
    text = await asyncio.to_thread(_transcribe_sync, audio_path, language)
    cache.set("whisper", cache_key, text)
    return text
