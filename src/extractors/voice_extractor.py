"""
VoiceExtractor — mensajes de voz de Telegram.

El handler del bot descarga el .ogg en disco y se lo pasa como FILE.
Detectamos por extensión .ogg/.oga (y por mime audio/* si llegara aquí desde otro origen).
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from src.extractors.base import (
    Extractor,
    ExtractedContent,
    IngestionPayload,
    SourceKind,
)
from src.utils.logger import get_logger
from src.utils.whisper_runner import transcribe

logger = get_logger(__name__)

VOICE_EXTS = {".ogg", ".oga", ".opus", ".m4a", ".mp3", ".wav"}


class VoiceExtractor(Extractor):
    name = "voice"

    def can_handle(self, payload: IngestionPayload) -> bool:
        if payload.kind is not SourceKind.FILE:
            return False
        ext = Path(payload.raw).suffix.lower()
        if ext in VOICE_EXTS:
            return True
        # También aceptamos por MIME si está disponible
        mime = (payload.metadata.get("mime") or "").lower()
        return mime.startswith("audio/")

    async def extract(self, payload: IngestionPayload) -> ExtractedContent:
        path = Path(payload.raw)
        original_name = payload.metadata.get("filename", path.name)
        is_telegram_voice = bool(payload.metadata.get("telegram_voice"))
        logger.info(
            "VoiceExtractor: %s (telegram_voice=%s)", original_name, is_telegram_voice
        )

        text = await transcribe(path)

        # Para notas de voz cortas, generamos un título genérico fechado;
        # el LLM lo mejorará a partir del contenido.
        if is_telegram_voice:
            from datetime import datetime
            title_hint = f"Nota de voz {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            source_type = "voice_note"
        else:
            title_hint = Path(original_name).stem
            source_type = "audio"

        return ExtractedContent(
            title_hint=title_hint,
            source_type=source_type,
            source_ref=original_name,
            text=text,
            extra={"original_filename": original_name},
        )
