"""
MediaExtractor — YouTube, Spotify, podcasts y, en general, cualquier cosa que yt-dlp soporte.

Estrategia inteligente:
  1. Si es YouTube y tiene transcripción oficial → la usamos (gratis, instantáneo).
  2. Si no, descargamos el audio con yt-dlp (mp3 ligero) y lo pasamos por faster-whisper.
"""
from __future__ import annotations

import json
import re

from src.extractors.base import (
    Extractor,
    ExtractedContent,
    IngestionPayload,
    SourceKind,
)
from src.extractors._helpers import (
    RE_SPOTIFY,
    RE_YOUTUBE,
    find_first_url,
    run_subprocess,
    temp_path,
)
from src.utils.logger import get_logger
from src.utils.whisper_runner import transcribe

logger = get_logger(__name__)


def _extract_youtube_id(url: str) -> str | None:
    patterns = [
        r"youtube\.com/watch\?v=([\w\-]+)",
        r"youtu\.be/([\w\-]+)",
        r"youtube\.com/shorts/([\w\-]+)",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


class MediaExtractor(Extractor):
    name = "media"

    def can_handle(self, payload: IngestionPayload) -> bool:
        if payload.kind is not SourceKind.URL:
            return False
        url = payload.raw
        return bool(RE_YOUTUBE.search(url) or RE_SPOTIFY.search(url))

    async def extract(self, payload: IngestionPayload) -> ExtractedContent:
        url = find_first_url(payload.raw) or payload.raw
        is_youtube = bool(RE_YOUTUBE.search(url))

        # 1) Intento rápido: transcripción oficial de YouTube
        if is_youtube:
            yt_id = _extract_youtube_id(url)
            if yt_id:
                text = await self._try_youtube_transcript(yt_id)
                if text:
                    meta = await self._yt_metadata(url)
                    return ExtractedContent(
                        title_hint=meta.get("title") or f"YouTube {yt_id}",
                        source_type="youtube",
                        source_ref=url,
                        text=text,
                        extra={
                            **{k: v for k, v in meta.items() if v is not None},
                            "video_id": yt_id,
                            "method": "transcript_api",
                            "doc_kind": "video",
                        },
                    )

        # 2) Fallback universal: descargar audio + Whisper
        logger.info("MediaExtractor: descargando audio para %s", url)
        audio_path = temp_path(".mp3")
        rc, _, err = await run_subprocess(
            "yt-dlp",
            "-x", "--audio-format", "mp3",
            "--audio-quality", "5",
            "--no-playlist",
            "--quiet",
            "-o", str(audio_path),
            url,
        )
        if rc != 0 or not audio_path.exists():
            raise RuntimeError(f"yt-dlp falló para {url}: {err[:300]}")

        try:
            text = await transcribe(audio_path)
        finally:
            audio_path.unlink(missing_ok=True)

        meta = await self._yt_metadata(url)
        source_type = "youtube" if is_youtube else "podcast/spotify"
        doc_kind = "video" if is_youtube else "podcast"
        return ExtractedContent(
            title_hint=meta.get("title") or url,
            source_type=source_type,
            source_ref=url,
            text=text,
            extra={
                **{k: v for k, v in meta.items() if v is not None},
                "method": "whisper",
                "doc_kind": doc_kind,
            },
        )

    @staticmethod
    async def _try_youtube_transcript(video_id: str) -> str:
        """Usa youtube-transcript-api si hay subtítulos disponibles.

        Compatible con youtube-transcript-api >=0.6 (nueva API basada en instancias).
        """
        try:
            import asyncio
            from youtube_transcript_api import YouTubeTranscriptApi

            def _fetch() -> str:
                api = YouTubeTranscriptApi()

                def _seg_text(seg) -> str:
                    """Compatibilidad: dict (API antigua) o FetchedTranscriptSnippet (API nueva)."""
                    return seg.text if hasattr(seg, "text") else seg["text"]

                # 1. Intentar fetch directo con preferencia de idiomas
                for langs in (["es", "en"], ["en", "es"]):
                    try:
                        transcript = api.fetch(video_id, languages=langs)
                        return " ".join(_seg_text(s) for s in transcript)
                    except Exception:
                        continue

                # 2. Fallback: listar y coger el primero disponible (cualquier idioma)
                try:
                    tr_list = api.list(video_id)
                    tr = next(iter(tr_list))
                    fetched = tr.fetch()
                    return " ".join(_seg_text(s) for s in fetched)
                except Exception:
                    pass

                return ""

            return await asyncio.to_thread(_fetch)
        except Exception as e:
            logger.info("Sin transcripción oficial (%s) — usaré Whisper.", e)
            return ""

    @staticmethod
    async def _yt_metadata(url: str) -> dict:
        """Saca metadata rica con yt-dlp --dump-json (no descarga el audio).

        Devuelve un dict con: title, author, channel_url, description,
        published, duration, view_count, source_tags, chapters.
        """
        rc, out, _ = await run_subprocess(
            "yt-dlp", "--dump-json", "--no-playlist", "--quiet", url
        )
        if rc != 0:
            return {}
        try:
            data = json.loads(out.splitlines()[0])
        except Exception:
            return {}

        meta: dict = {
            "title": data.get("title"),
            "author": data.get("uploader") or data.get("channel"),
            "channel_url": data.get("uploader_url") or data.get("channel_url"),
        }
        if desc := data.get("description"):
            # Las descripciones largas son ruido para el frontmatter; recortamos
            meta["description"] = desc.strip()[:800]
        # Fecha: yt-dlp da YYYYMMDD
        if upload_date := data.get("upload_date"):
            if len(upload_date) == 8 and upload_date.isdigit():
                meta["published"] = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:8]}"
                meta["year"] = int(upload_date[:4])
        # Duración en segundos → "1h 23m"
        if (dur := data.get("duration")) and isinstance(dur, (int, float)):
            h, rem = divmod(int(dur), 3600)
            m, _s = divmod(rem, 60)
            meta["duration"] = (f"{h}h {m}m" if h else f"{m}m")
            meta["duration_seconds"] = int(dur)
        if (vc := data.get("view_count")) is not None:
            meta["view_count"] = vc
        # Tags de YouTube
        if tags := data.get("tags"):
            meta["source_tags"] = [t for t in tags[:10] if isinstance(t, str)]
        # Capítulos (lista de {start_time, end_time, title})
        if chapters := data.get("chapters"):
            meta["chapters"] = [
                c.get("title") for c in chapters[:30] if c.get("title")
            ]
        return meta
