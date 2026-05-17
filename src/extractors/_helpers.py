"""Helpers reutilizables por varios extractores."""
from __future__ import annotations

import asyncio
import re
import uuid
from pathlib import Path

import httpx
import trafilatura

from src.config.settings import settings

# --- Patrones de URL ---
RE_YOUTUBE = re.compile(
    r"(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)[\w\-]+",
    re.IGNORECASE,
)
RE_TWITTER = re.compile(
    r"https?://(?:www\.)?(?:twitter\.com|x\.com)/[\w\d_]+/status/\d+",
    re.IGNORECASE,
)
RE_SPOTIFY = re.compile(
    r"https?://open\.spotify\.com/(?:episode|show|track)/[\w\d]+",
    re.IGNORECASE,
)
RE_GSHEETS = re.compile(
    r"https?://docs\.google\.com/spreadsheets/d/[\w\-]+",
    re.IGNORECASE,
)
RE_GENERIC_URL = re.compile(r"https?://\S+", re.IGNORECASE)


def find_first_url(text: str) -> str | None:
    m = RE_GENERIC_URL.search(text)
    return m.group(0) if m else None


async def fetch_html(url: str, timeout: float = 20.0) -> str:
    """GET con headers de navegador y timeout razonable."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Apple Silicon Mac OS X 14_5) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0 Safari/537.36"
        )
    }
    async with httpx.AsyncClient(
        timeout=timeout, follow_redirects=True, headers=headers
    ) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.text


def html_to_main_text(html: str, url: str | None = None) -> str:
    """Trafilatura es el extractor de cuerpo principal más fiable."""
    text = trafilatura.extract(
        html,
        url=url,
        favor_recall=True,
        include_comments=False,
        include_tables=True,
    )
    return (text or "").strip()


def truncate_for_llm(text: str, max_chars: int = 14_000) -> str:
    """
    Mantiene el contexto bajo control para no saturar el LLM local.
    14k chars ≈ 3.5k tokens, holgado para Llama3/Qwen2.5 con ventana 8k.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return text
    head = text[: int(max_chars * 0.7)]
    tail = text[-int(max_chars * 0.3):]
    return f"{head}\n\n[...contenido truncado por longitud...]\n\n{tail}"


def temp_path(suffix: str = "") -> Path:
    """Ruta temporal aleatoria dentro de TEMP_DIR."""
    settings.temp_dir.mkdir(parents=True, exist_ok=True)
    return settings.temp_dir / f"{uuid.uuid4().hex}{suffix}"


async def run_subprocess(*args: str) -> tuple[int, str, str]:
    """Ejecuta un proceso externo (yt-dlp, etc.) sin bloquear el event loop."""
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate()
    return proc.returncode, out.decode("utf-8", "ignore"), err.decode("utf-8", "ignore")
