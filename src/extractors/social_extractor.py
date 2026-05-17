"""
SocialExtractor — Twitter / X.

Twitter sin login es hostil al scraping. Estrategia en cascada:
  1. yt-dlp (funciona para tweets con media y a veces para texto). Si falla…
  2. oEmbed público de Twitter (sin auth, funciona para tweets de texto). Si falla…
  3. Devuelve text="" + extraction_error → el pipeline notifica al usuario sin crear nota.

Caso especial — Twitter Articles (x.com/i/article/...):
  El tweet solo contiene un enlace t.co al artículo. Se detecta porque el <p>
  del oEmbed queda vacío de texto real. Se intenta extraer con trafilatura;
  si falla (requiere JS), se señaliza como error sin crear nota.

Si en algún momento decides añadir una API key de la X API, basta con
añadir un método `_via_api` antes de los fallbacks.
"""
from __future__ import annotations

import json
import re
from urllib.parse import quote

import httpx
import trafilatura
from bs4 import BeautifulSoup

from src.extractors.base import (
    Extractor,
    ExtractedContent,
    IngestionPayload,
    SourceKind,
)
from src.extractors._helpers import RE_TWITTER, find_first_url, run_subprocess
from src.utils.logger import get_logger

logger = get_logger(__name__)

_OEMBED_ENDPOINT = "https://publish.twitter.com/oembed"
_RE_ONLY_URLS = re.compile(r"^(https?://\S+\s*)+$")


class SocialExtractor(Extractor):
    name = "social"

    def can_handle(self, payload: IngestionPayload) -> bool:
        return payload.kind is SourceKind.URL and bool(RE_TWITTER.search(payload.raw))

    async def extract(self, payload: IngestionPayload) -> ExtractedContent:
        url = find_first_url(payload.raw) or payload.raw
        logger.info("SocialExtractor: %s", url)

        text, author, error = await self._via_yt_dlp(url)

        if not text:
            logger.info("yt-dlp sin contenido; intentando oEmbed para: %s", url)
            text, author, error = await self._via_oembed(url)

        if not text and not error:
            error = (
                "No se pudo extraer el contenido del tweet.\n"
                "Causa probable: tweet protegido o cambios en la API de X/Twitter."
            )

        return ExtractedContent(
            title_hint=f"Tweet de {author or 'desconocido'}",
            source_type="tweet",
            source_ref=url,
            text=text,
            extra={"author": author, "extraction_error": error},
        )

    @staticmethod
    async def _via_yt_dlp(url: str) -> tuple[str, str | None, str | None]:
        rc, out, err = await run_subprocess(
            "yt-dlp", "--dump-json", "--skip-download", "--quiet", url
        )
        if rc != 0:
            logger.warning("yt-dlp tweet falló: %s", err[:200])
            return "", None, None
        try:
            data = json.loads(out.splitlines()[0])
        except Exception:
            return "", None, None
        body = data.get("description") or data.get("title") or ""
        author = data.get("uploader") or data.get("uploader_id")
        return body.strip(), author, None

    @staticmethod
    async def _via_oembed(url: str) -> tuple[str, str | None, str | None]:
        """API pública oEmbed de Twitter — funciona sin auth para tweets públicos."""
        oembed_url = f"{_OEMBED_ENDPOINT}?url={quote(url, safe='')}&omit_script=true"
        try:
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                r = await client.get(oembed_url)
                r.raise_for_status()
                data = r.json()
        except Exception as exc:
            logger.warning("oEmbed falló: %s", exc)
            return "", None, None

        html = data.get("html", "")
        author = data.get("author_name")

        soup = BeautifulSoup(html, "html.parser")
        p_tag = soup.find("p")
        if not p_tag:
            return "", author, None

        body = p_tag.get_text(separator=" ").strip()

        # Si el cuerpo es solo URLs, el tweet es un enlace puro (p.ej. Twitter Article).
        if _RE_ONLY_URLS.match(body):
            link_tag = p_tag.find("a", href=True)
            if link_tag:
                body, error = await SocialExtractor._via_linked_content(link_tag["href"])
                return body, author, error
            return "", author, None

        return body, author, None

    @staticmethod
    async def _via_linked_content(link_url: str) -> tuple[str, str | None]:
        """Sigue un enlace del tweet (t.co → destino) y extrae su contenido.
        Devuelve (texto, error_reason); uno de los dos siempre es vacío/None.
        """
        try:
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                r = await client.get(link_url)
                final_url = str(r.url)
                html = r.text
        except Exception as exc:
            logger.warning("No se pudo seguir el enlace del tweet: %s", exc)
            return "", None

        # Twitter Articles requieren JS — cascada de fallbacks.
        if "x.com/i/article" in final_url:
            logger.info("Tweet apunta a un Twitter Article: %s", final_url)
            return await SocialExtractor._extract_twitter_article(final_url, html)

        # Enlace externo — intentar extracción con trafilatura.
        extracted = trafilatura.extract(html, url=final_url, favor_recall=True) or ""
        if extracted:
            logger.info("Contenido extraído desde enlace externo: %s", final_url)
            return extracted, None

        return "", None

    @staticmethod
    async def _extract_twitter_article(article_url: str, html: str) -> tuple[str, str | None]:
        """Extrae un Twitter Article con cascada: trafilatura → archive.ph.

        X bloquea activamente los navegadores headless (Playwright incluido) sirviendo
        siempre el muro "JavaScript is disabled", por lo que el renderizado JS no es viable
        sin credenciales de sesión reales.
        """

        # 1. Trafilatura directo (funciona si algún día X deja de requerir JS)
        extracted = trafilatura.extract(html, url=article_url, favor_recall=True) or ""
        if extracted and len(extracted) > 100 and not _is_js_wall(extracted):
            return extracted, None

        # 2. archive.ph — funciona cuando el artículo ha sido archivado previamente
        logger.info("Intentando archive.ph para: %s", article_url)
        text = await SocialExtractor._via_archive_ph(article_url)
        if text:
            return text, None

        return "", (
            f"El tweet enlaza a un Twitter Article que no pudo extraerse automáticamente.\n"
            f"X bloquea el acceso sin autenticación y el artículo no está en archive.ph.\n"
            f"URL del artículo: {article_url}"
        )

    @staticmethod
    async def _via_archive_ph(url: str) -> str:
        """Busca la versión más reciente en archive.ph y extrae el texto."""
        archive_url = f"https://archive.ph/newest/{url}"
        try:
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                r = await client.get(archive_url)
                if r.status_code != 200:
                    return ""
                final_url = str(r.url)
                # archive.ph devuelve su propia página si no hay snapshot → la descartamos
                if "archive.ph" in final_url and "/newest/" in final_url:
                    return ""
                extracted = trafilatura.extract(r.text, url=final_url, favor_recall=True) or ""
                if extracted and not _is_js_wall(extracted):
                    logger.info("archive.ph: contenido extraído desde %s", final_url)
                    return extracted
        except Exception as exc:
            logger.warning("archive.ph falló: %s", exc)
        return ""


def _is_js_wall(text: str) -> bool:
    return "JavaScript is disabled" in text or "enable JavaScript" in text.lower()
