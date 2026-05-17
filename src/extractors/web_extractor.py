"""WebScraperExtractor — webs genéricas, papers HTML."""
from __future__ import annotations

import re

import pandas as pd

from src.extractors.base import (
    Extractor,
    ExtractedContent,
    IngestionPayload,
    SourceKind,
)
from src.extractors._helpers import (
    RE_GSHEETS,
    fetch_html,
    find_first_url,
    html_to_main_text,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class WebScraperExtractor(Extractor):
    """URLs HTTP/S genéricas (después de que extractores más específicos digan que no)."""
    name = "web"

    def can_handle(self, payload: IngestionPayload) -> bool:
        return payload.kind is SourceKind.URL

    async def extract(self, payload: IngestionPayload) -> ExtractedContent:
        url = find_first_url(payload.raw) or payload.raw
        logger.info("WebScraperExtractor: %s", url)
        html = await fetch_html(url)
        text = html_to_main_text(html, url=url)

        # Metadata enriquecida (Open Graph + meta + JSON-LD)
        meta = self._extract_web_metadata(html, url)
        title_hint = meta.get("title") or url
        title_hint = title_hint[:200]

        # Determinar doc_kind
        og_type = (meta.get("og_type") or "").lower()
        if "article" in og_type:
            doc_kind = "article"
        elif "video" in og_type:
            doc_kind = "video"
        elif "book" in og_type:
            doc_kind = "book"
        else:
            doc_kind = "web"
        meta["doc_kind"] = doc_kind
        meta["url"] = url

        return ExtractedContent(
            title_hint=title_hint,
            source_type="web",
            source_ref=url,
            text=text or "(No se pudo extraer contenido del cuerpo principal.)",
            extra=meta,
        )

    @staticmethod
    def _extract_web_metadata(html: str, url: str) -> dict:
        """Extrae meta-tags Open Graph, Twitter y JSON-LD (schema.org)."""
        from bs4 import BeautifulSoup
        meta: dict = {}
        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception:
            return meta

        def og(prop: str) -> str | None:
            tag = soup.find("meta", property=prop)
            if tag and tag.get("content"):
                return tag["content"].strip()
            return None

        def name_meta(name: str) -> str | None:
            tag = soup.find("meta", attrs={"name": name})
            if tag and tag.get("content"):
                return tag["content"].strip()
            return None

        # --- Título ---
        title = og("og:title") or name_meta("twitter:title")
        if not title:
            t = soup.find("title")
            if t and t.text:
                title = t.text.strip()
        if title:
            meta["title"] = title[:200]

        # --- Descripción ---
        desc = og("og:description") or name_meta("description") or name_meta("twitter:description")
        if desc:
            meta["description"] = desc[:600]

        # --- Sitio / publicación ---
        if site := og("og:site_name"):
            meta["site_name"] = site
        if og_type := og("og:type"):
            meta["og_type"] = og_type
        # Autor
        author = (
            og("article:author")
            or name_meta("author")
            or name_meta("twitter:creator")
        )
        if author:
            meta["author"] = author.lstrip("@")
        # Fecha publicación
        published = (
            og("article:published_time")
            or name_meta("article:published_time")
            or name_meta("date")
            or name_meta("pubdate")
        )
        if published:
            meta["published"] = published[:10]  # YYYY-MM-DD
            if len(published) >= 4 and published[:4].isdigit():
                meta["year"] = int(published[:4])
        # Tags / sección
        tags: list[str] = []
        for t in soup.find_all("meta", property="article:tag"):
            if t.get("content"):
                tags.append(t["content"].strip())
        if section := og("article:section"):
            tags.append(section)
        if tags:
            meta["source_tags"] = list(dict.fromkeys(tags))[:10]

        # --- JSON-LD (schema.org Article / NewsArticle / BlogPosting) ---
        try:
            import json as _json
            for script in soup.find_all("script", type="application/ld+json"):
                if not script.string:
                    continue
                try:
                    data = _json.loads(script.string)
                except Exception:
                    continue
                # Puede ser dict o lista de dicts
                items = data if isinstance(data, list) else [data]
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    tp = item.get("@type", "")
                    if isinstance(tp, list):
                        tp = " ".join(tp)
                    if any(k in tp for k in ("Article", "NewsArticle", "BlogPosting", "Recipe", "VideoObject", "Book")):
                        # Autor
                        if "author" not in meta:
                            a = item.get("author")
                            if isinstance(a, dict) and a.get("name"):
                                meta["author"] = a["name"]
                            elif isinstance(a, list) and a and isinstance(a[0], dict):
                                names = [x.get("name") for x in a if isinstance(x, dict) and x.get("name")]
                                if names:
                                    meta["authors"] = names[:10]
                            elif isinstance(a, str):
                                meta["author"] = a
                        # Editor / publisher
                        if "publisher" not in meta:
                            p = item.get("publisher")
                            if isinstance(p, dict) and p.get("name"):
                                meta["publisher"] = p["name"]
                        # Fecha
                        if "published" not in meta and item.get("datePublished"):
                            dp = item["datePublished"]
                            meta["published"] = dp[:10]
                            if len(dp) >= 4 and dp[:4].isdigit():
                                meta["year"] = int(dp[:4])
                        # Keywords
                        if "source_tags" not in meta and item.get("keywords"):
                            kw = item["keywords"]
                            if isinstance(kw, str):
                                meta["source_tags"] = [k.strip() for k in re.split(r"[;,]", kw) if k.strip()][:10]
                            elif isinstance(kw, list):
                                meta["source_tags"] = [str(k).strip() for k in kw][:10]
        except Exception:
            pass

        return meta


class GoogleSheetsExtractor(Extractor):
    """
    Hojas de Google públicas — convierte el enlace a su versión `export?format=csv`
    y lo carga con pandas.
    Recordatorio: la hoja debe ser 'Cualquiera con el enlace puede ver'.
    """
    name = "gsheets"

    def can_handle(self, payload: IngestionPayload) -> bool:
        return payload.kind is SourceKind.URL and bool(RE_GSHEETS.search(payload.raw))

    async def extract(self, payload: IngestionPayload) -> ExtractedContent:
        url = find_first_url(payload.raw) or payload.raw
        m = re.search(r"/spreadsheets/d/([\w\-]+)", url)
        if not m:
            raise ValueError(f"No se pudo extraer el sheet ID de {url}")
        sheet_id = m.group(1)

        # Detectar gid si está en la URL (#gid=… o &gid=…)
        gid_m = re.search(r"[#&]gid=(\d+)", url)
        gid = gid_m.group(1) if gid_m else "0"
        csv_url = (
            f"https://docs.google.com/spreadsheets/d/{sheet_id}"
            f"/export?format=csv&gid={gid}"
        )
        logger.info("GoogleSheetsExtractor → %s", csv_url)

        import asyncio
        df = await asyncio.to_thread(pd.read_csv, csv_url)

        head = df.head(50).to_markdown(index=False)
        text = (
            f"_Google Sheet con {len(df)} filas y {len(df.columns)} columnas. "
            f"Columnas: {', '.join(df.columns)}_\n\n{head}"
        )
        return ExtractedContent(
            title_hint=f"GSheet {sheet_id[:8]}",
            source_type="gsheets",
            source_ref=url,
            text=text,
            extra={"sheet_id": sheet_id, "gid": gid, "rows": len(df)},
        )
