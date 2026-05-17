"""
Descripción de imágenes con un modelo de visión local servido por Ollama.

Uso típico:
    from src.utils.vision import describe_image
    md = describe_image("/path/foto.jpg")  # devuelve markdown estructurado

Si el modelo no está disponible (no `pull`-eado, sin RAM, etc.), devuelve
cadena vacía y deja un warning en logs — el extractor seguirá adelante con
lo que tenga (OCR, en su defecto).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import ollama

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


VISION_PROMPT = """Analiza esta imagen y devuelve una descripción estructurada en {language}.

Devuelve EXCLUSIVAMENTE Markdown con esta estructura exacta, sin preámbulos:

## Descripción general
(2-4 frases capturando el contenido principal de la imagen)

## Elementos identificados
(lista con guiones `-` de objetos, personas, animales, escenas, formas o colores dominantes que VES con claridad)

## Texto visible
(transcribe TODO el texto que aparezca en la imagen, conservando saltos de línea cuando sean claros. Si no hay texto, escribe `(Ninguno)`.)

## Contexto interpretado
(2-3 frases sobre qué situación, lugar, momento o intención sugiere la imagen — etiqueta como interpretación, no como hecho)

## Etiquetas conceptuales
(5-8 palabras clave en {language}, minúsculas, separadas por comas, que describan el tema y categoría)

REGLAS:
- NO inventes detalles que no veas con claridad. Si algo es ambiguo, dilo explícitamente.
- NO añadas conversación, disculpas, ni cierre. Empieza directamente con `## Descripción general`.
- Si la imagen es ininteligible (toda negra, ruido, etc.), dilo en `## Descripción general` y deja vacíos los demás campos."""


def describe_image(
    path: str | Path,
    language: str = "español",
    *,
    model: Optional[str] = None,
) -> str:
    """Llama al modelo de visión configurado y devuelve la descripción markdown.

    Devuelve "" si el modelo no está disponible o falla.
    """
    model = model or settings.vision_model
    if not model:
        return ""
    import time as _time
    t0 = _time.time()
    logger.info("Vision: describiendo imagen %s con %s …", Path(path).name, model)
    try:
        # Timeout largo (5 min) — la primera carga del modelo puede tardar
        client = ollama.Client(host=settings.ollama_host, timeout=300.0)
        resp = client.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": VISION_PROMPT.format(language=language),
                    "images": [str(path)],
                }
            ],
            think=False,
            options={
                "temperature": 0.2,
                "num_predict": 1500,
                "top_p": 0.9,
            },
        )
        msg = resp.get("message", {})
        text = (msg.get("content") or msg.get("thinking") or "").strip()
        if not text:
            logger.warning("Vision: respuesta vacía de %s para %s", model, path)
            return ""
        # Limpieza ligera: a veces el modelo envuelve en ```markdown ... ```
        if text.startswith("```"):
            lines = text.splitlines()[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        logger.info(
            "Vision: descripción lista (%d chars en %.1fs)",
            len(text), _time.time() - t0,
        )
        return text
    except Exception as exc:
        logger.warning(
            "Vision desactivada (%s). Para activarla: `ollama pull %s` "
            "(o ajusta VISION_MODEL en .env). Detalle: %s",
            type(exc).__name__, model, exc,
        )
        return ""


def get_image_metadata(path: str | Path) -> dict:
    """Lee dimensiones y formato vía PyMuPDF. No requiere Pillow."""
    import pymupdf
    meta: dict = {}
    try:
        doc = pymupdf.open(str(path))
        try:
            if doc.page_count > 0:
                page = doc[0]
                meta["dimensions"] = f"{int(page.rect.width)}×{int(page.rect.height)}"
        finally:
            doc.close()
        # extensión sin punto, en mayúsculas
        ext = Path(path).suffix.lstrip(".").upper()
        if ext:
            meta["image_format"] = ext
    except Exception as exc:
        logger.debug("get_image_metadata falló para %s: %s", path, exc)
    return meta
