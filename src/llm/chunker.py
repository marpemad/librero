"""
Chunker para textos largos (Fase 1).

Estrategia:
  1. Si el texto tiene encabezados de sección (Markdown #/##, "Capítulo X", "Chapter X"),
     se divide por ellos. Es lo ideal para libros, papers y artículos largos.
  2. Si no, se hace split por tamaño con solapamiento (preserva continuidad entre chunks).

`estimate_processing()` devuelve una previsión legible para enseñarle al usuario
y pedirle confirmación antes de procesar libros muy largos.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.config.settings import settings


# Encabezados de sección (Markdown ATX + nombres comunes en libros/papers)
_SECTION_RE = re.compile(
    r"^(#{1,3}\s+.+|(?:Cap[ií]tulo|Chapter|Section|Secci[óo]n|Parte|Part)\s+\d+.*)$",
    re.MULTILINE | re.IGNORECASE,
)


@dataclass
class ChunkPlan:
    chunks: list[str]
    method: str            # "by_sections" | "by_size" | "single"
    total_chars: int
    estimated_minutes: float

    @property
    def n_chunks(self) -> int:
        return len(self.chunks)


def _split_by_sections(text: str) -> list[str]:
    """Divide en bloques cuyos encabezados marcan límites naturales."""
    matches = list(_SECTION_RE.finditer(text))
    if len(matches) < 2:
        return []  # No hay suficientes secciones para que merezca la pena
    chunks: list[str] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def _split_by_size(text: str, size: int, overlap: int) -> list[str]:
    """Split por tamaño con solapamiento. Intenta cortar en saltos de párrafo."""
    chunks: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + size, n)
        # Si no es el último, intenta retroceder hasta el último \n\n cercano
        if end < n:
            window = text.rfind("\n\n", i + size // 2, end)
            if window != -1:
                end = window
        chunks.append(text[i:end].strip())
        if end >= n:
            break
        i = max(0, end - overlap)
    return [c for c in chunks if c]


def _merge_small_chunks(chunks: list[str], min_size: int) -> list[str]:
    """Junta chunks consecutivos demasiado pequeños (típico tras split por secciones)."""
    if not chunks:
        return chunks
    merged: list[str] = [chunks[0]]
    for c in chunks[1:]:
        if len(merged[-1]) < min_size:
            merged[-1] = merged[-1] + "\n\n" + c
        else:
            merged.append(c)
    return merged


def make_plan(text: str) -> ChunkPlan:
    n = len(text)
    threshold = settings.chunk_threshold_chars

    # Caso fácil: contenido normal, una sola pasada al LLM.
    if n <= threshold:
        return ChunkPlan(
            chunks=[text],
            method="single",
            total_chars=n,
            estimated_minutes=_estimate_minutes(1),
        )

    size = settings.chunk_size_chars
    overlap = settings.chunk_overlap_chars

    # Caso ideal: hay secciones detectables → split natural.
    by_sec = _split_by_sections(text)
    if by_sec:
        # Si las secciones son demasiado pequeñas, agrupar; si demasiado grandes, re-split.
        by_sec = _merge_small_chunks(by_sec, min_size=size // 2)
        final: list[str] = []
        for c in by_sec:
            if len(c) > size * 1.5:
                final.extend(_split_by_size(c, size, overlap))
            else:
                final.append(c)
        return ChunkPlan(
            chunks=final,
            method="by_sections",
            total_chars=n,
            estimated_minutes=_estimate_minutes(len(final)),
        )

    # Fallback: split por tamaño.
    chunks = _split_by_size(text, size, overlap)
    return ChunkPlan(
        chunks=chunks,
        method="by_size",
        total_chars=n,
        estimated_minutes=_estimate_minutes(len(chunks)),
    )


def _estimate_minutes(n_chunks: int) -> float:
    """
    Estimación grosera: ~45s por chunk (qwen2.5:14b en M1 con num_predict=600 para mapa)
    + 60s extra para la fase de reduce.
    Si n_chunks==1, no hay reduce → solo una pasada de ~30s.
    """
    if n_chunks <= 1:
        return 0.5
    return (n_chunks * 0.75) + 1.0
