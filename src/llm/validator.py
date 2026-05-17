"""
Validador de la estructura de la nota generada por el LLM.

Devuelve una lista de problemas (strings legibles). Lista vacía = nota válida.
La lista se reusa para el prompt correctivo en el segundo intento.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Caracteres prohibidos en el H1 (incompatibles con nombres de fichero)
_BAD_TITLE_CHARS = re.compile(r'[:\\/?*<>|"]')

# Estructura mínima esperada — secciones OBLIGATORIAS.
# Las secciones nuevas (Citas, Personas, Glosario, Preguntas, Aplicación) son
# deseables pero NO obligatorias: el LLM puede omitirlas si no aplican y la nota
# se considera válida igualmente.
_REQUIRED_SECTIONS = ("## Resumen", "## Puntos Clave", "## Análisis Profundo")


@dataclass
class ValidationResult:
    ok: bool
    problems: list[str]
    title: str | None
    n_tags: int
    n_relations: int

    def as_correction_prompt(self) -> str:
        """Texto que mandamos al LLM para el segundo intento."""
        bullets = "\n".join(f"- {p}" for p in self.problems)
        return (
            "Tu respuesta anterior NO cumplió el formato exigido. "
            "Corrige específicamente estos problemas:\n"
            f"{bullets}\n\n"
            "Devuelve la nota completa de nuevo, ya corregida, "
            "respetando todas las reglas del system prompt."
        )


def validate_note(md: str) -> ValidationResult:
    problems: list[str] = []
    title: str | None = None
    n_tags = 0
    n_relations = 0

    text = md.strip()

    # 1) H1 al inicio
    first_line = text.splitlines()[0] if text else ""
    if not first_line.startswith("# "):
        problems.append("La primera línea debe ser un H1 (`# Título`).")
    else:
        title = first_line[2:].strip()
        if not title:
            problems.append("El H1 está vacío.")
        elif _BAD_TITLE_CHARS.search(title):
            problems.append(
                f"El título contiene caracteres prohibidos (`:`, `/`, `\\`, `?`, `*`, `<`, `>`, `|`, `\"`). "
                f"Título recibido: «{title}»."
            )

    # 2) Etiquetas
    tags_match = re.search(r"\*\*Etiquetas:\*\*\s*([^\n]+)", text)
    if not tags_match:
        problems.append("Falta la línea `**Etiquetas:** #tag1 #tag2 ...`.")
    else:
        tags = re.findall(r"#[\wáéíóúüñÁÉÍÓÚÜÑ\-]+", tags_match.group(1))
        n_tags = len(tags)
        if n_tags < 3:
            problems.append(f"Solo hay {n_tags} etiquetas (mínimo 3).")
        elif n_tags > 8:
            problems.append(f"Hay {n_tags} etiquetas (máximo 8).")

    # 3) Relaciones (wikilinks)
    rel_match = re.search(r"\*\*Relaciones:\*\*\s*([^\n]+)", text)
    if not rel_match:
        problems.append("Falta la línea `**Relaciones:** [[A]], [[B]], ...`.")
    else:
        wikilinks = re.findall(r"\[\[([^\]]+)\]\]", rel_match.group(1))
        n_relations = len(wikilinks)
        if n_relations < 2:
            problems.append(f"Solo hay {n_relations} wikilinks de relación (mínimo 2).")
        elif n_relations > 8:
            problems.append(f"Hay {n_relations} wikilinks (máximo 8).")

    # 4) Secciones obligatorias
    for section in _REQUIRED_SECTIONS:
        if section not in text:
            problems.append(f"Falta la sección obligatoria `{section}`.")

    return ValidationResult(
        ok=len(problems) == 0,
        problems=problems,
        title=title,
        n_tags=n_tags,
        n_relations=n_relations,
    )
