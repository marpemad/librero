"""
BaseLLMClient — lógica compartida entre todos los proveedores LLM.

Subclases concretas solo necesitan implementar `_chat(system, user, num_predict) -> str`.
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.extractors import ExtractedContent
from src.llm.chunker import ChunkPlan, make_plan
from src.llm.validator import ValidationResult, validate_note
from src.utils.language import detect_language, language_label
from src.utils.logger import get_logger

try:
    from src.vault_intel import VaultContext
except ImportError:
    VaultContext = None  # type: ignore[assignment,misc]

logger = get_logger(__name__)


# ─── Prompts (idénticos para todos los proveedores) ───────────────────────────

SYSTEM_PROMPT_TEMPLATE = """Eres un bibliotecario digital experto en construir notas Atómicas profundas para Obsidian (Zettelkasten + PARA). Tu objetivo: máxima densidad informativa, cero relleno, cero conversación.

REGLAS ABSOLUTAS — INCUMPLIRLAS INVALIDA TU RESPUESTA:
1. Devuelve EXCLUSIVAMENTE Markdown. CERO texto conversacional, CERO disculpas, CERO bloques ```.
2. La primera línea DEBE ser un H1 con el título de la nota: `# Título Conciso y Específico`.
3. Inmediatamente después, sin omitir ninguna sección obligatoria, en este orden:

   **Etiquetas:** #tag1 #tag2 #tag3 #tag4 #tag5 (entre 4 y 7, minúsculas, en {language})
   **Relaciones:** [[Concepto A]], [[Concepto B]], [[Concepto C]] (entre 3 y 6 wikilinks a notas plausibles)

   ## Resumen
   (3-5 frases densas que capturen TESIS CENTRAL + CONTEXTO + RELEVANCIA)

   ## Puntos Clave
   (lista con guiones `-`, entre 6 y 10 puntos. Cada punto: idea autosuficiente en 1-2 frases. Concretos, accionables o memorables.)

   ## Citas Destacadas
   (entre 2 y 5 citas literales del texto fuente, con `> ` markdown blockquote. Si NO hay citas claras en la fuente, escribe `_(Sin citas literales en la fuente.)_` y omite el resto de citas.)

   ## Personas y Entidades
   (lista con guiones `-` de personas, organizaciones o entidades mencionadas con relevancia. Formato: `- **Nombre** — papel/contexto en 1 frase`. Si no hay ninguna relevante, escribe `_(Sin personas o entidades destacadas.)_`)

   ## Glosario
   (entre 3 y 6 términos técnicos o conceptos clave del texto. Formato: `- **término** — definición en 1 frase, derivada del texto.` Si el contenido no usa terminología específica, escribe `_(Sin terminología técnica relevante.)_`)

   ## Análisis Profundo
   (4-7 párrafos de síntesis sustantiva. NO repitas los puntos clave. Conecta ideas, identifica supuestos, contrasta con otras corrientes/contextos, valora implicaciones, señala límites y posibles aplicaciones prácticas. Aporta perspectiva.)

   ## Preguntas para Profundizar
   (3-5 preguntas abiertas que el lector podría investigar a continuación, con guiones `-`.)

   ## Aplicación
   (1-3 frases sobre cómo aplicar las ideas a la vida personal, profesional o intelectual.)

4. Escribe SIEMPRE en {language} neutro, claro y preciso. Vocabulario rico pero sin pomposidad.
5. NO inventes datos: usa SOLO lo que aparezca en CONTENIDO o en DATOS DE LA FUENTE.
6. Si DATOS DE LA FUENTE incluye autor, año, editorial, canal, etc., RESPETA esos datos exactos en lugar de adivinar.
7. El título del H1 NO contiene `:`, `/`, `\\`, `?`, `*`, `<`, `>`, `|`, ni emojis.
8. Las wikilinks de Relaciones deben ser CONCEPTOS ATÓMICOS (no frases largas, no la propia nota).

EJEMPLO DEL FORMATO EXACTO (respeta los prefijos `**Etiquetas:**` y `**Relaciones:**` LITERALES):

# Disonancia Cognitiva

**Etiquetas:** #psicología #cognición #autoengaño #toma-de-decisiones #festinger
**Relaciones:** [[Sesgo de confirmación]], [[Heurísticas]], [[Racionalización]], [[Leon Festinger]]

## Resumen
…

(NO escribas "Tags:", NO escribas "# tag1 #tag2" como un H1, NO escribas las etiquetas
como bullets. Usa LITERALMENTE `**Etiquetas:**` y `**Relaciones:**` al inicio de cada línea.)
"""

USER_TEMPLATE_SINGLE = """Procesa el siguiente contenido extraído de la fuente "{source_type}".

Pista de título sugerido (úsala como inspiración, mejórala si puedes): {title_hint}
{source_facts}
------- CONTENIDO -------
{content}
------- FIN CONTENIDO -------

Genera la nota siguiendo EXACTAMENTE el formato del system prompt. No añadas nada antes del `# Título`."""

MAP_SYSTEM_PROMPT = """Eres un asistente que resume fragmentos de un texto largo.
Tu salida será combinada después con otros resúmenes para construir una nota final.

REGLAS:
- Devuelve SOLO una lista de 3-6 puntos clave en {language}, con guiones `-`.
- Cada punto, una idea concreta y autosuficiente, en 1-2 frases.
- NO inventes nada que no esté en el fragmento.
- NO uses encabezados, NO añadas introducción ni cierre.
- Si el fragmento es ruido o irrelevante, devuelve `- (sin contenido relevante)`."""

MAP_USER_TEMPLATE = """Fragmento {i} de {n} de un texto sobre "{title_hint}".

------- FRAGMENTO -------
{content}
------- FIN FRAGMENTO -------"""

REDUCE_USER_TEMPLATE = """A continuación tienes los puntos clave extraídos de {n} fragmentos
de un documento más extenso titulado «{title_hint}» (fuente: {source_type}).
{source_facts}
Sintetiza TODO en una única nota final, eliminando redundancias y agrupando ideas afines.
La nota debe seguir EXACTAMENTE el formato (todas las secciones obligatorias) del system prompt.

------- PUNTOS CLAVE POR FRAGMENTO -------
{map_outputs}
------- FIN -------

Genera ahora la nota final."""


@dataclass
class SynthResult:
    markdown: str
    language: str
    needs_review: bool
    validation_problems: list[str]
    chunks_used: int


class BaseLLMClient(ABC):
    """Clase base con toda la lógica de síntesis. Subclases implementan `_chat`."""

    # Subclases deben establecer estos atributos en __init__
    provider_name: str = "base"
    model_name: str = ""

    # ─── API pública ─────────────────────────────────────────────────────────

    def describe(self) -> str:
        """Devuelve una etiqueta legible del proveedor activo, ej. `gemini:gemini-2.0-flash`."""
        return f"{self.provider_name}:{self.model_name}"

    @staticmethod
    def prepare_plan(content: ExtractedContent) -> ChunkPlan:
        return make_plan(content.text)

    async def synthesize(
        self,
        content: ExtractedContent,
        plan: ChunkPlan | None = None,
        vault_ctx: "VaultContext | None" = None,
    ) -> SynthResult:
        plan = plan or make_plan(content.text)
        language = detect_language(content.text)
        lang_label = language_label(language)
        logger.info(
            "Síntesis [%s]: idioma=%s método=%s chunks=%d",
            self.__class__.__name__, language, plan.method, plan.n_chunks,
        )

        if plan.method == "single":
            note = await self._single_pass(content, lang_label, vault_ctx)
        else:
            note = await self._map_reduce(content, plan, lang_label, vault_ctx)

        result = validate_note(note)
        if not result.ok:
            logger.warning("Validación falló: %s — reintentando…", result.problems)
            note = await self._retry_correction(content, note, result, lang_label)
            result = validate_note(note)

        return SynthResult(
            markdown=note,
            language=language,
            needs_review=not result.ok,
            validation_problems=result.problems,
            chunks_used=plan.n_chunks,
        )

    @staticmethod
    def extract_h1_title(markdown: str) -> str:
        from src.llm.validator import _BAD_TITLE_CHARS
        for line in markdown.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                title = _BAD_TITLE_CHARS.sub("", title)
                return title or "Sin título"
        return "Sin título"

    # ─── Método abstracto — cada proveedor lo implementa ────────────────────

    @abstractmethod
    async def _chat(self, system: str, user: str, num_predict: int) -> str:
        """Llama al LLM y devuelve el texto de la respuesta."""
        ...

    # ─── Internos compartidos ─────────────────────────────────────────────────

    async def _single_pass(
        self,
        content: ExtractedContent,
        lang_label: str,
        vault_ctx: "VaultContext | None" = None,
    ) -> str:
        vault_hint = self._build_vault_hint(vault_ctx)
        source_facts = self._build_source_facts(content)
        user_msg = USER_TEMPLATE_SINGLE.format(
            source_type=content.source_type,
            title_hint=content.title_hint,
            source_facts=source_facts,
            content=content.text,
        ) + vault_hint
        return await self._chat(
            system=SYSTEM_PROMPT_TEMPLATE.format(language=lang_label),
            user=user_msg,
            num_predict=6000,
        )

    async def _map_reduce(
        self,
        content: ExtractedContent,
        plan: ChunkPlan,
        lang_label: str,
        vault_ctx: "VaultContext | None" = None,
    ) -> str:
        map_system = MAP_SYSTEM_PROMPT.format(language=lang_label)
        map_outputs: list[str] = []
        for i, chunk in enumerate(plan.chunks, start=1):
            logger.info("MAP %d/%d (%d chars)", i, plan.n_chunks, len(chunk))
            user_msg = MAP_USER_TEMPLATE.format(
                i=i, n=plan.n_chunks,
                title_hint=content.title_hint,
                content=chunk,
            )
            piece = await self._chat(system=map_system, user=user_msg, num_predict=1200)
            map_outputs.append(f"--- Fragmento {i} ---\n{piece.strip()}")

        logger.info("REDUCE: combinando %d resúmenes parciales", len(map_outputs))
        vault_hint = self._build_vault_hint(vault_ctx)
        source_facts = self._build_source_facts(content)
        reduce_user = REDUCE_USER_TEMPLATE.format(
            n=plan.n_chunks,
            title_hint=content.title_hint,
            source_type=content.source_type,
            source_facts=source_facts,
            map_outputs="\n\n".join(map_outputs),
        ) + vault_hint
        return await self._chat(
            system=SYSTEM_PROMPT_TEMPLATE.format(language=lang_label),
            user=reduce_user,
            num_predict=6000,
        )

    async def _retry_correction(
        self,
        content: ExtractedContent,
        previous_attempt: str,
        validation: ValidationResult,
        lang_label: str,
    ) -> str:
        correction = validation.as_correction_prompt()
        user_msg = (
            f"Procesa este contenido (fuente: {content.source_type}, título sugerido: "
            f"«{content.title_hint}»):\n\n{content.text[:8000]}\n\n"
            f"Tu intento anterior fue:\n\n{previous_attempt}\n\n{correction}"
        )
        return await self._chat(
            system=SYSTEM_PROMPT_TEMPLATE.format(language=lang_label),
            user=user_msg,
            num_predict=4096,
        )

    @staticmethod
    def _build_source_facts(content: ExtractedContent) -> str:
        e = content.extra or {}
        FIELDS = [
            ("doc_kind",        "Tipo"),
            ("book_title",      "Título original"),
            ("doc_title",       "Título original"),
            ("authors",         "Autores"),
            ("author",          "Autor"),
            ("publisher",       "Editorial"),
            ("year",            "Año"),
            ("published",       "Fecha de publicación"),
            ("isbn",            "ISBN"),
            ("doi",             "DOI"),
            ("original_language", "Idioma original"),
            ("site_name",       "Sitio"),
            ("url",             "URL"),
            ("channel_url",     "Canal"),
            ("video_id",        "Video ID"),
            ("duration",        "Duración"),
            ("view_count",      "Visualizaciones"),
            ("description",     "Descripción"),
            ("abstract",        "Abstract"),
            ("chapters",        "Capítulos"),
            ("source_tags",     "Tags de la fuente"),
            ("query",           "Consulta original"),
            ("sources",         "Fuentes consultadas"),
        ]
        seen_labels: set[str] = set()
        lines: list[str] = []
        for key, label in FIELDS:
            if label in seen_labels:
                continue
            val = e.get(key)
            if val is None or val == "" or val == []:
                continue
            seen_labels.add(label)
            if isinstance(val, list):
                rendered = ", ".join(str(v) for v in val[:10])
            else:
                rendered = str(val)
                if len(rendered) > 600:
                    rendered = rendered[:600].rstrip() + "…"
            lines.append(f"- {label}: {rendered}")

        if not lines:
            return ""
        return (
            "\n------- DATOS DE LA FUENTE (factuales — RESPÉTALOS, no inventes) -------\n"
            + "\n".join(lines)
            + "\n------- FIN DATOS -------\n"
        )

    @staticmethod
    def _build_vault_hint(vault_ctx: "VaultContext | None") -> str:
        if vault_ctx is None:
            return ""
        parts: list[str] = []
        if vault_ctx.suggested_wikilinks:
            wl = ", ".join(f"[[{t}]]" for t in vault_ctx.suggested_wikilinks[:6])
            parts.append(f"Wikilinks REALES ya existentes en el Vault (úsalos si aplican): {wl}")
        if vault_ctx.existing_tags:
            tags = " ".join(f"#{t}" for t in vault_ctx.existing_tags[:25])
            parts.append(
                f"Tags ya usados en el Vault (úsalos preferentemente, crea nuevos solo si ninguno encaja): {tags}"
            )
        if not parts:
            return ""
        return (
            "\n\n---\n**CONTEXTO DEL VAULT (importante para coherencia):**\n"
            + "\n".join(parts)
        )

    @staticmethod
    def _post_process(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        if not text.startswith("#"):
            idx = text.find("\n# ")
            if idx > -1:
                text = text[idx + 1:]
        text = text.strip()
        text = BaseLLMClient._fix_metadata_labels(text)
        text = BaseLLMClient._rescue_metadata(text)
        return text

    @staticmethod
    def _rescue_metadata(text: str) -> str:
        import re
        lines = text.splitlines()

        has_etiq = any(re.match(r"^\*\*Etiquetas:\*\*", l.strip()) for l in lines)
        has_rels = any(re.match(r"^\*\*Relaciones:\*\*", l.strip()) for l in lines)
        if has_etiq and has_rels:
            return text

        h1_idx = next(
            (i for i, l in enumerate(lines) if l.strip().startswith("# ")), -1,
        )
        if h1_idx < 0:
            return text

        end_idx = next(
            (i for i, l in enumerate(lines) if i > h1_idx and l.strip().startswith("## ")),
            min(len(lines), h1_idx + 30),
        )
        zone = list(enumerate(lines[h1_idx + 1:end_idx], start=h1_idx + 1))

        _TAG_RE = re.compile(r"#[\wáéíóúüñÁÉÍÓÚÜÑ\-]+")
        _WL_RE  = re.compile(r"\[\[[^\]]+\]\]")
        _ETIQ_PREFIX = re.compile(
            r"^\s*\*{0,2}\s*(tags?|etiquetas?|labels?|tema?s?)\s*\*{0,2}\s*:?\s*",
            re.IGNORECASE,
        )
        _REL_PREFIX = re.compile(
            r"^\s*\*{0,2}\s*(related|relations?|relaciones?|conexiones?|v[ií]nculos?|links?|enlaces?)\s*\*{0,2}\s*:?\s*",
            re.IGNORECASE,
        )

        tags_collected: list[str] = []
        tags_replace_indices: set[int] = set()
        if not has_etiq:
            for idx, raw in zone:
                stripped = raw.strip()
                if not stripped:
                    continue
                m = _ETIQ_PREFIX.match(stripped)
                if m:
                    rest = stripped[m.end():]
                    candidates = _TAG_RE.findall(rest)
                    if not candidates:
                        candidates = [
                            w if w.startswith("#") else f"#{w.strip(',;.')}"
                            for w in rest.split()
                            if w and re.match(r"^[#]?[\wáéíóúüñÁÉÍÓÚÜÑ\-]+[,;.]?$", w)
                        ]
                    if len(candidates) >= 2:
                        tags_collected = candidates
                        tags_replace_indices.add(idx)
                        for j in range(idx + 1, min(idx + 8, end_idx)):
                            sub = lines[j].strip()
                            if not sub or sub.startswith("##"):
                                break
                            bm = re.match(r"^[-*•]\s*(#[\wáéíóúüñÁÉÍÓÚÜÑ\-]+)\s*$", sub)
                            if bm:
                                tags_collected.append(bm.group(1))
                                tags_replace_indices.add(j)
                            else:
                                break
                        break

            if not tags_collected:
                for idx, raw in zone:
                    found = _TAG_RE.findall(raw)
                    if len(found) >= 3:
                        tags_collected = found
                        tags_replace_indices.add(idx)
                        break

            if not tags_collected:
                bulleted: list[tuple[int, str]] = []
                for idx, raw in zone:
                    sub = raw.strip()
                    bm = re.match(r"^[-*•]\s*(#[\wáéíóúüñÁÉÍÓÚÜÑ\-]+)\s*$", sub)
                    if bm:
                        bulleted.append((idx, bm.group(1)))
                if len(bulleted) >= 3:
                    tags_collected = [t for _, t in bulleted]
                    tags_replace_indices.update(i for i, _ in bulleted)

        rels_collected: list[str] = []
        rels_replace_indices: set[int] = set()
        if not has_rels:
            for idx, raw in zone:
                if idx in tags_replace_indices:
                    continue
                stripped = raw.strip()
                m = _REL_PREFIX.match(stripped)
                if m:
                    rest = stripped[m.end():]
                    wl = _WL_RE.findall(rest)
                    if len(wl) >= 1:
                        rels_collected = wl
                        rels_replace_indices.add(idx)
                        break

            if not rels_collected:
                for idx, raw in zone:
                    if idx in tags_replace_indices:
                        continue
                    wl = _WL_RE.findall(raw)
                    if len(wl) >= 2:
                        rels_collected = wl
                        rels_replace_indices.add(idx)
                        break

        if not tags_collected and not rels_collected:
            return text

        to_remove = tags_replace_indices | rels_replace_indices
        new_lines: list[str] = []
        for i, l in enumerate(lines):
            if i in to_remove:
                continue
            new_lines.append(l)
            if i == h1_idx and (tags_collected or rels_collected):
                if not (i + 1 < len(lines) and lines[i + 1].strip() == ""):
                    new_lines.append("")
                if tags_collected and not has_etiq:
                    seen: set[str] = set()
                    norm: list[str] = []
                    for t in tags_collected:
                        t = t if t.startswith("#") else f"#{t.lstrip('#')}"
                        if t not in seen:
                            seen.add(t)
                            norm.append(t)
                    new_lines.append(f"**Etiquetas:** {' '.join(norm[:8])}")
                if rels_collected and not has_rels:
                    seen2: set[str] = set()
                    norm2: list[str] = []
                    for w in rels_collected:
                        if w not in seen2:
                            seen2.add(w)
                            norm2.append(w)
                    new_lines.append(f"**Relaciones:** {', '.join(norm2[:8])}")
        return "\n".join(new_lines)

    @staticmethod
    def _fix_metadata_labels(text: str) -> str:
        import re

        _WORD = r"[\wáéíóúüñÁÉÍÓÚÜÑ\-]+"
        _TAG  = r"#" + _WORD
        _TAGS_LINE = re.compile(
            r"^#?\s*"
            r"(" + _WORD + r")"
            r"(?:\s+" + _TAG + r"){1,}"
            r"\s*$"
            r"|"
            r"^(" + _TAG + r"(?:\s+" + _TAG + r"){1,})\s*$"
        )
        _WIKILINKS_LINE = re.compile(
            r"^(\[\[[^\]]+\]\](?:[,\s]+\[\[[^\]]+\]\])*)\s*$"
        )
        _HAS_ETIQUETAS = re.compile(r"^\*\*Etiquetas:\*\*")
        _HAS_RELACIONES = re.compile(r"^\*\*Relaciones:\*\*")

        lines = text.splitlines()
        result: list[str] = []
        found_h1 = False

        for line in lines:
            stripped = line.strip()

            if stripped.startswith("# ") and not found_h1:
                found_h1 = True
                result.append(line)
                continue

            if not found_h1:
                result.append(line)
                continue

            if _HAS_ETIQUETAS.match(stripped) or _HAS_RELACIONES.match(stripped):
                result.append(line)
                continue

            if re.match(r"^\s*[-*•]\s|^\s*\d+\.\s", line):
                result.append(line)
                continue

            m_tags = _TAGS_LINE.match(stripped)
            if m_tags:
                clean = re.sub(r"^#\s+", "", stripped)
                words = clean.split()
                tags = [w if w.startswith("#") else f"#{w}" for w in words if w]
                if len(tags) >= 2:
                    result.append(f"**Etiquetas:** {' '.join(tags)}")
                    continue
                result.append(line)
                continue

            m_wl = _WIKILINKS_LINE.match(stripped)
            if m_wl:
                result.append(f"**Relaciones:** {m_wl.group(1)}")
                continue

            result.append(line)

        return "\n".join(result)
