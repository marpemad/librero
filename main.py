"""
Punto de entrada — bot de Telegram en Long Polling. (Fase 1)

Flujo por mensaje:
  1. Telegram → Router → IngestionPayload
  2. Factory selecciona Extractor → ExtractedContent
  3. (NUEVO) make_plan() decide si hacer chunking. Si el contenido es muy grande,
     bot pide confirmación al usuario con botones inline.
  4. LLMClient.synthesize() → SynthResult (con flag needs_review si toca) — Ollama o Gemini según .env
  5. ObsidianWriter.write() → archivo .md en el Vault
  6. Mensaje final al usuario indicando idioma, chunks usados y si necesita revisión.

Handlers nuevos:
  - handle_voice: mensajes de voz de Telegram → VoiceExtractor (vía FILE).
  - on_confirm_callback: respuesta a los botones de confirmación.
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.error import NetworkError, TimedOut
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from src.config.settings import settings
from src.extractors import ExtractorFactory, IngestionPayload, SourceKind
from src.llm import make_plan, make_llm_client
from src.queue import Job, JobQueue, JobStatus
from src.queue.job_queue import new_job_id as _new_queue_job_id
from src.router import MessageRouter
from src.notes import NoteManager, kind_emoji, normalize_kind
from src.tasks import (
    STATE_EMOJI as TASK_STATE_EMOJI,
    Task,
    TaskBoard,
    TaskState,
    extract_task_description,
)
from src.utils.logger import get_logger
from src.writer import ObsidianWriter

# Google Calendar — importación opcional: si las libs no están instaladas, el bot sigue
# funcionando sin integración de calendario.
try:
    from src.gcalendar import GoogleCalendarClient, NotifyDB, CalendarScheduler
    from src.gcalendar import format_agenda, format_event
    _GCAL_AVAILABLE = True
except ImportError:
    _GCAL_AVAILABLE = False

# Vault Intel (Phase 2) — importación opcional: si falla (modelo no disponible), el bot
# sigue funcionando sin deduplicación ni wikilinks reales.
try:
    from src.vault_intel import VaultIndexer, VaultSearcher, VaultContext
    _VAULT_INTEL_AVAILABLE = True
except ImportError:
    _VAULT_INTEL_AVAILABLE = False
    VaultContext = None  # type: ignore[assignment,misc]

logger = get_logger(__name__)


# ─── Singletons reutilizables ─────────────────────────────────────────────────
llm = make_llm_client()
writer = ObsidianWriter()
job_queue = JobQueue()  # cola serializada para evitar OOM con LLM concurrente
task_board = TaskBoard(settings.tasks_file_path)  # tablón Kanban en el Vault
note_manager = NoteManager(settings.lists_path)   # gestor de listas curadas

# Google Calendar — singletons (None si las libs no están instaladas)
gcal: "GoogleCalendarClient | None" = None
notify_db: "NotifyDB | None" = None
cal_scheduler: "CalendarScheduler | None" = None

# Vault Intel — se inicializan en _post_init (puede quedar None si el modelo no está)
vault_indexer: "VaultIndexer | None" = None
vault_searcher: "VaultSearcher | None" = None


async def _post_init(application: Application) -> None:
    """Inicializa el índice semántico del Vault y arranca la cola de trabajos."""
    global vault_indexer, vault_searcher

    # 1. Vault Intel
    if _VAULT_INTEL_AVAILABLE:
        try:
            idx: VaultIndexer = VaultIndexer()
            n = await asyncio.to_thread(idx.rebuild)
            vault_indexer = idx
            vault_searcher = VaultSearcher(idx)
            logger.info(
                "🧭 Vault Intel activo: %d notas indexadas (%d actualizadas)",
                idx.note_count(), n,
            )
        except Exception as exc:
            logger.warning(
                "⚠️ Vault Intel desactivado (%s). "
                "Para activar: ollama pull nomic-embed-text",
                exc,
            )

    # 2. Google Calendar
    global gcal, notify_db, cal_scheduler
    if _GCAL_AVAILABLE:
        try:
            gcal = GoogleCalendarClient(
                client_id=settings.gcal_client_id,
                client_secret=settings.gcal_client_secret,
                token_path=settings.gcal_token_path,
            )
            notify_db = NotifyDB(settings.gcal_notify_db_path)
            # Buscamos el primer chat_id autorizado para enviar notificaciones proactivas
            first_user_id = next(iter(settings.allowed_user_ids), None)
            if first_user_id:
                cal_scheduler = CalendarScheduler(
                    gcal=gcal,
                    notify_db=notify_db,
                    bot=application.bot,
                    chat_id=first_user_id,
                )
                cal_scheduler.start()
                logger.info("📅 Google Calendar módulo activo. Autenticado: %s", gcal.is_authenticated())
        except Exception as exc:
            logger.warning("⚠️ Google Calendar no disponible: %s", exc)
    else:
        logger.info("📅 Google Calendar desactivado (instala google-api-python-client apscheduler)")

    # 3. JobQueue worker — síntesis serializada para no saturar el LLM
    bot = application.bot

    async def _synthesis_handler(job: Job) -> None:
        """Procesa UN job: actualiza UI, llama al LLM, escribe la nota, indexa."""
        await _process_synthesis_job(bot, job)

    job_queue.start(_synthesis_handler)


# ─── Estado en memoria de jobs pendientes de confirmación ────────────────────
# Se guarda en `bot_data` (persistente durante la vida del proceso). Cada job tiene un id corto.
@dataclass
class PendingJob:
    payload: IngestionPayload
    chat_id: int


def _new_job_id() -> str:
    return uuid.uuid4().hex[:8]


def _put_job(app: Application, job: PendingJob) -> str:
    app.bot_data.setdefault("jobs", {})
    job_id = _new_job_id()
    app.bot_data["jobs"][job_id] = job
    return job_id


def _pop_job(app: Application, job_id: str) -> Optional[PendingJob]:
    return app.bot_data.get("jobs", {}).pop(job_id, None)


# ─── Helpers de seguridad ─────────────────────────────────────────────────────
def _is_authorized(update: Update) -> bool:
    user = update.effective_user
    if not user:
        return False
    return user.id in settings.allowed_user_ids


# ─── Helper: URL `obsidian://` para abrir notas con un click ─────────────────
def _make_obsidian_url(path: Path) -> str:
    """Construye un deep link `obsidian://open?vault=X&file=Y` para esta nota.

    En Mac/iOS abre Obsidian directamente; en otras plataformas el click es
    inerte si Obsidian no está instalado. La ruta se calcula relativa al Vault.
    """
    from urllib.parse import quote
    vault_root = settings.obsidian_vault_path.resolve()
    try:
        rel = path.resolve().relative_to(vault_root)
    except ValueError:
        # Por si la nota no está dentro del Vault (no debería pasar)
        rel = Path(path.name)
    vault_name = vault_root.name
    return f"obsidian://open?vault={quote(vault_name)}&file={quote(str(rel))}"


# ─── Helper: escape Markdown (parse_mode="Markdown") ─────────────────────────
# Telegram parse_mode="Markdown" interpreta `_*[`. Si un título o nombre de
# fichero los contiene, el mensaje peta con BadRequest. Escapamos esos chars
# antes de interpolar cualquier valor que no controlemos.
def _md(s: str | None) -> str:
    if not s:
        return ""
    return (
        str(s)
        .replace("\\", "\\\\")
        .replace("_", "\\_")
        .replace("*", "\\*")
        .replace("`", "\\`")
        .replace("[", "\\[")
    )


# ─── Comandos ─────────────────────────────────────────────────────────────────
async def cmd_start(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update):
        await update.message.reply_text("⛔ No autorizado.")
        return
    extractors = ", ".join(ExtractorFactory.registered())
    await update.message.reply_text(
        "📚 *Librero Automatizado v1*\n\n"
        "Mándame:\n"
        "• Una idea o concepto (texto plano)\n"
        "• Concepto *estructurado*: `Tipo. Autor. Título`\n"
        "    ej. `Libro. Robert Kiyosaki. Padre rico, padre pobre`\n"
        "    ej. `Película: Inception - Christopher Nolan`\n"
        "• Una URL (YouTube, Twitter/X, Spotify, web, paper, GSheets…)\n"
        "• Un archivo (PDF, EPUB, DOCX, CSV, TXT, MD)\n"
        "• 🖼 Una imagen (la describo con visión + OCR)\n"
        "• 🎤 Un mensaje de voz (lo transcribiré con Whisper)\n"
        "• 📋 *Tareas:* mensaje empezando por `task` o `tarea`\n"
        "    ej. `task comprar café`, `tarea: llamar al fontanero`\n\n"
        "*Comandos:*\n"
        "• /status — estado del bot y cola\n"
        "• /queue — qué hay en cola\n"
        "• /cancel `<id>` — cancelar un job pendiente\n"
        "• /weekly `[kind]` `[llm]` — resumen de los últimos 7 días\n"
        "• /random `[kind]` — nota aleatoria, opcionalmente filtrada\n"
        "• /search `<texto>` — búsqueda semántica (embeddings)\n"
        "• /find `<palabras>` — búsqueda literal AND (más rápida)\n"
        "• /read `<id>` — leer una nota por su id o slug\n"
        "• /tasks — tablón de tareas\n"
        "• /done `T003` · /wip `T003` · /todo `T003` · /del `T003`\n"
        "• /note `<tipo>` `[ítem1, ítem2]` — listas curadas\n"
        "    ej. `/note libros Isaac Asimov, Marcos Vázquez`\n"
        "    ej. `/note series` (ver lista) · `/note` (ver tipos)\n"
        "• /notedel `<tipo>` `<nombre>` — eliminar de una lista\n\n"
        "📅 *Google Calendar:*\n"
        "• /calauth — conectar Google Calendar (OAuth2)\n"
        "• /cal `[días]` — agenda de hoy o próximos N días\n"
        "• /calweek — agenda de la semana\n"
        "• /caladd `<título> [fecha] [hora]` — añadir evento\n"
        "• /calday — recibir resumen del día ahora\n"
        "• /calconfig `[HH:MM|on|off]` — configurar avisos\n\n"
        f"Extractores: `{extractors}`\n"
        f"LLM: `{llm.describe()}`",
        parse_mode="Markdown",
    )


_HELP_TEXT = """\
📚 *Librero — comandos disponibles*

━━━━━━━━━━━━━━━━━
*📥 Ingesta de contenido*
Mándame cualquiera de estos directamente:
  • Texto libre o idea
  • Concepto estructurado: `Tipo. Autor. Título`
    _ej._ `Libro. Kiyosaki. Padre rico, padre pobre`
  • URL (YouTube, Spotify, web, paper…)
  • Archivo adjunto (PDF, EPUB, DOCX, TXT, MD, CSV)
  • 🖼 Imagen (visión + OCR)
  • 🎤 Mensaje de voz (transcripción Whisper)

━━━━━━━━━━━━━━━━━
*⚙️ Sistema*
`/status` — estado del bot, cola y configuración
`/queue` — jobs en cola y el que se está procesando
`/cancel <id>` — cancelar un job pendiente

━━━━━━━━━━━━━━━━━
*🔍 Búsqueda y lectura*
`/search <texto>` — búsqueda semántica (embeddings)
`/find <palabras>` — búsqueda literal AND (sin LLM)
`/read <id>` — leer nota por id o slug del fichero
`/random [tipo]` — nota aleatoria del Vault
  _ej._ `/random` · `/random book` · `/random video`
`/weekly [tipo] [llm]` — resumen de los últimos 7 días
  _ej._ `/weekly` · `/weekly book` · `/weekly llm`

━━━━━━━━━━━━━━━━━
*📋 Tareas (Kanban)*
`/tasks` — tablón completo con botones rápidos
`/done <id>` — marcar tarea como completada
`/wip <id>` — marcar tarea en progreso
`/todo <id>` — devolver tarea a pendiente
`/del <id>` — borrar tarea
_Añadir:_ escribe `task <descripción>` o `tarea: <desc>`
  _ej._ `task comprar café` · `tarea: llamar al fontanero`

━━━━━━━━━━━━━━━━━
*📝 Listas curadas*
`/note` — ver todos los tipos de lista
`/note <tipo>` — ver una lista concreta
`/note <tipo> <ítem1>, <ítem2>` — añadir ítems
`/notedel <tipo> <nombre>` — eliminar un ítem
_Tipos:_ `libros` · `series` · `películas` · `podcasts`
         `música` · `juegos` · `artículos` · `otros` _(y cualquier nombre ad\\-hoc)_
  _ej._ `/note libros Isaac Asimov, Marcos Vázquez`
  _ej._ `/notedel series Dark`

━━━━━━━━━━━━━━━━━
*📅 Google Calendar*
`/calauth` — conectar Google Calendar (OAuth2, solo una vez)
`/cal [N]` — agenda de hoy (o próximos N días)
`/calweek` — agenda de los próximos 7 días
`/caladd <título> [fecha] [hora] [min]` — crear evento
  _ej._ `/caladd Dentista mañana 10:30`
  _ej._ `/caladd Reunión viernes 16:00 90`
`/calday` — recibir resumen del día ahora mismo
`/calconfig` — ver configuración de avisos
`/calconfig HH:MM` — cambiar hora del resumen diario
`/calconfig on|off` — activar/desactivar resumen semanal
_En cada evento hay botones para_ 🔔 _avisar,_ ✏️ _editar y_ 🗑 _borrar._
"""


async def cmd_help(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Muestra todos los comandos disponibles."""
    if not _is_authorized(update):
        return
    try:
        await update.message.reply_text(_HELP_TEXT, parse_mode="Markdown",
                                        disable_web_page_preview=True)
    except Exception:
        await update.message.reply_text(_HELP_TEXT, disable_web_page_preview=True)


async def cmd_status(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update):
        return
    stats = job_queue.stats()
    await update.message.reply_text(
        f"🟢 OK\nVault: `{settings.obsidian_vault_path}`\n"
        f"Inbox: `{settings.obsidian_inbox_folder}`\n"
        f"Modelo: `{llm.describe()}`\n"
        f"Umbral chunking: {settings.chunk_threshold_chars:,} chars\n"
        f"Umbral confirmación: {settings.confirm_threshold_chars:,} chars\n"
        f"\n*Cola:* corriendo={1 if job_queue.current_job() else 0} · "
        f"pendientes={stats['pending']} · ok={stats['done']} · "
        f"fallos={stats['failed']} · cancelados={stats['cancelled']}",
        parse_mode="Markdown",
    )


async def cmd_weekly(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Resumen de notas ingeridas/actualizadas en los últimos 7 días.

    Sintaxis:
      /weekly                  → todas las notas
      /weekly book             → solo libros (alias: libro, libros, book, books)
      /weekly video            → solo vídeos
      /weekly llm              → todas + meta-resumen LLM al final
      /weekly book llm         → libros + meta-resumen LLM
    """
    if not _is_authorized(update):
        return
    from src.vault_intel.digest import (
        collect_weekly_notes, weekly_digest, normalize_kind, llm_meta_summary,
    )

    args = [a.lower() for a in (ctx.args or [])]
    use_llm = "llm" in args
    args = [a for a in args if a != "llm"]
    kind_filter = normalize_kind(args[0]) if args else None

    if args and not kind_filter:
        await update.message.reply_text(
            f"⚠️ Tipo desconocido: `{args[0]}`. "
            "Tipos válidos: book, video, paper, podcast, article, web, image, concept, voice, tweet.",
            parse_mode="Markdown",
        )
        return

    digest = await asyncio.to_thread(
        weekly_digest, settings.obsidian_vault_path, 7, kind_filter
    )

    # Meta-resumen LLM (encolado para no bloquear el bot ni saltarse la cola)
    if use_llm:
        notes = await asyncio.to_thread(
            collect_weekly_notes, settings.obsidian_vault_path, 7, kind_filter
        )
        if not notes:
            # Nada que resumir → mandamos solo el digest base
            pass
        else:
            await update.message.reply_text(
                f"🧠 Generando meta-resumen LLM de {len(notes)} notas… "
                f"(tarda ~30s, va por la cola)"
            )
            meta = await asyncio.to_thread(
                llm_meta_summary, notes, 7, llm, "Español",
            )
            if meta:
                digest = digest + "\n\n━━━━━━━━━━━━━━━━━━━━\n\n" + meta

    await _send_long_markdown(
        update.message, digest,
        filename=f"weekly_{datetime_now_str()}.md",
        caption="📅 Resumen semanal (demasiado largo para un mensaje)",
    )


async def _send_long_markdown(target_msg, text: str, filename: str, caption: str) -> None:
    """Manda un texto Markdown. Si encaja en 4000 chars lo manda como mensaje;
    si no, como adjunto. Si el parser de Markdown peta, fallback a texto plano."""
    if len(text) <= 4000:
        try:
            await target_msg.reply_text(text, parse_mode="Markdown")
            return
        except Exception:
            await target_msg.reply_text(text)
            return
    from io import BytesIO
    buf = BytesIO(text.encode("utf-8"))
    buf.name = filename
    await target_msg.reply_document(document=buf, filename=filename, caption=caption)


async def cmd_random(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Devuelve una nota aleatoria del Vault.

    Sintaxis:
      /random           → cualquiera
      /random book      → solo libros
      /random video     → solo vídeos
    """
    if not _is_authorized(update):
        return
    from src.vault_intel.digest import (
        pick_random, format_note_for_preview, normalize_kind,
    )

    args = ctx.args or []
    kind_filter = normalize_kind(args[0]) if args else None
    if args and not kind_filter:
        await update.message.reply_text(
            f"⚠️ Tipo desconocido: `{args[0]}`. "
            "Tipos válidos: book, video, paper, podcast, article, web, image, concept, voice, tweet.",
            parse_mode="Markdown",
        )
        return

    summary = await asyncio.to_thread(pick_random, settings.obsidian_vault_path, kind_filter)
    if summary is None:
        if kind_filter:
            await update.message.reply_text(f"📭 No hay notas de tipo `{kind_filter}`.", parse_mode="Markdown")
        else:
            await update.message.reply_text("📭 Tu Vault está vacío.")
        return

    body, truncated = await asyncio.to_thread(format_note_for_preview, summary.path, 3500)

    header = (
        f"🎲 *Nota aleatoria* {summary.emoji}\n"
        f"📁 `{summary.path.name}`\n"
        f"🕒 _Modificada: {summary.mtime.strftime('%Y-%m-%d %H:%M')}_\n"
        f"━━━━━━━━━━━━━━━━━━━━\n\n"
    )
    full = header + body

    if len(full) <= 4000:
        try:
            await update.message.reply_text(full, parse_mode="Markdown")
            return
        except Exception:
            # Fallback sin formato (algún carácter conflictivo)
            await update.message.reply_text(full)
            return

    # Si aun así es demasiado largo, mandamos cabecera + archivo adjunto
    await update.message.reply_text(header, parse_mode="Markdown")
    from io import BytesIO
    full_text = summary.path.read_text(encoding="utf-8", errors="ignore")
    buf = BytesIO(full_text.encode("utf-8"))
    buf.name = summary.path.name
    await update.message.reply_document(document=buf, filename=summary.path.name)


def datetime_now_str() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M")


async def cmd_search(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Búsqueda semántica en el Vault usando embeddings (top-5 más similares).

    Sintaxis:  /search <texto libre>
    Requiere Vault Intel activo (modelo nomic-embed-text).
    """
    if not _is_authorized(update):
        return
    if vault_indexer is None:
        await update.message.reply_text(
            "⚠️ Vault Intel desactivado. Para activarlo:\n`ollama pull nomic-embed-text`\n"
            "y reinicia el bot.",
            parse_mode="Markdown",
        )
        return

    query = " ".join(ctx.args or []).strip()
    if not query:
        await update.message.reply_text(
            "Uso: `/search <texto>` — busca por significado, no solo palabras.",
            parse_mode="Markdown",
        )
        return

    progress = await update.message.reply_text("🔎 Buscando…")
    try:
        # Embedding del query + búsqueda coseno (todo en thread, no bloquear loop)
        def _do_search() -> list[tuple[float, str, str, list[str]]]:
            qvec = vault_indexer.embed_text(query)
            return vault_indexer.search(qvec, top_k=5)

        results = await asyncio.to_thread(_do_search)
        if not results:
            await progress.edit_text("📭 Sin resultados — el Vault está vacío o no se ha indexado.")
            return

        lines = [f"🔎 *Búsqueda:* {_md(query)}\n"]
        for i, (sim, path, title, tags) in enumerate(results, start=1):
            from pathlib import Path as _Path
            pct = int(sim * 100)
            tags_str = " ".join(f"#{t}" for t in (tags or [])[:5])
            tags_part = f"\n   {tags_str}" if tags_str else ""
            file_name = _Path(path).name
            lines.append(
                f"`{i}.` *{_md(title)}* — {pct}%\n"
                f"   📁 `{file_name}`{tags_part}"
            )
        await progress.edit_text("\n".join(lines), parse_mode="Markdown")
    except Exception as e:
        logger.exception("Fallo en /search")
        await progress.edit_text(f"❌ Error: `{_md(f'{type(e).__name__}: {e}')}`", parse_mode="Markdown")


# ─── /read <id> y /find <palabras> ──────────────────────────────────────────

async def cmd_read(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Lee una nota del Vault por su ID (frontmatter) o slug del fichero.

    Sintaxis:  /read <id|slug>
    Ej.:  /read a1b2c3d4   ·   /read trabajo-profundo
    """
    if not _is_authorized(update):
        return
    args = ctx.args or []
    if not args:
        await update.message.reply_text(
            "Uso: `/read <id>` (8 chars hex) o `/read <slug>` del fichero.",
            parse_mode="Markdown",
        )
        return

    target = args[0]
    from src.vault_intel.digest import find_note_by_id, format_note_for_preview

    summary = await asyncio.to_thread(find_note_by_id, settings.obsidian_vault_path, target)
    if summary is None:
        await update.message.reply_text(
            f"📭 No encontré ninguna nota con id/slug `{_md(target)}`.\n"
            "_Lista IDs con `/find <palabras>`._",
            parse_mode="Markdown",
        )
        return

    body, truncated = await asyncio.to_thread(format_note_for_preview, summary.path, 3300)

    obs_url = _make_obsidian_url(summary.path)
    header = (
        f"📖 *{_md(summary.title)}* {summary.emoji}\n"
        f"🆔 `{summary.note_id or '(sin id)'}`  ·  📁 `{summary.path.name}`\n"
        f"📂 [Abrir en Obsidian]({obs_url})\n"
        f"🕒 _Modificada: {summary.mtime.strftime('%Y-%m-%d %H:%M')}_\n"
        f"━━━━━━━━━━━━━━━━━━━━\n\n"
    )
    full = header + body

    if len(full) <= 4000:
        try:
            await update.message.reply_text(full, parse_mode="Markdown",
                                             disable_web_page_preview=True)
            return
        except Exception:
            await update.message.reply_text(full, disable_web_page_preview=True)
            return

    # Si se pasa de 4096 chars: cabecera + .md adjunto
    try:
        await update.message.reply_text(header, parse_mode="Markdown",
                                         disable_web_page_preview=True)
    except Exception:
        await update.message.reply_text(header, disable_web_page_preview=True)
    from io import BytesIO
    full_text = summary.path.read_text(encoding="utf-8", errors="ignore")
    buf = BytesIO(full_text.encode("utf-8"))
    buf.name = summary.path.name
    await update.message.reply_document(document=buf, filename=summary.path.name)


async def cmd_find(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Búsqueda por palabras clave (literal, AND).

    Sintaxis:  /find <una o más palabras>
    Devuelve top-8 notas con score, snippet del primer match e ID para `/read`.
    Distinto de `/search`: no usa embeddings, solo coincidencia textual.
    """
    if not _is_authorized(update):
        return
    args = ctx.args or []
    keywords = [a for a in args if a.strip()]
    if not keywords:
        await update.message.reply_text(
            "Uso: `/find <palabras>` — todas deben aparecer (AND).\n"
            "_Búsqueda semántica:_ `/search <texto>`",
            parse_mode="Markdown",
        )
        return

    progress = await update.message.reply_text(f"🔎 Buscando _{_md(' '.join(keywords))}_…", parse_mode="Markdown")
    from src.vault_intel.digest import find_by_keywords
    hits = await asyncio.to_thread(
        find_by_keywords, settings.obsidian_vault_path, keywords,
    )

    if not hits:
        await progress.edit_text(
            f"📭 Sin resultados para `{_md(' '.join(keywords))}`.",
            parse_mode="Markdown",
        )
        return

    lines = [f"🔎 *{len(hits)} resultados* para: _{_md(' '.join(keywords))}_\n"]
    for i, h in enumerate(hits, start=1):
        n = h.note
        emoji = n.emoji
        snippet = h.snippet[:140].strip()
        lines.append(
            f"`{i}.` {emoji} *{_md(n.title)}* — score *{h.score}*"
        )
        lines.append(f"   🆔 `{n.note_id or n.path.stem[:14]}`  ·  `{n.path.name}`")
        lines.append(f"   _{_md(snippet)}_")
        lines.append(f"   `/read {n.note_id or n.path.stem}`")
        lines.append("")

    text = "\n".join(lines).rstrip()
    if len(text) > 4000:
        text = text[:3900] + "\n\n_(resultados recortados)_"
    try:
        await progress.edit_text(text, parse_mode="Markdown",
                                  disable_web_page_preview=True)
    except Exception:
        await progress.edit_text(text, disable_web_page_preview=True)


# ─── Tareas ───────────────────────────────────────────────────────────────────

def _render_task_added_message(task: Task, source: str = "text") -> str:
    """Confirmación bonita tras añadir una tarea."""
    src_emoji = "🎤" if source == "voz" else "✏️"
    return (
        f"✅ *Tarea añadida* {src_emoji}\n"
        f"`{task.id}` — {_md(task.description)}\n\n"
        f"_Verla en el tablón:_ /tasks   ·   _Marcar hecha:_ `/done {task.id}`"
    )


async def _save_new_task(update: Update, description: str, source: str = "text") -> None:
    """Guarda una tarea y responde con confirmación."""
    if not description or len(description) < 2:
        await update.message.reply_text(
            "⚠️ Descripción de la tarea vacía. Ej: `task comprar café`",
            parse_mode="Markdown",
        )
        return
    task = await asyncio.to_thread(task_board.add, description)
    try:
        await update.message.reply_text(
            _render_task_added_message(task, source=source),
            parse_mode="Markdown",
        )
    except Exception:
        await update.message.reply_text(_render_task_added_message(task, source=source))


async def cmd_tasks(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Muestra el tablón con todas las tareas y sus estados."""
    if not _is_authorized(update):
        return
    # Recargamos por si se editó manualmente en Obsidian
    await asyncio.to_thread(task_board.load)
    text = task_board.render_telegram()

    # Botones inline para acciones rápidas sobre las primeras 3 pendientes
    pending = sorted(
        task_board.by_state()[TaskState.PENDING],
        key=lambda t: t.created_at, reverse=True,
    )[:3]
    in_prog = sorted(
        task_board.by_state()[TaskState.IN_PROGRESS],
        key=lambda t: t.updated_at, reverse=True,
    )[:3]
    rows: list[list[InlineKeyboardButton]] = []
    for t in in_prog:
        rows.append([
            InlineKeyboardButton(f"✅ {t.id} hecha",  callback_data=f"taskdone:{t.id}"),
            InlineKeyboardButton(f"🟦 {t.id} pausa", callback_data=f"taskpending:{t.id}"),
        ])
    for t in pending:
        rows.append([
            InlineKeyboardButton(f"🟨 {t.id} en curso", callback_data=f"taskwip:{t.id}"),
            InlineKeyboardButton(f"✅ {t.id} hecha",     callback_data=f"taskdone:{t.id}"),
        ])

    keyboard = InlineKeyboardMarkup(rows) if rows else None

    if len(text) > 4000:
        text = text[:3900] + "\n\n_(tablón recortado — mira el .md en Obsidian para todo)_"
    try:
        await update.message.reply_text(text, parse_mode="Markdown", reply_markup=keyboard)
    except Exception:
        await update.message.reply_text(text, reply_markup=keyboard)


async def _change_task_state_cmd(
    update: Update, ctx: ContextTypes.DEFAULT_TYPE,
    new_state: TaskState, action_name: str,
) -> None:
    """Helper común para /done /wip /todo."""
    if not _is_authorized(update):
        return
    args = ctx.args or []
    if not args:
        await update.message.reply_text(
            f"Uso: `/{action_name} <id_tarea>` (ej. `/{action_name} T003`)",
            parse_mode="Markdown",
        )
        return
    target_id = args[0]
    task = await asyncio.to_thread(task_board.update_state, target_id, new_state)
    if not task:
        await update.message.reply_text(
            f"⚠️ No encontré la tarea `{_md(target_id)}`. Lista con /tasks.",
            parse_mode="Markdown",
        )
        return
    emoji = TASK_STATE_EMOJI[new_state]
    await update.message.reply_text(
        f"{emoji} Tarea `{task.id}` → *{new_state.value}*\n_{_md(task.description)}_",
        parse_mode="Markdown",
    )


async def cmd_done(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Marca una tarea como completada. Uso: /done T003"""
    await _change_task_state_cmd(update, ctx, TaskState.DONE, "done")


async def cmd_wip(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Marca una tarea como en progreso. Uso: /wip T003"""
    await _change_task_state_cmd(update, ctx, TaskState.IN_PROGRESS, "wip")


async def cmd_todo(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Devuelve una tarea al estado pendiente. Uso: /todo T003"""
    await _change_task_state_cmd(update, ctx, TaskState.PENDING, "todo")


async def cmd_del(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Borra una tarea del tablón. Uso: /del T003"""
    if not _is_authorized(update):
        return
    args = ctx.args or []
    if not args:
        await update.message.reply_text(
            "Uso: `/del <id_tarea>` (ej. `/del T003`)", parse_mode="Markdown",
        )
        return
    target_id = args[0]
    # Resolvemos el id real (admite prefijos como T1 → T001)
    resolved = await asyncio.to_thread(task_board.get, target_id)
    if not resolved:
        await update.message.reply_text(
            f"⚠️ No encontré la tarea `{_md(target_id)}`.", parse_mode="Markdown",
        )
        return
    ok = await asyncio.to_thread(task_board.delete, resolved.id)
    if ok:
        await update.message.reply_text(
            f"🗑 Tarea `{resolved.id}` borrada\n_~{_md(resolved.description)}~_",
            parse_mode="Markdown",
        )
    else:
        await update.message.reply_text(f"⚠️ No se pudo borrar `{resolved.id}`.")


async def on_task_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Botones inline del tablón para cambios rápidos de estado."""
    if not _is_authorized(update):
        return
    query = update.callback_query
    await query.answer()
    action, _, task_id = query.data.partition(":")

    state_map = {
        "taskdone":    TaskState.DONE,
        "taskwip":     TaskState.IN_PROGRESS,
        "taskpending": TaskState.PENDING,
    }
    new_state = state_map.get(action)
    if not new_state:
        return
    task = await asyncio.to_thread(task_board.update_state, task_id, new_state)
    if not task:
        await query.edit_message_text("⚠️ Tarea no encontrada.")
        return
    # Reemplazamos el mensaje por la versión actualizada del tablón
    await asyncio.to_thread(task_board.load)
    text = task_board.render_telegram()
    pending = sorted(task_board.by_state()[TaskState.PENDING],
                     key=lambda t: t.created_at, reverse=True)[:3]
    in_prog = sorted(task_board.by_state()[TaskState.IN_PROGRESS],
                     key=lambda t: t.updated_at, reverse=True)[:3]
    rows: list[list[InlineKeyboardButton]] = []
    for t in in_prog:
        rows.append([
            InlineKeyboardButton(f"✅ {t.id} hecha",  callback_data=f"taskdone:{t.id}"),
            InlineKeyboardButton(f"🟦 {t.id} pausa", callback_data=f"taskpending:{t.id}"),
        ])
    for t in pending:
        rows.append([
            InlineKeyboardButton(f"🟨 {t.id} en curso", callback_data=f"taskwip:{t.id}"),
            InlineKeyboardButton(f"✅ {t.id} hecha",     callback_data=f"taskdone:{t.id}"),
        ])
    keyboard = InlineKeyboardMarkup(rows) if rows else None
    if len(text) > 4000:
        text = text[:3900] + "\n\n_(tablón recortado)_"
    try:
        await query.edit_message_text(text, parse_mode="Markdown", reply_markup=keyboard)
    except Exception:
        # Si la edición falla (mismo contenido, parser strict, etc.) lo ignoramos
        pass


async def cmd_queue(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Muestra el job que se está procesando y los pendientes."""
    if not _is_authorized(update):
        return
    current = job_queue.current_job()
    pending = job_queue.pending_jobs()

    lines: list[str] = ["📋 *Estado de la cola*\n"]
    if current:
        lines.append(
            f"▶️ *Procesando ahora:* `{current.short_id()}` — {_md(current.title[:50])} "
            f"(⏱ {current.elapsed:.0f}s)"
        )
    else:
        lines.append("▶️ _Nada en proceso_")

    if pending:
        lines.append("\n📥 *En cola:*")
        for i, j in enumerate(pending, start=1):
            pos = i + (1 if current else 0)
            lines.append(f"  `{pos}.` `{j.short_id()}` — {_md(j.title[:50])}")
    else:
        lines.append("\n📥 _Cola vacía_")

    lines.append("\n_Cancelar uno:_ `/cancel <id>`")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def cmd_cancel(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Cancela un job pendiente. Sintaxis: /cancel <short_id>"""
    if not _is_authorized(update):
        return
    args = ctx.args or []
    if not args:
        await update.message.reply_text(
            "Uso: `/cancel <id_corto>` — ver IDs con /queue",
            parse_mode="Markdown",
        )
        return

    target = args[0].strip().lower()
    # Permitimos prefijos: si dan "abc1" buscamos el job cuyo id empiece por eso
    matches = [
        j for j in job_queue.pending_jobs()
        if j.id.startswith(target) or j.short_id() == target
    ]
    if not matches:
        await update.message.reply_text(f"⚠️ No encontré job pendiente con id `{target}`.", parse_mode="Markdown")
        return
    if len(matches) > 1:
        await update.message.reply_text(
            f"⚠️ Prefijo ambiguo (`{target}` coincide con {len(matches)} jobs).",
            parse_mode="Markdown",
        )
        return

    job = matches[0]
    if job_queue.cancel(job.id):
        # Avisamos en el mensaje de progreso original también
        try:
            await ctx.bot.edit_message_text(
                chat_id=job.chat_id, message_id=job.progress_msg_id,
                text=f"❌ Cancelado por el usuario (job `{job.short_id()}`).",
                parse_mode="Markdown",
            )
        except Exception:
            pass
        await update.message.reply_text(f"✅ Job `{job.short_id()}` cancelado.", parse_mode="Markdown")
    else:
        await update.message.reply_text(f"⚠️ No se pudo cancelar `{job.short_id()}`.", parse_mode="Markdown")


async def on_qcancel_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Botón inline 'Cancelar' del mensaje 'En cola'."""
    if not _is_authorized(update):
        return
    query = update.callback_query
    await query.answer()
    _, _, job_id = query.data.partition(":")
    job = job_queue.get(job_id)
    if not job:
        await query.edit_message_text("⚠️ Job no encontrado (quizá ya empezó).")
        return
    if job.status != JobStatus.PENDING:
        await query.edit_message_text(
            f"⚠️ El job `{job.short_id()}` ya está {job.status.value}, no se puede cancelar.",
            parse_mode="Markdown",
        )
        return
    if job_queue.cancel(job.id):
        await query.edit_message_text(
            f"❌ Cancelado por el usuario (job `{job.short_id()}`).",
            parse_mode="Markdown",
        )


async def on_readnote_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Botón inline para leer una nota después de su creación/actualización."""
    if not _is_authorized(update):
        return
    query = update.callback_query
    await query.answer()
    _, _, note_id = query.data.partition(":")

    from src.vault_intel.digest import find_note_by_id, format_note_for_preview

    summary = await asyncio.to_thread(
        find_note_by_id, settings.obsidian_vault_path, note_id
    )
    if summary is None:
        await query.edit_message_text(f"⚠️ No encontré nota con id `{_md(note_id)}`")
        return

    body, truncated = await asyncio.to_thread(
        format_note_for_preview, summary.path, 3300
    )

    obs_url = _make_obsidian_url(summary.path)
    header = (
        f"📖 *{_md(summary.title)}* {summary.emoji}\n"
        f"🆔 `{summary.note_id or '(sin id)'}`  ·  📁 `{summary.path.name}`\n"
        f"📂 [Abrir en Obsidian]({obs_url})\n"
        f"🕒 _Modificada: {summary.mtime.strftime('%Y-%m-%d %H:%M')}_\n"
        f"━━━━━━━━━━━━━━━━━━━━\n\n"
    )
    full = header + body

    if len(full) <= 4000:
        try:
            await query.edit_message_text(full, parse_mode="Markdown",
                                          disable_web_page_preview=True)
        except Exception:
            await query.edit_message_text(full, disable_web_page_preview=True)
    else:
        # Si es demasiado largo: cabecera + archivo adjunto
        await query.edit_message_text(header, parse_mode="Markdown",
                                      disable_web_page_preview=True)
        from io import BytesIO
        full_text = summary.path.read_text(encoding="utf-8", errors="ignore")
        buf = BytesIO(full_text.encode("utf-8"))
        buf.name = summary.path.name
        await query.message.reply_document(document=buf, filename=summary.path.name)


# ─── Núcleo: extraer + (eventualmente confirmar) + sintetizar + escribir ─────
async def _run_pipeline(update: Update, ctx: ContextTypes.DEFAULT_TYPE, payload: IngestionPayload) -> None:
    msg = update.effective_message
    progress = await msg.reply_text("⏳ Extrayendo contenido…")

    try:
        # 1. Extracción
        extractor = ExtractorFactory.select(payload)

        # Feedback más útil para imágenes — el modelo de visión la primera vez
        # tarda en cargar (~30-60s). Sin este aviso parece que el bot está colgado.
        is_image = (
            payload.kind is SourceKind.FILE
            and Path(payload.raw).suffix.lower() in {
                ".jpg", ".jpeg", ".png", ".gif",
                ".bmp", ".tiff", ".tif", ".webp",
            }
        )
        if is_image:
            await progress.edit_text(
                "🖼 *Analizando imagen…*\n"
                f"_(visión: `{settings.vision_model}` + OCR Tesseract; "
                "la primera vez puede tardar ~60s mientras carga el modelo)_",
                parse_mode="Markdown",
            )
        else:
            await progress.edit_text(f"🔎 Extractor: *{extractor.name}*…", parse_mode="Markdown")

        content = await extractor.extract(payload)
        if not content.text or len(content.text.strip()) < 30:
            error_reason = content.extra.get("extraction_error") if content.extra else None
            user_msg = f"⚠️ {error_reason}" if error_reason else "⚠️ No se pudo extraer contenido útil."
            await progress.edit_text(user_msg)
            return

        # 1.5 ¿Era una nota de voz que dice "tarea …"? → añadir al tablón
        # (intercepta ANTES de ningún preview o cola de síntesis)
        if payload.metadata.get("telegram_voice"):
            if (desc := extract_task_description(content.text)):
                task = await asyncio.to_thread(task_board.add, desc)
                await progress.edit_text(
                    _render_task_added_message(task, source="voz"),
                    parse_mode="Markdown",
                )
                return

        # 2. Si es imagen → mostrar muestra de lo detectado y preguntar antes de seguir
        if content.extra.get("doc_kind") == "image":
            await _ask_image_confirmation(ctx, msg, content, progress)
            return

        # 2.b Si es un concepto con metadata estructurada → preview + confirmación
        if content.extra.get("concept_parsed"):
            await _ask_concept_confirmation(ctx, msg, content, progress)
            return

        # 3. Vault Intel: detección de duplicados + contexto para el LLM
        vault_ctx = None
        if vault_searcher is not None:
            await progress.edit_text("🧭 Buscando en el Vault…")
            vault_ctx = await asyncio.to_thread(
                vault_searcher.analyze, content.title_hint, content.text[:500]
            )
            if vault_ctx.duplicate:
                await _ask_dup_confirmation(
                    ctx, msg, content, make_plan(content.text), vault_ctx, progress
                )
                return

        # 4. Plan: ¿chunking? ¿confirmar?
        plan = make_plan(content.text)
        chars = len(content.text)
        await progress.edit_text(
            f"📊 Extraído: {chars:,} chars → método: `{plan.method}` "
            f"({plan.n_chunks} chunk{'s' if plan.n_chunks != 1 else ''})",
            parse_mode="Markdown",
        )

        # 5. ¿Pedir confirmación? Para contenido grande o con chunking.
        needs_confirmation = (
            chars >= settings.confirm_threshold_chars or plan.method != "single"
        )
        if needs_confirmation:
            await _ask_confirmation(ctx, msg, content, plan, progress_msg=progress, vault_ctx=vault_ctx)
            return

        # 6. Procesar directamente
        await _enqueue_synthesis(ctx, progress, content, plan, vault_ctx)

    except Exception as e:
        logger.exception("Fallo procesando mensaje")
        await progress.edit_text(f"❌ Error: `{_md(f'{type(e).__name__}: {e}')}`", parse_mode="Markdown")


def _extract_image_preview(text: str) -> dict:
    """Saca un preview compacto del texto que produce el extractor de imágenes.

    El texto entrante tiene esta forma (tras `_parse_image`):
        ## Caption del usuario        ← opcional
        ...
        ---
        ## Análisis visual
        ## Descripción general
        ...
        ## Elementos identificados
        - ...
        ## Texto visible
        ...
        ## Contexto interpretado
        ...
        ## Etiquetas conceptuales
        ...
        ---
        ## Texto extraído (OCR Tesseract)
        ...

    Devuelve un dict con: descripcion, elementos (list), texto_visible, etiquetas, ocr.
    """
    import re as _re
    out = {
        "descripcion": "", "elementos": [], "texto_visible": "",
        "etiquetas": "", "ocr": "",
    }

    def _section(header: str, src: str) -> str:
        """Devuelve el contenido entre `## header` y la siguiente `## ` o `---`."""
        pat = _re.compile(
            rf"##\s+{_re.escape(header)}\s*\n(.+?)(?=\n##\s|\n---\s*\n|\Z)",
            _re.DOTALL,
        )
        m = pat.search(src)
        return m.group(1).strip() if m else ""

    out["descripcion"]     = _section("Descripción general", text)
    elementos_raw          = _section("Elementos identificados", text)
    out["texto_visible"]   = _section("Texto visible", text)
    out["etiquetas"]       = _section("Etiquetas conceptuales", text)
    out["ocr"]             = _section("Texto extraído (OCR Tesseract)", text)

    # Parsear bullets de elementos
    if elementos_raw:
        out["elementos"] = [
            ln.lstrip("-*• ").strip()
            for ln in elementos_raw.splitlines()
            if ln.strip().startswith(("-", "*", "•"))
        ]
    return out


async def _ask_image_confirmation(ctx, original_msg, content, progress_msg) -> None:
    """Muestra un preview de lo detectado en la imagen y pide confirmación."""
    preview = _extract_image_preview(content.text)

    # Construir el mensaje con secciones cortas
    parts: list[str] = ["🖼 *He analizado la imagen*\n"]

    if preview["descripcion"]:
        desc = preview["descripcion"]
        if len(desc) > 350:
            desc = desc[:350].rstrip() + "…"
        parts.append(f"*Descripción:*\n{_md(desc)}\n")

    if preview["elementos"]:
        elems = preview["elementos"][:6]
        elems_str = "\n".join(f"  • {_md(e)}" for e in elems)
        more = f"\n  _… y {len(preview['elementos']) - 6} más_" if len(preview["elementos"]) > 6 else ""
        parts.append(f"*Elementos detectados:*\n{elems_str}{more}\n")

    # Texto visible: priorizar OCR (es literal); si no hay, usa el del modelo
    text_to_show = preview["ocr"] or preview["texto_visible"]
    text_to_show = (text_to_show or "").strip()
    if text_to_show and text_to_show != "(Ninguno)":
        snippet = text_to_show
        truncated = False
        if len(snippet) > 400:
            snippet = snippet[:400].rstrip() + "…"
            truncated = True
        # Bloque de cita
        quoted = "\n".join(f"> {_md(line)}" for line in snippet.splitlines() if line.strip())
        parts.append(f"*Texto leído:*\n{quoted}")
        if truncated:
            parts.append("_(texto recortado)_\n")
        else:
            parts.append("")

    if preview["etiquetas"]:
        tags_clean = preview["etiquetas"].replace("\n", " ").strip()
        if len(tags_clean) > 200:
            tags_clean = tags_clean[:200] + "…"
        parts.append(f"*Etiquetas sugeridas:* {_md(tags_clean)}")

    parts.append("\n¿Genero la nota completa con análisis profundo?")

    msg_text = "\n".join(parts)

    # Guardamos el contenido para retomarlo con un id corto
    job_id = _new_job_id()
    ctx.application.bot_data.setdefault("image_jobs", {})
    ctx.application.bot_data["image_jobs"][job_id] = {"content": content}

    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ Sí, generar nota", callback_data=f"imgok:{job_id}"),
            InlineKeyboardButton("❌ Descartar", callback_data=f"imgno:{job_id}"),
        ],
        [InlineKeyboardButton("📋 Ver análisis completo", callback_data=f"imgfull:{job_id}")],
    ])

    # Si el preview es demasiado largo, recortar y mandar
    if len(msg_text) > 4000:
        msg_text = msg_text[:3900] + "\n\n_(preview recortado)_"
    try:
        await progress_msg.edit_text(msg_text, parse_mode="Markdown", reply_markup=keyboard)
    except Exception:
        # Fallback sin formato si algún carácter rompe el parser
        await progress_msg.edit_text(msg_text, reply_markup=keyboard)


async def on_image_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Maneja los botones del preview de imagen."""
    if not _is_authorized(update):
        return
    query = update.callback_query
    await query.answer()
    action, _, job_id = query.data.partition(":")

    bucket = ctx.application.bot_data.get("image_jobs", {})

    if action == "imgfull":
        # No popea — el usuario quiere seguir decidiendo después de leer
        item = bucket.get(job_id)
        if not item:
            await query.message.reply_text("⚠️ Sesión expirada.")
            return
        full = item["content"].text
        # Mandamos el análisis completo como mensaje extra
        if len(full) <= 4000:
            try:
                await query.message.reply_text(full, parse_mode="Markdown")
            except Exception:
                await query.message.reply_text(full)
        else:
            from io import BytesIO
            buf = BytesIO(full.encode("utf-8"))
            buf.name = "analisis_imagen.md"
            await query.message.reply_document(
                document=buf, filename=buf.name,
                caption="📋 Análisis completo de la imagen",
            )
        return

    item = bucket.pop(job_id, None)
    if not item:
        await query.edit_message_text("⚠️ Sesión expirada. Vuelve a mandar la imagen.")
        return

    if action == "imgno":
        await query.edit_message_text("❌ Descartado. La imagen no se ha guardado.")
        return

    # imgok → continuamos el pipeline normal: vault intel → plan → cola
    content = item["content"]
    progress = query.message

    # Vault Intel
    vault_ctx = None
    if vault_searcher is not None:
        await progress.edit_text("🧭 Buscando en el Vault…")
        vault_ctx = await asyncio.to_thread(
            vault_searcher.analyze, content.title_hint, content.text[:500]
        )
        if vault_ctx.duplicate:
            await _ask_dup_confirmation(
                ctx, query.message, content, make_plan(content.text), vault_ctx, progress
            )
            return

    plan = make_plan(content.text)
    chars = len(content.text)
    needs_confirmation = (
        chars >= settings.confirm_threshold_chars or plan.method != "single"
    )
    if needs_confirmation:
        await _ask_confirmation(ctx, query.message, content, plan, progress, vault_ctx)
    else:
        await _enqueue_synthesis(ctx, progress, content, plan, vault_ctx)


async def _ask_concept_confirmation(ctx, original_msg, content, progress_msg) -> None:
    """Muestra metadata del concepto + sample de fuentes encontradas y pide confirmación.

    Solo aplica cuando el input fue estructurado (parsed por concept_parser).
    """
    extra = content.extra or {}

    # Iconos por tipo
    KIND_EMOJI = {
        "book": "📚", "movie": "🎬", "video": "📺", "podcast": "🎙",
        "paper": "🎓", "article": "📰", "tv": "📺", "song": "🎵",
        "album": "💿", "game": "🎮", "person": "👤", "place": "📍",
        "concept": "💡",
    }

    parts: list[str] = ["💡 *Concepto estructurado detectado*\n"]

    kind = extra.get("doc_kind")
    if kind:
        parts.append(f"*Tipo:* {KIND_EMOJI.get(kind, '💡')} `{kind}`")
    title = extra.get("book_title") or extra.get("doc_title")
    if title:
        parts.append(f"*Título:* {_md(title)}")
    author = extra.get("author") or (
        extra.get("authors")[0] if extra.get("authors") else None
    )
    if author:
        parts.append(f"*Autor:* {_md(author)}")

    # Búsqueda usada
    parts.append(f"*Búsqueda DDG:* `{_md(content.title_hint)}`")

    # Fuentes
    sources = extra.get("sources") or []
    if sources:
        parts.append(f"\n*Fuentes recuperadas* ({len(sources)}):")
        for s in sources[:5]:
            parts.append(f"  • {_md(s[:80])}")
    else:
        parts.append("\n_⚠️ No se encontraron fuentes — el LLM trabajará con su conocimiento general._")

    # Muestra del cuerpo (saltándose la cabecera de metadata factual)
    body = content.text
    # Quitar el "# Investigación estructurada" + sus campos
    import re as _re
    body = _re.sub(
        r"^#\s+(Investigación estructurada|Concepto investigado:.*?)\n(\*\*[^\n]+\n)*",
        "", body, count=1, flags=_re.MULTILINE,
    ).strip()

    if body:
        sample = body[:600]
        if len(body) > 600:
            sample = sample.rstrip() + "…"
        parts.append(f"\n*Muestra del contenido recuperado:*\n```\n{sample}\n```")

    parts.append("\n¿Genero la nota completa?")

    msg_text = "\n".join(parts)

    # Guardamos para retomarlo en el callback
    job_id = _new_job_id()
    ctx.application.bot_data.setdefault("concept_jobs", {})
    ctx.application.bot_data["concept_jobs"][job_id] = {"content": content}

    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ Sí, generar nota", callback_data=f"cptok:{job_id}"),
            InlineKeyboardButton("❌ Descartar",       callback_data=f"cptno:{job_id}"),
        ],
        [InlineKeyboardButton("📋 Ver fuentes completas", callback_data=f"cptfull:{job_id}")],
    ])

    if len(msg_text) > 4000:
        msg_text = msg_text[:3900] + "\n\n_(preview recortado)_"
    try:
        await progress_msg.edit_text(msg_text, parse_mode="Markdown", reply_markup=keyboard)
    except Exception:
        await progress_msg.edit_text(msg_text, reply_markup=keyboard)


async def on_concept_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Maneja los botones del preview del concepto estructurado."""
    if not _is_authorized(update):
        return
    query = update.callback_query
    await query.answer()
    action, _, job_id = query.data.partition(":")

    bucket = ctx.application.bot_data.get("concept_jobs", {})

    if action == "cptfull":
        # No popea — solo muestra; el usuario sigue decidiendo después
        item = bucket.get(job_id)
        if not item:
            await query.message.reply_text("⚠️ Sesión expirada.")
            return
        full = item["content"].text
        if len(full) <= 4000:
            try:
                await query.message.reply_text(full, parse_mode="Markdown")
            except Exception:
                await query.message.reply_text(full)
        else:
            from io import BytesIO
            buf = BytesIO(full.encode("utf-8"))
            buf.name = "fuentes_concepto.md"
            await query.message.reply_document(
                document=buf, filename=buf.name,
                caption="📋 Fuentes completas recuperadas",
            )
        return

    item = bucket.pop(job_id, None)
    if not item:
        await query.edit_message_text("⚠️ Sesión expirada. Vuelve a mandar el concepto.")
        return

    if action == "cptno":
        await query.edit_message_text("❌ Descartado. No se ha generado nota.")
        return

    # cptok → continuar pipeline normal: vault intel → plan → cola
    content = item["content"]
    progress = query.message

    vault_ctx = None
    if vault_searcher is not None:
        await progress.edit_text("🧭 Buscando en el Vault…")
        vault_ctx = await asyncio.to_thread(
            vault_searcher.analyze, content.title_hint, content.text[:500]
        )
        if vault_ctx.duplicate:
            await _ask_dup_confirmation(
                ctx, query.message, content, make_plan(content.text), vault_ctx, progress
            )
            return

    plan = make_plan(content.text)
    chars = len(content.text)
    needs_confirmation = (
        chars >= settings.confirm_threshold_chars or plan.method != "single"
    )
    if needs_confirmation:
        await _ask_confirmation(ctx, query.message, content, plan, progress, vault_ctx)
    else:
        await _enqueue_synthesis(ctx, progress, content, plan, vault_ctx)


async def _ask_confirmation(
    ctx, original_msg, content, plan, progress_msg, vault_ctx=None,
    synthesis_mode: str = "create", target_paths: Optional[list[str]] = None,
):
    """Manda botones de confirmación. Guarda el contenido extraído en bot_data."""
    job_id = _new_job_id()
    ctx.application.bot_data.setdefault("ready_jobs", {})
    ctx.application.bot_data["ready_jobs"][job_id] = {
        "content": content, "plan": plan, "vault_ctx": vault_ctx,
        "synthesis_mode": synthesis_mode, "target_paths": target_paths,
    }

    minutes = plan.estimated_minutes
    mode_hint = {
        "overwrite": " _(sobreescribiendo la nota existente)_",
        "update":    " _(añadiendo sección a nota(s) existente(s))_",
    }.get(synthesis_mode, "")
    text = (
        f"⏱ Este contenido es grande:\n\n"
        f"• Caracteres: *{plan.total_chars:,}*\n"
        f"• Método: `{plan.method}` en *{plan.n_chunks}* fragmento(s)\n"
        f"• Tiempo estimado: *~{minutes:.1f} min*{mode_hint}\n\n"
        f"¿Continúo?"
    )
    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("✅ Sí, procesar", callback_data=f"go:{job_id}"),
        InlineKeyboardButton("❌ Cancelar", callback_data=f"no:{job_id}"),
    ]])
    await progress_msg.edit_text(text, parse_mode="Markdown", reply_markup=keyboard)


async def _ask_dup_confirmation(ctx, msg, content, plan, vault_ctx, progress_msg):
    """Pregunta al usuario qué hacer cuando se detecta una nota muy similar.

    Opciones:
      ✍️ Sobreescribir — reemplaza la nota existente con la nueva síntesis
      📄 Crear nueva   — crea una nota nueva (comportamiento por defecto)
      🔄 Actualizar    — añade una sección a la nota existente (y si hay varias, a todas)
      ❌ Cancelar      — descarta sin hacer nada
    """
    dup = vault_ctx.duplicate
    related = vault_ctx.related or []
    # Todas las notas similares: la principal + las relacionadas por encima del umbral bajo
    all_similar = [dup] + related

    job_id = _new_job_id()
    ctx.application.bot_data.setdefault("dup_jobs", {})
    ctx.application.bot_data["dup_jobs"][job_id] = {
        "content": content, "plan": plan, "vault_ctx": vault_ctx,
    }

    sim_pct = int(dup.similarity * 100)
    lines = [f"🔍 *Notas similares encontradas en el Vault:*\n"]
    lines.append(f"  📝 *{_md(dup.title)}* — {sim_pct}%")
    for r in related[:3]:
        pct = int(r.similarity * 100)
        lines.append(f"  └ *{_md(r.title)}* — {pct}%")

    lines.append("\n*¿Qué hago con el nuevo contenido?*")
    if len(all_similar) > 1:
        lines.append(
            f"_('Actualizar' añadirá una sección a las {len(all_similar)} notas similares)_"
        )

    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✍️ Sobreescribir",        callback_data=f"dup_overwrite:{job_id}"),
            InlineKeyboardButton("📄 Crear nueva",           callback_data=f"dup_new:{job_id}"),
        ],
        [
            InlineKeyboardButton("🔄 Actualizar existente(s)", callback_data=f"dup_update:{job_id}"),
            InlineKeyboardButton("❌ Cancelar",                callback_data=f"dup_cancel:{job_id}"),
        ],
    ])
    await progress_msg.edit_text(
        "\n".join(lines),
        parse_mode="Markdown",
        reply_markup=keyboard,
    )


async def on_dup_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Maneja la respuesta del usuario al aviso de duplicado.

    Acciones reconocidas:
      dup_new      — crea nota nueva (pipeline normal)
      dup_overwrite — sobreescribe la nota existente con la nueva síntesis
      dup_update   — añade sección de actualización a todas las notas similares
      dup_cancel   — descarta
    """
    if not _is_authorized(update):
        return
    query = update.callback_query
    await query.answer()
    action, _, job_id = query.data.partition(":")

    bucket = ctx.application.bot_data.get("dup_jobs", {})
    item = bucket.pop(job_id, None)
    if not item:
        await query.edit_message_text("⚠️ Sesión expirada. Vuelve a mandar el contenido.")
        return

    if action == "dup_cancel":
        await query.edit_message_text("❌ Procesamiento cancelado.")
        return

    content = item["content"]
    plan = item["plan"]
    vault_ctx = item.get("vault_ctx")
    progress = query.message

    if action == "dup_overwrite":
        # Sobreescribir la nota principal más similar
        dup = vault_ctx.duplicate if vault_ctx else None
        target_paths = [dup.path] if dup else None
        chars = len(content.text)
        needs_confirmation = (
            chars >= settings.confirm_threshold_chars or plan.method != "single"
        )
        if needs_confirmation:
            await _ask_confirmation(
                ctx, progress, content, plan, progress, vault_ctx,
                synthesis_mode="overwrite", target_paths=target_paths,
            )
        else:
            await _enqueue_synthesis(
                ctx, progress, content, plan, vault_ctx,
                synthesis_mode="overwrite", target_paths=target_paths,
            )
        return

    if action == "dup_update":
        # Actualizar todas las notas similares (duplicado + relacionadas)
        dup = vault_ctx.duplicate if vault_ctx else None
        related = (vault_ctx.related or []) if vault_ctx else []
        all_similar = ([dup] if dup else []) + related
        target_paths = [n.path for n in all_similar] if all_similar else None
        chars = len(content.text)
        needs_confirmation = (
            chars >= settings.confirm_threshold_chars or plan.method != "single"
        )
        if needs_confirmation:
            await _ask_confirmation(
                ctx, progress, content, plan, progress, vault_ctx,
                synthesis_mode="update", target_paths=target_paths,
            )
        else:
            await _enqueue_synthesis(
                ctx, progress, content, plan, vault_ctx,
                synthesis_mode="update", target_paths=target_paths,
            )
        return

    # "dup_new" → pipeline normal ignorando el duplicado
    chars = len(content.text)
    needs_confirmation = (
        chars >= settings.confirm_threshold_chars or plan.method != "single"
    )
    if needs_confirmation:
        await _ask_confirmation(ctx, progress, content, plan, progress, vault_ctx)
    else:
        await _enqueue_synthesis(ctx, progress, content, plan, vault_ctx)


async def on_confirm_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update):
        return
    query = update.callback_query
    await query.answer()
    action, _, job_id = query.data.partition(":")

    bucket = ctx.application.bot_data.get("ready_jobs", {})
    item = bucket.pop(job_id, None)
    if not item:
        await query.edit_message_text("⚠️ Sesión expirada. Vuelve a mandar el contenido.")
        return

    if action == "no":
        await query.edit_message_text("❌ Procesamiento cancelado.")
        return

    content = item["content"]
    plan = item["plan"]
    vault_ctx = item.get("vault_ctx")
    synthesis_mode = item.get("synthesis_mode", "create")
    target_paths = item.get("target_paths")
    await _enqueue_synthesis(
        ctx, query.message, content, plan, vault_ctx,
        synthesis_mode=synthesis_mode, target_paths=target_paths,
    )


async def _enqueue_synthesis(
    ctx, progress_msg, content, plan, vault_ctx=None,
    synthesis_mode: str = "create",
    target_paths: Optional[list[str]] = None,
):
    """Encola la síntesis. El worker la procesará cuando le toque turno.

    `synthesis_mode`:
      "create"    — crea nota nueva (por defecto)
      "overwrite" — sobreescribe la nota en `target_paths[0]`
      "update"    — añade sección de actualización a todas las notas en `target_paths`

    Si la cola está vacía y no hay nada corriendo, el job arrancará en milisegundos
    y la UI saltará directamente a "🧠 Sintetizando…". Si hay otros por delante,
    la UI muestra la posición y un botón para cancelar.
    """
    user = progress_msg.chat  # en private chats coincide con el user
    title = (content.title_hint or "Nota").strip()[:80]
    job = Job(
        id=_new_queue_job_id(),
        title=f"{title} ({content.source_type})",
        chat_id=progress_msg.chat_id,
        user_id=user.id if user else 0,
        progress_msg_id=progress_msg.message_id,
        payload={
            "content": content, "plan": plan, "vault_ctx": vault_ctx,
            "synthesis_mode": synthesis_mode, "target_paths": target_paths,
        },
    )
    pos = await job_queue.enqueue(job)

    # Si pos==1 y no hay running, arrancará casi al instante → no spameamos UI;
    # el handler editará el mensaje cuando empiece.
    if pos == 1 and job_queue.current_job() is None:
        await progress_msg.edit_text(
            f"🟢 Encolado y empezando enseguida…\n_Job_ `{job.short_id()}`",
            parse_mode="Markdown",
        )
        return

    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("❌ Cancelar", callback_data=f"qcancel:{job.id}"),
    ]])
    pending = pos - 1  # cuántos van por delante (incluido el que está corriendo)
    await progress_msg.edit_text(
        f"📥 *En cola*\n"
        f"• Posición: *{pos}* (hay {pending} por delante)\n"
        f"• Job: `{job.short_id()}` — {_md(title[:40])}\n\n"
        f"Recibirás aviso cuando empiece. Usa /queue para ver el estado.",
        parse_mode="Markdown",
        reply_markup=keyboard,
    )


async def _process_synthesis_job(bot, job: Job) -> None:
    """Handler ejecutado por el worker para cada job. Tiene acceso al bot
    para actualizar el mensaje de progreso del usuario.

    Modos:
      "create"    — nota nueva (por defecto)
      "overwrite" — sobreescribe la nota existente en target_paths[0]
      "update"    — añade sección de actualización a cada nota en target_paths
    """
    payload = job.payload
    content = payload["content"]
    plan = payload["plan"]
    vault_ctx = payload.get("vault_ctx")
    synthesis_mode: str = payload.get("synthesis_mode", "create")
    target_paths: list[str] | None = payload.get("target_paths")
    chat_id = job.chat_id
    msg_id = job.progress_msg_id

    async def _edit(text: str, **kw):
        try:
            await bot.edit_message_text(
                chat_id=chat_id, message_id=msg_id, text=text,
                parse_mode="Markdown", **kw,
            )
        except Exception as e:  # mensaje borrado, demasiado viejo, etc.
            logger.debug("edit_message_text falló (%s) — sigo", e)

    # Indicador de modo en el mensaje de progreso
    mode_label = {
        "overwrite": "sobreescribiendo",
        "update":    "actualizando",
    }.get(synthesis_mode, "")
    mode_suffix = f" _({mode_label})_" if mode_label else ""

    try:
        await _edit(
            f"🧠 *Sintetizando* con `{llm.describe()}`\n"
            f"(`{plan.method}`, {plan.n_chunks} chunk(s)) — job `{job.short_id()}`{mode_suffix}"
        )
        result = await llm.synthesize(content, plan=plan, vault_ctx=vault_ctx)

        related_titles = (
            [n.title for n in vault_ctx.related] if vault_ctx and vault_ctx.related else None
        )
        title = llm.extract_h1_title(result.markdown)
        flag = "⚠️ Revisar" if result.needs_review else "✅"
        extra = ""
        if result.needs_review:
            bullets = "\n".join(f"  • {_md(p)}" for p in result.validation_problems[:3])
            extra = f"\n\n_Problemas de validación:_\n{bullets}"
        elapsed = job.elapsed

        # ── Modo: sobreescribir ──────────────────────────────────────────
        if synthesis_mode == "overwrite" and target_paths:
            target_path = Path(target_paths[0])
            path, note_id = writer.overwrite(
                existing_path=target_path,
                title=title,
                body_md=result.markdown,
                content=content,
                language=result.language,
                needs_review=result.needs_review,
                review_notes=result.validation_problems if result.needs_review else None,
                related=related_titles,
            )
            if vault_indexer is not None:
                await asyncio.to_thread(vault_indexer.index_note, path)
            obs_url = _make_obsidian_url(path)
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("📖 Leer nota", callback_data=f"readnote:{note_id}")]
            ])
            await _edit(
                f"{flag} *{_md(title)}* ♻️ _sobreescrita_\n"
                f"🆔 `{note_id}`\n"
                f"📂 [Abrir en Obsidian]({obs_url})\n"
                f"📁 `{path.name}`\n"
                f"🌐 Idioma: `{result.language}` · 📦 Chunks: {result.chunks_used} · "
                f"⏱ {elapsed:.0f}s{extra}",
                reply_markup=keyboard
            )
            return

        # ── Modo: actualizar (append) ────────────────────────────────────
        if synthesis_mode == "update" and target_paths:
            # Sección a añadir: cuerpo de la síntesis sin el H1 (que ya tiene la nota)
            import re as _re
            update_body = _re.sub(r"^#\s+.+\n", "", result.markdown, count=1).strip()
            updated_paths: list[Path] = []
            first_note_id = ""
            for tp in target_paths:
                p = Path(tp)
                up, nid = writer.append_update(p, update_body)
                if vault_indexer is not None:
                    await asyncio.to_thread(vault_indexer.index_note, up)
                updated_paths.append(up)
                if not first_note_id:
                    first_note_id = nid

            n = len(updated_paths)
            names = ", ".join(f"`{p.name}`" for p in updated_paths[:3])
            obs_url = _make_obsidian_url(updated_paths[0])
            keyboard = None
            if first_note_id:
                keyboard = InlineKeyboardMarkup([
                    [InlineKeyboardButton("📖 Leer nota", callback_data=f"readnote:{first_note_id}")]
                ])
            await _edit(
                f"{flag} *{_md(title)}* 🔄 _actualizada{'s' if n > 1 else ''}_\n"
                f"📝 {n} nota{'s' if n > 1 else ''} actualizada{'s' if n > 1 else ''}: {names}\n"
                f"📂 [Abrir primera en Obsidian]({obs_url})\n"
                f"🌐 Idioma: `{result.language}` · 📦 Chunks: {result.chunks_used} · "
                f"⏱ {elapsed:.0f}s{extra}",
                reply_markup=keyboard
            )
            return

        # ── Modo: crear (por defecto) ────────────────────────────────────
        path, note_id = writer.write(
            title=title,
            body_md=result.markdown,
            content=content,
            language=result.language,
            needs_review=result.needs_review,
            review_notes=result.validation_problems if result.needs_review else None,
            related=related_titles,
        )
        if vault_indexer is not None:
            await asyncio.to_thread(vault_indexer.index_note, path)

        obs_url = _make_obsidian_url(path)
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("📖 Leer nota", callback_data=f"readnote:{note_id}")]
        ])
        await _edit(
            f"{flag} *{_md(title)}*\n"
            f"🆔 `{note_id}`\n"
            f"📂 [Abrir en Obsidian]({obs_url})\n"
            f"📁 `{path.name}`\n"
            f"🌐 Idioma: `{result.language}` · 📦 Chunks: {result.chunks_used} · "
            f"⏱ {elapsed:.0f}s{extra}",
            reply_markup=keyboard
        )
    except Exception as e:
        logger.exception("Job %s falló durante síntesis", job.short_id())
        await _edit(f"❌ Job `{job.short_id()}` falló: `{_md(f'{type(e).__name__}: {e}')}`")
        raise  # para que el worker marque el job como FAILED


# ─── Gestor de listas curadas (/note, /notedel) ───────────────────────────────

async def cmd_note(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Gestiona listas curadas (libros, series, películas, …).

    Uso:
      /note                           → ver todos los tipos
      /note <tipo>                    → ver esa lista
      /note <tipo> <ítem1>, <ítem2>   → añadir ítems

    Ejemplos:
      /note libros
      /note libros Isaac Asimov, Marcos Vázquez
      /note series Dark, Severance, The Bear
      /note películas Inception, Interstellar
    """
    if not _is_authorized(update):
        return

    args_raw = " ".join(ctx.args or []).strip()

    # Sin argumentos → listado de todos los tipos disponibles
    if not args_raw:
        kinds = await asyncio.to_thread(note_manager.list_kinds)
        if not kinds:
            await update.message.reply_text(
                "📋 *No hay listas todavía.*\n\n"
                "Crea una con:\n`/note <tipo> <ítem1>, <ítem2>`\n\n"
                "Tipos predefinidos: `libros`, `series`, `películas`, `podcasts`, "
                "`música`, `juegos`, `artículos`, `otros`",
                parse_mode="Markdown",
            )
            return
        lines = ["📋 *Tus listas curadas:*\n"]
        for kind, n in kinds:
            emoji = kind_emoji(kind)
            lines.append(
                f"  {emoji} *{kind}* — {n} elemento{'s' if n != 1 else ''} · `/note {kind}`"
            )
        lines.append(
            "\n_Añadir:_ `/note <tipo> <ítem1>, <ítem2>`\n"
            "_Borrar:_ `/notedel <tipo> <nombre_parcial>`"
        )
        try:
            await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
        except Exception:
            await update.message.reply_text("\n".join(lines))
        return

    # Separar el tipo del resto (primer token = tipo)
    parts = args_raw.split(None, 1)
    raw_kind = parts[0]
    rest = parts[1].strip() if len(parts) > 1 else ""
    canonical = normalize_kind(raw_kind)

    # Solo tipo, sin ítems → mostrar la lista
    if not rest:
        text = await asyncio.to_thread(note_manager.render_telegram, canonical)
        try:
            await update.message.reply_text(text, parse_mode="Markdown")
        except Exception:
            await update.message.reply_text(text)
        return

    # Tipo + ítems → añadir
    items = [i.strip() for i in rest.split(",") if i.strip()]
    added, was_created = await asyncio.to_thread(note_manager.add_items, canonical, items)

    emoji = kind_emoji(canonical)
    action = "creada ✨" if was_created else "actualizada"

    if not added:
        already = ", ".join(_md(i) for i in items[:5])
        await update.message.reply_text(
            f"{emoji} *{canonical.capitalize()}* — sin cambios\n"
            f"_Todos los ítems ya existían:_ {already}",
            parse_mode="Markdown",
        )
        return

    total = len(await asyncio.to_thread(note_manager.list_items, canonical))
    items_str = "\n".join(f"  ✅ {_md(i)}" for i in added)
    skipped = len(items) - len(added)
    skip_note = f"\n_({skipped} ya existía{'n' if skipped > 1 else ''})_" if skipped else ""

    obs_path = note_manager.note_path_for(canonical)
    obs_url = _make_obsidian_url(obs_path)

    try:
        await update.message.reply_text(
            f"{emoji} *{canonical.capitalize()}* {action}\n\n"
            f"*Añadido{'s' if len(added) > 1 else ''}:*\n{items_str}{skip_note}\n\n"
            f"_Total: {total} elemento{'s' if total != 1 else ''}_\n"
            f"📂 [Abrir en Obsidian]({obs_url})  ·  `/note {canonical}`",
            parse_mode="Markdown",
            disable_web_page_preview=True,
        )
    except Exception:
        await update.message.reply_text(
            f"{emoji} {canonical.capitalize()} {action}: "
            + ", ".join(added)
            + f" (total {total})"
        )


async def cmd_notedel(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Elimina un ítem de una lista por coincidencia parcial.

    Uso: /notedel <tipo> <nombre_parcial>
    Ej:  /notedel libros Asimov
         /notedel series Dark
    """
    if not _is_authorized(update):
        return
    args = ctx.args or []
    if len(args) < 2:
        await update.message.reply_text(
            "Uso: `/notedel <tipo> <nombre_parcial>`\n"
            "Ej: `/notedel libros Asimov` · `/notedel series Dark`",
            parse_mode="Markdown",
        )
        return

    raw_kind = args[0]
    query = " ".join(args[1:])
    canonical = normalize_kind(raw_kind)
    removed = await asyncio.to_thread(note_manager.remove_item, canonical, query)

    emoji = kind_emoji(canonical)
    if removed is None:
        await update.message.reply_text(
            f"⚠️ No encontré ningún ítem con `{_md(query)}` en *{canonical}*.\n"
            f"_Ver lista:_ `/note {canonical}`",
            parse_mode="Markdown",
        )
    else:
        total = len(await asyncio.to_thread(note_manager.list_items, canonical))
        await update.message.reply_text(
            f"{emoji} Eliminado de *{canonical}*:\n"
            f"  ~{_md(removed)}~\n\n"
            f"_Quedan {total} elemento{'s' if total != 1 else ''}._ · `/note {canonical}`",
            parse_mode="Markdown",
        )


# ─── Google Calendar ─────────────────────────────────────────────────────────

def _gcal_check(update: Update) -> bool:
    """Devuelve True si el módulo gcal está disponible y envía error si no."""
    if not _GCAL_AVAILABLE or gcal is None:
        asyncio.ensure_future(update.message.reply_text(
            "⚠️ Google Calendar no disponible.\n"
            "Instala: `pip install google-api-python-client google-auth-oauthlib google-auth-httplib2 apscheduler`",
            parse_mode="Markdown",
        ))
        return False
    return True


async def cmd_calauth(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Autenticación OAuth2 con Google Calendar.

    1. Genera la URL de autorización.
    2. Abre un servidor local temporal en GCAL_AUTH_PORT.
    3. Usuario abre la URL en el navegador y acepta.
    4. Bot captura el código y guarda el token.
    """
    if not _is_authorized(update):
        return
    if not _GCAL_AVAILABLE or gcal is None:
        await update.message.reply_text(
            "⚠️ Módulo Google Calendar no disponible.\n"
            "Ejecuta: `pip install google-api-python-client google-auth-oauthlib google-auth-httplib2`",
            parse_mode="Markdown",
        )
        return
    if not settings.gcal_client_id or not settings.gcal_client_secret:
        await update.message.reply_text(
            "⚠️ Faltan credenciales de Google en el `.env`.\n\n"
            "Añade estas dos líneas:\n"
            "```\nGCAL_CLIENT_ID=tu_client_id\nGCAL_CLIENT_SECRET=tu_client_secret\n```\n\n"
            "Cómo obtenerlas:\n"
            "1. [console.cloud.google.com](https://console.cloud.google.com)\n"
            "2. APIs & Services → Credentials → Create credentials → OAuth client ID\n"
            "3. Tipo: *Desktop app* → copias Client ID y Client Secret",
            parse_mode="Markdown",
            disable_web_page_preview=True,
        )
        return
    if gcal.is_authenticated():
        await update.message.reply_text(
            "✅ Ya estás autenticado con Google Calendar.\n"
            "_Para re-autenticar borra el token y vuelve a ejecutar /calauth._",
            parse_mode="Markdown",
        )
        return

    port = settings.gcal_auth_port
    try:
        auth_url, flow = gcal.get_auth_url(redirect_port=port)
    except Exception as exc:
        await update.message.reply_text(f"❌ Error generando URL de auth: `{_md(str(exc))}`", parse_mode="Markdown")
        return

    msg = await update.message.reply_text(
        f"🔐 *Autenticación con Google Calendar*\n\n"
        f"1. Abre este enlace en tu navegador:\n{auth_url}\n\n"
        f"2. Acepta los permisos.\n"
        f"3. Google te redirigirá a `localhost:{port}` — espera la confirmación aquí.\n\n"
        f"_Timeout: 2 minutos._",
        parse_mode="Markdown",
        disable_web_page_preview=True,
    )

    import threading
    from http.server import HTTPServer, BaseHTTPRequestHandler
    from urllib.parse import urlparse, parse_qs as _parse_qs

    loop = asyncio.get_running_loop()
    result_q: asyncio.Queue = asyncio.Queue(maxsize=1)

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            params = _parse_qs(urlparse(self.path).query)
            code = params.get("code", [None])[0]
            error = params.get("error", [None])[0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(
                b"<html><body style='font-family:sans-serif;padding:40px'>"
                b"<h2>Autenticado con Google Calendar</h2>"
                b"<p>Puedes cerrar esta ventana y volver al bot de Telegram.</p>"
                b"</body></html>"
            )
            asyncio.run_coroutine_threadsafe(result_q.put((code, error)), loop)

        def log_message(self, *args): pass  # silencia logs del servidor HTTP

    server = HTTPServer(("localhost", port), _Handler)
    threading.Thread(target=server.handle_request, daemon=True).start()

    try:
        code, error = await asyncio.wait_for(result_q.get(), timeout=120)
    except asyncio.TimeoutError:
        server.server_close()
        await msg.edit_text("⏱ Timeout de autenticación. Vuelve a ejecutar /calauth.")
        return
    finally:
        server.server_close()

    if error or not code:
        await msg.edit_text(f"❌ Error de autenticación: `{_md(error or 'sin código')}`", parse_mode="Markdown")
        return

    try:
        gcal.exchange_code(flow, code)
        # Actualiza el chat_id del scheduler con este usuario
        global cal_scheduler
        if cal_scheduler:
            cal_scheduler.chat_id = update.effective_user.id
        await msg.edit_text(
            "✅ *¡Autenticado con Google Calendar!*\n\n"
            "Ahora puedes usar:\n"
            "• /cal — agenda de hoy\n"
            "• /calweek — agenda de la semana\n"
            "• /caladd — añadir evento",
            parse_mode="Markdown",
        )
    except Exception as exc:
        await msg.edit_text(f"❌ Error canjeando código: `{_md(str(exc))}`", parse_mode="Markdown")


def _cal_event_keyboard(event: dict, marked: bool) -> InlineKeyboardMarkup:
    """Botones inline para un evento de calendario."""
    event_id = event.get("id", "")
    bell_label = "🔕 Quitar aviso" if marked else "🔔 Avisarme"
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(bell_label, callback_data=f"cal_notify:{event_id}"),
            InlineKeyboardButton("✏️ Editar", callback_data=f"cal_edit:{event_id}"),
        ],
        [
            InlineKeyboardButton("🗑 Borrar evento", callback_data=f"cal_del:{event_id}"),
        ],
    ])


async def cmd_cal(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Muestra la agenda del día actual (o de los próximos N días).

    Uso:
      /cal        → hoy
      /cal 3      → próximos 3 días
    """
    if not _is_authorized(update):
        return
    if not _gcal_check(update):
        return
    if not gcal.is_authenticated():
        await update.message.reply_text(
            "🔐 Aún no has conectado Google Calendar.\nEjecuta /calauth primero.",
        )
        return

    days = 1
    if ctx.args:
        try:
            days = max(1, min(int(ctx.args[0]), 30))
        except ValueError:
            pass

    progress = await update.message.reply_text("📅 Cargando agenda…")
    try:
        from datetime import datetime, timedelta, timezone
        now_local = datetime.now()
        day_start = now_local.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
        day_end = day_start + timedelta(days=days)

        events = await gcal.list_events(day_start, day_end, max_results=25)
        marked_ids = {r["event_id"] for r in notify_db.get_all_marked()} if notify_db else set()

        weekday_es = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
        if days == 1:
            day_name = weekday_es[now_local.weekday()]
            title = f"📅 Hoy — {day_name} {now_local.strftime('%d/%m')}"
        else:
            end_date = now_local + timedelta(days=days - 1)
            title = f"📅 {now_local.strftime('%d/%m')} – {end_date.strftime('%d/%m')}"

        agenda_text = format_agenda(events, title, marked_ids, show_date=(days > 1))
        await progress.delete()

        if not events:
            await update.message.reply_text(agenda_text, parse_mode="Markdown")
            return

        # Enviamos cada evento con sus botones (max 5 para no saturar)
        await update.message.reply_text(agenda_text, parse_mode="Markdown")
        for ev in events[:5]:
            marked = ev.get("id", "") in marked_ids
            ev_text = format_event(ev, show_date=(days > 1), marked=marked)
            keyboard = _cal_event_keyboard(ev, marked)
            await update.message.reply_text(ev_text, parse_mode="Markdown", reply_markup=keyboard)

    except Exception as exc:
        logger.exception("Error en /cal")
        await progress.edit_text(f"❌ Error: `{_md(str(exc))}`", parse_mode="Markdown")


async def cmd_calweek(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Muestra la agenda de los próximos 7 días con botones de acción."""
    if not _is_authorized(update):
        return
    ctx.args = ["7"]
    await cmd_cal(update, ctx)


async def cmd_caladd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Añade un evento a Google Calendar.

    Uso:
      /caladd <título> [fecha] [hora] [duración_min]

    Ejemplos:
      /caladd Dentista mañana 10:30
      /caladd Reunión viernes 16:00 60
      /caladd Cumpleaños Ana 2026-06-15
      /caladd Llamada hoy 15:00
    """
    if not _is_authorized(update):
        return
    if not _gcal_check(update):
        return
    if not gcal.is_authenticated():
        await update.message.reply_text("🔐 Ejecuta /calauth primero.")
        return

    raw = " ".join(ctx.args or []).strip()
    if not raw:
        await update.message.reply_text(
            "Uso: `/caladd <título> [fecha] [hora] [duración_min]`\n\n"
            "Ejemplos:\n"
            "• `/caladd Dentista mañana 10:30`\n"
            "• `/caladd Reunión viernes 16:00 60`\n"
            "• `/caladd Cumpleaños Ana 2026-06-15`",
            parse_mode="Markdown",
        )
        return

    import dateparser
    from datetime import datetime, timedelta, timezone as _tz

    # Estrategia: parsear desde el final hacia el título.
    # Buscamos un entero de duración al final, luego hora, luego fecha.
    parts = raw.rsplit(None, 4)  # máx 5 tokens desde la derecha

    duration_min = 60
    start_dt = None
    title = raw

    # Intentamos parsear con dateparser tomando cada sufijo posible
    for i in range(1, min(5, len(parts))):
        candidate_date = " ".join(parts[-i:])
        candidate_title = " ".join(parts[:-i]).strip()
        if not candidate_title:
            continue
        parsed = dateparser.parse(
            candidate_date,
            languages=["es", "en"],
            settings={"PREFER_DATES_FROM": "future", "RETURN_AS_TIMEZONE_AWARE": True},
        )
        if parsed:
            start_dt = parsed
            title = candidate_title
            # ¿hay un entero solo al final que podría ser duración?
            last = parts[-1]
            if last.isdigit() and i > 1:
                duration_min = int(last)
                # re-parsear sin ese último token
                candidate_date2 = " ".join(parts[-i:-1])
                parsed2 = dateparser.parse(
                    candidate_date2,
                    languages=["es", "en"],
                    settings={"PREFER_DATES_FROM": "future", "RETURN_AS_TIMEZONE_AWARE": True},
                )
                if parsed2:
                    start_dt = parsed2
            break

    if start_dt is None:
        # Fallback: evento hoy a las 12:00
        now = datetime.now()
        start_dt = now.replace(hour=12, minute=0, second=0, microsecond=0).astimezone()
        await update.message.reply_text(
            f"⚠️ No pude interpretar la fecha/hora. Creando el evento hoy a las 12:00.\n"
            f"Usa el formato: `/caladd {raw} mañana 10:00`",
            parse_mode="Markdown",
        )

    end_dt = start_dt + timedelta(minutes=duration_min)

    try:
        event = await gcal.create_event(title=title, start=start_dt, end=end_dt)
        event_id = event.get("id", "")
        event_url = event.get("htmlLink", "")

        # Enlace al evento en Google Calendar
        link_part = f"\n[Abrir en Google Calendar]({event_url})" if event_url else ""
        await update.message.reply_text(
            f"✅ *Evento creado*{link_part}\n\n"
            f"📅 *{_md(title)}*\n"
            f"🕐 {start_dt.strftime('%A %d/%m/%Y %H:%M')} ({duration_min} min)",
            parse_mode="Markdown",
            disable_web_page_preview=True,
            reply_markup=_cal_event_keyboard(event, marked=False),
        )
    except Exception as exc:
        logger.exception("Error creando evento")
        await update.message.reply_text(f"❌ Error: `{_md(str(exc))}`", parse_mode="Markdown")


async def cmd_calday(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Envía el resumen diario ahora (sin esperar al horario programado)."""
    if not _is_authorized(update):
        return
    if not _gcal_check(update):
        return
    if cal_scheduler:
        await cal_scheduler.trigger_daily_summary()
    else:
        await update.message.reply_text("⚠️ Scheduler no activo.")


async def cmd_calconfig(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Configura el resumen diario automático.

    Uso:
      /calconfig          → muestra configuración actual
      /calconfig 08:30    → cambia hora del resumen diario
      /calconfig off      → desactiva resumen semanal
      /calconfig on       → activa resumen semanal
    """
    if not _is_authorized(update):
        return
    if not _gcal_check(update):
        return
    if not notify_db:
        await update.message.reply_text("⚠️ Módulo de notificaciones no disponible.")
        return

    args = ctx.args or []
    if not args:
        hhmm = notify_db.get_daily_summary_time()
        weekly = notify_db.get_weekly_summary_enabled()
        await update.message.reply_text(
            f"📅 *Configuración de calendario*\n\n"
            f"⏰ Resumen diario: `{hhmm}`\n"
            f"📆 Resumen semanal (lunes): {'✅ activo' if weekly else '❌ inactivo'}\n\n"
            f"Cambiar hora: `/calconfig 08:30`\n"
            f"Activar/desactivar semanal: `/calconfig on` o `/calconfig off`",
            parse_mode="Markdown",
        )
        return

    arg = args[0].lower()
    if arg == "off":
        notify_db.set_weekly_summary_enabled(False)
        await update.message.reply_text("📆 Resumen semanal desactivado.")
    elif arg == "on":
        notify_db.set_weekly_summary_enabled(True)
        if cal_scheduler:
            cal_scheduler._reschedule_weekly_summary()
        await update.message.reply_text("📆 Resumen semanal activado (lunes por la mañana).")
    elif ":" in arg:
        # Formato HH:MM
        try:
            h, m = map(int, arg.split(":"))
            assert 0 <= h <= 23 and 0 <= m <= 59
        except (ValueError, AssertionError):
            await update.message.reply_text("⚠️ Formato inválido. Usa `HH:MM` (ej. `08:30`).", parse_mode="Markdown")
            return
        hhmm = f"{h:02d}:{m:02d}"
        if cal_scheduler:
            cal_scheduler.update_daily_summary_time(hhmm)
        else:
            notify_db.set_daily_summary_time(hhmm)
        await update.message.reply_text(f"✅ Resumen diario configurado a las `{hhmm}`.", parse_mode="Markdown")
    else:
        await update.message.reply_text(
            "Uso: `/calconfig HH:MM` · `/calconfig on` · `/calconfig off`",
            parse_mode="Markdown",
        )


# ─── Callbacks de calendario ──────────────────────────────────────────────────

async def on_cal_notify_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Toggle notificación para un evento. Callback: cal_notify:<event_id>"""
    if not _is_authorized(update):
        return
    query = update.callback_query
    await query.answer()
    _, _, event_id = query.data.partition(":")

    if not notify_db or not gcal:
        await query.answer("⚠️ Módulo no disponible", show_alert=True)
        return

    if notify_db.is_marked(event_id):
        notify_db.unmark(event_id)
        await query.answer("🔕 Aviso eliminado")
        new_marked = False
    else:
        # Obtenemos el evento para guardar start_iso y título
        try:
            event = await gcal.get_event(event_id)
            start_dt = GoogleCalendarClient.event_start_dt(event)
            start_iso = start_dt.isoformat() if start_dt else ""
            title = event.get("summary", "")
            notify_db.mark(event_id, title=title, start_iso=start_iso, minutes_before=15)
            await query.answer("🔔 Te avisaré 15 min antes")
            new_marked = True
        except Exception as exc:
            await query.answer(f"❌ Error: {exc}", show_alert=True)
            return

    # Actualiza el botón del mensaje
    try:
        event = await gcal.get_event(event_id)
        ev_text = format_event(event, marked=new_marked)
        keyboard = _cal_event_keyboard(event, marked=new_marked)
        await query.edit_message_text(ev_text, parse_mode="Markdown", reply_markup=keyboard)
    except Exception:
        pass


async def on_cal_del_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Borra un evento de Google Calendar. Callback: cal_del:<event_id>"""
    if not _is_authorized(update):
        return
    query = update.callback_query
    _, _, event_id = query.data.partition(":")

    if not gcal:
        await query.answer("⚠️ Módulo no disponible", show_alert=True)
        return

    # Pedimos confirmación con botones
    await query.answer()
    await query.edit_message_reply_markup(
        reply_markup=InlineKeyboardMarkup([
            [
                InlineKeyboardButton("✅ Sí, borrar", callback_data=f"cal_del_confirm:{event_id}"),
                InlineKeyboardButton("❌ Cancelar", callback_data=f"cal_del_cancel:{event_id}"),
            ]
        ])
    )


async def on_cal_del_confirm_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Confirma el borrado de un evento."""
    if not _is_authorized(update):
        return
    query = update.callback_query
    _, _, event_id = query.data.partition(":")
    await query.answer()
    try:
        await gcal.delete_event(event_id)
        if notify_db:
            notify_db.unmark(event_id)
        await query.edit_message_text("🗑 Evento borrado.")
    except Exception as exc:
        await query.edit_message_text(f"❌ Error al borrar: `{_md(str(exc))}`", parse_mode="Markdown")


async def on_cal_del_cancel_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Cancela el borrado — restaura los botones originales."""
    if not _is_authorized(update):
        return
    query = update.callback_query
    _, _, event_id = query.data.partition(":")
    await query.answer("Borrado cancelado")
    try:
        event = await gcal.get_event(event_id)
        marked = notify_db.is_marked(event_id) if notify_db else False
        keyboard = _cal_event_keyboard(event, marked=marked)
        await query.edit_message_reply_markup(reply_markup=keyboard)
    except Exception:
        pass


async def on_cal_edit_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Inicia edición de un evento. Callback: cal_edit:<event_id>

    El usuario responde con el nuevo título o "HH:MM nuevo título" para cambiar hora y título.
    """
    if not _is_authorized(update):
        return
    query = update.callback_query
    _, _, event_id = query.data.partition(":")
    await query.answer()

    try:
        event = await gcal.get_event(event_id)
    except Exception as exc:
        await query.edit_message_text(f"❌ No encontré el evento: `{_md(str(exc))}`", parse_mode="Markdown")
        return

    title = event.get("summary", "")
    # Guardamos el estado de edición
    ctx.application.bot_data.setdefault("cal_edit", {})[str(query.message.chat_id)] = {
        "event_id": event_id,
        "msg_id": query.message.message_id,
    }

    await query.edit_message_text(
        f"✏️ *Editando evento:* {_md(title)}\n\n"
        f"Responde con el nuevo texto. Formatos:\n"
        f"• `nuevo título` — cambia solo el título\n"
        f"• `10:30 nuevo título` — cambia hora y título\n"
        f"• `10:30` — cambia solo la hora\n\n"
        f"_Escribe /cancel para cancelar._",
        parse_mode="Markdown",
    )


async def _handle_cal_edit_reply(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> bool:
    """Procesa la respuesta al flujo de edición de un evento.
    Devuelve True si ha gestionado el mensaje (para que handle_text no lo procese)."""
    chat_id = str(update.effective_chat.id)
    edit_state = ctx.application.bot_data.get("cal_edit", {})
    if chat_id not in edit_state:
        return False

    state = edit_state.pop(chat_id)
    text = (update.message.text or "").strip()

    if text.lower() == "/cancel":
        await update.message.reply_text("❌ Edición cancelada.")
        return True

    event_id = state["event_id"]
    import re as _re

    new_title = None
    new_start = None

    # ¿Empieza con HH:MM?
    time_match = _re.match(r"^(\d{1,2}:\d{2})\s*(.*)", text)
    if time_match:
        hhmm_str = time_match.group(1)
        rest = time_match.group(2).strip()
        try:
            h, m = map(int, hhmm_str.split(":"))
            # Obtenemos el evento para saber su fecha actual
            event = await gcal.get_event(event_id)
            current_start = GoogleCalendarClient.event_start_dt(event)
            if current_start:
                from datetime import timezone as _tz
                new_start = current_start.replace(hour=h, minute=m, second=0, microsecond=0)
        except (ValueError, Exception):
            pass
        if rest:
            new_title = rest
    else:
        new_title = text

    if not new_title and not new_start:
        await update.message.reply_text("⚠️ No entendí el formato. Edición cancelada.")
        return True

    try:
        updated = await gcal.update_event(event_id, title=new_title, start=new_start)
        marked = notify_db.is_marked(event_id) if notify_db else False
        ev_text = format_event(updated, marked=marked)
        await update.message.reply_text(
            f"✅ *Evento actualizado*\n\n{ev_text}",
            parse_mode="Markdown",
            reply_markup=_cal_event_keyboard(updated, marked=marked),
        )
    except Exception as exc:
        await update.message.reply_text(f"❌ Error al editar: `{_md(str(exc))}`", parse_mode="Markdown")

    return True


# ─── Handlers de mensajes ─────────────────────────────────────────────────────
async def handle_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update):
        return
    text = (update.message.text or "").strip()
    if not text:
        return

    # 0a. ¿Estamos en flujo de edición de evento de calendario?
    if _GCAL_AVAILABLE and gcal is not None:
        if await _handle_cal_edit_reply(update, ctx):
            return

    # 0b. ¿Es un mensaje-tarea? Prefijo "task"/"tarea"/"#task"/"/task"
    if (desc := extract_task_description(text)):
        await _save_new_task(update, desc, source="text")
        return

    payload = MessageRouter.from_text(text)
    await _run_pipeline(update, ctx, payload)


async def handle_document(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update):
        return
    doc = update.message.document
    if not doc:
        return

    settings.temp_dir.mkdir(parents=True, exist_ok=True)
    local_path = settings.temp_dir / f"{doc.file_unique_id}_{doc.file_name}"

    progress = await update.message.reply_text(
        f"📥 Descargando `{doc.file_name}`…", parse_mode="Markdown"
    )
    file = await ctx.bot.get_file(doc.file_id)
    await file.download_to_drive(custom_path=str(local_path))
    await progress.delete()

    payload = MessageRouter.from_file(
        local_path=local_path, mime=doc.mime_type, filename=doc.file_name or local_path.name,
    )
    try:
        await _run_pipeline(update, ctx, payload)
    finally:
        try:
            Path(local_path).unlink(missing_ok=True)
        except Exception:
            pass


async def handle_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Imágenes enviadas como FOTO (Telegram las comprime a JPEG).
    Para imágenes a calidad original el usuario debe mandarlas como `Documento`,
    eso ya lo gestiona handle_document."""
    if not _is_authorized(update):
        return
    photos = update.message.photo
    if not photos:
        return
    # photo es una lista con diferentes resoluciones; cogemos la mayor
    photo = max(photos, key=lambda p: (p.width or 0) * (p.height or 0))

    settings.temp_dir.mkdir(parents=True, exist_ok=True)
    local_path = settings.temp_dir / f"photo_{photo.file_unique_id}.jpg"

    progress = await update.message.reply_text("📥 Descargando imagen…")
    file = await ctx.bot.get_file(photo.file_id)
    await file.download_to_drive(custom_path=str(local_path))
    await progress.delete()

    payload = MessageRouter.from_file(
        local_path=local_path,
        mime="image/jpeg",
        filename=local_path.name,
    )
    payload.metadata["telegram_photo"] = True
    # Caption del usuario como pista contextual al título
    if update.message.caption:
        payload.metadata["caption"] = update.message.caption.strip()

    try:
        await _run_pipeline(update, ctx, payload)
    finally:
        try:
            Path(local_path).unlink(missing_ok=True)
        except Exception:
            pass


async def handle_voice(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Mensaje de voz nativo de Telegram (.ogg con codec opus)."""
    if not _is_authorized(update):
        return
    voice = update.message.voice or update.message.audio
    if not voice:
        return

    settings.temp_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".ogg" if update.message.voice else ".m4a"
    local_path = settings.temp_dir / f"voice_{voice.file_unique_id}{suffix}"

    progress = await update.message.reply_text("📥 Descargando audio…")
    file = await ctx.bot.get_file(voice.file_id)
    await file.download_to_drive(custom_path=str(local_path))
    await progress.delete()

    payload = MessageRouter.from_file(
        local_path=local_path,
        mime=getattr(voice, "mime_type", "audio/ogg"),
        filename=local_path.name,
    )
    payload.metadata["telegram_voice"] = True
    try:
        await _run_pipeline(update, ctx, payload)
    finally:
        try:
            Path(local_path).unlink(missing_ok=True)
        except Exception:
            pass


# ─── Error handler global ─────────────────────────────────────────────────────

async def error_handler(update: object, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Captura todos los errores no manejados de la librería de Telegram.

    - NetworkError / TimedOut: errores de red transitorios — la librería ya
      reintenta automáticamente, solo logueamos un WARNING sin traceback.
    - Resto: logueamos como ERROR con traceback completo para debugging.
    """
    err = ctx.error
    if isinstance(err, (NetworkError, TimedOut)):
        logger.warning("⚠️ Red: %s (reintentando automáticamente…)", err)
        return
    logger.exception("❌ Error inesperado en el bot", exc_info=err)


# ─── Bootstrap ────────────────────────────────────────────────────────────────
def main() -> None:
    settings.validate()
    logger.info(
        "Iniciando bot. Vault=%s | LLM=%s | Extractores=%s",
        settings.obsidian_vault_path,
        llm.describe(),
        ExtractorFactory.registered(),
    )

    app = (
        Application.builder()
        .token(settings.telegram_token)
        .post_init(_post_init)
        .build()
    )

    app.add_error_handler(error_handler)
    app.add_handler(CommandHandler("start",  cmd_start))
    app.add_handler(CommandHandler("help",   cmd_help))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("queue",  cmd_queue))
    app.add_handler(CommandHandler("cancel", cmd_cancel))
    app.add_handler(CommandHandler("weekly", cmd_weekly))
    app.add_handler(CommandHandler("random", cmd_random))
    app.add_handler(CommandHandler("search", cmd_search))
    app.add_handler(CommandHandler("find",   cmd_find))
    app.add_handler(CommandHandler("read",   cmd_read))
    # Tareas
    app.add_handler(CommandHandler("tasks",    cmd_tasks))
    app.add_handler(CommandHandler("done",     cmd_done))
    app.add_handler(CommandHandler("wip",      cmd_wip))
    app.add_handler(CommandHandler("todo",     cmd_todo))
    app.add_handler(CommandHandler("del",      cmd_del))
    # Listas curadas
    app.add_handler(CommandHandler("note",     cmd_note))
    app.add_handler(CommandHandler("notedel",  cmd_notedel))
    # Google Calendar
    app.add_handler(CommandHandler("calauth",   cmd_calauth))
    app.add_handler(CommandHandler("cal",       cmd_cal))
    app.add_handler(CommandHandler("calweek",   cmd_calweek))
    app.add_handler(CommandHandler("caladd",    cmd_caladd))
    app.add_handler(CommandHandler("calday",    cmd_calday))
    app.add_handler(CommandHandler("calconfig", cmd_calconfig))
    app.add_handler(CallbackQueryHandler(on_cal_notify_callback,     pattern=r"^cal_notify:"))
    app.add_handler(CallbackQueryHandler(on_cal_edit_callback,       pattern=r"^cal_edit:"))
    app.add_handler(CallbackQueryHandler(on_cal_del_callback,        pattern=r"^cal_del:"))
    app.add_handler(CallbackQueryHandler(on_cal_del_confirm_callback, pattern=r"^cal_del_confirm:"))
    app.add_handler(CallbackQueryHandler(on_cal_del_cancel_callback,  pattern=r"^cal_del_cancel:"))
    app.add_handler(CallbackQueryHandler(on_dup_callback,     pattern=r"^dup_(new|cancel|overwrite|update):"))
    app.add_handler(CallbackQueryHandler(on_qcancel_callback, pattern=r"^qcancel:"))
    app.add_handler(CallbackQueryHandler(on_readnote_callback, pattern=r"^readnote:"))
    app.add_handler(CallbackQueryHandler(on_image_callback,   pattern=r"^img(ok|no|full):"))
    app.add_handler(CallbackQueryHandler(on_concept_callback, pattern=r"^cpt(ok|no|full):"))
    app.add_handler(CallbackQueryHandler(on_task_callback,    pattern=r"^task(done|wip|pending):"))
    app.add_handler(CallbackQueryHandler(on_confirm_callback, pattern=r"^(go|no):"))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Bot escuchando en Long Polling…")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
