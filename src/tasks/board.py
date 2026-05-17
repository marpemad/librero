"""
TaskBoard — tablón Kanban persistido como Markdown.

El tablón vive en un único fichero `.md` del Vault (ruta configurable en
`settings.tasks_file_path`). Estructura del fichero:

    ---
    updated: "2026-05-01T08:30:00"
    type: "task_board"
    ---

    # 📋 Tablón de tareas

    ## 🟦 Pendientes (3)
    - [ ] **T003** · 2026-05-01 · Llamar al fontanero
    - [ ] **T002** · 2026-05-01 · Comprar café #personal
    - [ ] **T001** · 2026-04-30 · Refactorizar el extractor #dev

    ## 🟨 En progreso (1)
    - [ ] **T004** · 2026-04-30 · Vault Intel embeddings #dev

    ## ✅ Completadas (2)
    - [x] **T000** · 2026-04-29 · Setup del bot

Lo escribimos así para que sea legible y editable a mano en Obsidian, y
también parseable de vuelta por nosotros.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TaskState(str, Enum):
    PENDING     = "pending"
    IN_PROGRESS = "in_progress"
    DONE        = "done"


STATE_EMOJI: dict[TaskState, str] = {
    TaskState.PENDING:     "🟦",
    TaskState.IN_PROGRESS: "🟨",
    TaskState.DONE:        "✅",
}

STATE_LABEL: dict[TaskState, str] = {
    TaskState.PENDING:     "Pendientes",
    TaskState.IN_PROGRESS: "En progreso",
    TaskState.DONE:        "Completadas",
}


# --- Detección de mensajes-tarea ---------------------------------------------

# "task ...", "tarea ...", "#task ...", "/task ..." (acepta `:`, `.`, `,`, `-` opcional)
_TASK_PREFIX_RE = re.compile(
    r"^\s*(?:#?(?:task|tarea|tareas)|/task)\s*[:\.\-,]?\s+(.+)",
    re.IGNORECASE | re.DOTALL,
)


def extract_task_description(text: str) -> Optional[str]:
    """Si `text` es un mensaje-tarea, devuelve la descripción. Si no, None.

    Detecta:
      - "task comprar leche"
      - "tarea: llamar al fontanero"
      - "Tarea, refactorizar extractor"
      - "#task escribir post"
      - "/task arreglar bug"
    """
    if not text:
        return None
    m = _TASK_PREFIX_RE.match(text.strip())
    if not m:
        return None
    desc = m.group(1).strip()
    # Limpiamos puntuación final típica del habla transcrita ("..." / ".")
    desc = desc.rstrip(".").strip()
    return desc or None


# --- Modelo ------------------------------------------------------------------

@dataclass
class Task:
    id: str
    description: str
    state: TaskState = TaskState.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    tags: list[str] = field(default_factory=list)

    def to_md_line(self) -> str:
        date = self.created_at.strftime("%Y-%m-%d")
        check = "x" if self.state is TaskState.DONE else " "
        tags_str = (" " + " ".join(f"#{t}" for t in self.tags)) if self.tags else ""
        return f"- [{check}] **{self.id}** · {date} · {self.description}{tags_str}"


# Parser de líneas de tarea (tolerante con backticks, dobles asteriscos, fechas)
_TASK_LINE_RE = re.compile(
    r"""
    ^\-\s\[(?P<check>[\sxX])\]\s+
    \*\*(?P<id>T\d+)\*\*\s*[·\-]\s*
    (?P<date>\d{4}-\d{2}-\d{2})\s*[·\-]\s*
    (?P<rest>.+?)\s*$
    """,
    re.VERBOSE,
)
_TAG_TOKEN_RE = re.compile(r"\s+(#[\w\-áéíóúüñÁÉÍÓÚÜÑ]+)")


# --- Tablón ------------------------------------------------------------------

class TaskBoard:
    """Carga, modifica y persiste el tablón de tareas."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._tasks: list[Task] = []
        self.load()

    # ---- I/O ----

    def load(self) -> None:
        if not self.path.exists():
            self._tasks = []
            return
        try:
            text = self.path.read_text(encoding="utf-8")
        except OSError as e:
            logger.warning("No se pudo leer el tablón: %s", e)
            self._tasks = []
            return

        tasks: list[Task] = []
        current_state: Optional[TaskState] = None
        for raw_line in text.splitlines():
            stripped = raw_line.strip()
            # Detectar cambio de sección por H2
            if stripped.startswith("## "):
                low = stripped.lower()
                if "pendient" in low:
                    current_state = TaskState.PENDING
                elif "progreso" in low or "progress" in low or "curso" in low:
                    current_state = TaskState.IN_PROGRESS
                elif "complet" in low or "done" in low or "hech" in low:
                    current_state = TaskState.DONE
                else:
                    current_state = None
                continue
            # Línea de tarea
            if current_state is None:
                continue
            m = _TASK_LINE_RE.match(raw_line)
            if not m:
                continue
            rest = m.group("rest")
            # Extraer tags al final de `rest`
            tags = [t.lstrip("#") for t in _TAG_TOKEN_RE.findall(rest)]
            description = _TAG_TOKEN_RE.sub("", rest).rstrip(" ·-")
            try:
                created = datetime.fromisoformat(m.group("date"))
            except ValueError:
                created = datetime.now()
            tasks.append(Task(
                id=m.group("id"),
                description=description.strip(),
                state=current_state,
                created_at=created,
                updated_at=created,
                tags=tags,
            ))
        self._tasks = tasks

    def save(self) -> None:
        self.path.write_text(self.render_markdown(), encoding="utf-8")

    # ---- API ----

    def add(self, description: str, tags: Optional[list[str]] = None) -> Task:
        new_id = self._next_id()
        task = Task(id=new_id, description=description.strip(), tags=tags or [])
        self._tasks.append(task)
        self.save()
        logger.info("Tarea %s añadida: %s", new_id, description[:60])
        return task

    def update_state(self, task_id: str, new_state: TaskState) -> Optional[Task]:
        t = self.get(task_id)
        if not t:
            return None
        t.state = new_state
        t.updated_at = datetime.now()
        self.save()
        return t

    def delete(self, task_id: str) -> bool:
        before = len(self._tasks)
        self._tasks = [t for t in self._tasks if t.id.lower() != task_id.lower()]
        if len(self._tasks) < before:
            self.save()
            return True
        return False

    def get(self, task_id: str) -> Optional[Task]:
        target = task_id.lower().lstrip("#")
        if not target:
            return None
        # Match exacto o por prefijo (T01 → T001 si es único)
        for t in self._tasks:
            if t.id.lower() == target:
                return t
        prefix_matches = [t for t in self._tasks if t.id.lower().startswith(target)]
        if len(prefix_matches) == 1:
            return prefix_matches[0]
        return None

    def all_tasks(self) -> list[Task]:
        return list(self._tasks)

    def by_state(self) -> dict[TaskState, list[Task]]:
        out: dict[TaskState, list[Task]] = {s: [] for s in TaskState}
        for t in self._tasks:
            out[t.state].append(t)
        return out

    def stats(self) -> dict[str, int]:
        bs = self.by_state()
        return {state.value: len(items) for state, items in bs.items()}

    # ---- Render ----

    def render_markdown(self) -> str:
        """Markdown que se persiste en disco (legible en Obsidian)."""
        lines = [
            "---",
            f'updated: "{datetime.now().isoformat(timespec="seconds")}"',
            'type: "task_board"',
            "---",
            "",
            "# 📋 Tablón de tareas",
            "",
        ]
        order = (TaskState.PENDING, TaskState.IN_PROGRESS, TaskState.DONE)
        bs = self.by_state()
        for state in order:
            tasks = bs[state]
            if not tasks:
                continue
            emoji = STATE_EMOJI[state]
            lines.append(f"## {emoji} {STATE_LABEL[state]} ({len(tasks)})")
            lines.append("")
            # Más recientes primero
            for t in sorted(tasks, key=lambda x: x.updated_at, reverse=True):
                lines.append(t.to_md_line())
            lines.append("")
        if not any(bs[s] for s in order):
            lines.append("_(sin tareas)_")
        return "\n".join(lines).rstrip() + "\n"

    def render_telegram(self, max_pending: int = 15, max_done: int = 5) -> str:
        """Resumen bonito para `/tasks` en Telegram (parse_mode=Markdown)."""
        bs = self.by_state()
        pending = sorted(bs[TaskState.PENDING],     key=lambda t: t.created_at, reverse=True)
        in_prog = sorted(bs[TaskState.IN_PROGRESS], key=lambda t: t.updated_at, reverse=True)
        done    = sorted(bs[TaskState.DONE],        key=lambda t: t.updated_at, reverse=True)

        if not (pending or in_prog or done):
            return "📋 *Tablón de tareas*\n\n_Sin tareas todavía. Manda un mensaje empezando por_ `task` _o_ `tarea`."

        lines: list[str] = ["📋 *Tablón de tareas*", ""]
        # Cabecera con contadores
        lines.append(
            f"🟦 *{len(pending)}* pendientes  ·  "
            f"🟨 *{len(in_prog)}* en curso  ·  "
            f"✅ *{len(done)}* hechas"
        )
        lines.append("━━━━━━━━━━━━━━━━━━━━")

        if in_prog:
            lines.append(f"\n🟨 *En progreso* ({len(in_prog)})")
            for t in in_prog:
                lines.append(_render_task_line(t))

        if pending:
            lines.append(f"\n🟦 *Pendientes* ({len(pending)})")
            for t in pending[:max_pending]:
                lines.append(_render_task_line(t))
            if len(pending) > max_pending:
                lines.append(f"  _… y {len(pending) - max_pending} más_")

        if done:
            recent_done = done[:max_done]
            lines.append(f"\n✅ *Últimas completadas* ({len(recent_done)} de {len(done)})")
            for t in recent_done:
                date = t.updated_at.strftime("%d %b")
                desc = _md_escape(t.description)
                lines.append(f"  `{t.id}` · {date} · ~{desc}~")

        lines.append("\n_Cambiar estado:_ `/done T003` · `/wip T002` · `/del T005`")
        return "\n".join(lines)

    # ---- Internal ----

    def _next_id(self) -> str:
        max_n = 0
        for t in self._tasks:
            m = re.match(r"T(\d+)", t.id)
            if m:
                max_n = max(max_n, int(m.group(1)))
        return f"T{max_n + 1:03d}"


# --- Helpers de render -------------------------------------------------------

def _md_escape(s: str) -> str:
    """Escape mínimo para Markdown V1 de Telegram."""
    return (
        s.replace("\\", "\\\\")
         .replace("_", "\\_")
         .replace("*", "\\*")
         .replace("`", "\\`")
         .replace("[", "\\[")
    )


def _render_task_line(t: Task) -> str:
    date = t.created_at.strftime("%d %b")
    desc = _md_escape(t.description)
    tags = (" " + " ".join(f"#{tag}" for tag in t.tags)) if t.tags else ""
    return f"  `{t.id}` · _{date}_ · {desc}{tags}"
