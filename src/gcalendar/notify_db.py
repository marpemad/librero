"""
SQLite para gestionar eventos marcados para notificación y la configuración
del resumen diario/semanal.

Tabla events_notify:
  event_id        TEXT PK   — ID del evento en Google Calendar
  calendar_id     TEXT      — normalmente "primary"
  title           TEXT      — título (cache)
  start_iso       TEXT      — start ISO (cache)
  minutes_before  INTEGER   — minutos de antelación para la notificación
  notified        INTEGER   — 1 si ya se envió la notificación para esta ocurrencia

Tabla settings:
  key   TEXT PK
  value TEXT
"""
from __future__ import annotations

import sqlite3
from pathlib import Path


class NotifyDB:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS events_notify (
                    event_id       TEXT PRIMARY KEY,
                    calendar_id    TEXT NOT NULL DEFAULT 'primary',
                    title          TEXT NOT NULL DEFAULT '',
                    start_iso      TEXT NOT NULL DEFAULT '',
                    minutes_before INTEGER NOT NULL DEFAULT 15,
                    notified       INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS settings (
                    key   TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )

    # ─── Marcado de eventos ───────────────────────────────────────────────────

    def mark(
        self,
        event_id: str,
        title: str,
        start_iso: str,
        minutes_before: int = 15,
        calendar_id: str = "primary",
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO events_notify (event_id, calendar_id, title, start_iso, minutes_before, notified)
                VALUES (?, ?, ?, ?, ?, 0)
                ON CONFLICT(event_id) DO UPDATE SET
                    title=excluded.title,
                    start_iso=excluded.start_iso,
                    minutes_before=excluded.minutes_before,
                    notified=0
                """,
                (event_id, calendar_id, title, start_iso, minutes_before),
            )

    def unmark(self, event_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM events_notify WHERE event_id=?", (event_id,))

    def is_marked(self, event_id: str) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM events_notify WHERE event_id=?", (event_id,)
            ).fetchone()
            return row is not None

    def get_all_marked(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM events_notify WHERE notified=0"
            ).fetchall()
            return [dict(r) for r in rows]

    def mark_as_notified(self, event_id: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE events_notify SET notified=1 WHERE event_id=?", (event_id,)
            )

    def reset_notified(self, event_id: str) -> None:
        """Resetea el flag de notificado (para cuando el evento se repite o se mueve)."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE events_notify SET notified=0 WHERE event_id=?", (event_id,)
            )

    # ─── Settings ─────────────────────────────────────────────────────────────

    def get_setting(self, key: str, default: str = "") -> str:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT value FROM settings WHERE key=?", (key,)
            ).fetchone()
            return row["value"] if row else default

    def set_setting(self, key: str, value: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

    # Helpers para configuración de resumen
    def get_daily_summary_time(self) -> str:
        """Hora de resumen diario en formato HH:MM (local)."""
        return self.get_setting("daily_summary_time", "08:00")

    def set_daily_summary_time(self, hhmm: str) -> None:
        self.set_setting("daily_summary_time", hhmm)

    def get_weekly_summary_enabled(self) -> bool:
        return self.get_setting("weekly_summary_enabled", "1") == "1"

    def set_weekly_summary_enabled(self, enabled: bool) -> None:
        self.set_setting("weekly_summary_enabled", "1" if enabled else "0")
