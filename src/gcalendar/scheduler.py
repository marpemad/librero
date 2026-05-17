"""
Scheduler de calendario basado en APScheduler (AsyncIOScheduler).

Jobs registrados:
  - poll_notifications   — cada 5 min: revisa eventos marcados y dispara recordatorios
  - daily_summary        — configurable (default 08:00): agenda del día
  - weekly_summary       — lunes a la misma hora: agenda de la semana

Uso desde main.py:
    cal_scheduler = CalendarScheduler(gcal, notify_db, bot, chat_id)
    cal_scheduler.start()
    ...
    cal_scheduler.stop()  # en shutdown
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Callable, Awaitable

from src.utils.logger import get_logger

if TYPE_CHECKING:
    from telegram import Bot
    from .gcal_client import GoogleCalendarClient
    from .notify_db import NotifyDB

logger = get_logger(__name__)


class CalendarScheduler:
    def __init__(
        self,
        gcal: "GoogleCalendarClient",
        notify_db: "NotifyDB",
        bot: "Bot",
        chat_id: int,
    ) -> None:
        self.gcal = gcal
        self.notify_db = notify_db
        self.bot = bot
        self.chat_id = chat_id
        self._scheduler = None
        self._send_message: Callable[[str], Awaitable[None]] | None = None

    def _get_scheduler(self):
        if self._scheduler is None:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler  # type: ignore[import-untyped]
            self._scheduler = AsyncIOScheduler(timezone="Europe/Madrid")
        return self._scheduler

    async def _send(self, text: str) -> None:
        await self.bot.send_message(
            chat_id=self.chat_id,
            text=text,
            parse_mode="Markdown",
        )

    # ─── Arranque ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        sched = self._get_scheduler()

        # Poll de notificaciones cada 5 minutos
        sched.add_job(
            self._poll_notifications,
            trigger="interval",
            minutes=5,
            id="poll_notifications",
            replace_existing=True,
            misfire_grace_time=120,
        )

        # Resumen diario
        self._reschedule_daily_summary()

        # Resumen semanal (lunes)
        self._reschedule_weekly_summary()

        sched.start()
        logger.info("CalendarScheduler arrancado")

    def stop(self) -> None:
        if self._scheduler and self._scheduler.running:
            self._scheduler.shutdown(wait=False)

    # ─── Resumen diario ───────────────────────────────────────────────────────

    def _reschedule_daily_summary(self) -> None:
        hhmm = self.notify_db.get_daily_summary_time()
        try:
            h, m = map(int, hhmm.split(":"))
        except ValueError:
            h, m = 8, 0

        sched = self._get_scheduler()
        sched.add_job(
            self._send_daily_summary,
            trigger="cron",
            hour=h,
            minute=m,
            id="daily_summary",
            replace_existing=True,
            misfire_grace_time=300,
        )
        logger.info("Resumen diario programado a las %s", hhmm)

    def update_daily_summary_time(self, hhmm: str) -> None:
        self.notify_db.set_daily_summary_time(hhmm)
        self._reschedule_daily_summary()

    # ─── Resumen semanal ──────────────────────────────────────────────────────

    def _reschedule_weekly_summary(self) -> None:
        if not self.notify_db.get_weekly_summary_enabled():
            return
        hhmm = self.notify_db.get_daily_summary_time()
        try:
            h, m = map(int, hhmm.split(":"))
        except ValueError:
            h, m = 8, 0

        sched = self._get_scheduler()
        sched.add_job(
            self._send_weekly_summary,
            trigger="cron",
            day_of_week="mon",
            hour=h,
            minute=m,
            id="weekly_summary",
            replace_existing=True,
            misfire_grace_time=300,
        )

    # ─── Jobs async ───────────────────────────────────────────────────────────

    async def _poll_notifications(self) -> None:
        """Revisa los eventos marcados y envía recordatorio si toca."""
        if not self.gcal.is_authenticated():
            return
        marked = self.notify_db.get_all_marked()
        if not marked:
            return

        now = datetime.now(timezone.utc)
        for row in marked:
            event_id = row["event_id"]
            minutes_before = row["minutes_before"]
            start_iso = row["start_iso"]

            try:
                start_dt = datetime.fromisoformat(start_iso)
            except ValueError:
                continue

            if start_dt.tzinfo is None:
                start_dt = start_dt.replace(tzinfo=timezone.utc)

            trigger_at = start_dt - timedelta(minutes=minutes_before)
            window_start = now - timedelta(minutes=5)

            if window_start <= trigger_at <= now:
                try:
                    event = await self.gcal.get_event(event_id, row["calendar_id"])
                    from .notifier import format_event_notification
                    msg = format_event_notification(event, minutes_before)
                    await self._send(msg)
                    self.notify_db.mark_as_notified(event_id)
                    logger.info("Notificación enviada para evento %s", event_id)
                except Exception as exc:
                    logger.warning("Error al notificar evento %s: %s", event_id, exc)

    async def _send_daily_summary(self) -> None:
        if not self.gcal.is_authenticated():
            return
        try:
            now_local = datetime.now()
            day_start = now_local.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
            day_end = day_start + timedelta(days=1)
            events = await self.gcal.list_events(day_start, day_end, max_results=20)
            marked_ids = {r["event_id"] for r in self.notify_db.get_all_marked()}

            from .notifier import format_agenda
            weekday_es = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
            day_name = weekday_es[now_local.weekday()]
            title = f"📅 Agenda de hoy — {day_name} {now_local.strftime('%d/%m')}"
            msg = format_agenda(events, title, marked_ids)
            await self._send(msg)
        except Exception as exc:
            logger.warning("Error al enviar resumen diario: %s", exc)

    async def _send_weekly_summary(self) -> None:
        if not self.gcal.is_authenticated():
            return
        try:
            now_local = datetime.now()
            week_start = now_local.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
            week_end = week_start + timedelta(days=7)
            events = await self.gcal.list_events(week_start, week_end, max_results=50)
            marked_ids = {r["event_id"] for r in self.notify_db.get_all_marked()}

            from .notifier import format_agenda
            title = f"📅 Agenda de la semana — {now_local.strftime('%d/%m')} al {(now_local + timedelta(days=6)).strftime('%d/%m')}"
            msg = format_agenda(events, title, marked_ids, show_date=True)
            await self._send(msg)
        except Exception as exc:
            logger.warning("Error al enviar resumen semanal: %s", exc)

    # ─── Trigger manual ───────────────────────────────────────────────────────

    async def trigger_daily_summary(self) -> None:
        await self._send_daily_summary()

    async def trigger_weekly_summary(self) -> None:
        await self._send_weekly_summary()
