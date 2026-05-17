"""
Formateadores de eventos y agenda para mensajes de Telegram.
"""
from __future__ import annotations

from datetime import datetime, timezone


def _fmt_dt(event: dict) -> str:
    """Devuelve una representación legible del inicio del evento."""
    start = event.get("start", {})
    if "dateTime" in start:
        dt = datetime.fromisoformat(start["dateTime"])
        return dt.strftime("%H:%M")
    if "date" in start:
        return "Todo el día"
    return "?"


def _fmt_date(event: dict) -> str:
    start = event.get("start", {})
    if "dateTime" in start:
        dt = datetime.fromisoformat(start["dateTime"])
        return dt.strftime("%d %b")
    if "date" in start:
        return datetime.fromisoformat(start["date"]).strftime("%d %b")
    return ""


def format_event(event: dict, show_date: bool = False, marked: bool = False) -> str:
    """Formatea un evento individual como línea de texto."""
    title = event.get("summary", "(Sin título)")
    time_str = _fmt_dt(event)
    date_str = f" · {_fmt_date(event)}" if show_date else ""
    location = event.get("location", "")
    loc_str = f"\n   📍 {location}" if location else ""
    desc = event.get("description", "")
    desc_str = f"\n   📝 {desc[:80]}{'…' if len(desc) > 80 else ''}" if desc else ""
    bell = " 🔔" if marked else ""
    return f"🕐 *{time_str}*{date_str} — {title}{bell}{loc_str}{desc_str}"


def format_agenda(
    events: list[dict],
    title: str,
    marked_ids: set[str] | None = None,
    show_date: bool = False,
) -> str:
    """Formatea una lista de eventos como agenda completa."""
    marked_ids = marked_ids or set()
    if not events:
        return f"*{title}*\n\n_No hay eventos programados._"

    lines = [f"*{title}*\n"]
    for ev in events:
        marked = ev.get("id", "") in marked_ids
        lines.append(format_event(ev, show_date=show_date, marked=marked))
    return "\n".join(lines)


def format_event_notification(event: dict, minutes_before: int) -> str:
    """Mensaje de notificación de evento próximo."""
    title = event.get("summary", "(Sin título)")
    time_str = _fmt_dt(event)
    location = event.get("location", "")
    loc_str = f"\n📍 {location}" if location else ""

    if minutes_before >= 60:
        hours = minutes_before // 60
        mins = minutes_before % 60
        when = f"en {hours}h" + (f" {mins}min" if mins else "")
    else:
        when = f"en {minutes_before} min"

    return (
        f"🔔 *Recordatorio* — {when}\n\n"
        f"📅 *{title}*\n"
        f"🕐 {time_str}{loc_str}"
    )
