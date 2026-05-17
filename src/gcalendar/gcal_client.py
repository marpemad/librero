"""
Cliente Google Calendar API con OAuth2.

Flujo de autenticación (primera vez):
  1. /calauth  → bot genera URL y arranca servidor local temporal
  2. Usuario abre URL en el navegador y acepta permisos
  3. Google redirige a localhost → bot captura el token y lo guarda en GCAL_TOKEN_PATH
  4. Todas las llamadas siguientes usan el token guardado (se renueva automáticamente)
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)

SCOPES = ["https://www.googleapis.com/auth/calendar"]


class GoogleCalendarClient:
    def __init__(self, client_id: str, client_secret: str, token_path: Path) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_path = token_path
        self._service: Any = None

    # ─── Auth ─────────────────────────────────────────────────────────────────

    def is_authenticated(self) -> bool:
        return self.token_path.exists()

    def get_auth_url(self, redirect_port: int = 8765) -> tuple[str, Any]:
        """Devuelve (url, flow). Guarda el flow para llamar a exchange_code()."""
        from google_auth_oauthlib.flow import Flow  # type: ignore[import-untyped]

        client_config = {
            "installed": {
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [f"http://localhost:{redirect_port}"],
            }
        }
        flow = Flow.from_client_config(
            client_config,
            scopes=SCOPES,
            redirect_uri=f"http://localhost:{redirect_port}",
        )
        url, _ = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            prompt="consent",
        )
        return url, flow

    def exchange_code(self, flow: Any, code: str) -> None:
        """Intercambia el código de autorización por tokens y los guarda."""
        import json
        flow.fetch_token(code=code)
        creds = flow.credentials
        self.token_path.parent.mkdir(parents=True, exist_ok=True)
        self.token_path.write_text(
            json.dumps(
                {
                    "token": creds.token,
                    "refresh_token": creds.refresh_token,
                    "token_uri": creds.token_uri,
                    "client_id": creds.client_id,
                    "client_secret": creds.client_secret,
                    "scopes": list(creds.scopes or SCOPES),
                    "expiry": creds.expiry.isoformat() if creds.expiry else None,
                }
            )
        )
        self._service = None  # fuerza reconstrucción

    def _get_service(self) -> Any:
        if self._service is not None:
            return self._service

        import json
        from google.oauth2.credentials import Credentials  # type: ignore[import-untyped]
        from google.auth.transport.requests import Request  # type: ignore[import-untyped]
        from googleapiclient.discovery import build  # type: ignore[import-untyped]

        data = json.loads(self.token_path.read_text())
        creds = Credentials(
            token=data["token"],
            refresh_token=data.get("refresh_token"),
            token_uri=data.get("token_uri", "https://oauth2.googleapis.com/token"),
            client_id=data["client_id"],
            client_secret=data["client_secret"],
            scopes=data.get("scopes", SCOPES),
        )
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            # Guarda el token renovado
            data["token"] = creds.token
            data["expiry"] = creds.expiry.isoformat() if creds.expiry else None
            self.token_path.write_text(json.dumps(data))

        self._service = build("calendar", "v3", credentials=creds, cache_discovery=False)
        return self._service

    # ─── Listado ──────────────────────────────────────────────────────────────

    async def list_events(
        self,
        time_min: datetime | None = None,
        time_max: datetime | None = None,
        max_results: int = 20,
        calendar_id: str = "primary",
    ) -> list[dict]:
        if time_min is None:
            time_min = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        if time_max is None:
            time_max = time_min + timedelta(days=1)

        def _call() -> list[dict]:
            svc = self._get_service()
            result = (
                svc.events()
                .list(
                    calendarId=calendar_id,
                    timeMin=time_min.isoformat(),
                    timeMax=time_max.isoformat(),
                    maxResults=max_results,
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            )
            return result.get("items", [])

        return await asyncio.to_thread(_call)

    async def get_event(self, event_id: str, calendar_id: str = "primary") -> dict:
        def _call() -> dict:
            return self._get_service().events().get(calendarId=calendar_id, eventId=event_id).execute()

        return await asyncio.to_thread(_call)

    # ─── Creación ─────────────────────────────────────────────────────────────

    async def create_event(
        self,
        title: str,
        start: datetime,
        end: datetime | None = None,
        description: str = "",
        calendar_id: str = "primary",
    ) -> dict:
        if end is None:
            end = start + timedelta(hours=1)

        def _to_gcal_dt(dt: datetime) -> dict:
            if dt.hour == 0 and dt.minute == 0 and dt.second == 0:
                # Evento de día completo
                return {"date": dt.strftime("%Y-%m-%d")}
            return {"dateTime": dt.isoformat(), "timeZone": "Europe/Madrid"}

        body = {
            "summary": title,
            "description": description,
            "start": _to_gcal_dt(start),
            "end": _to_gcal_dt(end),
        }

        def _call() -> dict:
            return self._get_service().events().insert(calendarId=calendar_id, body=body).execute()

        return await asyncio.to_thread(_call)

    # ─── Edición ──────────────────────────────────────────────────────────────

    async def update_event(
        self,
        event_id: str,
        title: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        description: str | None = None,
        calendar_id: str = "primary",
    ) -> dict:
        def _call() -> dict:
            svc = self._get_service()
            event = svc.events().get(calendarId=calendar_id, eventId=event_id).execute()
            if title is not None:
                event["summary"] = title
            if description is not None:
                event["description"] = description
            if start is not None:
                event["start"] = {"dateTime": start.isoformat(), "timeZone": "Europe/Madrid"}
            if end is not None:
                event["end"] = {"dateTime": end.isoformat(), "timeZone": "Europe/Madrid"}
            return svc.events().update(calendarId=calendar_id, eventId=event_id, body=event).execute()

        return await asyncio.to_thread(_call)

    # ─── Borrado ──────────────────────────────────────────────────────────────

    async def delete_event(self, event_id: str, calendar_id: str = "primary") -> None:
        def _call() -> None:
            self._get_service().events().delete(calendarId=calendar_id, eventId=event_id).execute()

        await asyncio.to_thread(_call)

    # ─── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def event_start_dt(event: dict) -> datetime | None:
        """Devuelve el start del evento como datetime con tz UTC (o None si todo-día)."""
        start = event.get("start", {})
        if "dateTime" in start:
            dt = datetime.fromisoformat(start["dateTime"])
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        return None  # todo-día

    @staticmethod
    def event_end_dt(event: dict) -> datetime | None:
        end = event.get("end", {})
        if "dateTime" in end:
            dt = datetime.fromisoformat(end["dateTime"])
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        return None
