# 📚 Librero

Bot personal de Telegram que ingiere contenido de cualquier fuente —YouTube, podcasts, PDFs, EPUBs, webs, papers, imágenes, mensajes de voz, conceptos estructurados— y lo sintetiza con un **LLM** (Ollama local o Google Gemini) en **notas Markdown ricas e interconectadas** dentro de tu Vault de Obsidian.

Incluye además **integración con Google Calendar** (leer, crear, editar eventos; notificaciones programadas; resúmenes diarios y semanales), **tablón Kanban de tareas** y **listas curadas**.

```
Telegram → Router → Extractor → LLM (Ollama|Gemini) → Markdown enriquecido en Obsidian
                                       ↑
                            Vault Intel: embeddings + dedup + wikilinks reales
```

---

## ✨ Qué sabe hacer

### 📥 Ingesta de contenido

| Entrada                          | Extractor              | Estrategia                                       |
|----------------------------------|------------------------|--------------------------------------------------|
| Texto libre (concepto)           | `ConceptExtractor`     | `ddgs` (DuckDuckGo) + scrape top-3               |
| Concepto estructurado            | `ConceptExtractor`     | `Tipo. Autor. Título` → búsqueda afilada + preview |
| YouTube                          | `MediaExtractor`       | Transcript API → fallback Whisper                |
| Spotify, podcasts, RSS audio     | `MediaExtractor`       | yt-dlp + faster-whisper                          |
| Twitter / X                      | `SocialExtractor`      | yt-dlp metadata                                  |
| Google Sheets (público)          | `GoogleSheetsExtractor`| Export URL → CSV                                 |
| Webs / papers HTML               | `WebScraperExtractor`  | trafilatura + Open Graph + JSON-LD               |
| 🎤 Mensajes de voz / audio       | `VoiceExtractor`       | faster-whisper                                   |
| PDF / DOCX / EPUB                | `DocumentExtractor`    | PyMuPDF / python-docx / EbookLib                 |
| 🖼 Imágenes                       | `DocumentExtractor`    | Vision LLM (`qwen2.5vl`) + OCR Tesseract         |
| CSV / TXT / MD                   | `DocumentExtractor`    | pandas / lectura directa                         |

### 🧠 Síntesis LLM intercambiable

Eliges el proveedor en `.env` con `LLM_PROVIDER`:

| Proveedor | Pros | Contras |
|---|---|---|
| **`ollama`** | 100% local, privado, gratis | Necesita GPU/Apple Silicon decente; modelos grandes son lentos en CPU |
| **`gemini`** | Rápido, capaz, ligero en hardware | Requiere API key; los datos pasan por Google |

La capa de prompts, chunking, validación y post-proceso es idéntica: solo cambia el cliente que llama al modelo.

### 📅 Google Calendar

- `/cal [N]` — agenda de hoy o próximos N días
- `/calweek` — agenda de los próximos 7 días
- `/caladd <título> [fecha] [hora] [min]` — añadir evento con fecha en lenguaje natural (`mañana 10:30`, `viernes`, `2026-06-15`…)
- `/calday` — resumen del día ahora
- `/calconfig HH:MM | on | off` — configurar horario del resumen y activar/desactivar el semanal
- Botones por evento: **🔔 Avisarme** (notif. 15 min antes) · **✏️ Editar** · **🗑 Borrar**
- **Scheduler automático**: cada 5 min revisa avisos pendientes; resumen diario a la hora que elijas; resumen semanal los lunes.

### 📋 Tareas Kanban

Tablón Markdown en el Vault (`Tablón.md`), editable manualmente y desde el bot:

```
task comprar café
tarea: llamar al fontanero
```

`/tasks`, `/done T003`, `/wip T003`, `/todo T003`, `/del T003`.

### 📝 Listas curadas

`/note libros Asimov, Marcos Vázquez` · `/note series` · `/notedel libros Asimov`.

### 🔍 Búsqueda y lectura

- `/search <texto>` — semántica (embeddings)
- `/find <palabras>` — literal AND (sin LLM)
- `/read <id>` — abre una nota concreta
- `/random [tipo]` — nota al azar
- `/weekly [tipo] [llm]` — resumen de la última semana, con meta-resumen LLM opcional

### 🧭 Vault Intel (embeddings)

Si tienes `nomic-embed-text` en Ollama:

- **Dedup** automático cuando subes algo parecido a una nota existente
- **Wikilinks reales** entre notas (sin enlaces rotos)
- **Tags coherentes** (reutiliza los que ya tienes)

Funciona también si solo usas Gemini para síntesis: el embedder es independiente del LLM principal.

### 🛡 Robustez

- Cola serializada (`/queue`) para que el LLM no se sature
- Cache por hash de PDFs / audios / imágenes
- Validación de la nota generada + reintento correctivo
- Chunking map-reduce automático para textos >20k chars
- Confirmación interactiva antes de procesar contenido pesado

---

## 🚀 Instalación

### Opción A — Docker (recomendado para servidores)

```bash
git clone https://github.com/tu_usuario/librero
cd librero
cp .env.example .env
nano .env          # rellenar TELEGRAM_BOT_TOKEN, ALLOWED_USER_IDS, etc.

# Modo cloud (Gemini, ligero):
docker compose up -d bot

# Modo local (con Ollama):
docker compose --profile local up -d
```

Los datos persistentes viven en `./data/` (DBs, token, Vault). Mantén esa carpeta entre actualizaciones.

### Opción B — UmbrelOS 1.7.2+

Ver [`umbrel/README.md`](umbrel/README.md) para instrucciones detalladas. Resumen:

1. Construir y publicar la imagen en GHCR / Docker Hub (multiarch amd64+arm64).
2. **App Store → Community App Stores → Add** y pega la URL de tu fork.
3. Instalar `Librero` y configurar `.env` desde el host de Umbrel.

### Opción C — Instalación nativa (desarrollo)

```bash
# 1. Dependencias del sistema (Mac M1):
brew install ffmpeg tesseract
brew install ollama && ollama serve &   # solo si LLM_PROVIDER=ollama

# Modelos Ollama (solo si los vas a usar):
ollama pull qwen3:14b              # síntesis principal
ollama pull nomic-embed-text       # /search, dedup, wikilinks reales
ollama pull qwen2.5vl:7b           # opcional: visión de imágenes

# 2. Python:
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Configuración:
cp .env.example .env && nano .env

# 4. Arrancar:
python main.py
```

---

## 🎛 Configuración (`.env`)

```bash
# ─── Telegram ───
TELEGRAM_BOT_TOKEN=...            # BotFather
ALLOWED_USER_IDS=123456789        # tu user_id (separados por comas)

# ─── LLM: elige uno ───
LLM_PROVIDER=gemini               # "ollama" o "gemini"

# Si LLM_PROVIDER=ollama:
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen3:14b

# Si LLM_PROVIDER=gemini:
GEMINI_API_KEY=...                # https://aistudio.google.com/apikey
GEMINI_MODEL=gemini-2.0-flash     # o gemini-2.5-pro

# ─── Obsidian ───
OBSIDIAN_VAULT_PATH=/path/al/Vault
OBSIDIAN_INBOX_FOLDER=00_Inbox

# ─── Google Calendar (opcional) ───
GCAL_CLIENT_ID=...
GCAL_CLIENT_SECRET=...
GCAL_DAILY_SUMMARY_TIME=08:00     # hora del resumen diario automático

# ─── Modelos auxiliares (opcionales) ───
EMBED_MODEL=nomic-embed-text      # vault intel (vacío = desactivar)
VISION_MODEL=qwen2.5vl:7b         # imágenes (vacío = solo OCR)
WHISPER_MODEL=small               # tiny|base|small|medium|large-v3
```

Ver `.env.example` para la lista completa.

---

## 📅 Setup de Google Calendar

1. Ve a [console.cloud.google.com](https://console.cloud.google.com)
2. **APIs & Services → Enable APIs** → activa "Google Calendar API"
3. **Credentials → Create credentials → OAuth client ID** → tipo *Desktop app*
4. Copia el Client ID y Client Secret al `.env`:
   ```bash
   GCAL_CLIENT_ID=tu_client_id.apps.googleusercontent.com
   GCAL_CLIENT_SECRET=tu_client_secret
   ```
5. En Telegram: `/calauth` → abre el navegador → acepta permisos → listo.

El token se guarda automáticamente en `data/config/gcal_token.json` (Docker) o `~/.config/librero/gcal_token.json` (nativo).

---

## ⌨️ Comandos del bot

| Comando | Qué hace |
|---|---|
| `/start` · `/help` | Bienvenida y todos los comandos |
| `/status` · `/queue` · `/cancel <id>` | Estado, cola de síntesis, cancelar |
| `/search <texto>` · `/find <palabras>` · `/read <id>` | Buscar y leer notas |
| `/random [tipo]` · `/weekly [tipo] [llm]` | Notas aleatorias y resúmenes |
| `/tasks` · `/done <id>` · `/wip <id>` · `/todo <id>` · `/del <id>` | Tablón Kanban |
| `/note [tipo] [items]` · `/notedel <tipo> <item>` | Listas curadas |
| `/calauth` | Conectar Google Calendar (una sola vez) |
| `/cal [N]` · `/calweek` · `/caladd ...` · `/calday` · `/calconfig` | Calendario |

Y al margen de los comandos, mándale:
- Cualquier texto, URL o archivo → genera nota
- Mensaje de voz → transcribe + genera nota
- Foto → describe con visión + OCR + genera nota
- `task <descripción>` o `tarea: <desc>` → añade al tablón

---

## 📐 Arquitectura

```
librero/
├── main.py                          # Bot Telegram + cola + handlers
├── requirements.txt
├── Dockerfile · docker-compose.yml  # Despliegue
├── umbrel/                          # App de UmbrelOS
├── src/
│   ├── config/settings.py
│   ├── router/                      # Texto/URL/FILE → IngestionPayload
│   ├── extractors/                  # Strategy pattern por fuente
│   ├── llm/
│   │   ├── base_client.py           # Lógica compartida + prompts
│   │   ├── ollama_client.py         # Backend Ollama
│   │   ├── gemini_client.py         # Backend Gemini
│   │   ├── chunker.py · validator.py
│   ├── writer/obsidian_writer.py    # Markdown con frontmatter rico
│   ├── queue/                       # Cola FIFO single-worker
│   ├── vault_intel/                 # Embeddings + búsqueda + dedup + digest
│   ├── gcalendar/                   # Google Calendar + scheduler + notifier
│   ├── tasks/                       # Tablón Kanban
│   ├── notes/                       # Listas curadas
│   └── utils/                       # Logger, cache, language, vision, whisper
└── tests/
```

### Añadir un nuevo proveedor LLM

Hereda de `BaseLLMClient` e implementa `_chat()`:

```python
class MiClient(BaseLLMClient):
    provider_name = "mi-proveedor"
    def __init__(self):
        self.model_name = settings.mi_modelo

    async def _chat(self, system: str, user: str, num_predict: int) -> str:
        # ... llama a tu API
        return self._post_process(respuesta)
```

Y añade el caso en `make_llm_client()` (`src/llm/__init__.py`).

---

## 🧪 Tests offline

```bash
PYTHONPATH=. python tests/test_factory.py
PYTHONPATH=. python tests/test_chunker.py
PYTHONPATH=. python tests/test_validator.py
```

---

## 🛟 Operativa

- **`/cancel`** solo cancela jobs `PENDING`. Lo que ya se está procesando no se aborta a media síntesis.
- **`yt-dlp` envejece rápido**: si YouTube empieza a fallar con `HTTP 400`, `pip install --upgrade yt-dlp`.
- **Notas con `⚠`**: filtra `needs_review: true` en Obsidian con Dataview para una bandeja de revisión.
- **Caches** en `data/cache/`: borrarlas no rompe nada, solo fuerza re-procesado.
- **Logs Docker**: `docker compose logs -f bot`.
- **Reiniciar tras cambiar `.env`**: `docker compose restart bot`.

---

## 📄 Licencia

Personal — adáptalo a tu gusto.
