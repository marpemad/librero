# Librero — App de UmbrelOS

Esta carpeta contiene los ficheros necesarios para empaquetar Librero como app
de UmbrelOS 1.7.2 o superior.

## Estructura

```
umbrel/
├── umbrel-app.yml      # Manifest de la app (metadata, descripción, versión)
├── docker-compose.yml  # Compose adaptado al runtime de Umbrel
└── README.md           # Este fichero
```

## Instalación en Umbrel

### 1. Construir y publicar la imagen Docker

Antes de instalar la app, la imagen `ghcr.io/marcperez/librero:latest` (o tu
fork) debe estar disponible públicamente. Construye y publica desde la raíz del
repo:

```bash
# Multiarch para Pi (arm64) y x86 (amd64)
docker buildx create --use --name librero-builder 2>/dev/null || true
docker buildx build \
    --platform linux/amd64,linux/arm64 \
    --tag ghcr.io/TU_USUARIO/librero:latest \
    --push \
    .
```

### 2. Añadir el app store

En la UI de Umbrel:

1. **App Store → Community App Stores → Add**
2. Pega la URL de tu repo: `https://github.com/TU_USUARIO/librero`
3. La app `Librero` aparecerá en tu app store comunitario.

### 3. Configurar `.env` antes de la primera ejecución

Tras instalar la app, edita el `.env` desde el host de Umbrel:

```bash
ssh umbrel@umbrel.local
cd ~/umbrel/app-data/librero
cp ../../app-source/community/librero/.env.example .env
nano .env
```

Variables mínimas a configurar:

```bash
TELEGRAM_BOT_TOKEN=...      # token de BotFather
ALLOWED_USER_IDS=...        # tu user_id de Telegram

LLM_PROVIDER=gemini         # o "ollama" si has activado el servicio
GEMINI_API_KEY=...          # https://aistudio.google.com/apikey

# Google Calendar (opcional)
GCAL_CLIENT_ID=...
GCAL_CLIENT_SECRET=...
```

Reinicia la app desde la UI y ya está corriendo.

### 4. Tu Vault de Obsidian

Por defecto el Vault vive dentro del data dir de la app:
`~/umbrel/app-data/librero/data/vault`. Apunta tu cliente Obsidian a esta carpeta
(via Syncthing, SMB o lo que prefieras).

Si ya tienes el Vault en otra ubicación del Umbrel, edita el `docker-compose.yml`
y cambia el bind mount para que apunte ahí:

```yaml
volumes:
  - /home/umbrel/MiVault:/data/vault
```

## Notas

- **Modo cloud (Gemini)** es el recomendado para hardware modesto (Pi 5, NUC).
  No necesita GPU ni descargar modelos grandes.
- **Modo local (Ollama)** funciona si tienes un Mac mini M1 o un host potente.
  Recuerda descomentar el servicio `ollama` en el compose y hacer
  `ollama pull qwen3:14b` (o el modelo que prefieras) dentro del contenedor.
- El puerto `8765` solo se usa puntualmente durante `/calauth` (OAuth de Google
  Calendar). El bot no expone una UI web.
