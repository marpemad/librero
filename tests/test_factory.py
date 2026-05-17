"""
Smoke test del Strategy Pattern — sin dependencias externas (no llama a Ollama
ni descarga nada). Verifica que el factory selecciona el extractor correcto
para cada tipo de payload.

Ejecutar: python -m tests.test_factory
"""
from src.extractors import ExtractorFactory, IngestionPayload, SourceKind


CASES = [
    (IngestionPayload(SourceKind.TEXT, "qué es la entropía cruzada"),                 "concept"),
    (IngestionPayload(SourceKind.URL,  "https://www.youtube.com/watch?v=dQw4w9WgXcQ"),"media"),
    (IngestionPayload(SourceKind.URL,  "https://youtu.be/dQw4w9WgXcQ"),               "media"),
    (IngestionPayload(SourceKind.URL,  "https://open.spotify.com/episode/abc123"),    "media"),
    (IngestionPayload(SourceKind.URL,  "https://x.com/jack/status/20"),               "social"),
    (IngestionPayload(SourceKind.URL,  "https://twitter.com/jack/status/20"),         "social"),
    (IngestionPayload(SourceKind.URL,  "https://docs.google.com/spreadsheets/d/abc"), "gsheets"),
    (IngestionPayload(SourceKind.URL,  "https://arxiv.org/abs/1706.03762"),           "web"),
    (IngestionPayload(SourceKind.FILE, "/tmp/foo.pdf",  metadata={"filename": "foo.pdf"}),  "document"),
    (IngestionPayload(SourceKind.FILE, "/tmp/foo.epub", metadata={"filename": "foo.epub"}), "document"),
    (IngestionPayload(SourceKind.FILE, "/tmp/foo.csv",  metadata={"filename": "foo.csv"}),  "document"),
    # Fase 1: voz
    (IngestionPayload(SourceKind.FILE, "/tmp/voice.ogg", metadata={"filename": "voice.ogg", "telegram_voice": True}), "voice"),
    (IngestionPayload(SourceKind.FILE, "/tmp/audio.m4a", metadata={"filename": "audio.m4a"}), "voice"),
    (IngestionPayload(SourceKind.FILE, "/tmp/audio.mp3", metadata={"filename": "audio.mp3"}), "voice"),
]


def main() -> None:
    print(f"Extractores registrados: {ExtractorFactory.registered()}\n")
    failures = 0
    for payload, expected in CASES:
        try:
            chosen = ExtractorFactory.select(payload).name
        except Exception as e:
            chosen = f"<error: {e}>"
        ok = chosen == expected
        flag = "✅" if ok else "❌"
        print(f"{flag} {payload.kind.name:5} {payload.raw[:50]:50} → {chosen:10} (esperado: {expected})")
        if not ok:
            failures += 1
    print()
    if failures:
        raise SystemExit(f"{failures} casos fallaron")
    print("Todos los casos pasaron 🎉")


if __name__ == "__main__":
    main()
