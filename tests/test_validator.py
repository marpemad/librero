"""Smoke tests del validador estructural — sin red, sin LLM."""
from src.llm.validator import validate_note


VALID_NOTE = """# Mi Concepto

**Etiquetas:** #aprendizaje #ia #productividad
**Relaciones:** [[Memoria]], [[Atención]], [[Cognición]]

## Resumen
Esto es un resumen de prueba con suficiente contenido.

## Puntos Clave
- Punto uno
- Punto dos
- Punto tres
- Punto cuatro

## Análisis Profundo
Párrafo uno.

Párrafo dos.

Párrafo tres.
"""


def test_valid():
    r = validate_note(VALID_NOTE)
    assert r.ok, f"Debería ser válida. Problemas: {r.problems}"
    assert r.title == "Mi Concepto"
    assert r.n_tags == 3
    assert r.n_relations == 3


def test_missing_h1():
    r = validate_note(VALID_NOTE.replace("# Mi Concepto", "Mi Concepto"))
    assert not r.ok
    assert any("H1" in p for p in r.problems)


def test_too_few_tags():
    bad = VALID_NOTE.replace("#aprendizaje #ia #productividad", "#solo")
    r = validate_note(bad)
    assert not r.ok
    assert any("etiqueta" in p.lower() for p in r.problems)


def test_too_few_relations():
    bad = VALID_NOTE.replace(
        "[[Memoria]], [[Atención]], [[Cognición]]", "[[Solo]]"
    )
    r = validate_note(bad)
    assert not r.ok
    assert any("wikilink" in p.lower() for p in r.problems)


def test_missing_section():
    bad = VALID_NOTE.replace("## Análisis Profundo", "## Otro")
    r = validate_note(bad)
    assert not r.ok
    assert any("Análisis Profundo" in p for p in r.problems)


def test_forbidden_chars_in_title():
    bad = VALID_NOTE.replace("# Mi Concepto", "# Concepto: con dos puntos / y barra")
    r = validate_note(bad)
    assert not r.ok
    assert any("prohibidos" in p for p in r.problems)


if __name__ == "__main__":
    funcs = [
        test_valid, test_missing_h1, test_too_few_tags,
        test_too_few_relations, test_missing_section, test_forbidden_chars_in_title,
    ]
    for f in funcs:
        try:
            f()
            print(f"✅ {f.__name__}")
        except AssertionError as e:
            print(f"❌ {f.__name__}: {e}")
            raise
    print(f"\n{len(funcs)} tests pasados 🎉")
