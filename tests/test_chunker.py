"""Smoke tests del chunker — sin dependencias externas."""
from src.llm.chunker import make_plan


def test_short_text_single_pass():
    plan = make_plan("Texto corto." * 50)  # ~600 chars
    assert plan.method == "single"
    assert plan.n_chunks == 1


def test_long_text_no_sections_size_split():
    text = "Lorem ipsum dolor sit amet. " * 2000  # ~56k chars
    plan = make_plan(text)
    assert plan.method == "by_size"
    assert plan.n_chunks > 1


def test_long_text_with_sections():
    sections = [f"# Sección {i}\n\n{'contenido ' * 1000}\n\n" for i in range(8)]
    text = "".join(sections)
    plan = make_plan(text)
    assert plan.method == "by_sections"
    assert plan.n_chunks >= 4


def test_book_chapters():
    chapters = [f"Capítulo {i}\n\n{'palabras ' * 1500}\n\n" for i in range(6)]
    text = "".join(chapters)
    plan = make_plan(text)
    assert plan.method == "by_sections"


def test_estimated_minutes_grows_with_chunks():
    short_plan = make_plan("a" * 100)
    long_plan = make_plan("\n\n".join([f"# Sec {i}\n\n{'x ' * 5000}" for i in range(10)]))
    assert long_plan.estimated_minutes > short_plan.estimated_minutes


if __name__ == "__main__":
    funcs = [
        test_short_text_single_pass, test_long_text_no_sections_size_split,
        test_long_text_with_sections, test_book_chapters,
        test_estimated_minutes_grows_with_chunks,
    ]
    for f in funcs:
        try:
            f()
            print(f"✅ {f.__name__}")
        except AssertionError as e:
            print(f"❌ {f.__name__}: {e}")
            raise
    print(f"\n{len(funcs)} tests pasados 🎉")
