import pytest

from vendas_agent.chunking import chunk_document, split_body, split_sections
from vendas_agent.models import Document


@pytest.mark.parametrize(
    ("max_chars", "overlap"),
    [(0, 0), (-5, 0), (100, 100), (100, 150), (100, -1)],
)
def test_invalid_parameters_raise(max_chars: int, overlap: int) -> None:
    with pytest.raises(ValueError, match="max_chars|overlap"):
        chunk_document(Document("a.md", "texto"), max_chars=max_chars, overlap=overlap)


def test_empty_document_has_no_chunks() -> None:
    assert chunk_document(Document("a.md", "   \n\n ")) == []


def test_sections_use_heading_as_title() -> None:
    text = "intro\n\n# Planos\n\nbarato\n\n## Pro\n\ncaro"
    assert split_sections(text) == [("Geral", "intro"), ("Planos", "barato"), ("Pro", "caro")]


def test_short_paragraphs_are_packed_together() -> None:
    assert split_body("um\n\ndois\n\ntres", max_chars=100, overlap=10) == ["um\n\ndois\n\ntres"]


def test_long_paragraph_is_sliced_with_overlap() -> None:
    words = " ".join(f"palavra{i}" for i in range(60))
    pieces = split_body(words, max_chars=100, overlap=20)
    assert len(pieces) > 1
    assert all(len(p) <= 100 for p in pieces)
    # overlap: the end of a piece reappears at the start of the next one
    assert pieces[0].split()[-1] in pieces[1]
    # nothing is lost
    assert {w for p in pieces for w in p.split()} == set(words.split())


def test_chunks_keep_source_section_and_position() -> None:
    doc = Document("planos.md", "# A\n\nconteudo a\n\n# B\n\nconteudo b")
    chunks = chunk_document(doc)
    assert [(c.source, c.section, c.position) for c in chunks] == [
        ("planos.md", "A", 0),
        ("planos.md", "B", 1),
    ]
    assert chunks[0].citation == "planos.md › A"
