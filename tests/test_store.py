import pytest

from vendas_agent.models import Chunk
from vendas_agent.store import InMemoryVectorStore, cosine


def _chunk(name: str) -> Chunk:
    return Chunk(text=name, source="x.md", section=name, position=0)


def test_cosine_basics() -> None:
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_search_returns_best_first_and_respects_k() -> None:
    store = InMemoryVectorStore()
    store.add([_chunk("a"), _chunk("b"), _chunk("c")], [[1.0, 0.0], [0.7, 0.7], [0.0, 1.0]])
    hits = store.search([1.0, 0.0], k=2)
    assert [h.chunk.section for h in hits] == ["a", "b"]
    assert hits[0].score >= hits[1].score


def test_search_with_non_positive_k_is_empty() -> None:
    store = InMemoryVectorStore()
    store.add([_chunk("a")], [[1.0]])
    assert store.search([1.0], k=0) == []


def test_add_validates_lengths_and_clear_empties() -> None:
    store = InMemoryVectorStore()
    with pytest.raises(ValueError, match="same length"):
        store.add([_chunk("a")], [])
    store.add([_chunk("a")], [[1.0]])
    store.clear()
    assert len(store) == 0
