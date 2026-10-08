"""Vector stores. Same Protocol for the in-memory store and for PgVector."""

import heapq
import math
from collections.abc import Sequence
from typing import Protocol

from vendas_agent.models import Chunk, Retrieved


class VectorStore(Protocol):
    def clear(self) -> None: ...

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None: ...

    def search(self, vector: Sequence[float], k: int) -> list[Retrieved]: ...


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class InMemoryVectorStore:
    """Exact (brute-force) cosine search. Fine for a few thousand chunks."""

    def __init__(self) -> None:
        self._items: list[tuple[Chunk, list[float]]] = []

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        self._items.extend((c, list(v)) for c, v in zip(chunks, vectors, strict=True))

    def search(self, vector: Sequence[float], k: int) -> list[Retrieved]:
        if k <= 0:
            return []
        scored = (Retrieved(chunk, cosine(vector, vec)) for chunk, vec in self._items)
        return heapq.nlargest(k, scored, key=lambda r: r.score)
