"""Embedders. ``HashingEmbedder`` works offline (great for tests and demos);
``OpenAIEmbedder`` gives real semantic embeddings."""

from collections.abc import Sequence
import hashlib
import math
import re
import unicodedata
from typing import Any, Protocol


class Embedder(Protocol):
    @property
    def dimension(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a o as os e em no na nos nas um uma uns umas de da do das dos para por com que "
    "se ao aos eu me meu minha voce voces vc tem ter ha eh sao ser como mais ou ja "
    "foi sua seu suas seus isso essa esse esta este".split()
)


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def tokenize(text: str) -> list[str]:
    """Lowercase, strip accents, drop stopwords and apply a crude stemmer
    (singularize + 6-char prefix) so 'integração'/'integrações' match."""
    tokens = []
    for raw in _TOKEN.findall(strip_accents(text).lower()):
        if raw in _STOPWORDS or (len(raw) < 2 and not raw.isdigit()):
            continue
        if len(raw) > 3 and raw.endswith("s"):
            raw = raw[:-1]
        tokens.append(raw[:6])
    return tokens


class HashingEmbedder:
    """Bag-of-words feature hashing. Deterministic, dependency-free, no network."""

    def __init__(self, dimension: int = 512) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            sign = 1.0 if (value >> 63) & 1 else -1.0
            vector[value % self._dimension] += sign
        norm = math.sqrt(sum(x * x for x in vector))
        if norm == 0:
            return vector
        return [x / norm for x in vector]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]


class OpenAIEmbedder:
    """Thin wrapper over the OpenAI embeddings API (``pip install .[openai]``)."""

    _DIMENSIONS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}

    def __init__(self, model: str = "text-embedding-3-small", client: Any | None = None) -> None:
        if model not in self._DIMENSIONS:
            raise ValueError(f"unknown embedding model {model!r}")
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self._client = client
        self._model = model

    @property
    def dimension(self) -> int:
        return self._DIMENSIONS[self._model]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(model=self._model, input=list(texts))
        return [list(item.embedding) for item in response.data]
