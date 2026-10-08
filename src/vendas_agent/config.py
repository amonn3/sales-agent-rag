"""Runtime settings, read from environment variables (never hard-coded secrets)."""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

EmbedderName = Literal["hashing", "openai"]
StoreName = Literal["memory", "pgvector"]

_EMBEDDERS: tuple[str, ...] = ("hashing", "openai")
_STORES: tuple[str, ...] = ("memory", "pgvector")

# Cosine scores live on different scales depending on the embedder.
DEFAULT_MIN_SIMILARITY: dict[str, float] = {"hashing": 0.12, "openai": 0.30}


@dataclass(frozen=True, slots=True)
class Settings:
    anthropic_model: str = "claude-sonnet-5-5"
    embedder: EmbedderName = "hashing"
    openai_embedding_model: str = "text-embedding-3-small"
    store: StoreName = "memory"
    database_url: str = "postgresql://postgres:postgres@localhost:5432/vendas"
    top_k: int = 4
    min_similarity: float | None = None
    max_history_turns: int = 6

    @property
    def effective_min_similarity(self) -> float:
        if self.min_similarity is not None:
            return self.min_similarity
        return DEFAULT_MIN_SIMILARITY[self.embedder]

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        e = os.environ if env is None else env
        embedder = e.get("EMBEDDER", "hashing")
        store = e.get("STORE", "memory")
        if embedder not in _EMBEDDERS:
            raise ValueError(f"EMBEDDER must be one of {_EMBEDDERS}, got {embedder!r}")
        if store not in _STORES:
            raise ValueError(f"STORE must be one of {_STORES}, got {store!r}")
        raw_min = e.get("MIN_SIMILARITY")
        defaults = cls()
        return cls(
            anthropic_model=e.get("ANTHROPIC_MODEL", defaults.anthropic_model),
            embedder=cast(EmbedderName, embedder),
            openai_embedding_model=e.get("OPENAI_EMBEDDING_MODEL", defaults.openai_embedding_model),
            store=cast(StoreName, store),
            database_url=e.get("DATABASE_URL", defaults.database_url),
            top_k=int(e.get("TOP_K", defaults.top_k)),
            min_similarity=float(raw_min) if raw_min else None,
            max_history_turns=int(e.get("MAX_HISTORY_TURNS", defaults.max_history_turns)),
        )
