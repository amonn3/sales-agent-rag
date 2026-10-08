"""PgVector-backed store (``pip install .[pgvector]``, needs the pgvector extension).

One table serves many clients: every row carries a ``kb_id`` so each client's
knowledge base is isolated and can be re-indexed independently.
"""

import re
from collections.abc import Sequence

import psycopg

from vendas_agent.models import Chunk, Retrieved

_KB_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")


def to_vector_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vector) + "]"


class PgVectorStore:
    def __init__(self, dsn: str, kb_id: str, dimension: int) -> None:
        if not _KB_ID.match(kb_id):
            raise ValueError("kb_id must be lowercase letters, digits, '-' or '_'")
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._dsn = dsn
        self._kb_id = kb_id
        self._dimension = dimension
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        # ``dimension`` is an int validated above, so formatting it in is safe.
        with psycopg.connect(self._dsn) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS kb_chunks (
                    id        BIGSERIAL PRIMARY KEY,
                    kb_id     TEXT NOT NULL,
                    source    TEXT NOT NULL,
                    section   TEXT NOT NULL,
                    position  INT  NOT NULL,
                    content   TEXT NOT NULL,
                    embedding vector({self._dimension}) NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS kb_chunks_kb_idx ON kb_chunks (kb_id)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS kb_chunks_embedding_idx "
                "ON kb_chunks USING hnsw (embedding vector_cosine_ops)"
            )

    def clear(self) -> None:
        with psycopg.connect(self._dsn) as conn:
            conn.execute("DELETE FROM kb_chunks WHERE kb_id = %s", (self._kb_id,))

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        rows = [
            (self._kb_id, c.source, c.section, c.position, c.text, to_vector_literal(v))
            for c, v in zip(chunks, vectors, strict=True)
        ]
        with psycopg.connect(self._dsn) as conn, conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO kb_chunks (kb_id, source, section, position, content, embedding) "
                "VALUES (%s, %s, %s, %s, %s, %s::vector)",
                rows,
            )

    def search(self, vector: Sequence[float], k: int) -> list[Retrieved]:
        literal = to_vector_literal(vector)
        with psycopg.connect(self._dsn) as conn:
            rows = conn.execute(
                """
                SELECT source, section, position, content,
                       1 - (embedding <=> %s::vector) AS score
                FROM kb_chunks
                WHERE kb_id = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (literal, self._kb_id, literal, k),
            ).fetchall()
        return [
            Retrieved(Chunk(text=r[3], source=r[0], section=r[1], position=r[2]), float(r[4]))
            for r in rows
        ]
