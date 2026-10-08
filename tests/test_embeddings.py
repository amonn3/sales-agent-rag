import math

import pytest

from vendas_agent.embeddings import HashingEmbedder, tokenize
from vendas_agent.store import cosine


def test_tokenize_ignores_accents_case_and_plurals() -> None:
    assert tokenize("Integrações") == tokenize("integracao")
    assert tokenize("Planos") == tokenize("plano")


def test_tokenize_drops_stopwords() -> None:
    assert tokenize("o preço de um plano") == tokenize("preço plano")


def test_embeddings_are_deterministic_and_normalized() -> None:
    embedder = HashingEmbedder(dimension=64)
    a, b = embedder.embed(["plano pro whatsapp", "plano pro whatsapp"])
    assert a == b
    assert len(a) == embedder.dimension == 64
    assert math.isclose(sum(x * x for x in a), 1.0)


def test_similar_texts_score_higher_than_unrelated() -> None:
    embedder = HashingEmbedder()
    query, related, unrelated = embedder.embed(
        [
            "quanto custa o plano pro",
            "o plano pro custa 129 reais",
            "backups diarios criptografados",
        ]
    )
    assert cosine(query, related) > cosine(query, unrelated)


def test_empty_text_gives_zero_vector() -> None:
    assert set(HashingEmbedder(dimension=8).embed([""])[0]) == {0.0}


def test_invalid_dimension() -> None:
    with pytest.raises(ValueError, match="dimension"):
        HashingEmbedder(dimension=0)
