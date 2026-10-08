"""Intent classification: a keyword baseline and an LLM-based classifier."""

from collections.abc import Sequence
from typing import Protocol

from vendas_agent.embeddings import strip_accents
from vendas_agent.llm import LLMClient
from vendas_agent.models import Intent, Message


class IntentClassifier(Protocol):
    def classify(self, message: str, history: Sequence[Message]) -> Intent: ...


def _normalize(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else " " for ch in strip_accents(text).lower())
    return f" {' '.join(cleaned.split())} "


# Checked in order: the first group with a match wins.
_RULES: tuple[tuple[Intent, tuple[str, ...]], ...] = (
    (
        Intent.HUMAN,
        (
            "humano",
            "atendente",
            "vendedor",
            "especialista",
            "falar com alguem",
            "falar com uma pessoa",
            "falar com um",
        ),
    ),
    (
        Intent.BUYING,
        (
            "quero contratar",
            "quero assinar",
            "quero comprar",
            "contratar",
            "assinar",
            "fechar",
            "como comeco",
            "quero comecar",
        ),
    ),
    (
        Intent.OBJECTION,
        (
            "caro",
            "concorrente",
            "ja uso",
            "nao preciso",
            "vou pensar",
            "pensar",
            "nao sei se",
            "dificil",
        ),
    ),
    (
        Intent.PRICING,
        (
            "preco",
            "precos",
            "quanto custa",
            "quanto e",
            "valor",
            "valores",
            "plano",
            "planos",
            "mensalidade",
            "desconto",
        ),
    ),
)
_GREETINGS = ("oi", "ola", "bom dia", "boa tarde", "boa noite", "e ai", "hello", "hi")


class KeywordIntentClassifier:
    """Cheap, deterministic baseline. Also the fallback for the LLM classifier."""

    def classify(self, message: str, history: Sequence[Message]) -> Intent:
        text = _normalize(message)
        for intent, phrases in _RULES:
            if any(f" {p} " in text for p in phrases):
                return intent
        words = len(text.split())
        if words <= 4 and any(f" {g} " in text for g in _GREETINGS):
            return Intent.GREETING
        return Intent.QUESTION


_LABELS = "greeting, question, pricing, objection, buying, human"
_CLASSIFIER_SYSTEM = (
    "Classifique a última mensagem do cliente em UMA categoria: "
    "greeting (cumprimento), question (dúvida sobre o produto), pricing (preço/planos), "
    "objection (objeção ou hesitação), buying (quer contratar/comprar), "
    f"human (quer falar com uma pessoa). Responda apenas com uma palavra de: {_LABELS}."
)


class LLMIntentClassifier:
    """Asks the LLM for a label; falls back to ``fallback`` if the answer is unusable."""

    def __init__(self, llm: LLMClient, fallback: IntentClassifier | None = None) -> None:
        self._llm = llm
        self._fallback = fallback or KeywordIntentClassifier()

    def classify(self, message: str, history: Sequence[Message]) -> Intent:
        answer = self._llm.complete(
            _CLASSIFIER_SYSTEM, [*history[-4:], {"role": "user", "content": message}]
        ).lower()
        # earliest label mentioned in the answer wins
        found = [(answer.find(i.value), i) for i in Intent if i.value in answer]
        if not found:
            return self._fallback.classify(message, history)
        return min(found, key=lambda pair: pair[0])[1]
