from collections.abc import Sequence

import pytest

from tests.conftest import FakeLLM
from vendas_agent.intent import KeywordIntentClassifier, LLMIntentClassifier
from vendas_agent.models import Intent, Message

NO_HISTORY: Sequence[Message] = []


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Oi!", Intent.GREETING),
        ("Bom dia", Intent.GREETING),
        ("Quanto custa o plano Pro?", Intent.PRICING),
        ("Tem desconto anual?", Intent.PRICING),
        ("Acho o plano muito caro", Intent.OBJECTION),
        ("Vou pensar e te falo", Intent.OBJECTION),
        ("Quero contratar o Pro", Intent.BUYING),
        ("Como faço para assinar?", Intent.BUYING),
        ("Quero falar com um atendente", Intent.HUMAN),
        ("Vocês integram com Gmail?", Intent.QUESTION),
        ("Oi, queria saber como funciona a integração com Gmail", Intent.QUESTION),
    ],
)
def test_keyword_classifier(message: str, expected: Intent) -> None:
    assert KeywordIntentClassifier().classify(message, NO_HISTORY) is expected


@pytest.mark.parametrize(
    ("llm_answer", "expected"),
    [
        ("pricing", Intent.PRICING),
        ("  Buying.\n", Intent.BUYING),
        ("Acho que é objection, não pricing", Intent.OBJECTION),
        ("???", Intent.PRICING),  # unusable answer -> keyword fallback
    ],
)
def test_llm_classifier_parses_label_or_falls_back(llm_answer: str, expected: Intent) -> None:
    classifier = LLMIntentClassifier(FakeLLM(llm_answer))
    assert classifier.classify("quanto custa?", NO_HISTORY) is expected
