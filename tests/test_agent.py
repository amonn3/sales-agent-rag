from dataclasses import replace

import pytest

from tests.conftest import FakeLLM
from vendas_agent.agent import (
    AgentState,
    SalesAgent,
    SalesNodes,
    build_system_prompt,
    route_after_classify,
    route_after_retrieve,
)
from vendas_agent.models import Intent, KnowledgeBase, Retrieved


@pytest.mark.parametrize(
    ("intent", "expected"),
    [
        (Intent.GREETING, "greet"),
        (Intent.HUMAN, "escalate"),
        (Intent.QUESTION, "retrieve"),
        (Intent.PRICING, "retrieve"),
        (Intent.OBJECTION, "retrieve"),
        (Intent.BUYING, "retrieve"),
    ],
)
def test_route_after_classify(intent: Intent, expected: str) -> None:
    assert route_after_classify({"intent": intent}) == expected


@pytest.mark.parametrize(("grounded", "expected"), [(True, "respond"), (False, "fallback")])
def test_route_after_retrieve(grounded: bool, expected: str) -> None:
    assert route_after_retrieve({"grounded": grounded}) == expected


def test_retrieve_finds_relevant_chunks(nodes: SalesNodes) -> None:
    state: AgentState = {"message": "integração com whatsapp", "intent": Intent.QUESTION}
    result = nodes.retrieve(state)
    assert result["grounded"] is True
    assert result["retrieved"][0].chunk.source == "integracoes.md"


def test_retrieve_is_not_grounded_for_off_topic(nodes: SalesNodes) -> None:
    result = nodes.retrieve({"message": "passagens aéreas baratas", "intent": Intent.QUESTION})
    assert result["grounded"] is False
    assert result["retrieved"] == []


def test_pricing_is_grounded_by_plans_even_without_doc_hits(nodes: SalesNodes) -> None:
    result = nodes.retrieve({"message": "xyzzy plugh", "intent": Intent.PRICING})
    assert result["grounded"] is True  # product.toml has plans


def test_system_prompt_contains_rules_plans_and_context(kb: KnowledgeBase, nodes: SalesNodes) -> None:
    hits = nodes.store.search(nodes.embedder.embed(["plano pro"])[0], 1)
    prompt = build_system_prompt(kb.product, Intent.PRICING, hits)
    assert "Nimbus CRM" in prompt
    assert "Pro: R$ 129,00/mês" in prompt
    assert "<contexto>" in prompt and "[1] (fonte:" in prompt
    assert "Nunca prometa funcionalidades" in prompt  # client-specific rule
    assert kb.product.escalation_contact in prompt


def test_prompt_handles_missing_context_and_plans(kb: KnowledgeBase) -> None:
    product = replace(kb.product, plans=())
    prompt = build_system_prompt(product, Intent.QUESTION, [])
    assert "nenhum plano cadastrado" in prompt
    assert "<contexto>\n\n</contexto>" in prompt


def test_greeting_does_not_call_llm(agent: SalesAgent, fake_llm: FakeLLM, kb: KnowledgeBase) -> None:
    reply = agent.ask("Oi, bom dia!")
    assert reply.intent is Intent.GREETING
    assert reply.text == kb.product.greeting
    assert fake_llm.calls == []


def test_human_request_escalates_without_llm(agent: SalesAgent, fake_llm: FakeLLM) -> None:
    reply = agent.ask("Quero falar com um atendente")
    assert reply.escalated is True
    assert "vendas@nimbus.example" in reply.text
    assert fake_llm.calls == []


def test_off_topic_question_falls_back_without_calling_llm(
    agent: SalesAgent, fake_llm: FakeLLM
) -> None:
    reply = agent.ask("Vocês vendem passagens aéreas?")
    assert reply.grounded is False
    assert "Não encontrei" in reply.text
    assert fake_llm.calls == []  # guardrail: no context -> the LLM is never asked to improvise


def test_grounded_answer_cites_sources_and_uses_llm(agent: SalesAgent, fake_llm: FakeLLM) -> None:
    reply = agent.ask("Vocês integram com WhatsApp?")
    assert reply.text == "resposta fake"
    assert any("integracoes.md" in s for s in reply.sources)
    system, messages = fake_llm.calls[0]
    assert "WhatsApp" in system
    assert messages[-1] == {"role": "user", "content": "Vocês integram com WhatsApp?"}


def test_buying_appends_call_to_action(agent: SalesAgent, kb: KnowledgeBase) -> None:
    reply = agent.ask("Quero contratar o plano Pro")
    assert reply.intent is Intent.BUYING
    assert reply.text.endswith(kb.product.call_to_action)


def test_history_is_passed_and_trimmed(agent: SalesAgent, fake_llm: FakeLLM) -> None:
    agent.ask("Vocês integram com WhatsApp?")
    agent.ask("Quanto custa o plano Pro?")
    agent.ask("Tem desconto no plano anual?")
    _, messages = fake_llm.calls[-1]
    assert len(messages) == 5  # 2 turns of history (max_history_turns=2) + current message
    assert len(agent.history) == 4
    agent.reset()
    assert agent.history == []


def test_empty_message_is_rejected(agent: SalesAgent) -> None:
    with pytest.raises(ValueError, match="empty"):
        agent.ask("   ")


def test_retrieved_type_is_exposed(nodes: SalesNodes) -> None:
    hits = nodes.retrieve({"message": "plano pro", "intent": Intent.PRICING})["retrieved"]
    assert all(isinstance(h, Retrieved) for h in hits)
