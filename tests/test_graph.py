import pytest

from vendas_agent.agent import SalesNodes, run_sequential
from vendas_agent.graph import build_flow, langgraph_available

QUESTIONS = [
    "Oi, bom dia!",
    "Quanto custa o plano Pro?",
    "Vocês integram com WhatsApp?",
    "Quero falar com um atendente",
    "Vocês vendem passagens aéreas?",
    "Quero contratar o plano Pro",
]


def test_unknown_engine_is_rejected(nodes: SalesNodes) -> None:
    with pytest.raises(ValueError, match="unknown engine"):
        build_flow(nodes, "nope")


def test_sequential_engine_runs(nodes: SalesNodes) -> None:
    flow = build_flow(nodes, "sequential")
    state = flow({"message": "Vocês integram com WhatsApp?", "history": []})
    assert state["intent"].value == "question"
    assert state["reply"]


@pytest.mark.skipif(not langgraph_available(), reason="langgraph not installed")
@pytest.mark.parametrize("question", QUESTIONS)
def test_langgraph_matches_sequential(nodes: SalesNodes, question: str) -> None:
    expected = run_sequential(nodes, {"message": question, "history": []})
    actual = build_flow(nodes, "langgraph")({"message": question, "history": []})
    for key in ("intent", "reply", "sources", "grounded", "escalated"):
        assert actual.get(key) == expected.get(key), key
