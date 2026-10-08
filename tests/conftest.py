from collections.abc import Sequence
from pathlib import Path

import pytest

from vendas_agent.agent import SalesAgent, SalesNodes, run_sequential
from vendas_agent.embeddings import HashingEmbedder
from vendas_agent.indexing import index_knowledge_base
from vendas_agent.intent import KeywordIntentClassifier
from vendas_agent.knowledge import load_knowledge_base
from vendas_agent.models import KnowledgeBase, Message
from vendas_agent.store import InMemoryVectorStore

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_KB = ROOT / "examples" / "nimbus"
EVAL_DATASET = ROOT / "evals" / "dataset.jsonl"


class FakeLLM:
    """Records every call so tests can assert on prompts. Never touches the network."""

    def __init__(self, answer: str = "resposta fake") -> None:
        self.answer = answer
        self.calls: list[tuple[str, list[Message]]] = []

    def complete(self, system: str, messages: Sequence[Message]) -> str:
        self.calls.append((system, list(messages)))
        return self.answer


@pytest.fixture
def kb() -> KnowledgeBase:
    return load_knowledge_base(EXAMPLE_KB)


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def nodes(kb: KnowledgeBase, fake_llm: FakeLLM) -> SalesNodes:
    embedder = HashingEmbedder()
    store = InMemoryVectorStore()
    index_knowledge_base(kb, embedder, store)
    return SalesNodes(
        product=kb.product,
        classifier=KeywordIntentClassifier(),
        embedder=embedder,
        store=store,
        llm=fake_llm,
    )


@pytest.fixture
def agent(nodes: SalesNodes) -> SalesAgent:
    return SalesAgent(lambda state: run_sequential(nodes, state), max_history_turns=2)
