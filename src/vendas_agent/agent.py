"""The sales agent: pure node functions + routers + a thin ``SalesAgent`` facade.

The nodes are plain functions over a state dict, so they are trivial to unit-test.
``graph.py`` wires the *same* nodes into a LangGraph; ``run_sequential`` below
runs them without LangGraph. Both produce identical results.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypedDict

from vendas_agent.embeddings import Embedder
from vendas_agent.intent import IntentClassifier
from vendas_agent.llm import LLMClient
from vendas_agent.models import AgentReply, Intent, Message, ProductConfig, Retrieved
from vendas_agent.store import VectorStore
from vendas_agent.tracing import traced


class AgentState(TypedDict, total=False):
    message: str
    history: list[Message]
    intent: Intent
    retrieved: list[Retrieved]
    grounded: bool
    reply: str
    sources: list[str]
    escalated: bool


_INTENT_HINTS: dict[Intent, str] = {
    Intent.QUESTION: "Responda a dúvida com precisão e, se fizer sentido, conecte ao valor para o cliente.",
    Intent.PRICING: "Apresente os planos relevantes com preços exatos e ajude a escolher o mais adequado.",
    Intent.OBJECTION: "Reconheça a preocupação com empatia, responda com fatos do contexto e não pressione.",
    Intent.BUYING: "O cliente quer comprar: confirme o plano de interesse e explique o próximo passo.",
    Intent.GREETING: "Cumprimente e pergunte como pode ajudar.",
    Intent.HUMAN: "Encaminhe para um especialista humano.",
}


def build_system_prompt(
    product: ProductConfig, intent: Intent, retrieved: Sequence[Retrieved]
) -> str:
    rules = [
        "Use SOMENTE as informações do contexto e dos planos abaixo. "
        "Nunca invente preços, descontos, prazos ou funcionalidades.",
        "Se a informação não estiver disponível, diga que não tem certeza e ofereça falar "
        f"com um especialista ({product.escalation_contact}).",
        "Seja objetivo (até 4 frases) e termine com no máximo uma pergunta para avançar a conversa.",
        "Cite as fontes usadas com o número entre colchetes, por exemplo [1].",
        *product.extra_rules,
    ]
    plans = "\n".join(
        f"- {p.name}: {p.price_label}/mês — {', '.join(p.features)}" for p in product.plans
    )
    passages = "\n\n".join(
        f"[{i}] (fonte: {r.chunk.citation})\n{r.chunk.text}" for i, r in enumerate(retrieved, 1)
    )
    return (
        f"Você é consultor(a) de vendas da {product.company}, vendendo o {product.name}.\n"
        f"Tom de voz: {product.tone}. Responda em {product.language}.\n\n"
        "Regras:\n" + "\n".join(f"- {r}" for r in rules) + "\n\n"
        f"Objetivo desta resposta: {_INTENT_HINTS[intent]}\n\n"
        f"Planos disponíveis:\n{plans or '- (nenhum plano cadastrado)'}\n\n"
        f"<contexto>\n{passages}\n</contexto>"
    )


@dataclass(slots=True)
class SalesNodes:
    """All the steps of the flow. Dependencies are injected (Protocols, not vendors)."""

    product: ProductConfig
    classifier: IntentClassifier
    embedder: Embedder
    store: VectorStore
    llm: LLMClient
    top_k: int = 4
    min_similarity: float = 0.12

    def classify(self, state: AgentState) -> AgentState:
        return {"intent": self.classifier.classify(state["message"], state.get("history", []))}

    def retrieve(self, state: AgentState) -> AgentState:
        vector = self.embedder.embed([state["message"]])[0]
        hits = self.store.search(vector, self.top_k)
        relevant = [h for h in hits if h.score >= self.min_similarity]
        # Pricing questions can be answered from the product config even if the docs miss.
        has_plans = state["intent"] is Intent.PRICING and bool(self.product.plans)
        return {"retrieved": relevant, "grounded": bool(relevant) or has_plans}

    def respond(self, state: AgentState) -> AgentState:
        intent = state["intent"]
        retrieved = state.get("retrieved", [])
        system = build_system_prompt(self.product, intent, retrieved)
        messages: list[Message] = [
            *state.get("history", []),
            {"role": "user", "content": state["message"]},
        ]
        text = self.llm.complete(system, messages).strip()
        if intent is Intent.BUYING:
            text = f"{text}\n\n{self.product.call_to_action}"
        sources = list(dict.fromkeys(r.chunk.citation for r in retrieved))
        return {"reply": text, "sources": sources, "escalated": False}

    def fallback(self, state: AgentState) -> AgentState:
        text = (
            "Não encontrei essa informação com segurança na nossa documentação. "
            f"Posso te conectar com um especialista: {self.product.escalation_contact}."
        )
        return {"reply": text, "sources": [], "escalated": False, "grounded": False}

    def greet(self, state: AgentState) -> AgentState:
        return {"reply": self.product.greeting, "sources": [], "escalated": False}

    def escalate(self, state: AgentState) -> AgentState:
        text = (
            "Claro! Vou te passar para um especialista da nossa equipe: "
            f"{self.product.escalation_contact}."
        )
        return {"reply": text, "sources": [], "escalated": True}


def route_after_classify(state: AgentState) -> Literal["greet", "escalate", "retrieve"]:
    intent = state["intent"]
    if intent is Intent.GREETING:
        return "greet"
    if intent is Intent.HUMAN:
        return "escalate"
    return "retrieve"


def route_after_retrieve(state: AgentState) -> Literal["respond", "fallback"]:
    return "respond" if state.get("grounded", False) else "fallback"


def run_sequential(nodes: SalesNodes, state: AgentState) -> AgentState:
    """Run the flow with plain Python (no LangGraph needed)."""
    state.update(nodes.classify(state))
    step = route_after_classify(state)
    if step == "greet":
        state.update(nodes.greet(state))
    elif step == "escalate":
        state.update(nodes.escalate(state))
    else:
        state.update(nodes.retrieve(state))
        if route_after_retrieve(state) == "respond":
            state.update(nodes.respond(state))
        else:
            state.update(nodes.fallback(state))
    return state


class Flow(Protocol):
    def __call__(self, state: AgentState) -> AgentState: ...


class SalesAgent:
    """Holds the conversation history and runs one turn per ``ask`` call."""

    def __init__(self, flow: Flow, max_history_turns: int = 6) -> None:
        self._flow = flow
        self._max_messages = max(0, max_history_turns) * 2
        self._history: list[Message] = []

    @property
    def history(self) -> list[Message]:
        return list(self._history)

    def reset(self) -> None:
        self._history.clear()

    @traced("sales-agent-turn")
    def ask(self, message: str) -> AgentReply:
        message = message.strip()
        if not message:
            raise ValueError("message must not be empty")
        state = self._flow({"message": message, "history": list(self._history)})
        reply = AgentReply(
            text=state["reply"],
            intent=state["intent"],
            sources=tuple(state.get("sources", [])),
            grounded=state.get("grounded", True),
            escalated=state.get("escalated", False),
        )
        self._history.extend(
            [{"role": "user", "content": message}, {"role": "assistant", "content": reply.text}]
        )
        if self._max_messages:
            self._history = self._history[-self._max_messages :]
        else:
            self._history.clear()
        return reply
