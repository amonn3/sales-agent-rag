"""Core domain types. Everything here is immutable and fully typed."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, TypedDict


class Intent(StrEnum):
    GREETING = "greeting"
    QUESTION = "question"
    PRICING = "pricing"
    OBJECTION = "objection"
    BUYING = "buying"
    HUMAN = "human"


class Message(TypedDict):
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True, slots=True)
class Document:
    source: str
    text: str


@dataclass(frozen=True, slots=True)
class Chunk:
    text: str
    source: str
    section: str
    position: int

    @property
    def embedding_text(self) -> str:
        """Section title helps retrieval, so it is embedded together with the body."""
        return f"{self.section}\n{self.text}"

    @property
    def citation(self) -> str:
        return f"{self.source} › {self.section}"


@dataclass(frozen=True, slots=True)
class Retrieved:
    chunk: Chunk
    score: float


@dataclass(frozen=True, slots=True)
class Plan:
    name: str
    price_monthly: float
    features: tuple[str, ...]

    @property
    def price_label(self) -> str:
        return f"R$ {self.price_monthly:.2f}".replace(".", ",")


@dataclass(frozen=True, slots=True)
class ProductConfig:
    """Everything that makes the agent specific to one client's product."""

    name: str
    company: str
    language: str
    tone: str
    greeting: str
    call_to_action: str
    escalation_contact: str
    plans: tuple[Plan, ...]
    extra_rules: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class KnowledgeBase:
    product: ProductConfig
    documents: tuple[Document, ...]


@dataclass(frozen=True, slots=True)
class AgentReply:
    text: str
    intent: Intent
    sources: tuple[str, ...]
    grounded: bool
    escalated: bool
