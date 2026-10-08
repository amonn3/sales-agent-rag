"""Composition root: the only place that knows about concrete implementations."""

import os
import re
from pathlib import Path
from typing import Literal

from vendas_agent.agent import SalesAgent, SalesNodes
from vendas_agent.config import Settings
from vendas_agent.embeddings import Embedder, HashingEmbedder, OpenAIEmbedder
from vendas_agent.graph import build_flow
from vendas_agent.indexing import index_knowledge_base
from vendas_agent.intent import IntentClassifier, KeywordIntentClassifier, LLMIntentClassifier
from vendas_agent.knowledge import load_knowledge_base
from vendas_agent.llm import AnthropicClient, ExtractiveDemoLLM, LLMClient
from vendas_agent.models import KnowledgeBase
from vendas_agent.store import InMemoryVectorStore, VectorStore

LLMMode = Literal["auto", "demo", "anthropic"]


def make_embedder(settings: Settings) -> Embedder:
    if settings.embedder == "openai":
        return OpenAIEmbedder(settings.openai_embedding_model)
    return HashingEmbedder()


def make_llm(settings: Settings, mode: LLMMode = "auto") -> tuple[LLMClient, IntentClassifier]:
    use_real = mode == "anthropic" or (mode == "auto" and bool(os.environ.get("ANTHROPIC_API_KEY")))
    if not use_real:
        return ExtractiveDemoLLM(), KeywordIntentClassifier()
    llm = AnthropicClient(settings.anthropic_model)
    return llm, LLMIntentClassifier(llm, fallback=KeywordIntentClassifier())


def kb_id_from_path(root: Path) -> str:
    slug = re.sub(r"[^a-z0-9_-]+", "-", root.resolve().name.lower()).strip("-")
    return slug or "default"


def make_store(settings: Settings, embedder: Embedder, root: Path) -> VectorStore:
    if settings.store == "pgvector":
        from vendas_agent.pgstore import PgVectorStore

        return PgVectorStore(settings.database_url, kb_id_from_path(root), embedder.dimension)
    return InMemoryVectorStore()


def build_agent(
    kb_root: Path,
    settings: Settings,
    *,
    llm_mode: LLMMode = "auto",
    engine: str = "auto",
    reindex: bool = True,
) -> SalesAgent:
    kb: KnowledgeBase = load_knowledge_base(kb_root)
    embedder = make_embedder(settings)
    store = make_store(settings, embedder, kb_root)
    if reindex or settings.store == "memory":
        index_knowledge_base(kb, embedder, store)
    llm, classifier = make_llm(settings, llm_mode)
    nodes = SalesNodes(
        product=kb.product,
        classifier=classifier,
        embedder=embedder,
        store=store,
        llm=llm,
        top_k=settings.top_k,
        min_similarity=settings.effective_min_similarity,
    )
    return SalesAgent(build_flow(nodes, engine), settings.max_history_turns)
