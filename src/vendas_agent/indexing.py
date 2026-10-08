from vendas_agent.chunking import chunk_document
from vendas_agent.embeddings import Embedder
from vendas_agent.models import Chunk, KnowledgeBase
from vendas_agent.store import VectorStore


def index_knowledge_base(
    kb: KnowledgeBase,
    embedder: Embedder,
    store: VectorStore,
    max_chars: int = 700,
    overlap: int = 80,
) -> int:
    """Chunk, embed and (re)write the whole knowledge base. Returns the chunk count."""
    chunks: list[Chunk] = [
        chunk
        for doc in kb.documents
        for chunk in chunk_document(doc, max_chars=max_chars, overlap=overlap)
    ]
    vectors = embedder.embed([c.embedding_text for c in chunks])
    store.clear()
    store.add(chunks, vectors)
    return len(chunks)
