"""LLM clients behind a Protocol, so the agent never depends on a vendor."""

import logging
import re
from collections.abc import Sequence
from typing import Any, Protocol

from vendas_agent.models import Message
from vendas_agent.utils import timed

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    def complete(self, system: str, messages: Sequence[Message]) -> str: ...


class AnthropicClient:
    """Anthropic Messages API (``pip install .[anthropic]``, needs ANTHROPIC_API_KEY)."""

    def __init__(
        self,
        model: str,
        max_tokens: int = 600,
        temperature: float = 0.3,
        client: Any | None = None,
    ) -> None:
        if client is None:
            from anthropic import Anthropic

            client = Anthropic()
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    @timed(logger)
    def complete(self, system: str, messages: Sequence[Message]) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            system=system,
            messages=[{"role": m["role"], "content": m["content"]} for m in messages],
        )
        return "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        ).strip()


_CONTEXT_BLOCK = re.compile(r"<contexto>\s*(.*?)\s*</contexto>", re.DOTALL)
_PASSAGE_HEADER = re.compile(r"^\[\d+\][^\n]*\n", re.MULTILINE)


class ExtractiveDemoLLM:
    """Offline stand-in used when no API key is configured.

    It does not generate text: it quotes the best retrieved passage from the prompt.
    Good enough to demo retrieval, routing and guardrails without any network.
    """

    def complete(self, system: str, messages: Sequence[Message]) -> str:
        match = _CONTEXT_BLOCK.search(system)
        if not match:
            return "Posso te ajudar com mais detalhes sobre o produto."
        passages = [p.strip() for p in _PASSAGE_HEADER.split(match.group(1)) if p.strip()]
        if not passages:
            return "Posso te ajudar com mais detalhes sobre o produto."
        return f"Com base na nossa documentação: {passages[0]}"
