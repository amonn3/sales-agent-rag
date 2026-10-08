import logging
from pathlib import Path

import pytest

from tests.conftest import EXAMPLE_KB
from vendas_agent.bootstrap import build_agent, kb_id_from_path
from vendas_agent.config import Settings
from vendas_agent.llm import AnthropicClient, ExtractiveDemoLLM
from vendas_agent.utils import timed


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def test_timed_logs_and_preserves_result_and_name() -> None:
    logger = logging.getLogger("test-timed")
    logger.setLevel(logging.INFO)
    handler = _ListHandler()
    logger.addHandler(handler)

    @timed(logger)
    def add(a: int, b: int) -> int:
        """doc"""
        return a + b

    assert add(1, b=2) == 3
    assert add.__name__ == "add" and add.__doc__ == "doc"
    assert any("add took" in m for m in handler.messages)


def test_timed_logs_even_when_function_raises() -> None:
    logger = logging.getLogger("test-timed-raise")
    logger.setLevel(logging.INFO)
    handler = _ListHandler()
    logger.addHandler(handler)

    @timed(logger)
    def boom() -> None:
        raise RuntimeError("x")

    with pytest.raises(RuntimeError):
        boom()
    assert handler.messages


def test_settings_defaults_and_env_overrides() -> None:
    assert Settings.from_env({}).effective_min_similarity == 0.12
    s = Settings.from_env({"EMBEDDER": "openai", "TOP_K": "7", "STORE": "pgvector"})
    assert (s.embedder, s.top_k, s.store, s.effective_min_similarity) == (
        "openai",
        7,
        "pgvector",
        0.30,
    )
    assert Settings.from_env({"MIN_SIMILARITY": "0.5"}).effective_min_similarity == 0.5


@pytest.mark.parametrize("env", [{"EMBEDDER": "x"}, {"STORE": "redis"}])
def test_settings_reject_unknown_values(env: dict[str, str]) -> None:
    with pytest.raises(ValueError):
        Settings.from_env(env)


def test_kb_id_is_a_safe_slug() -> None:
    assert kb_id_from_path(Path("/tmp/My Client!!")) == "my-client"


def test_demo_llm_quotes_best_passage() -> None:
    system = "x\n<contexto>\n[1] (fonte: a.md › A)\nPrimeiro trecho\n\n[2] (fonte: b.md › B)\nSegundo\n</contexto>"
    assert "Primeiro trecho" in ExtractiveDemoLLM().complete(system, [])
    assert "mais detalhes" in ExtractiveDemoLLM().complete("sem contexto", [])


class _Block:
    def __init__(self, type_: str, text: str = "") -> None:
        self.type = type_
        self.text = text


class _FakeAnthropic:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] = {}
        self.messages = self

    def create(self, **kwargs: object) -> object:
        self.kwargs = kwargs

        class _Response:
            content = (_Block("text", " olá "), _Block("tool_use"), _Block("text", "mundo"))

        return _Response()


def test_anthropic_client_builds_request_and_joins_text_blocks() -> None:
    fake = _FakeAnthropic()
    client = AnthropicClient("some-model", client=fake)
    out = client.complete("sys", [{"role": "user", "content": "oi"}])
    assert out == "olá mundo"
    assert fake.kwargs["system"] == "sys"
    assert fake.kwargs["model"] == "some-model"
    assert fake.kwargs["messages"] == [{"role": "user", "content": "oi"}]


def test_build_agent_end_to_end_offline() -> None:
    agent = build_agent(EXAMPLE_KB, Settings(), llm_mode="demo", engine="sequential")
    reply = agent.ask("Quanto custa o plano Business?")
    assert "299" in reply.text
