"""Tiny evaluation harness: run a dataset of questions and check the outcome.

Dataset format (JSON Lines), one case per line::

    {"question": "Quanto custa o plano Pro?", "intent": "pricing", "source": "planos.md"}
    {"question": "Vendem passagens aéreas?", "intent": "question", "grounded": false}
"""

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from vendas_agent.agent import SalesAgent
from vendas_agent.models import AgentReply, Intent


@dataclass(frozen=True, slots=True)
class EvalCase:
    question: str
    intent: Intent
    source: str | None = None  # substring that must appear in one cited source
    grounded: bool | None = None
    escalated: bool | None = None


@dataclass(frozen=True, slots=True)
class CaseResult:
    case: EvalCase
    reply: AgentReply
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures


@dataclass(frozen=True, slots=True)
class EvalReport:
    results: tuple[CaseResult, ...]

    @property
    def pass_rate(self) -> float:
        return sum(r.passed for r in self.results) / len(self.results) if self.results else 0.0


def load_cases(path: Path) -> list[EvalCase]:
    cases = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            cases.append(
                EvalCase(
                    question=raw["question"],
                    intent=Intent(raw["intent"]),
                    source=raw.get("source"),
                    grounded=raw.get("grounded"),
                    escalated=raw.get("escalated"),
                )
            )
        except (KeyError, ValueError) as exc:
            raise ValueError(f"{path}:{number}: invalid eval case ({exc})") from exc
    return cases


def check(case: EvalCase, reply: AgentReply) -> tuple[str, ...]:
    failures = []
    if reply.intent is not case.intent:
        failures.append(f"intent: expected {case.intent.value}, got {reply.intent.value}")
    if case.source and not any(case.source in s for s in reply.sources):
        failures.append(f"source: none of {list(reply.sources)} contains {case.source!r}")
    if case.grounded is not None and reply.grounded != case.grounded:
        failures.append(f"grounded: expected {case.grounded}, got {reply.grounded}")
    if case.escalated is not None and reply.escalated != case.escalated:
        failures.append(f"escalated: expected {case.escalated}, got {reply.escalated}")
    return tuple(failures)


def run_eval(make_agent: Callable[[], SalesAgent], cases: Iterable[EvalCase]) -> EvalReport:
    """Each case runs on a fresh agent so conversations never leak into each other."""
    results = []
    for case in cases:
        reply = make_agent().ask(case.question)
        results.append(CaseResult(case, reply, check(case, reply)))
    return EvalReport(tuple(results))
