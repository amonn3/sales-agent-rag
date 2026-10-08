from pathlib import Path

import pytest

from tests.conftest import EVAL_DATASET
from vendas_agent.agent import SalesAgent, SalesNodes, run_sequential
from vendas_agent.evaluation import EvalCase, check, load_cases, run_eval
from vendas_agent.models import AgentReply, Intent


def test_dataset_passes_on_example_kb(nodes: SalesNodes) -> None:
    """Regression guard for retrieval + routing: change chunking/thresholds -> this tells you."""
    report = run_eval(
        lambda: SalesAgent(lambda s: run_sequential(nodes, s)), load_cases(EVAL_DATASET)
    )
    failures = [(r.case.question, r.failures) for r in report.results if not r.passed]
    assert failures == []
    assert report.pass_rate == 1.0


def test_check_reports_each_mismatch() -> None:
    reply = AgentReply("x", Intent.QUESTION, ("a.md › A",), grounded=True, escalated=False)
    case = EvalCase("q", Intent.PRICING, source="b.md", grounded=False, escalated=True)
    assert len(check(case, reply)) == 4


def test_load_cases_reports_line_number(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    path.write_text('{"question": "ok", "intent": "question"}\n{"question": "x", "intent": "???"}\n')
    with pytest.raises(ValueError, match=r"d.jsonl:2"):
        load_cases(path)
