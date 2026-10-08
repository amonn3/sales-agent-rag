"""Command line interface: chat, ask, ingest, eval."""

import argparse
from collections.abc import Sequence
import logging
from pathlib import Path

from vendas_agent.agent import SalesAgent
from vendas_agent.bootstrap import build_agent
from vendas_agent.config import Settings
from vendas_agent.evaluation import load_cases, run_eval
from vendas_agent.models import AgentReply

DEFAULT_KB = Path("examples/nimbus")
DEFAULT_EVAL = Path("evals/dataset.jsonl")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vendas-agent", description=__doc__)
    parser.add_argument("--kb", type=Path, default=DEFAULT_KB, help="knowledge base folder")
    parser.add_argument("--llm", choices=["auto", "demo", "anthropic"], default="auto")
    parser.add_argument("--engine", choices=["auto", "langgraph", "sequential"], default="auto")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("chat", help="interactive conversation")
    ask = sub.add_parser("ask", help="answer a single message")
    ask.add_argument("message")
    sub.add_parser("ingest", help="(re)index the knowledge base into the configured store")
    ev = sub.add_parser("eval", help="run the evaluation dataset")
    ev.add_argument("--dataset", type=Path, default=DEFAULT_EVAL)
    return parser


def _print_reply(reply: AgentReply) -> None:
    print(f"\n{reply.text}")
    if reply.sources:
        print("\n  fontes: " + " | ".join(reply.sources))
    print(f"  [intenção: {reply.intent.value}]\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    settings = Settings.from_env()

    if args.command == "ingest":
        build_agent(args.kb, settings, llm_mode="demo", engine="sequential", reindex=True)
        print(f"Indexed {args.kb} into the '{settings.store}' store.")
        return 0

    agent = build_agent(args.kb, settings, llm_mode=args.llm, engine=args.engine)

    if args.command == "ask":
        _print_reply(agent.ask(args.message))
        return 0

    if args.command == "eval":

        def fresh_agent() -> SalesAgent:
            agent.reset()  # reuse the indexed agent, but never share conversation state
            return agent

        report = run_eval(fresh_agent, load_cases(args.dataset))
        for result in report.results:
            mark = "PASS" if result.passed else "FAIL"
            print(f"{mark}  {result.case.question}")
            for failure in result.failures:
                print(f"        - {failure}")
        print(f"\npass rate: {report.pass_rate:.0%}")
        return 0 if report.pass_rate == 1.0 else 1

    print("Digite 'sair' para encerrar.\n")
    while True:
        try:
            message = input("você> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if message.lower() in {"sair", "exit", "quit"}:
            return 0
        if message:
            _print_reply(agent.ask(message))
