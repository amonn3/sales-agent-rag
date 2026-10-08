"""Load a client's knowledge base from a folder.

Layout (this is what makes the agent customizable per client)::

    my-client/
    ├── product.toml   # product name, tone, plans, rules, escalation contact
    └── docs/          # any number of .md / .txt files
"""

from collections.abc import Mapping
from pathlib import Path
import tomllib
from typing import Any

from vendas_agent.models import Document, KnowledgeBase, Plan, ProductConfig

DOC_SUFFIXES = {".md", ".txt"}


class KnowledgeBaseError(ValueError):
    """Raised when a knowledge base folder is missing or malformed."""


def _table(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise KnowledgeBaseError(f"product.toml: missing [{key}] table")
    return value


def _text(table: Mapping[str, Any], key: str, default: str | None = None) -> str:
    value = table.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise KnowledgeBaseError(f"product.toml: '{key}' must be a non-empty string")
    return value.strip()


def _str_list(table: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = table.get(key, [])
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise KnowledgeBaseError(f"product.toml: '{key}' must be a list of strings")
    return tuple(value)


def _parse_plan(raw: Mapping[str, Any]) -> Plan:
    price = raw.get("price_monthly")
    if isinstance(price, bool) or not isinstance(price, int | float) or price < 0:
        raise KnowledgeBaseError("product.toml: plan 'price_monthly' must be a number >= 0")
    return Plan(
        name=_text(raw, "name"),
        price_monthly=float(price),
        features=_str_list(raw, "features"),
    )


def load_product(path: Path) -> ProductConfig:
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except FileNotFoundError as exc:
        raise KnowledgeBaseError(f"{path} not found") from exc
    except tomllib.TOMLDecodeError as exc:
        raise KnowledgeBaseError(f"{path} is not valid TOML: {exc}") from exc

    product = _table(data, "product")
    sales = _table(data, "sales")
    name = _text(product, "name")
    company = _text(product, "company")
    raw_plans = data.get("plans", [])
    if not isinstance(raw_plans, list):
        raise KnowledgeBaseError("product.toml: [[plans]] must be an array of tables")

    return ProductConfig(
        name=name,
        company=company,
        language=_text(product, "language", "pt-BR"),
        tone=_text(product, "tone", "consultivo, simpático e direto"),
        greeting=_text(
            product,
            "greeting",
            f"Olá! Sou o assistente da {company}. Como posso te ajudar com o {name}?",
        ),
        call_to_action=_text(sales, "call_to_action"),
        escalation_contact=_text(sales, "escalation_contact"),
        plans=tuple(_parse_plan(p) for p in raw_plans),
        extra_rules=_str_list(sales, "extra_rules"),
    )


def load_documents(docs_dir: Path) -> tuple[Document, ...]:
    if not docs_dir.is_dir():
        raise KnowledgeBaseError(f"{docs_dir} is not a directory")
    documents = []
    for path in sorted(docs_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in DOC_SUFFIXES:
            text = path.read_text(encoding="utf-8").strip()
            if text:
                documents.append(Document(source=path.relative_to(docs_dir).as_posix(), text=text))
    if not documents:
        raise KnowledgeBaseError(f"no .md/.txt documents found in {docs_dir}")
    return tuple(documents)


def load_knowledge_base(root: Path) -> KnowledgeBase:
    return KnowledgeBase(
        product=load_product(root / "product.toml"),
        documents=load_documents(root / "docs"),
    )
