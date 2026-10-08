from pathlib import Path

import pytest

from vendas_agent.knowledge import KnowledgeBaseError, load_documents, load_knowledge_base, load_product
from vendas_agent.models import KnowledgeBase

VALID = """
[product]
name = "X"
company = "Acme"
[sales]
call_to_action = "compre"
escalation_contact = "a@b.c"
"""


def test_example_kb_loads(kb: KnowledgeBase) -> None:
    assert kb.product.name == "Nimbus CRM"
    assert [p.name for p in kb.product.plans] == ["Starter", "Pro", "Business"]
    assert kb.product.plans[1].price_label == "R$ 129,00"
    assert {d.source for d in kb.documents} >= {"planos.md", "integracoes.md"}


def test_defaults_are_applied(tmp_path: Path) -> None:
    path = tmp_path / "product.toml"
    path.write_text(VALID, encoding="utf-8")
    product = load_product(path)
    assert product.language == "pt-BR"
    assert "Acme" in product.greeting
    assert product.plans == ()


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("not = [valid", "not valid TOML"),
        ('[product]\nname = "X"\ncompany = "Y"\n', r"\[sales\]"),
        (VALID.replace('name = "X"', 'name = ""'), "'name'"),
        (VALID + '[[plans]]\nname = "P"\nprice_monthly = -1\n', "price_monthly"),
        (VALID + '[[plans]]\nname = "P"\nprice_monthly = "9"\n', "price_monthly"),
    ],
)
def test_invalid_product_files(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / "product.toml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(KnowledgeBaseError, match=message):
        load_product(path)


def test_missing_files_and_empty_docs(tmp_path: Path) -> None:
    with pytest.raises(KnowledgeBaseError, match="not found"):
        load_product(tmp_path / "product.toml")
    (tmp_path / "docs").mkdir()
    with pytest.raises(KnowledgeBaseError, match="no .md/.txt"):
        load_documents(tmp_path / "docs")
    with pytest.raises(KnowledgeBaseError):
        load_knowledge_base(tmp_path)


def test_only_supported_non_empty_docs_are_loaded(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    (docs / "sub").mkdir(parents=True)
    (docs / "a.md").write_text("# A\n\ntexto", encoding="utf-8")
    (docs / "sub" / "b.txt").write_text("texto b", encoding="utf-8")
    (docs / "image.png").write_bytes(b"\x89PNG")
    (docs / "empty.md").write_text("  \n", encoding="utf-8")
    assert [d.source for d in load_documents(docs)] == ["a.md", "sub/b.txt"]
