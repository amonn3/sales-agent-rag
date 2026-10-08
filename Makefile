.PHONY: install fmt check test run eval db

install:
	pip install -e ".[dev]"

fmt:  ## auto-fix style
	ruff check . --fix
	ruff format .

check:  ## what CI runs
	ruff check .
	ruff format --check .
	mypy
	pytest

test:
	pytest

run:
	python -m vendas_agent chat

eval:
	python -m vendas_agent eval

db:
	docker compose up -d db
