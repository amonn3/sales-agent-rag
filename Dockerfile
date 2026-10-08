# Stage 1: build a wheel with all runtime extras
FROM python:3.12-slim AS builder
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir --prefix=/install ".[anthropic,openai,graph,pgvector]"

# Stage 2: slim, non-root runtime image
FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /install /usr/local
COPY examples/ examples/
COPY evals/ evals/
RUN useradd --create-home appuser
USER appuser
ENTRYPOINT ["python", "-m", "vendas_agent"]
CMD ["chat"]
