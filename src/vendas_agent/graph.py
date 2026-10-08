"""Wire the same nodes into a LangGraph (``pip install .[graph]``).

START → classify ─┬─ greet ───────────────┐
                  ├─ escalate ────────────┤
                  └─ retrieve ─┬─ respond ─┤→ END
                               └─ fallback ┘
"""

from typing import cast

from vendas_agent.agent import (
    AgentState,
    Flow,
    SalesNodes,
    route_after_classify,
    route_after_retrieve,
    run_sequential,
)


def langgraph_available() -> bool:
    try:
        import langgraph.graph  # noqa: F401
    except ImportError:
        return False
    return True


def build_langgraph_flow(nodes: SalesNodes) -> Flow:
    from langgraph.graph import END, START, StateGraph

    builder = StateGraph(AgentState)
    builder.add_node("classify", nodes.classify)
    builder.add_node("greet", nodes.greet)
    builder.add_node("escalate", nodes.escalate)
    builder.add_node("retrieve", nodes.retrieve)
    builder.add_node("respond", nodes.respond)
    builder.add_node("fallback", nodes.fallback)

    builder.add_edge(START, "classify")
    builder.add_conditional_edges(
        "classify",
        route_after_classify,
        {"greet": "greet", "escalate": "escalate", "retrieve": "retrieve"},
    )
    builder.add_conditional_edges(
        "retrieve", route_after_retrieve, {"respond": "respond", "fallback": "fallback"}
    )
    for terminal in ("greet", "escalate", "respond", "fallback"):
        builder.add_edge(terminal, END)

    compiled = builder.compile()

    def flow(state: AgentState) -> AgentState:
        return cast(AgentState, compiled.invoke(state))

    return flow


def build_flow(nodes: SalesNodes, engine: str = "auto") -> Flow:
    """``engine``: 'langgraph', 'sequential' or 'auto' (LangGraph if installed)."""
    if engine not in {"auto", "langgraph", "sequential"}:
        raise ValueError(f"unknown engine {engine!r}")
    use_graph = engine == "langgraph" or (engine == "auto" and langgraph_available())
    if use_graph:
        return build_langgraph_flow(nodes)

    def flow(state: AgentState) -> AgentState:
        return run_sequential(nodes, state)

    return flow
