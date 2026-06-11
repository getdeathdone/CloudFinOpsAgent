"""Integration tests for LangGraph flow, retry routing, and fallback behavior."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage

from cloud_finops_agent.graph.router import END_ROUTE, error_and_flow_router
from cloud_finops_agent.graph.state import FinOpsGraphState, create_initial_state, validate_graph_state
from cloud_finops_agent.models.errors import AgentError, AgentErrorType, AgentName


@pytest.mark.asyncio
async def test_graph_happy_path_reaches_executor_and_end(
    localstack_seeded: None,
    qdrant_seeded: None,
    patch_paid_llms: None,
) -> None:
    """Graph should complete discovery, analysis, and execution with real LocalStack/Qdrant."""

    from cloud_finops_agent.graph.blueprint import compile_finops_graph

    graph = await compile_finops_graph()
    final_state = validate_graph_state(
        await graph.ainvoke(create_initial_state(current_agent="agent-discovery"))
    )

    assert final_state.current_agent == "end"
    assert final_state.infrastructure_snapshot is not None
    assert final_state.infrastructure_snapshot.resource_count > 0
    assert final_state.optimization_plan is not None
    assert final_state.optimization_plan.finding_count == 1
    assert final_state.errors == []
    assert "Cloud FinOps Audit Report" in str(final_state.messages[-1].content)


@pytest.mark.asyncio
async def test_self_correction_retry_path_routes_back_to_failed_node(
    monkeypatch: pytest.MonkeyPatch,
    initial_state: FinOpsGraphState,
) -> None:
    """A transient Discovery failure should be captured and routed back to discovery."""

    from cloud_finops_agent.graph import blueprint
    from cloud_finops_agent.graph.blueprint import compile_finops_graph
    from cloud_finops_agent.models.errors import AgentError as ErrorModel

    attempts = {"count": 0}

    async def flaky_discovery_node(state: FinOpsGraphState) -> dict[str, Any]:
        attempts["count"] += 1
        if attempts["count"] == 1:
            return {
                "errors": [
                    ErrorModel(
                        agent_name=AgentName.DISCOVERY,
                        error_type=AgentErrorType.AWS_API_ERROR,
                        message="transient LocalStack API failure",
                    )
                ],
                "messages": [AIMessage(content="agent-discovery failed with AWS_API_ERROR")],
                "current_agent": "analyst",
            }
        return {
            "messages": [AIMessage(content="discovery recovered")],
            "infrastructure_snapshot": None,
            "current_agent": "end",
        }

    monkeypatch.setattr(blueprint, "discovery_node", flaky_discovery_node)
    graph = await compile_finops_graph()
    final_state = validate_graph_state(await graph.ainvoke(initial_state))

    assert attempts["count"] == 2
    assert final_state.errors
    assert final_state.errors[-1].retry_count == 0
    assert final_state.current_agent == "end"


def test_router_retries_failed_agent_before_limit(initial_state: FinOpsGraphState) -> None:
    """Router should send a recoverable Discovery error back to discovery before the retry limit."""

    initial_state["current_agent"] = "analyst"
    initial_state["errors"] = [
        AgentError(
            agent_name=AgentName.DISCOVERY,
            error_type=AgentErrorType.AWS_API_ERROR,
            message="temporary failure",
        )
    ]

    route = error_and_flow_router(initial_state)

    assert route == "discovery"
    assert initial_state["errors"][-1].retry_count == 0


@pytest.mark.asyncio
async def test_fallback_path_routes_to_executor_after_retry_limit(
    monkeypatch: pytest.MonkeyPatch,
    initial_state: FinOpsGraphState,
) -> None:
    """Persistent failures should end with a fallback report instead of an unhandled exception."""

    from cloud_finops_agent.graph import blueprint
    from cloud_finops_agent.graph.blueprint import compile_finops_graph
    from cloud_finops_agent.models.errors import AgentError as ErrorModel

    async def always_failing_discovery_node(_state: FinOpsGraphState) -> dict[str, Any]:
        return {
            "errors": [
                ErrorModel(
                    agent_name=AgentName.DISCOVERY,
                    error_type=AgentErrorType.AWS_API_ERROR,
                    message="persistent LocalStack API failure",
                )
            ],
            "messages": [AIMessage(content="agent-discovery failed with AWS_API_ERROR")],
            "current_agent": "analyst",
        }

    monkeypatch.setattr(blueprint, "discovery_node", always_failing_discovery_node)
    graph = await compile_finops_graph()
    final_state = validate_graph_state(await graph.ainvoke(initial_state))

    assert final_state.current_agent == "end"
    assert final_state.optimization_plan is None
    assert len(final_state.errors) == 3
    assert final_state.errors[-1].retry_count == 2
    assert "could not be completed" in str(final_state.messages[-1].content)
    assert error_and_flow_router(final_state.to_langgraph_state()) == END_ROUTE
