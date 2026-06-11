"""StateGraph blueprint for the Cloud FinOps multi-agent workflow."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from cloud_finops_agent.graph.nodes import analyst_node, discovery_node, executor_node
from cloud_finops_agent.graph.router import END_ROUTE, error_and_flow_router
from cloud_finops_agent.graph.state import FinOpsGraphState


async def compile_finops_graph() -> Any:
    """Compile and return the Cloud FinOps LangGraph application."""

    workflow = StateGraph(FinOpsGraphState)
    workflow.add_node("discovery", discovery_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("executor", executor_node)

    route_map = {
        "discovery": "discovery",
        "analyst": "analyst",
        "executor": "executor",
        END_ROUTE: END,
    }
    workflow.set_entry_point("discovery")
    workflow.add_conditional_edges("discovery", error_and_flow_router, route_map)
    workflow.add_conditional_edges("analyst", error_and_flow_router, route_map)
    workflow.add_conditional_edges("executor", error_and_flow_router, route_map)

    return workflow.compile()
