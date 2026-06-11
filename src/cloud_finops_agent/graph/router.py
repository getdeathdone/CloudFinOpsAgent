"""Conditional routing logic for the Cloud FinOps LangGraph workflow."""

from __future__ import annotations

from langchain_core.messages import BaseMessage

from cloud_finops_agent.graph.state import FinOpsGraphState, validate_graph_state
from cloud_finops_agent.models.errors import AgentError, AgentName

MAX_ERROR_RETRIES = 2
END_ROUTE = "__end__"

AGENT_ERROR_TO_NODE: dict[str, str] = {
    AgentName.DISCOVERY.value: "discovery",
    AgentName.ANALYST.value: "analyst",
    AgentName.EXECUTOR.value: "executor",
    AgentName.GRAPH.value: "executor",
}

FLOW_TO_NODE: dict[str, str] = {
    "agent-discovery": "discovery",
    "discovery": "discovery",
    "analyst": "analyst",
    "agent-analyst": "analyst",
    "executor": "executor",
    "agent-executor": "executor",
    "end": END_ROUTE,
    END_ROUTE: END_ROUTE,
}


def _message_text(message: BaseMessage | object | None) -> str:
    """Return message content as plain text for lightweight routing checks."""

    if message is None:
        return ""
    content = getattr(message, "content", message)
    return content if isinstance(content, str) else str(content)


def _same_error(left: AgentError, right: AgentError) -> bool:
    """Return whether two errors represent the same retry family."""

    return (
        str(left.agent_name) == str(right.agent_name)
        and left.error_type == right.error_type
        and left.message == right.message
    )


def _effective_retry_count(errors: list[AgentError], latest_error: AgentError) -> int:
    """Calculate retry count even when nodes append fresh error objects."""

    matching_attempts = sum(1 for error in errors if _same_error(error, latest_error))
    return max(latest_error.retry_count, matching_attempts - 1)


def _error_is_recovered(state: FinOpsGraphState, error: AgentError) -> bool:
    """Return whether a later successful node update made this error obsolete."""

    agent_name = str(error.agent_name)
    if agent_name == AgentName.DISCOVERY.value:
        return state.get("infrastructure_snapshot") is not None
    if agent_name == AgentName.ANALYST.value:
        return state.get("optimization_plan") is not None
    if agent_name == AgentName.EXECUTOR.value:
        latest_message = state.get("messages", [])[-1] if state.get("messages") else None
        latest_text = _message_text(latest_message)
        return state.get("current_agent") == "end" and "failed with" not in latest_text
    return False


def _node_for_error(error: AgentError) -> str:
    """Map an AgentError to the graph node that should be retried."""

    return AGENT_ERROR_TO_NODE.get(str(error.agent_name), "executor")


def error_and_flow_router(state: FinOpsGraphState) -> str:
    """Route graph execution based on errors and the current flow marker.

    Errors are accumulated in state for observability, so this router treats an error as active
    only while its producing stage has not successfully created its expected artifact.
    """

    validated_state = validate_graph_state(state)
    current_agent = validated_state.current_agent
    latest_message = validated_state.messages[-1] if validated_state.messages else None
    latest_text = _message_text(latest_message)

    if current_agent == "end" and "failed with" not in latest_text:
        return END_ROUTE

    if validated_state.errors:
        latest_error = validated_state.errors[-1]
        if not _error_is_recovered(state, latest_error):
            retry_count = _effective_retry_count(validated_state.errors, latest_error)
            latest_error.retry_count = retry_count
            if latest_error.recoverable and retry_count < MAX_ERROR_RETRIES:
                return _node_for_error(latest_error)
            return "executor"

    return FLOW_TO_NODE.get(current_agent, END_ROUTE)
