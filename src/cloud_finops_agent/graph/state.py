"""LangGraph state contract for the Cloud FinOps multi-agent workflow."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from cloud_finops_agent.models.analysis import OptimizationPlan
from cloud_finops_agent.models.aws_resources import AWSInfrastructureSnapshot
from cloud_finops_agent.models.errors import AgentError


class FinOpsGraphState(TypedDict):
    """Mutable LangGraph state shared by Discovery, Analyst, and Executor nodes."""

    messages: Annotated[list[BaseMessage], add_messages]
    infrastructure_snapshot: AWSInfrastructureSnapshot | None
    optimization_plan: OptimizationPlan | None
    errors: Annotated[list[AgentError], operator.add]
    current_agent: str


class FinOpsGraphStateModel(BaseModel):
    """Pydantic validation mirror for FinOpsGraphState.

    LangGraph consumes the TypedDict annotations for reducer behavior, while nodes can use this
    model to validate state snapshots at boundaries before making AWS, Qdrant, or LLM calls.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", validate_assignment=True)

    messages: list[BaseMessage] = Field(default_factory=list)
    infrastructure_snapshot: AWSInfrastructureSnapshot | None = None
    optimization_plan: OptimizationPlan | None = None
    errors: list[AgentError] = Field(default_factory=list)
    current_agent: str = Field(default="graph", min_length=1)

    def to_langgraph_state(self) -> FinOpsGraphState:
        """Convert the validated model into the TypedDict consumed by LangGraph."""

        return {
            "messages": self.messages,
            "infrastructure_snapshot": self.infrastructure_snapshot,
            "optimization_plan": self.optimization_plan,
            "errors": self.errors,
            "current_agent": self.current_agent,
        }


FinOpsGraphStateAdapter: TypeAdapter[FinOpsGraphState] = TypeAdapter(FinOpsGraphState)


def create_initial_state(*, current_agent: str = "agent-discovery") -> FinOpsGraphState:
    """Create an empty, validated graph state for a new FinOps audit run."""

    return FinOpsGraphStateModel(current_agent=current_agent).to_langgraph_state()


def validate_graph_state(state: FinOpsGraphState | dict[str, object]) -> FinOpsGraphStateModel:
    """Validate a raw LangGraph state dictionary using the Pydantic mirror model."""

    return FinOpsGraphStateModel.model_validate(state)
