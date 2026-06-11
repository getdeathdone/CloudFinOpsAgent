"""Error contracts passed between LangGraph nodes."""

from __future__ import annotations

import traceback
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator


class AgentName(StrEnum):
    """Canonical graph node names used in errors and traces."""

    DISCOVERY = "agent-discovery"
    ANALYST = "agent-analyst"
    EXECUTOR = "agent-executor"
    GRAPH = "graph"


class AgentErrorType(StrEnum):
    """Recoverable error categories understood by the graph router."""

    AWS_API_ERROR = "AWS_API_ERROR"
    LLM_PARSING_ERROR = "LLM_PARSING_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    TOOL_ERROR = "TOOL_ERROR"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


class AgentError(BaseModel):
    """Structured failure record that lets the graph retry or reroute safely."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    agent_name: AgentName | str = Field(
        validation_alias=AliasChoices("agent_name", "agent"),
        min_length=1,
        description="Agent or graph component where the failure occurred.",
    )
    error_type: AgentErrorType = Field(description="Machine-readable error category.")
    message: str = Field(min_length=1, description="Human-readable error message.")
    stack_trace: str | None = Field(
        default=None,
        validation_alias=AliasChoices("stack_trace", "traceback"),
        description="Python traceback or provider diagnostic text, when available.",
    )
    retry_count: int = Field(default=0, ge=0, description="Number of correction attempts already made.")
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    recoverable: bool = Field(
        default=True,
        description="Whether graph routing may attempt a retry or correction path.",
    )

    @model_validator(mode="before")
    @classmethod
    def fallback_empty_message(cls, data: Any) -> Any:
        """Ensure message is never empty to satisfy min_length=1 constraint."""

        if isinstance(data, dict):
            message = data.get("message")
            if message is not None and not str(message).strip():
                data["message"] = "Unknown error occurred (empty exception message)"
        return data

    @field_validator("error_type", mode="before")
    @classmethod
    def normalize_error_type(cls, value: object) -> object:
        """Normalize lowercase or hyphenated error type strings."""

        if isinstance(value, str):
            return value.strip().upper().replace("-", "_").replace(" ", "_")
        return value

    @field_validator("occurred_at")
    @classmethod
    def ensure_utc_occurred_at(cls, value: datetime) -> datetime:
        """Normalize error timestamps to UTC."""

        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    def increment_retry(self) -> AgentError:
        """Return a copy of this error with an incremented retry counter."""

        return self.model_copy(update={"retry_count": self.retry_count + 1})

    @classmethod
    def from_exception(
        cls,
        *,
        agent_name: AgentName | str,
        error_type: AgentErrorType,
        exception: Exception,
        retry_count: int = 0,
        recoverable: bool = True,
    ) -> AgentError:
        """Create an AgentError from a caught exception with a formatted stack trace."""

        msg = str(exception).strip()
        if not msg:
            msg = "Unknown error occurred (empty exception message)"

        return cls(
            agent_name=agent_name,
            error_type=error_type,
            message=msg,
            stack_trace="".join(
                traceback.format_exception(type(exception), exception, exception.__traceback__)
            ),
            retry_count=retry_count,
            recoverable=recoverable,
        )
