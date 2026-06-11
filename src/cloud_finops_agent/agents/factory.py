"""Factories for LLM-backed agents and Langfuse tracing."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import structlog
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from cloud_finops_agent.config.settings import Settings, get_settings
from cloud_finops_agent.models.analysis import OptimizationPlan
from cloud_finops_agent.tools.aws_discovery_tools import discover_ec2_instances, discover_s3_buckets

logger = structlog.get_logger(__name__)


def _secret_value(value: object) -> str | None:
    """Return a secret value when a pydantic SecretStr-like object is configured."""

    if value is None:
        return None
    get_secret_value = getattr(value, "get_secret_value", None)
    if callable(get_secret_value):
        secret = str(get_secret_value())
        return secret or None
    secret = str(value)
    return secret or None


def create_langfuse_callbacks(settings: Settings | None = None) -> list[Any]:
    """Create Langfuse callback handlers when credentials are configured."""

    resolved_settings = settings or get_settings()
    public_key = _secret_value(resolved_settings.LANGFUSE_PUBLIC_KEY)
    secret_key = _secret_value(resolved_settings.LANGFUSE_SECRET_KEY)
    if not public_key or not secret_key:
        logger.info("langfuse_callbacks_disabled", reason="missing_credentials")
        return []

    try:
        from langfuse.callback import CallbackHandler
    except ImportError:
        logger.warning("langfuse_callback_import_failed")
        return []

    return [
        CallbackHandler(
            public_key=public_key,
            secret_key=secret_key,
            host=str(resolved_settings.LANGFUSE_HOST),
        )
    ]


def create_llm(settings: Settings | None = None) -> BaseChatModel:
    """Create the configured chat model provider from Settings."""

    resolved_settings = settings or get_settings()
    model_name = resolved_settings.LLM_MODEL_NAME
    if model_name.lower().startswith("claude"):
        from langchain_anthropic import ChatAnthropic

        api_key = _secret_value(resolved_settings.ANTHROPIC_API_KEY)
        kwargs: dict[str, Any] = {"model": model_name, "temperature": 0}
        if api_key:
            kwargs["api_key"] = api_key
        return ChatAnthropic(**kwargs)

    from langchain_openai import ChatOpenAI

    api_key = _secret_value(resolved_settings.OPENAI_API_KEY)
    kwargs = {"model": model_name, "temperature": 0}
    if api_key:
        kwargs["api_key"] = api_key
    return ChatOpenAI(**kwargs)


def create_discovery_agent(settings: Settings | None = None) -> Runnable[Any, Any]:
    """Create Discovery LLM with AWS discovery tools bound."""

    tools = get_discovery_tools()
    return create_llm(settings).bind_tools(tools)


def create_analyst_agent(settings: Settings | None = None) -> Runnable[Any, OptimizationPlan]:
    """Create Analyst LLM constrained to return OptimizationPlan."""

    return create_llm(settings).with_structured_output(OptimizationPlan)


def create_executor_agent(settings: Settings | None = None) -> BaseChatModel:
    """Create Executor LLM for Markdown report and remediation code generation."""

    return create_llm(settings)


def get_discovery_tools() -> Sequence[BaseTool]:
    """Return tools available to Agent-Discovery."""

    return (discover_ec2_instances, discover_s3_buckets)
