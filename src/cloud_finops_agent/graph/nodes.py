"""Async LangGraph nodes for the Cloud FinOps workflow."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import structlog
from botocore.exceptions import ClientError
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from pydantic import ValidationError
from qdrant_client import AsyncQdrantClient

from cloud_finops_agent.agents.factory import (
    create_analyst_agent,
    create_discovery_agent,
    create_executor_agent,
    create_langfuse_callbacks,
    get_discovery_tools,
)
from cloud_finops_agent.config.settings import Settings, get_settings
from cloud_finops_agent.graph.state import FinOpsGraphState, validate_graph_state
from cloud_finops_agent.models.analysis import OptimizationPlan
from cloud_finops_agent.models.aws_resources import (
    AWSInfrastructureSnapshot,
    EC2InstanceModel,
    S3BucketModel,
)
from cloud_finops_agent.models.errors import AgentError, AgentErrorType, AgentName
from cloud_finops_agent.prompts.agent_prompts import (
    ANALYST_AGENT_PROMPT,
    DISCOVERY_AGENT_PROMPT,
    EXECUTOR_AGENT_PROMPT,
)

logger = structlog.get_logger(__name__)

DEFAULT_RULE_SEARCH_LIMIT = 5
DETERMINISTIC_EMBEDDING_SIZE = 384


def _build_langfuse_config(*, settings: Settings, run_name: str) -> dict[str, Any]:
    """Build LangChain Runnable config with Langfuse callbacks when available."""

    callbacks = create_langfuse_callbacks(settings)
    return {
        "callbacks": callbacks,
        "run_name": run_name,
        "tags": ["cloud-finops-agent", run_name],
    }


def _classify_exception(exception: Exception) -> AgentErrorType:
    """Map runtime exceptions to graph error categories."""

    if isinstance(exception, ClientError):
        return AgentErrorType.AWS_API_ERROR
    if isinstance(exception, ValidationError):
        return AgentErrorType.VALIDATION_ERROR
    if isinstance(exception, json.JSONDecodeError):
        return AgentErrorType.LLM_PARSING_ERROR
    if isinstance(exception, (ValueError, TypeError)):
        return AgentErrorType.VALIDATION_ERROR
    return AgentErrorType.UNKNOWN_ERROR


def _error_update(
    *,
    agent_name: AgentName,
    exception: Exception,
    current_agent: str,
) -> dict[str, Any]:
    """Create a LangGraph state update containing a structured AgentError."""

    error = AgentError.from_exception(
        agent_name=agent_name,
        error_type=_classify_exception(exception),
        exception=exception,
        recoverable=True,
    )
    logger.exception(
        "graph_node_failed",
        agent_name=agent_name,
        error_type=error.error_type,
        current_agent=current_agent,
    )
    return {
        "errors": [error],
        "messages": [
            AIMessage(
                content=(
                    f"{agent_name} failed with {error.error_type}: {error.message}. "
                    "The graph router can inspect state['errors'] and decide whether to retry."
                )
            )
        ],
        "current_agent": current_agent,
    }


def _message_content(message: BaseMessage | Any) -> str:
    """Safely stringify LangChain message content."""

    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)


async def _invoke_tool(tool: BaseTool, args: Mapping[str, Any] | None = None) -> Any:
    """Invoke a LangChain tool asynchronously with normalized arguments."""

    return await tool.ainvoke(dict(args or {}))


async def _run_discovery_tools_from_ai_message(
    *,
    ai_message: BaseMessage,
    tools_by_name: Mapping[str, BaseTool],
) -> dict[str, Any]:
    """Execute tool calls requested by the Discovery LLM."""

    outputs: dict[str, Any] = {}
    tool_calls = getattr(ai_message, "tool_calls", None) or []
    for tool_call in tool_calls:
        name = str(tool_call.get("name", ""))
        if name not in tools_by_name:
            logger.warning("unknown_discovery_tool_requested", tool_name=name)
            continue
        args = tool_call.get("args") or {}
        outputs[name] = await _invoke_tool(tools_by_name[name], args)
    return outputs


async def _run_missing_discovery_tools(
    *,
    outputs: dict[str, Any],
    tools_by_name: Mapping[str, BaseTool],
) -> dict[str, Any]:
    """Ensure the Discovery node always collects the full required inventory."""

    for required_name in ("discover_ec2_instances", "discover_s3_buckets"):
        if required_name not in outputs:
            logger.info("running_required_discovery_tool", tool_name=required_name)
            outputs[required_name] = await _invoke_tool(tools_by_name[required_name])
    return outputs


def _build_snapshot_from_tool_outputs(
    *,
    outputs: Mapping[str, Any],
    settings: Settings,
) -> AWSInfrastructureSnapshot:
    """Convert Discovery tool outputs into an AWSInfrastructureSnapshot."""

    ec2_instances = [
        EC2InstanceModel.model_validate(item)
        for item in outputs.get("discover_ec2_instances", [])
    ]
    s3_buckets = [
        S3BucketModel.model_validate(item)
        for item in outputs.get("discover_s3_buckets", [])
    ]
    return AWSInfrastructureSnapshot(
        collected_at=datetime.now(UTC),
        region=settings.AWS_REGION,
        s3_buckets=s3_buckets,
        ec2_instances=ec2_instances,
        cloudwatch_metrics=[],
    )


async def discovery_node(state: FinOpsGraphState) -> dict[str, Any]:
    """Collect AWS infrastructure data and store it as AWSInfrastructureSnapshot."""

    settings = get_settings()
    try:
        validated_state = validate_graph_state(state)
        tools = get_discovery_tools()
        tools_by_name = {tool.name: tool for tool in tools}
        discovery_agent = create_discovery_agent(settings)
        prompt_messages = [
            SystemMessage(content=DISCOVERY_AGENT_PROMPT),
            HumanMessage(
                content=(
                    "Run a full infrastructure discovery for EC2 and S3. "
                    f"Previous errors: {len(validated_state.errors)}."
                )
            ),
        ]
        ai_message = await discovery_agent.ainvoke(
            prompt_messages,
            config=_build_langfuse_config(settings=settings, run_name="agent-discovery"),
        )
        outputs = await _run_discovery_tools_from_ai_message(
            ai_message=ai_message,
            tools_by_name=tools_by_name,
        )
        outputs = await _run_missing_discovery_tools(outputs=outputs, tools_by_name=tools_by_name)
        snapshot = _build_snapshot_from_tool_outputs(outputs=outputs, settings=settings)
        summary = (
            "Discovery completed: "
            f"{len(snapshot.ec2_instances)} EC2 instances, "
            f"{len(snapshot.s3_buckets)} S3 buckets, "
            f"estimated monthly cost ${snapshot.total_estimated_monthly_cost}."
        )
        logger.info(
            "discovery_node_completed",
            ec2_count=len(snapshot.ec2_instances),
            s3_count=len(snapshot.s3_buckets),
            total_estimated_monthly_cost=str(snapshot.total_estimated_monthly_cost),
        )
        return {
            "messages": [ai_message, AIMessage(content=summary)],
            "infrastructure_snapshot": snapshot,
            "current_agent": "analyst",
        }
    except Exception as exc:
        return _error_update(
            agent_name=AgentName.DISCOVERY,
            exception=exc,
            current_agent="analyst",
        )


def _snapshot_to_query(snapshot: AWSInfrastructureSnapshot) -> str:
    """Create a compact text query for FinOps rule retrieval."""

    ec2_lines = [
        (
            f"EC2 {instance.instance_id} type={instance.instance_type} "
            f"state={instance.state} cpu_avg={instance.cpu_average_percent} "
            f"cost={instance.estimated_monthly_cost}"
        )
        for instance in snapshot.ec2_instances
    ]
    s3_lines = [
        (
            f"S3 {bucket.bucket_name} size_bytes={bucket.size_bytes} "
            f"objects={bucket.object_count} versioning={bucket.versioning_enabled} "
            f"cost={bucket.estimated_monthly_cost}"
        )
        for bucket in snapshot.s3_buckets
    ]
    return "\n".join([*ec2_lines, *s3_lines]) or "No AWS resources were discovered."


def _deterministic_embedding(text: str, vector_size: int = DETERMINISTIC_EMBEDDING_SIZE) -> list[float]:
    """Create the same local hash embedding family used by the seed script."""

    vector = [0.0] * vector_size
    tokens = [token for token in text.lower().replace(".", " ").replace(",", " ").split() if token]
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], byteorder="big") % vector_size
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


async def _embed_query(*, query: str, settings: Settings) -> list[float]:
    """Embed a FinOps retrieval query using OpenAI when configured or deterministic local vectors."""

    if settings.OPENAI_API_KEY is not None and settings.OPENAI_API_KEY.get_secret_value():
        from langchain_openai import OpenAIEmbeddings

        embedder = OpenAIEmbeddings(model="text-embedding-3-small")
        return await embedder.aembed_query(query)
    return _deterministic_embedding(query)


async def _retrieve_finops_rules(
    *,
    query: str,
    settings: Settings,
    limit: int = DEFAULT_RULE_SEARCH_LIMIT,
) -> list[dict[str, Any]]:
    """Search Qdrant for FinOps rules relevant to the infrastructure snapshot."""

    query_vector = await _embed_query(query=query, settings=settings)
    client = AsyncQdrantClient(url=str(settings.QDRANT_URL))
    try:
        # Using query_points as the universal method for Qdrant 1.10+
        results = await client.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )
        rules: list[dict[str, Any]] = []
        for result in results.points:
            payload = dict(result.payload or {})
            payload["qdrant_point_id"] = str(result.id)
            payload["relevance_score"] = float(result.score)
            rules.append(payload)
        return rules
    finally:
        await client.close()


def _parse_optimization_plan_fallback(content: str) -> OptimizationPlan:
    """Attempt to extract OptimizationPlan from raw LLM text (fallback for Ollama)."""

    # Try to find JSON block in Markdown or raw text
    json_match = re.search(r"(\{.*\})", content, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            return OptimizationPlan.model_validate(data)
        except (json.JSONDecodeError, ValidationError):
            logger.warning("failed_to_parse_json_from_fallback_content")

    # If no valid JSON found, create a mock plan using the text as summary
    return OptimizationPlan(
        plan_id=f"plan-fallback-{hashlib.md5(content.encode()).hexdigest()[:8]}",
        summary=content.strip() or "No summary provided by the analyst model.",
        findings=[],
    )


async def analyst_node(state: FinOpsGraphState) -> dict[str, Any]:
    """Analyze the infrastructure snapshot with retrieved FinOps rules."""

    settings = get_settings()
    try:
        validated_state = validate_graph_state(state)
        snapshot = validated_state.infrastructure_snapshot
        if snapshot is None:
            raise ValueError("infrastructure_snapshot is required before analyst_node can run.")

        retrieval_query = _snapshot_to_query(snapshot)
        rules = await _retrieve_finops_rules(query=retrieval_query, settings=settings)
        analyst_agent = create_analyst_agent(settings)
        response = await analyst_agent.ainvoke(
            [
                SystemMessage(content=ANALYST_AGENT_PROMPT),
                HumanMessage(
                    content=(
                        "Analyze this infrastructure snapshot and produce an OptimizationPlan.\n\n"
                        f"Snapshot JSON:\n{snapshot.model_dump_json()}\n\n"
                        f"Retrieved FinOps rules JSON:\n{json.dumps(rules, ensure_ascii=False)}"
                    )
                ),
            ],
            config=_build_langfuse_config(settings=settings, run_name="agent-analyst"),
        )
        
        # Handle cases where the LLM doesn't support native structured output (e.g. ChatOllama)
        if isinstance(response, OptimizationPlan):
            plan = response
        else:
            content = _message_content(response)
            plan = _parse_optimization_plan_fallback(content)

        logger.info(
            "analyst_node_completed",
            finding_count=plan.finding_count,
            total_potential_monthly_savings=str(plan.total_potential_monthly_savings),
            retrieved_rule_count=len(rules),
        )
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Analysis completed: "
                        f"{plan.finding_count} findings, "
                        f"${plan.total_potential_monthly_savings} potential monthly savings."
                    )
                )
            ],
            "optimization_plan": plan,
            "current_agent": "executor",
        }
    except Exception as exc:
        return _error_update(
            agent_name=AgentName.ANALYST,
            exception=exc,
            current_agent="executor",
        )


async def executor_node(state: FinOpsGraphState) -> dict[str, Any]:
    """Generate the final Markdown report and conservative remediation code."""

    settings = get_settings()
    try:
        validated_state = validate_graph_state(state)
        plan = validated_state.optimization_plan
        if plan is None:
            if validated_state.errors:
                error_lines = "\n".join(
                    (
                        f"- {error.agent_name}: {error.error_type} "
                        f"(retry_count={error.retry_count}) - {error.message}"
                    )
                    for error in validated_state.errors
                )
                fallback_report = (
                    "# Cloud FinOps Audit Report\n\n"
                    "## Executive Summary\n\n"
                    "The audit could not be completed because one or more technical errors "
                    "prevented the graph from producing a validated optimization plan.\n\n"
                    "## Technical Errors\n\n"
                    f"{error_lines}\n\n"
                    "## Remediation Code\n\n"
                    "No automated remediation code was generated because the infrastructure "
                    "analysis did not complete successfully. Re-run the audit after resolving "
                    "the errors above.\n"
                )
                return {
                    "messages": [AIMessage(content=fallback_report)],
                    "current_agent": "end",
                }
            raise ValueError("optimization_plan is required before executor_node can run.")

        snapshot = validated_state.infrastructure_snapshot
        executor_agent = create_executor_agent(settings)
        response = await executor_agent.ainvoke(
            [
                SystemMessage(content=EXECUTOR_AGENT_PROMPT),
                HumanMessage(
                    content=(
                        "Generate the final Cloud FinOps audit report and remediation code.\n\n"
                        f"OptimizationPlan JSON:\n{plan.model_dump_json()}\n\n"
                        f"Infrastructure snapshot JSON:\n"
                        f"{snapshot.model_dump_json() if snapshot is not None else '{}'}"
                    )
                ),
            ],
            config=_build_langfuse_config(settings=settings, run_name="agent-executor"),
        )
        report = _message_content(response)
        logger.info(
            "executor_node_completed",
            report_length=len(report),
            finding_count=plan.finding_count,
        )
        return {
            "messages": [AIMessage(content=report)],
            "current_agent": "end",
        }
    except Exception as exc:
        return _error_update(
            agent_name=AgentName.EXECUTOR,
            exception=exc,
            current_agent="end",
        )
