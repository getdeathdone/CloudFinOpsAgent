"""Pytest fixtures for Cloud FinOps Agent integration tests."""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from langchain_core.messages import AIMessage
from pydantic import SecretStr

try:
    from langchain_core.language_models.fake import FakeListChatModel
except ImportError:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

from cloud_finops_agent.config.settings import Environment, Settings, get_settings
from cloud_finops_agent.graph.state import FinOpsGraphState, create_initial_state
from cloud_finops_agent.models.analysis import (
    FinOpsFinding,
    FindingSeverity,
    FindingType,
    KnowledgeBaseRuleReference,
    OptimizationPlan,
)
from cloud_finops_agent.models.aws_resources import AWSResourceType


@pytest.fixture(scope="session")
def test_settings() -> Iterator[Settings]:
    """Configure deterministic local settings for integration tests."""

    previous_values = {
        key: os.environ.get(key)
        for key in (
            "ENVIRONMENT",
            "AWS_REGION",
            "LOCALSTACK_ENDPOINT_URL",
            "QDRANT_URL",
            "QDRANT_COLLECTION",
            "LLM_MODEL_NAME",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
        )
    }
    os.environ["ENVIRONMENT"] = "dev"
    os.environ["AWS_REGION"] = "us-east-1"
    os.environ["LOCALSTACK_ENDPOINT_URL"] = "http://localhost:4566"
    os.environ["QDRANT_URL"] = "http://localhost:6333"
    os.environ["QDRANT_COLLECTION"] = "aws_finops_rules"
    os.environ["LLM_MODEL_NAME"] = "gpt-4o-mini"
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ.pop("ANTHROPIC_API_KEY", None)
    get_settings.cache_clear()

    yield Settings(
        ENVIRONMENT=Environment.DEV,
        AWS_REGION="us-east-1",
        LOCALSTACK_ENDPOINT_URL="http://localhost:4566",
        QDRANT_URL="http://localhost:6333",
        QDRANT_COLLECTION="aws_finops_rules",
        LLM_MODEL_NAME="gpt-4o-mini",
        OPENAI_API_KEY=None,
        ANTHROPIC_API_KEY=None,
        LANGFUSE_PUBLIC_KEY=SecretStr(""),
        LANGFUSE_SECRET_KEY=SecretStr(""),
    )

    for key, value in previous_values.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    get_settings.cache_clear()


@pytest.fixture(scope="session")
def localstack_seeded(test_settings: Settings) -> None:
    """Verify LocalStack is available and seed demo AWS resources."""

    boto3 = pytest.importorskip("boto3")
    botocore_exceptions = pytest.importorskip("botocore.exceptions")
    BotoCoreError = botocore_exceptions.BotoCoreError
    ClientError = botocore_exceptions.ClientError

    s3 = boto3.client("s3", **test_settings.get_aws_client_kwargs())
    try:
        s3.list_buckets()
    except (BotoCoreError, ClientError) as exc:
        pytest.skip(f"LocalStack is not available at {test_settings.LOCALSTACK_ENDPOINT_URL}: {exc}")

    from scripts.seed_localstack import main as seed_localstack

    seed_localstack()


@pytest.fixture(scope="session")
def qdrant_seeded(test_settings: Settings) -> None:
    """Verify Qdrant is available and initialize the FinOps rules collection."""

    qdrant_client = pytest.importorskip("qdrant_client")
    AsyncQdrantClient = qdrant_client.AsyncQdrantClient

    async def check_and_seed() -> None:
        client = AsyncQdrantClient(url=str(test_settings.QDRANT_URL))
        try:
            await client.get_collections()
        except Exception as exc:
            await client.close()
            pytest.skip(f"Qdrant is not available at {test_settings.QDRANT_URL}: {exc}")
        await client.close()

        from scripts.init_knowledge_base import initialize_knowledge_base

        await initialize_knowledge_base()

    asyncio.run(check_and_seed())


@pytest.fixture()
def initial_state() -> FinOpsGraphState:
    """Return a fresh graph state for each test."""

    return create_initial_state(current_agent="agent-discovery")


@pytest.fixture()
def mock_llm() -> FakeListChatModel:
    """Return a FakeListChatModel that never calls paid model providers."""

    return FakeListChatModel(
        responses=[
            "Discovery completed with mocked LLM tool planning.",
            (
                '{"plan_id":"test-plan","findings":[],"summary":"No paid LLM was called; '
                'structured analyst output is supplied by test doubles."}'
            ),
            "# Cloud FinOps Audit Report\n\nMock executor report.",
        ]
    )


@pytest.fixture()
def optimization_plan() -> OptimizationPlan:
    """Return a deterministic optimization plan used by mocked Analyst agents."""

    finding = FinOpsFinding(
        finding_id="finding-idle-ec2-demo",
        resource_id="localstack-demo-resource",
        resource_type=AWSResourceType.EC2_INSTANCE,
        finding_type=FindingType.IDLE_RESOURCE,
        severity=FindingSeverity.MEDIUM,
        current_monthly_cost=Decimal("10.0000"),
        potential_monthly_savings=Decimal("8.0000"),
        confidence_score=0.91,
        rationale=(
            "The instance shows sustained CPU utilization below the idle threshold from the "
            "FinOps rules retrieved from Qdrant."
        ),
        recommendation="Stop the idle instance after validating ownership and workload schedule.",
        rule_reference=KnowledgeBaseRuleReference(
            rule_id="ec2-idle-cpu-7d",
            title="Stop or downsize idle EC2 instances",
            qdrant_point_id="1",
            relevance_score=0.95,
        ),
    )
    return OptimizationPlan(
        plan_id="test-optimization-plan",
        findings=[finding],
        summary="One idle EC2 optimization opportunity was detected.",
    )


class FakeDiscoveryAgent:
    """Discovery test double that lets the node execute real bound tools itself."""

    async def ainvoke(self, *_args: Any, **_kwargs: Any) -> AIMessage:
        """Return a discovery planning message without paid LLM calls."""

        return AIMessage(content="Discovery tools should be executed by the graph node.")


class FakeAnalystAgent:
    """Analyst test double returning a valid OptimizationPlan."""

    def __init__(self, plan: OptimizationPlan) -> None:
        self._plan = plan

    async def ainvoke(self, *_args: Any, **_kwargs: Any) -> OptimizationPlan:
        """Return deterministic structured output."""

        return self._plan


class FakeExecutorAgent:
    """Executor test double returning a Markdown report."""

    def __init__(self, report: str) -> None:
        self._report = report

    async def ainvoke(self, *_args: Any, **_kwargs: Any) -> AIMessage:
        """Return deterministic final Markdown."""

        return AIMessage(content=self._report)


@pytest.fixture()
def patch_paid_llms(monkeypatch: pytest.MonkeyPatch, optimization_plan: OptimizationPlan) -> None:
    """Patch graph nodes so integration tests never call OpenAI or Anthropic."""

    from cloud_finops_agent.graph import nodes

    monkeypatch.setattr(nodes, "create_discovery_agent", lambda _settings=None: FakeDiscoveryAgent())
    monkeypatch.setattr(nodes, "create_analyst_agent", lambda _settings=None: FakeAnalystAgent(optimization_plan))
    monkeypatch.setattr(
        nodes,
        "create_executor_agent",
        lambda _settings=None: FakeExecutorAgent(
            "# Cloud FinOps Audit Report\n\n"
            "## Executive Summary\n\n"
            "The mocked executor generated this report without paid LLM calls.\n\n"
            "```python\nprint('review remediation before execution')\n```"
        ),
    )
    monkeypatch.setattr(nodes, "create_langfuse_callbacks", lambda _settings=None: [])
