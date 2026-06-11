# Cloud FinOps AI Agent

Cloud FinOps AI Agent is a production-style multi-agent system for auditing, analyzing, and optimizing AWS infrastructure. It uses LangGraph for orchestration, LocalStack for safe local AWS emulation, Qdrant for FinOps knowledge retrieval, and Langfuse for LLM observability.

The project is intentionally built as an enterprise-grade pet project: strict Pydantic contracts, async I/O, typed graph state, recoverable error contracts, deterministic local test paths, and clean module boundaries.

## Architecture Overview

The system is a cyclic LangGraph workflow built around a shared `FinOpsGraphState`.

```text
START
  |
  v
discovery
  |
  v
analyst
  |
  v
executor
  |
  v
END
```

Every node is followed by a conditional router:

```text
node -> error_and_flow_router -> next node | retry failed node | fallback executor | END
```

Core agents:

- `Agent-Discovery` scans AWS resources through async tools backed by `aioboto3`.
- `Agent-Analyst` retrieves optimization rules from Qdrant and returns a strict `OptimizationPlan`.
- `Agent-Executor` generates a Markdown report and conservative `boto3` remediation code.

Shared graph state:

- `messages`: accumulated LangChain messages.
- `infrastructure_snapshot`: normalized `AWSInfrastructureSnapshot` from Discovery.
- `optimization_plan`: structured `OptimizationPlan` from Analyst.
- `errors`: accumulated `AgentError` contracts.
- `current_agent`: flow marker used by the router.

## Tech Stack

- Python 3.11+
- LangGraph
- LangChain
- Pydantic v2
- pydantic-settings
- aioboto3 / boto3
- Qdrant
- LocalStack
- Langfuse
- Typer
- Rich
- structlog
- pytest / pytest-asyncio
- Docker Compose

## Local Deployment & Quick Start

Create a local environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
Copy-Item .env.example .env
```

Start local infrastructure:

```powershell
docker compose up -d
```

Services:

- LocalStack: http://localhost:4566
- Qdrant: http://localhost:6333
- Langfuse: http://localhost:3000

Initialize the knowledge base:

```powershell
python scripts/init_knowledge_base.py
```

Seed LocalStack with demo infrastructure:

```powershell
python scripts/seed_localstack.py
```

Run the audit:

```powershell
cloud-finops-agent run --env dev --model gpt-4o-mini
```

Equivalent module command:

```powershell
python -m cloud_finops_agent.main run --env dev --model gpt-4o-mini
```

## Resiliency & Self-Correction

All graph nodes catch operational failures and convert them into structured `AgentError` objects. Errors are appended to graph state instead of crashing the process.

The router uses those errors to decide what happens next:

- transient `Agent-Discovery` failure routes back to `discovery`;
- transient `Agent-Analyst` failure routes back to `analyst`;
- transient `Agent-Executor` failure routes back to `executor`;
- persistent failures are capped by `MAX_ERROR_RETRIES`;
- exhausted retries route to `executor`, which emits a fallback report explaining why the audit could not complete.

This makes the graph safe for demos and realistic production hardening: failures are observable, retryable, and reportable.

## Testing

Integration tests avoid paid LLM calls by patching the LLM factory with deterministic fake agents. They still interact with real LocalStack and Qdrant when those services are running.

Run tests:

```powershell
pytest
```

If LocalStack or Qdrant is not available, infrastructure-dependent tests are skipped with an explicit reason.

## Project Layout

```text
src/cloud_finops_agent/
├── agents/
│   └── factory.py
├── config/
│   └── settings.py
├── graph/
│   ├── blueprint.py
│   ├── nodes.py
│   ├── router.py
│   └── state.py
├── infrastructure/
│   └── aws_factory.py
├── models/
│   ├── analysis.py
│   ├── aws_resources.py
│   └── errors.py
├── prompts/
│   └── agent_prompts.py
└── tools/
    └── aws_discovery_tools.py
```
