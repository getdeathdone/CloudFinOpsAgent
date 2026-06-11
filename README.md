# Cloud FinOps AI Agent

Production-ready pet project for automated AWS infrastructure audit and FinOps optimization with LangGraph, LocalStack, Qdrant, and Langfuse.

## Local Infrastructure

```powershell
Copy-Item .env.example .env
docker compose up -d
```

Services:

- LocalStack: http://localhost:4566
- Qdrant: http://localhost:6333
- Langfuse: http://localhost:3000

## Project Layout

Application code will live under `src/cloud_finops_agent`. Agent implementations are intentionally left for the next step.
