"""CLI entrypoint for running the Cloud FinOps AI Agent."""

from __future__ import annotations

import asyncio
import os
from typing import Annotated

import structlog
import typer
from langchain_core.messages import BaseMessage
from rich.console import Console
from rich.panel import Panel

from cloud_finops_agent.config.settings import get_settings
from cloud_finops_agent.graph.blueprint import compile_finops_graph
from cloud_finops_agent.graph.state import FinOpsGraphState, create_initial_state, validate_graph_state

app = typer.Typer(
    name="cloud-finops-agent",
    help="Run the Cloud FinOps AI Agent against LocalStack or AWS.",
    no_args_is_help=True,
)
console = Console()


def configure_logging() -> None:
    """Configure structured console logging."""

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.dev.ConsoleRenderer(),
        ],
    )


def _message_text(message: BaseMessage | object) -> str:
    """Return final graph message content as text."""

    content = getattr(message, "content", message)
    return content if isinstance(content, str) else str(content)


def _latest_report(state: FinOpsGraphState) -> str:
    """Extract the final Markdown report from graph messages."""

    messages = state.get("messages", [])
    if not messages:
        return "No report was generated."
    return _message_text(messages[-1])


async def run_audit_async(*, model: str | None, environment: str | None) -> FinOpsGraphState:
    """Compile and run the LangGraph audit asynchronously."""

    if model:
        os.environ["LLM_MODEL_NAME"] = model
    if environment:
        os.environ["ENVIRONMENT"] = environment
    get_settings.cache_clear()

    initial_state = create_initial_state(current_agent="agent-discovery")
    graph = await compile_finops_graph()
    result = await graph.ainvoke(initial_state)
    return validate_graph_state(result).to_langgraph_state()


@app.command("run")
def run_audit(
    model: Annotated[
        str | None,
        typer.Option("--model", help="LLM model name, for example gpt-4o-mini or claude-3-5-sonnet-latest."),
    ] = None,
    environment: Annotated[
        str | None,
        typer.Option("--env", help="Runtime environment: dev or prod."),
    ] = None,
) -> None:
    """Run a full Cloud FinOps audit and print the final Markdown report."""

    configure_logging()
    console.print(Panel.fit("Cloud FinOps AI Agent audit started", style="cyan"))
    final_state = asyncio.run(run_audit_async(model=model, environment=environment))
    validated_state = validate_graph_state(final_state)

    if validated_state.optimization_plan is not None:
        console.print(
            Panel.fit(
                (
                    f"Findings: {validated_state.optimization_plan.finding_count}\n"
                    "Potential monthly savings: "
                    f"${validated_state.optimization_plan.total_potential_monthly_savings}"
                ),
                title="Optimization Summary",
                style="green",
            )
        )

    if validated_state.errors:
        error_summary = "\n".join(
            f"{error.agent_name}: {error.error_type} retry_count={error.retry_count}"
            for error in validated_state.errors
        )
        console.print(Panel(error_summary, title="Errors", style="yellow"))

    console.print(Panel(_latest_report(final_state), title="Final Report", style="white"))


def main() -> None:
    """Run the Typer application."""

    app()


if __name__ == "__main__":
    main()
