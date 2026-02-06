"""CLI entry point for the LangGraph Helper Agent."""

import argparse
import sys
from typing import Any, Optional

from langchain_core.messages import HumanMessage
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from src.agent.graph import create_agent
from src.config import AgentMode, Settings

console = Console()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LangGraph Helper Agent - Get help with LangGraph and LangChain",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single question in offline mode
  python -m src.main --mode offline "How do I add persistence?"
  
  # Interactive chat in online mode
  python -m src.main --mode online --interactive
  
  # Use environment variable for mode
  export AGENT_MODE=online
  python -m src.main "What's new in LangGraph?"
        """,
    )

    parser.add_argument(
        "question",
        nargs="?",
        help="Question to ask (omit for interactive mode)",
    )

    parser.add_argument(
        "--mode",
        "-m",
        choices=["offline", "online"],
        default=None,
        help="Operating mode (default: from AGENT_MODE env var, or 'offline')",
    )

    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run in interactive chat mode",
    )

    parser.add_argument(
        "--show-graph",
        action="store_true",
        help="Display the agent graph structure and exit",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show additional information (contexts, confidence)",
    )

    return parser.parse_args()


def display_welcome(settings: Settings) -> None:
    """Display welcome message with current configuration."""
    resolved_mode = settings.mode or AgentMode.OFFLINE
    mode_color = "green" if resolved_mode == AgentMode.OFFLINE else "blue"
    mode_text = f"[bold {mode_color}]{resolved_mode.value.upper()}[/]"

    console.print(
        Panel(
            f"""[bold]LangGraph Helper Agent[/bold]
        
Mode: {mode_text}
Model: {settings.llm_model}
Retrieval docs: {settings.retrieval_k}

Type your question or 'quit' to exit.
Type 'help' for available commands.""",
            title="Welcome",
            border_style="cyan",
        )
    )


def display_response(
    response: str, verbose: bool = False, confidence: Optional[float] = None
) -> None:
    """Display the agent's response."""
    # Render as markdown for nice formatting
    md = Markdown(response)
    console.print(Panel(md, title="Answer", border_style="green"))

    if verbose and confidence is not None:
        confidence_color = (
            "green" if confidence >= 0.7 else "yellow" if confidence >= 0.5 else "red"
        )
        console.print(f"[dim]Confidence: [{confidence_color}]{confidence:.2f}[/][/dim]")


def run_single_query(
    agent: Any, question: str, mode: AgentMode, verbose: bool = False
) -> None:
    """Run a single query through the agent."""
    with console.status("[bold cyan]Thinking...[/]"):
        result = agent.invoke(
            {
                "messages": [HumanMessage(content=question)],
                "retrieved_contexts": [],
                "mode": mode.value,
                "retrieval_score": None,
                "web_search_results": None,
            }
        )

    # Extract response
    if result["messages"]:
        last_message = result["messages"][-1]
        response = last_message.content

        display_response(response, verbose)

        if verbose and result.get("retrieved_contexts"):
            console.print(
                f"\n[dim]Retrieved {len(result['retrieved_contexts'])} context chunks[/dim]"
            )
            if result.get("web_search_results"):
                console.print(
                    f"[dim]Web search results: {len(result['web_search_results'])}[/dim]"
                )


def run_interactive(agent: Any, settings: Settings, verbose: bool = False) -> None:
    """Run interactive chat mode."""
    display_welcome(settings)

    while True:
        try:
            user_input = Prompt.ask("\n[bold cyan]You[/]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        # Handle commands
        user_input_lower = user_input.lower().strip()

        if user_input_lower in ("quit", "exit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break

        if user_input_lower == "help":
            console.print(
                """
[bold]Available commands:[/]
  help     - Show this help message
  graph    - Display the agent graph structure
  mode     - Show current mode
  quit     - Exit the chat
            """
            )
            continue

        if user_input_lower == "graph":
            console.print("[yellow]Graph visualization not available[/yellow]")
            continue

        if user_input_lower == "mode":
            current_mode = settings.mode or AgentMode.OFFLINE
            console.print(f"Current mode: [bold]{current_mode.value}[/bold]")
            continue

        if not user_input.strip():
            continue

        # Run query
        run_single_query(agent, user_input, settings.mode, verbose)


def run_agent(question: str, mode: str = "offline", verbose: bool = False) -> None:
    """Run a single question through the agent (for scripting).

    Args:
        question: Question to ask
        mode: "offline" or "online"
        verbose: Show additional information
    """
    # Create settings
    mode_enum = AgentMode(mode)
    settings = Settings(mode=mode_enum)

    # Create agent (cached globally if needed)
    agent = create_agent(settings)

    # Run query
    result = agent.invoke(
        {
            "messages": [HumanMessage(content=question)],
            "retrieved_contexts": [],
            "mode": mode,
            "retrieval_score": None,
            "web_search_results": None,
        }
    )

    # Extract and display response
    if result["messages"]:
        last_message = result["messages"][-1]
        response = last_message.content

        display_response(response, verbose)

        if verbose and result.get("retrieved_contexts"):
            console.print(
                f"\n[dim]Retrieved {len(result['retrieved_contexts'])} context chunks[/dim]"
            )
            if result.get("web_search_results"):
                console.print(
                    f"[dim]Web search results: {len(result['web_search_results'])}[/dim]"
                )


def main() -> int:
    """Main entry point."""
    args = parse_args()

    try:
        # Create settings
        mode = AgentMode(args.mode) if args.mode else None
        settings = Settings(mode=mode)
        resolved_mode = settings.mode or AgentMode.OFFLINE

        # Show graph and exit if requested
        if args.show_graph:
            console.print("[yellow]Graph visualization not available[/yellow]")
            return 0

        # Create agent
        with console.status("[bold cyan]Initializing agent...[/]"):
            agent = create_agent(settings)

        # Run in appropriate mode
        if args.interactive or args.question is None:
            run_interactive(agent, settings, args.verbose)
        else:
            run_single_query(agent, args.question, resolved_mode, args.verbose)

        return 0

    except ValueError as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        return 1
    except FileNotFoundError as e:
        console.print(f"[red]File not found:[/red] {e}")
        console.print(
            "[dim]Run 'python scripts/build_vectorstore.py' to create the vector store.[/dim]"
        )
        return 1
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
        return 130
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        if args.verbose:
            console.print_exception()
        return 1


if __name__ == "__main__":
    sys.exit(main())
