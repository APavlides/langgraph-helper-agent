#!/usr/bin/env python3
"""Run example questions and save outputs to files."""
import json
import os
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.graph import create_agent
from src.config import AgentMode, Settings


def run_questions(questions, outfile: Path, mode: str | None = None) -> None:
    """Run questions in the same process to reuse loaded models."""
    agent_mode = mode or "offline"
    settings = Settings(mode=AgentMode(agent_mode))
    agent = create_agent(settings)

    with outfile.open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(f"\n=== {q['id']} ===\nQ: {q['question']}\n\n")

            try:
                result = agent.invoke(
                    {
                        "messages": [HumanMessage(content=q["question"])],
                        "retrieved_contexts": [],
                        "mode": agent_mode,
                        "retrieval_score": None,
                        "web_search_results": None,
                    }
                )
                if result.get("messages"):
                    response = result["messages"][-1].content
                    f.write(response + "\n")
                else:
                    f.write("No response returned.\n")
            except Exception as e:
                f.write(f"Error: {e}\n")

            f.flush()  # Ensure output is written immediately


def main() -> None:
    data = json.load(open("evaluation/dataset.json", "r", encoding="utf-8"))
    questions = data["questions"]

    outdir = Path("example_outputs")
    outdir.mkdir(exist_ok=True)

    offline_file = outdir / "offline_outputs.txt"
    run_questions(questions, offline_file, mode="offline")
    print(f"Wrote {offline_file}")

    tavily_key = os.environ.get("TAVILY_API_KEY")
    if tavily_key:
        online_file = outdir / "online_outputs.txt"
        run_questions(questions, online_file, mode="online")
        print(f"Wrote {online_file}")
    else:
        print("TAVILY_API_KEY not set; skipped online outputs.")


if __name__ == "__main__":
    main()
