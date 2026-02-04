#!/usr/bin/env python3
"""Run example questions and save outputs to files."""
import json
import os
import sys
from io import StringIO
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.main import run_agent


def run_questions(questions, outfile: Path, mode: str | None = None) -> None:
    """Run questions in the same process to reuse loaded models."""
    with outfile.open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(f"\n=== {q['id']} ===\nQ: {q['question']}\n\n")
            
            # Capture stdout
            old_stdout = sys.stdout
            sys.stdout = captured_output = StringIO()
            
            try:
                # Run in same process to reuse models
                run_agent(q["question"], mode=mode or "offline")
            except Exception as e:
                print(f"Error: {e}")
            finally:
                sys.stdout = old_stdout
                
            f.write(captured_output.getvalue())
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
