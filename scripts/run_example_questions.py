#!/usr/bin/env python3
"""Run example questions and save outputs to files."""
import json
import os
import subprocess
import sys
from pathlib import Path


def run_questions(questions, outfile: Path, mode: str | None = None) -> None:
    args_prefix = [sys.executable, "-m", "src.main"]
    if mode:
        args_prefix += ["--mode", mode]

    with outfile.open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(f"\n=== {q['id']} ===\nQ: {q['question']}\n\n")
            result = subprocess.run(
                args_prefix + [q["question"]],
                capture_output=True,
                text=True,
            )
            f.write(result.stdout)
            if result.stderr:
                f.write("\n[stderr]\n" + result.stderr + "\n")


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
