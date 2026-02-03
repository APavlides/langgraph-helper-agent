"""Main evaluation runner for the LangGraph Helper Agent."""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Ragas imports - these will need to be installed
try:
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        context_precision,
        faithfulness,
        answer_relevancy,
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    print("Warning: Ragas not installed. Install with: pip install ragas datasets")

from langchain_core.messages import HumanMessage

from src.agent.graph import create_agent
from src.config import Settings, AgentMode
from evaluation.metrics import (
    EvaluationResult,
    AggregateMetrics,
    calculate_topic_coverage,
    check_code_presence,
    calculate_code_validity,
    check_snippet_presence,
    calculate_aggregate_metrics,
)


def load_evaluation_dataset(path: str = "evaluation/dataset.json") -> dict:
    """Load the evaluation dataset from JSON file."""
    with open(path) as f:
        return json.load(f)


def run_agent_query(
    agent, 
    question: str, 
    mode: AgentMode
) -> tuple[str, list[str], float]:
    """
    Run a single query through the agent.
    
    Args:
        agent: Compiled LangGraph agent
        question: User question to ask
        mode: Operating mode (offline/online)
        
    Returns:
        Tuple of (answer, retrieved_contexts, latency_ms)
    """
    start = time.perf_counter()
    
    # Prepare initial state
    initial_state = {
        "messages": [HumanMessage(content=question)],
        "retrieved_contexts": [],
        "mode": mode.value,
        "needs_web_search": False,
        "confidence_score": None,
        "web_search_results": None,
    }
    
    # Invoke agent
    result = agent.invoke(initial_state)
    
    latency_ms = (time.perf_counter() - start) * 1000
    
    # Extract answer from result
    answer = ""
    if result.get("messages"):
        last_message = result["messages"][-1]
        answer = last_message.content if hasattr(last_message, "content") else str(last_message)
    
    # Extract retrieved contexts
    contexts = result.get("retrieved_contexts", [])
    
    return answer, contexts, latency_ms


def evaluate_single_question(
    agent,
    question_data: dict,
    mode: AgentMode,
) -> EvaluationResult:
    """
    Evaluate the agent on a single question.
    
    Args:
        agent: Compiled LangGraph agent
        question_data: Question metadata from dataset
        mode: Operating mode
        
    Returns:
        EvaluationResult with all metrics
    """
    question = question_data["question"]
    
    try:
        # Run the agent
        answer, contexts, latency_ms = run_agent_query(agent, question, mode)
        
        # Create result object
        result = EvaluationResult(
            question_id=question_data["id"],
            question=question,
            answer=answer,
            contexts=contexts,
            mode=mode.value,
            latency_ms=latency_ms,
        )
        
        # Calculate custom metrics
        result.topic_coverage = calculate_topic_coverage(
            answer, 
            question_data.get("expected_topics", [])
        )
        
        # Check code presence if expected
        if question_data.get("expected_code", False):
            result.code_presence = 1.0 if check_code_presence(answer) else 0.0
        
        # Calculate code validity
        result.code_validity = calculate_code_validity(answer)
        
        return result
        
    except Exception as e:
        # Return error result
        return EvaluationResult(
            question_id=question_data["id"],
            question=question,
            answer="",
            contexts=[],
            mode=mode.value,
            error=str(e),
        )


def run_ragas_evaluation(results: list[EvaluationResult]) -> list[EvaluationResult]:
    """
    Run Ragas metrics on successful results.
    
    Args:
        results: List of evaluation results
        
    Returns:
        Updated results with Ragas metrics
    """
    if not RAGAS_AVAILABLE:
        print("Skipping Ragas evaluation - library not installed")
        return results
    
    # Filter successful results with contexts
    valid_results = [r for r in results if r.error is None and r.contexts]
    
    if not valid_results:
        print("No valid results with contexts for Ragas evaluation")
        return results
    
    print(f"Running Ragas evaluation on {len(valid_results)} results...")
    
    try:
        # Prepare dataset for Ragas
        data = {
            "question": [r.question for r in valid_results],
            "answer": [r.answer for r in valid_results],
            "contexts": [r.contexts for r in valid_results],
        }
        dataset = Dataset.from_dict(data)
        
        # Run Ragas evaluation
        ragas_results = ragas_evaluate(
            dataset,
            metrics=[context_precision, faithfulness, answer_relevancy],
        )
        
        # Map results back
        for i, result in enumerate(valid_results):
            if "context_precision" in ragas_results:
                result.context_relevancy = float(ragas_results["context_precision"][i])
            if "faithfulness" in ragas_results:
                result.faithfulness = float(ragas_results["faithfulness"][i])
            if "answer_relevancy" in ragas_results:
                result.answer_relevancy = float(ragas_results["answer_relevancy"][i])
                
    except Exception as e:
        print(f"Ragas evaluation failed: {e}")
    
    return results


def generate_report(
    results: list[EvaluationResult],
    aggregate: AggregateMetrics,
    mode: str,
    output_path: str,
) -> None:
    """
    Generate and save the evaluation report.
    
    Args:
        results: List of individual results
        aggregate: Aggregated metrics
        mode: Evaluation mode
        output_path: Path to save JSON report
    """
    report = {
        "metadata": {
            "timestamp": datetime.utcnow().isoformat(),
            "mode": mode,
            "total_questions": aggregate.total_questions,
            "successful_questions": aggregate.successful_questions,
            "failed_questions": aggregate.failed_questions,
        },
        "aggregate_metrics": {
            "success_rate": aggregate.successful_questions / aggregate.total_questions if aggregate.total_questions > 0 else 0,
            "avg_context_relevancy": aggregate.avg_context_relevancy,
            "avg_faithfulness": aggregate.avg_faithfulness,
            "avg_answer_relevancy": aggregate.avg_answer_relevancy,
            "avg_topic_coverage": aggregate.avg_topic_coverage,
            "avg_code_validity": aggregate.avg_code_validity,
            "avg_aggregate_score": aggregate.avg_aggregate_score,
            "avg_latency_ms": aggregate.avg_latency_ms,
        },
        "scores_by_category": aggregate.scores_by_category,
        "scores_by_difficulty": aggregate.scores_by_difficulty,
        "individual_results": [
            {
                "question_id": r.question_id,
                "question": r.question,
                "answer": r.answer[:500] + "..." if len(r.answer) > 500 else r.answer,
                "num_contexts": len(r.contexts),
                "context_relevancy": r.context_relevancy,
                "faithfulness": r.faithfulness,
                "answer_relevancy": r.answer_relevancy,
                "topic_coverage": r.topic_coverage,
                "code_validity": r.code_validity,
                "latency_ms": r.latency_ms,
                "error": r.error,
            }
            for r in results
        ],
    }
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save JSON report
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"Report saved to: {output_path}")


def print_summary(aggregate: AggregateMetrics, mode: str) -> None:
    """Print evaluation summary to console."""
    print("\n" + "=" * 60)
    print(f"EVALUATION SUMMARY - {mode.upper()} MODE")
    print("=" * 60)
    print(f"Questions: {aggregate.successful_questions}/{aggregate.total_questions} successful")
    print(f"\nAggregate Metrics:")
    print(f"  Avg Aggregate Score:    {aggregate.avg_aggregate_score:.3f}")
    print(f"  Avg Context Relevancy:  {aggregate.avg_context_relevancy:.3f}")
    print(f"  Avg Faithfulness:       {aggregate.avg_faithfulness:.3f}")
    print(f"  Avg Answer Relevancy:   {aggregate.avg_answer_relevancy:.3f}")
    print(f"  Avg Topic Coverage:     {aggregate.avg_topic_coverage:.3f}")
    print(f"  Avg Code Validity:      {aggregate.avg_code_validity:.3f}")
    print(f"  Avg Latency:            {aggregate.avg_latency_ms:.0f}ms")
    
    if aggregate.scores_by_category:
        print(f"\nScores by Category:")
        for cat, score in sorted(aggregate.scores_by_category.items()):
            print(f"  {cat:20s}: {score:.3f}")
    
    if aggregate.scores_by_difficulty:
        print(f"\nScores by Difficulty:")
        for diff, score in sorted(aggregate.scores_by_difficulty.items()):
            print(f"  {diff:20s}: {score:.3f}")
    
    print("=" * 60)


def main():
    """Main entry point for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate LangGraph Helper Agent"
    )
    parser.add_argument(
        "--mode",
        choices=["offline", "online"],
        default="offline",
        help="Agent mode to evaluate (default: offline)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for JSON report (default: evaluation/reports/report-{mode}-{timestamp}.json)",
    )
    parser.add_argument(
        "--dataset",
        default="evaluation/dataset.json",
        help="Path to evaluation dataset",
    )
    parser.add_argument(
        "--skip-ragas",
        action="store_true",
        help="Skip Ragas metrics (faster, but less comprehensive)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output for each question",
    )
    
    args = parser.parse_args()
    
    # Set default output path
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output = f"evaluation/reports/report-{args.mode}-{timestamp}.json"
    
    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    dataset = load_evaluation_dataset(args.dataset)
    questions = dataset["questions"]
    print(f"Loaded {len(questions)} questions")
    
    # Create agent
    print(f"\nInitializing agent in {args.mode} mode...")
    mode = AgentMode(args.mode)
    try:
        settings = Settings(mode=mode)
        agent = create_agent(settings)
    except Exception as e:
        print(f"Failed to create agent: {e}")
        return 1
    
    # Evaluate each question
    print(f"\nRunning evaluation...")
    results = []
    
    for i, q in enumerate(questions):
        print(f"  [{i+1}/{len(questions)}] {q['id']}: {q['question'][:50]}...")
        
        result = evaluate_single_question(agent, q, mode)
        results.append(result)
        
        if args.verbose:
            if result.error:
                print(f"    ❌ ERROR: {result.error}")
            else:
                print(f"    ✓ Topic coverage: {result.topic_coverage:.2f}, Latency: {result.latency_ms:.0f}ms")
    
    # Run Ragas evaluation
    if not args.skip_ragas:
        print("\nRunning Ragas metrics...")
        results = run_ragas_evaluation(results)
    
    # Calculate aggregates
    questions_metadata = {q["id"]: q for q in questions}
    aggregate = calculate_aggregate_metrics(results, questions_metadata)
    
    # Generate report
    generate_report(results, aggregate, args.mode, args.output)
    
    # Print summary
    print_summary(aggregate, args.mode)
    
    return 0


if __name__ == "__main__":
    exit(main())
