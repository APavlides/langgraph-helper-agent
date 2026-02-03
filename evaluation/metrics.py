"""Custom evaluation metrics for the LangGraph Helper Agent."""

from dataclasses import dataclass
from typing import Any
import re


@dataclass
class EvaluationResult:
    """Result of evaluating a single question."""
    question_id: str
    question: str
    answer: str
    contexts: list[str]
    
    # Ragas metrics (0-1 scale)
    context_relevancy: float | None = None
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    
    # Custom metrics (0-1 scale)
    topic_coverage: float | None = None
    code_presence: float | None = None
    code_validity: float | None = None
    
    # Metadata
    mode: str = "offline"
    latency_ms: float | None = None
    error: str | None = None


def calculate_topic_coverage(
    answer: str, 
    expected_topics: list[str]
) -> float:
    """
    Calculate what fraction of expected topics are mentioned in the answer.
    
    This metric checks if the answer covers the key concepts that should
    be addressed for a given question. Topics are matched case-insensitively.
    
    Args:
        answer: The generated answer text
        expected_topics: List of topics that should be covered
        
    Returns:
        Float between 0 and 1 representing coverage
        
    Examples:
        >>> calculate_topic_coverage("Use MemorySaver", ["MemorySaver", "SqliteSaver"])
        0.5
        >>> calculate_topic_coverage("Use MemorySaver or SqliteSaver", ["MemorySaver", "SqliteSaver"])
        1.0
    """
    if not expected_topics:
        return 1.0
    
    answer_lower = answer.lower()
    found = sum(
        1 for topic in expected_topics 
        if topic.lower() in answer_lower
    )
    return found / len(expected_topics)


def check_code_presence(answer: str) -> bool:
    """
    Check if the answer contains code blocks.
    
    Detects both fenced code blocks (```...```) and inline code (`...`).
    
    Args:
        answer: The generated answer text
        
    Returns:
        True if code is present, False otherwise
    """
    patterns = [
        r"```[\s\S]*?```",  # Fenced code blocks
        r"`[^`]+`",          # Inline code
    ]
    return any(re.search(p, answer) for p in patterns)


def extract_code_blocks(answer: str) -> list[str]:
    """
    Extract all fenced code blocks from an answer.
    
    Args:
        answer: The generated answer text
        
    Returns:
        List of code block contents (without the ``` delimiters)
    """
    pattern = r"```(?:python)?\n?([\s\S]*?)```"
    matches = re.findall(pattern, answer)
    return [m.strip() for m in matches if m.strip()]


def validate_python_syntax(code: str) -> bool:
    """
    Check if code has valid Python syntax.
    
    Args:
        code: Python code string to validate
        
    Returns:
        True if syntax is valid, False otherwise
    """
    try:
        compile(code, "<string>", "exec")
        return True
    except SyntaxError:
        return False


def calculate_code_validity(answer: str) -> float:
    """
    Calculate what fraction of code blocks have valid Python syntax.
    
    This metric helps ensure that code examples in answers are actually
    runnable and not just plausible-looking pseudocode.
    
    Args:
        answer: The generated answer text
        
    Returns:
        Float between 0 and 1. Returns 1.0 if no code blocks (not applicable).
    """
    code_blocks = extract_code_blocks(answer)
    if not code_blocks:
        return 1.0
    
    valid = sum(1 for code in code_blocks if validate_python_syntax(code))
    return valid / len(code_blocks)


def check_snippet_presence(
    answer: str, 
    expected_snippets: list[str]
) -> float:
    """
    Check how many expected code snippets appear in the answer.
    
    This is more specific than topic coverage - it checks for exact
    code patterns that should appear in a correct answer.
    
    Args:
        answer: The generated answer
        expected_snippets: Code patterns that should appear
        
    Returns:
        Float between 0 and 1
    """
    if not expected_snippets:
        return 1.0
    
    found = sum(
        1 for snippet in expected_snippets
        if snippet in answer
    )
    return found / len(expected_snippets)


def calculate_aggregate_score(result: EvaluationResult) -> float:
    """
    Calculate a weighted aggregate score for a single evaluation result.
    
    Weighting:
    - Ragas metrics: 60% total (20% each for context_relevancy, faithfulness, answer_relevancy)
    - Topic coverage: 25%
    - Code validity: 15%
    
    Args:
        result: EvaluationResult with metrics filled in
        
    Returns:
        Weighted average score between 0 and 1
    """
    scores = []
    weights = []
    
    if result.context_relevancy is not None:
        scores.append(result.context_relevancy)
        weights.append(0.20)
    
    if result.faithfulness is not None:
        scores.append(result.faithfulness)
        weights.append(0.20)
    
    if result.answer_relevancy is not None:
        scores.append(result.answer_relevancy)
        weights.append(0.20)
    
    if result.topic_coverage is not None:
        scores.append(result.topic_coverage)
        weights.append(0.25)
    
    if result.code_validity is not None:
        scores.append(result.code_validity)
        weights.append(0.15)
    
    if not scores:
        return 0.0
    
    # Normalize weights to sum to 1
    total_weight = sum(weights)
    return sum(s * w for s, w in zip(scores, weights)) / total_weight


@dataclass
class AggregateMetrics:
    """Aggregate metrics across all evaluation questions."""
    total_questions: int
    successful_questions: int
    failed_questions: int
    
    # Average Ragas metrics
    avg_context_relevancy: float
    avg_faithfulness: float
    avg_answer_relevancy: float
    
    # Average custom metrics
    avg_topic_coverage: float
    avg_code_validity: float
    
    # Overall
    avg_aggregate_score: float
    avg_latency_ms: float
    
    # Breakdown
    scores_by_category: dict[str, float]
    scores_by_difficulty: dict[str, float]


def calculate_aggregate_metrics(
    results: list[EvaluationResult],
    questions_metadata: dict[str, dict[str, Any]]
) -> AggregateMetrics:
    """
    Calculate aggregate metrics from a list of evaluation results.
    
    Args:
        results: List of individual evaluation results
        questions_metadata: Dictionary mapping question_id to metadata
            (should include 'category' and 'difficulty' keys)
            
    Returns:
        AggregateMetrics with computed averages and breakdowns
    """
    successful = [r for r in results if r.error is None]
    
    def safe_avg(values: list[float | None]) -> float:
        """Calculate average of non-None values."""
        valid = [v for v in values if v is not None]
        return sum(valid) / len(valid) if valid else 0.0
    
    # Group by category and difficulty
    scores_by_category: dict[str, list[float]] = {}
    scores_by_difficulty: dict[str, list[float]] = {}
    
    for result in successful:
        score = calculate_aggregate_score(result)
        meta = questions_metadata.get(result.question_id, {})
        
        category = meta.get("category", "unknown")
        difficulty = meta.get("difficulty", "unknown")
        
        scores_by_category.setdefault(category, []).append(score)
        scores_by_difficulty.setdefault(difficulty, []).append(score)
    
    return AggregateMetrics(
        total_questions=len(results),
        successful_questions=len(successful),
        failed_questions=len(results) - len(successful),
        avg_context_relevancy=safe_avg([r.context_relevancy for r in successful]),
        avg_faithfulness=safe_avg([r.faithfulness for r in successful]),
        avg_answer_relevancy=safe_avg([r.answer_relevancy for r in successful]),
        avg_topic_coverage=safe_avg([r.topic_coverage for r in successful]),
        avg_code_validity=safe_avg([r.code_validity for r in successful]),
        avg_aggregate_score=safe_avg([calculate_aggregate_score(r) for r in successful]),
        avg_latency_ms=safe_avg([r.latency_ms for r in successful]),
        scores_by_category={k: sum(v)/len(v) for k, v in scores_by_category.items()},
        scores_by_difficulty={k: sum(v)/len(v) for k, v in scores_by_difficulty.items()},
    )
