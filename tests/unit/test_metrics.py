"""Unit tests for evaluation metrics."""

import pytest

from evaluation.metrics import (
    calculate_topic_coverage,
    check_code_presence,
    extract_code_blocks,
    validate_python_syntax,
    calculate_code_validity,
    check_snippet_presence,
    EvaluationResult,
    calculate_aggregate_score,
)


class TestTopicCoverage:
    """Tests for topic coverage calculation."""
    
    def test_full_coverage(self):
        answer = "You can use MemorySaver or SqliteSaver as a checkpointer."
        topics = ["MemorySaver", "SqliteSaver", "checkpointer"]
        assert calculate_topic_coverage(answer, topics) == 1.0
    
    def test_partial_coverage(self):
        answer = "Use MemorySaver for persistence."
        topics = ["MemorySaver", "SqliteSaver", "PostgresSaver"]
        assert calculate_topic_coverage(answer, topics) == pytest.approx(1/3)
    
    def test_no_coverage(self):
        answer = "I don't know about that topic."
        topics = ["MemorySaver", "checkpointer"]
        assert calculate_topic_coverage(answer, topics) == 0.0
    
    def test_empty_topics(self):
        answer = "Some answer"
        assert calculate_topic_coverage(answer, []) == 1.0
    
    def test_case_insensitive(self):
        answer = "Use MEMORYSAVER for persistence."
        topics = ["memorysaver"]
        assert calculate_topic_coverage(answer, topics) == 1.0


class TestCodePresence:
    """Tests for code presence detection."""
    
    def test_fenced_code_block(self):
        answer = """Here's an example:
```python
from langgraph import StateGraph
```
"""
        assert check_code_presence(answer) is True
    
    def test_inline_code(self):
        answer = "Use `MemorySaver()` for persistence."
        assert check_code_presence(answer) is True
    
    def test_no_code(self):
        answer = "LangGraph is a framework for building AI agents."
        assert check_code_presence(answer) is False


class TestExtractCodeBlocks:
    """Tests for code block extraction."""
    
    def test_single_block(self):
        answer = """Example:
```python
x = 1
```
"""
        blocks = extract_code_blocks(answer)
        assert len(blocks) == 1
        assert "x = 1" in blocks[0]
    
    def test_multiple_blocks(self):
        answer = """First:
```python
x = 1
```
Second:
```python
y = 2
```
"""
        blocks = extract_code_blocks(answer)
        assert len(blocks) == 2
    
    def test_no_blocks(self):
        answer = "Just text, no code."
        assert extract_code_blocks(answer) == []


class TestValidatePythonSyntax:
    """Tests for Python syntax validation."""
    
    def test_valid_syntax(self):
        code = "from langgraph import StateGraph\ngraph = StateGraph()"
        assert validate_python_syntax(code) is True
    
    def test_invalid_syntax(self):
        code = "def foo(\n  # missing closing"
        assert validate_python_syntax(code) is False
    
    def test_empty_code(self):
        assert validate_python_syntax("") is True


class TestCodeValidity:
    """Tests for code validity calculation."""
    
    def test_all_valid(self):
        answer = """
```python
x = 1
```
```python
y = 2
```
"""
        assert calculate_code_validity(answer) == 1.0
    
    def test_mixed_validity(self):
        answer = """
```python
x = 1
```
```python
def broken(
```
"""
        assert calculate_code_validity(answer) == 0.5
    
    def test_no_code_blocks(self):
        answer = "No code here."
        assert calculate_code_validity(answer) == 1.0  # Not applicable


class TestSnippetPresence:
    """Tests for expected snippet presence."""
    
    def test_all_present(self):
        answer = "Use from langgraph.checkpoint.memory import MemorySaver"
        snippets = ["MemorySaver", "langgraph.checkpoint"]
        assert check_snippet_presence(answer, snippets) == 1.0
    
    def test_partial_present(self):
        answer = "Import MemorySaver for persistence."
        snippets = ["MemorySaver", "SqliteSaver"]
        assert check_snippet_presence(answer, snippets) == 0.5
    
    def test_empty_snippets(self):
        assert check_snippet_presence("any answer", []) == 1.0


class TestAggregateScore:
    """Tests for aggregate score calculation."""
    
    def test_full_scores(self):
        result = EvaluationResult(
            question_id="test-1",
            question="Test?",
            answer="Answer",
            contexts=["ctx"],
            context_relevancy=0.8,
            faithfulness=0.9,
            answer_relevancy=0.7,
            topic_coverage=0.8,
            code_validity=1.0,
        )
        score = calculate_aggregate_score(result)
        # Weighted average: 0.8*0.2 + 0.9*0.2 + 0.7*0.2 + 0.8*0.25 + 1.0*0.15
        expected = (0.8*0.2 + 0.9*0.2 + 0.7*0.2 + 0.8*0.25 + 1.0*0.15)
        assert score == pytest.approx(expected)
    
    def test_partial_scores(self):
        result = EvaluationResult(
            question_id="test-1",
            question="Test?",
            answer="Answer",
            contexts=[],
            topic_coverage=0.5,
            code_validity=0.5,
        )
        score = calculate_aggregate_score(result)
        assert score == pytest.approx(0.5)  # Both metrics are 0.5
    
    def test_no_scores(self):
        result = EvaluationResult(
            question_id="test-1",
            question="Test?",
            answer="Answer",
            contexts=[],
        )
        score = calculate_aggregate_score(result)
        assert score == 0.0
