# RAGAS Evaluation Guide

## Overview

The agent includes RAGAS (Retrieval Augmented Generation Assessment) evaluation to measure RAG performance quality:

- **Context Precision**: How relevant are retrieved documents?
- **Faithfulness**: Is the answer grounded in the context?
- **Answer Relevancy**: Does the answer address the question?

## Setup

### 1. Install Dependencies

```bash
pip install -e ".[dev]"
```

### 2. Get Google API Key (for RAGAS metrics)

RAGAS uses Google Gemini as an LLM judge to evaluate answers:

1. Visit: https://aistudio.google.com/app/apikey
2. Create a new API key
3. Add to `.env`:
   ```bash
   GOOGLE_API_KEY=your_key_here
   ```

**Note:** This is ONLY for evaluation metrics, not for agent queries (which use Ollama).

**Free Tier:** Limited RPM/RPD and may not complete the full dataset. Use paid tier or reduce dataset size/metrics for reliable full runs.

## Running Evaluation

### Local Evaluation

```bash
# Offline mode (15 questions)
python -m evaluation.evaluate --mode offline

# Online mode (requires TAVILY_API_KEY)
python -m evaluation.evaluate --mode online

# With custom output
python -m evaluation.evaluate --mode offline --output my-report.json
```

### What Gets Evaluated

Test dataset includes 15 questions across categories:

- Persistence & checkpointers
- State management
- Tools & streaming
- Error handling
- Human-in-the-loop
- Deployment & debugging

See: `evaluation/dataset.json`

### Understanding Results

**Report Location:** `evaluation/reports/`

**Key Metrics:**

```json
{
  "aggregate_metrics": {
    "success_rate": 1.0, // % questions answered
    "avg_aggregate_score": 0.85, // Overall score (0-1)
    "avg_context_relevancy": 0.9, // RAGAS: Context precision
    "avg_faithfulness": 0.88, // RAGAS: Answer faithfulness
    "avg_answer_relevancy": 0.92, // RAGAS: Answer relevancy
    "avg_topic_coverage": 0.75, // % expected topics mentioned
    "avg_code_validity": 0.95, // % valid code snippets
    "avg_latency_ms": 2500 // Average response time
  }
}
```

**Score Breakdown:**

- `0.9 - 1.0`: Excellent
- `0.8 - 0.9`: Good
- `0.7 - 0.8`: Acceptable
- `< 0.7`: Needs improvement

## Customizing Evaluation

### Add Questions

Edit `evaluation/dataset.json`:

```json
{
  "id": "my-question-001",
  "question": "How do I ...?",
  "category": "custom",
  "difficulty": "medium",
  "expected_topics": ["topic1", "topic2"],
  "expected_code": true,
  "ground_truth_snippets": ["code pattern"],
  "reference_answer": "Expected answer..."
}
```

### Adjust Metrics

Edit `evaluation/metrics.py` to change:

- Scoring weights
- Topic coverage calculation
- Code validity checks

## Cost & Configuration

See [docs/RAGAS_EVALUATION.md](docs/RAGAS_EVALUATION.md) for:
- Configuring the Gemini model via environment variables
- Cost estimation
- Free tier limits and troubleshooting

## Troubleshooting

### "GOOGLE_API_KEY not found"

Add to `.env`:

```bash
GOOGLE_API_KEY=your_key_here
```

### "Rate limit exceeded"

Google free tier limits:

- 1500 requests/day
- 60 requests/minute

Wait or upgrade to paid tier.

### "No valid results with contexts"

Vector store not built. Run:

```bash
python scripts/build_vectorstore.py
```

### Low scores

Check:

1. **Vector store quality:** Rebuild with fresh docs
2. **Retrieval K:** Increase in config.yaml (default: 5)
3. **LLM model:** Try larger Ollama model (llama3.2:7b)
4. **Test questions:** Ensure they match documentation scope

## Best Practices

1. **Run locally first** before CI/CD
2. **Baseline evaluation** before major changes
3. **Track trends** over time (score history)
4. **Review failures** in individual_results
5. **Update dataset** as documentation evolves

## Resources

- RAGAS Docs: https://docs.ragas.io/
- Google AI Studio: https://aistudio.google.com/
- Evaluation Metrics: `evaluation/metrics.py`
- Test Dataset: `evaluation/dataset.json`
