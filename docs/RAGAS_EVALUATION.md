# RAGAS Evaluation Guide

## Overview

RAGAS (Retrieval-Augmented Generation Assessment) provides advanced evaluation metrics using Google Gemini AI. The free tier is useful for smoke tests, but it often cannot complete the full dataset due to RPM/RPD limits.

## Why RAGAS is Disabled by Default

- **Requires API key**: Needs Google AI Studio API key (free tier available)
- **Adds evaluation time**: RAGAS adds ~2-5 minutes to evaluation
- **Rate limits**: Free tier has low RPM/RPD and can time out before completing the dataset
- **Optional advanced metrics**: Custom metrics provide good baseline evaluation

## Getting Started (Free Tier)

1. **Get a free API key** from Google AI Studio:
   - Visit: https://aistudio.google.com/app/apikey
   - Click "Create API Key"
   - Copy the key

2. **Set the API key** in your .env file:

   ```bash
   GOOGLE_API_KEY=your-api-key-here
   ```

3. **Run evaluation with RAGAS**:
   ```bash
   docker compose --env-file .env run --rm -e OLLAMA_BASE_URL=http://host.docker.internal:11434 dev -c "python -m evaluation.evaluate --ragas"
   ```

## Free Tier Limits

Google AI Studio free tier provides limited RPM/RPD. For larger evaluations, free-tier limits can lead to timeouts and incomplete RAGAS results.

Quota resets daily at midnight PT. Monitor usage at: https://ai.google.dev/rate-limit

## Configuration

Customize via environment variables:

```bash
# Override Gemini model (default: gemini-2.5-flash)
export GOOGLE_GEMINI_MODEL=gemini-2.5-flash

# Or use a different model available on your account
export GOOGLE_GEMINI_MODEL=gemini-2.5-flash-lite
```

## Troubleshooting

### Quota Exhausted (429 Error)

If you see `RESOURCE_EXHAUSTED` or `429`:

1. **Wait**: Quota resets daily at midnight PT
2. **Check usage**: https://ai.google.dev/rate-limit
3. **Upgrade** (optional): https://ai.google.dev/pricing for higher limits

### Model Not Found (404 Error)

Try these models:

- `gemini-2.5-flash` (default)
- `gemini-2.5-flash-lite`
- `gemini-3-flash`

### API Key Issues

- Verify key is set: `echo $GOOGLE_API_KEY`
- Get new key: https://aistudio.google.com/app/apikey
- Check key hasn't expired or been revoked

## RAGAS Metrics

When enabled, RAGAS evaluates:

- **Context Precision**: Quality of retrieved documents
- **Faithfulness**: How faithful the answer is to retrieved context
- **Answer Relevancy**: How relevant the answer is to the question

## Custom Metrics (Always Available)

Default evaluation uses these (no API needed):

- **Topic Coverage**: Fraction of expected topics mentioned
- **Code Validity**: Python syntax validation of code examples
- **Code Presence**: Whether code examples were provided
- **Aggregate Score**: Combined metric score

## Cost Considerations

Free tier is often insufficient to complete the full dataset because of RPM/RPD limits. For reliable full runs, use a paid tier or reduce dataset size/metrics.
