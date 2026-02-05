# RAGAS Evaluation Guide

## Overview

RAGAS (Retrieval-Augmented Generation Assessment) provides advanced evaluation metrics using Google Gemini AI. The free tier from Google AI Studio works great for demo projects.

## Why RAGAS is Disabled by Default

- **Requires API key**: Needs Google AI Studio API key (free tier available)
- **Adds evaluation time**: RAGAS adds ~2-5 minutes to evaluation
- **Rate limits**: Free tier has daily quotas (sufficient for demos)
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

Google AI Studio free tier provides:
- **Rate limits**: 15 requests per minute (RPM)
- **Daily quota**: 1,500 requests per day (RPD)
- **Models**: gemini-1.5-flash, gemini-1.5-pro, gemini-pro
- **Sufficient** for running evaluations a few times per day

Quota resets daily at midnight PT. Monitor usage at: https://ai.google.dev/rate-limit

## Configuration

Customize via environment variables:

```bash
# Override Gemini model (default: gemini-1.5-flash)
export GOOGLE_GEMINI_MODEL=gemini-1.5-flash

# Or use a different model
export GOOGLE_GEMINI_MODEL=gemini-1.5-pro
```

## Troubleshooting

### Quota Exhausted (429 Error)

If you see `RESOURCE_EXHAUSTED` or `429`:

1. **Wait**: Quota resets daily at midnight PT
2. **Check usage**: https://ai.google.dev/rate-limit
3. **Upgrade** (optional): https://ai.google.dev/pricing for higher limits

### Model Not Found (404 Error)

Try these models:
- `gemini-1.5-flash` (recommended, fastest)
- `gemini-1.5-pro` (better quality, slower)
- `gemini-pro` (older, good compatibility)

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

Free tier is sufficient for:
- Running evaluation 5-10 times per day
- Demo projects and development
- Small datasets (<50 questions)

For production or large-scale evaluation, consider upgrading to paid tier.
