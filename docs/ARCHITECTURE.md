# Architecture

## Overview

This project implements a **LangGraph-based RAG agent** with dual operating modes (offline/online). The architecture emphasizes reproducibility, modularity, and intelligent routing between local documentation and web search.

## Graph Design

Built with **LangGraph** (≥ 0.2.0) as a state machine with the following nodes:

1. **retrieve** - Fetches relevant docs from FAISS vector store, reranks with cross-encoder for quality assessment
2. **generate** - Creates answer using ChatOllama (llama3.2:3b)
3. **web_search_and_generate** - Searches web via Tavily API + generates enhanced answer (online mode)
4. **route_after_retrieve** - Decides whether to use docs or web search based on reranker scores

### Routing Logic

```
Query → Retrieve + Rerank → Route
                           ├─> [reranker score < 0.0 & online mode] → Web Search + Generate → END
                           └─> [else] → Generate → END
```

**Reranking:** Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` to assess retrieval quality. Scores < 0.0 indicate poor relevance, triggering web search in online mode.

## State Management

Uses TypedDict state schema with message history:

- `messages` - Conversation history (HumanMessage, AIMessage)
- `retrieved_contexts` - Documentation chunks from vector store
- `retrieval_score` - Cross-encoder reranking score (negative = poor relevance)
- `mode` - offline or online
- `web_search_results` - Search results (online mode only)

## Technology Stack

### Required Dependencies

- **LangGraph** ≥ 0.2.0 - State machine framework
- **LangChain** ≥ 0.3.0 - LLM abstractions
- **langchain-ollama** ≥ 0.1.0 - Ollama integration
- **sentence-transformers** ≥ 2.2.0 - Cross-encoder reranking

### Core Components

- **Ollama** - Local LLM (llama3.2:3b) and embeddings (nomic-embed-text)
- **FAISS** - Vector store with 10,943 chunks from official docs
- **Cross-encoder** - ms-marco-MiniLM-L-6-v2 for retrieval quality assessment
- **Python** 3.10+

### Optional Components

- **Tavily API** - Web search (online mode)
- **RAGAS + Google Gemini** - Advanced evaluation metrics

## Offline Mode Architecture

**Data Flow:**

1. User query → Embed with `nomic-embed-text`
2. FAISS retrieves top-5 relevant chunks
3. Cross-encoder reranks for quality
4. LLM generates answer using retrieved context
5. No external API calls

**Data Sources:**

- LangGraph docs: `https://langchain-ai.github.io/langgraph/llms.txt`
- LangChain docs: `https://docs.langchain.com/llms.txt`
- Pre-indexed FAISS (10,943 chunks, 1000 char chunks, 200 overlap)

## Online Mode Architecture

**Data Flow:**

1. User query → Embed with `nomic-embed-text`
2. FAISS retrieves top-5 relevant chunks
3. Cross-encoder reranks and scores
4. If score < 0.0 (poor relevance):
   - Trigger Tavily web search
   - Combine web results with docs
   - LLM generates enhanced answer
5. Otherwise:
   - Use docs only (same as offline)

**Tavily Integration:**

- AI-optimized search results
- Structured output (title, URL, snippet)
- Free tier: 1,000 searches/month

## Evaluation Architecture

### Custom Metrics (No LLM Required)

Rule-based evaluation using:

- **Topic Coverage** - Keyword matching
- **Code Presence** - Regex detection
- **Code Validity** - Python AST parsing
- **Aggregate Score** - Weighted average

### RAGAS Metrics (LLM-as-Judge)

Optional semantic evaluation using Google Gemini:

- **Context Precision** - Retrieval quality
- **Faithfulness** - Grounding in context
- **Answer Relevancy** - Question alignment

## Configuration System

**Priority Order:**

1. Direct arguments to `Settings` class
2. Environment variables (`.env`)
3. `config.yaml` values
4. Hardcoded defaults

See [CONFIGURATION.md](CONFIGURATION.md) for details.

## Docker Architecture

**Multi-stage build:**

1. **Builder stage** - Compiles dependencies (~1GB)
2. **Runtime stage** - Minimal image with app code (~500MB)

**Services:**

- `setup` - Downloads docs, builds vector store
- `agent-offline` - Runs CLI in offline mode
- `agent-online` - Runs CLI with web search
- `evaluate` - Runs evaluation suite
- `dev` - Development shell

See [DOCKER.md](DOCKER.md) for details.

## Security Considerations

- ✅ Non-root user (`agent`) runs containers
- ✅ Read-only data volumes (`:ro`)
- ✅ No hardcoded credentials
- ✅ Environment variable configuration
- ✅ Minimal attack surface (slim base images)

## Performance Characteristics

**Offline Mode:**

- Latency: ~2-5s per query (local LLM)
- Throughput: Limited by Ollama (CPU/GPU dependent)
- Cost: $0 (no API calls)

**Online Mode:**

- Latency: ~5-10s per query (includes web search)
- Throughput: Limited by Tavily rate limits (free: 60 RPM)
- Cost: Free tier sufficient for development

## Extensibility

**Adding New Data Sources:**

1. Add text files to `data/` directory
2. Update `scripts/build_vectorstore.py`
3. Rebuild vector store
4. Document in README

**Adding New Metrics:**

1. Define metric function in `evaluation/metrics.py`
2. Update `EvaluationResult` dataclass
3. Call metric in `evaluate_single_question()`
4. Document in [EVALUATION.md](EVALUATION.md)

**Adding New LLM Providers:**

1. Install provider package (e.g., `langchain-anthropic`)
2. Update `src/config.py` with provider settings
3. Modify `src/agent/graph.py` to use new provider
4. Update configuration documentation
