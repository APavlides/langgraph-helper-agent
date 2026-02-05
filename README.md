# LangGraph Helper Agent

**Summary:** A LangGraph-based RAG agent that answers LangGraph and LangChain questions using local documentation (offline) or web search (online).

## Quick Start (Docker/Compose Only)

```bash
# 1) Install & start Ollama
brew install ollama
ollama pull nomic-embed-text
ollama pull llama3.2:3b
ollama serve

# 2) Clone repo
git clone https://github.com/APavlides/langgraph-helper-agent.git
cd langgraph-helper-agent

# 3) Build & prepare data (uses .env)
docker compose --env-file .env run --rm setup

# 4) Run (offline) — default
docker compose --env-file .env up agent-offline

# 5) Run (online)
docker compose --env-file .env up agent-online
```

**Key Features:**

- Dual operating modes (offline/online)
- Local LLM via Ollama (no API costs for core functionality)
- RAG with FAISS vector store
- LangGraph state machine for routing and decision-making

## Architecture Overview

### Graph Design

Built with **LangGraph** (≥ 0.2.0) as a state machine with the following nodes:

1. **retrieve** - Fetches relevant docs from FAISS vector store, reranks with cross-encoder for quality assessment
2. **generate** - Creates answer using ChatOllama (llama3.2:3b)
3. **web_search_and_generate** - Searches web via Tavily API + generates enhanced answer (online mode)
4. **route_after_retrieve** - Decides whether to use docs or web search based on reranker scores

**Routing Logic:**

```
Query → Retrieve + Rerank → Route
                           ├─> [reranker score < 0.0 & online mode] → Web Search + Generate → END
                           └─> [else] → Generate → END
```

**Reranking:** Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` to assess retrieval quality. Scores < 0.0 indicate poor relevance, triggering web search in online mode.

### State Management

Uses TypedDict state schema with message history:

- `messages` - Conversation history (HumanMessage, AIMessage)
- `retrieved_contexts` - Documentation chunks from vector store
- `retrieval_score` - Cross-encoder reranking score (negative = poor relevance)
- `mode` - offline or online
- `web_search_results` - Search results (online mode only)

### Technology Stack

**Required (V1):**

- LangGraph ≥ 0.2.0 - State machine framework
- LangChain ≥ 0.3.0 - LLM abstractions
- langchain-ollama ≥ 0.1.0 - Ollama integration
- sentence-transformers ≥ 2.2.0 - Cross-encoder reranking

**Core Components:**

- **Ollama** - Local LLM (llama3.2:3b) and embeddings (nomic-embed-text)
- **FAISS** - Vector store with 10,943 chunks from official docs
- **Cross-encoder** - ms-marco-MiniLM-L-6-v2 for retrieval quality assessment
- **Python** 3.10+

**Optional:**

- Tavily API - Web search (online mode)
- RAGAS + Google Gemini - Evaluation metrics

## Operating Modes

### Offline Mode (Default)

**How it works:**

1. Query is embedded using local `nomic-embed-text` model
2. Top-5 relevant docs retrieved from FAISS vector store
3. Answer generated with `llama3.2:3b` using retrieved context
4. No external API calls required

**Data sources:**

- LangGraph docs: `https://langchain-ai.github.io/langgraph/llms.txt`
- LangChain docs: `https://docs.langchain.com/llms.txt`
- Pre-indexed into FAISS (committed to repo)

**Use when:**

- No internet connection
- Want fast, cost-free responses
- Official documentation is sufficient

### Online Mode

**How it works:**

1. Query is embedded and retrieves top docs from FAISS
2. Cross-encoder reranks results to assess true relevance
3. If reranker score < 0.0, triggers web search
4. Combines web results with docs for comprehensive answer
5. Returns enriched response with current information

**Services used:**

- **Tavily Search API** - Web search with AI-optimized results
  - Why: Provides recent updates, blog posts, discussions
  - Free tier: 1,000 searches/month

**Use when:**

- Need current information beyond official docs
- Checking recent updates or community discussions
- Question not well-covered in official documentation

### Switching Between Modes

**Via environment variable:**

```bash
export AGENT_MODE=offline  # or 'online'
python -m src.main "Your question"
```

**Via CLI flag:**

```bash
python -m src.main --mode online "Your question"
```

**Set API keys in .env (required):**

```bash
TAVILY_API_KEY=your_key
```

## Data Freshness Strategy

### Offline Mode

**Data preparation (Docker):**

1. Downloaded official llms.txt files from LangChain/LangGraph docs
2. Split into chunks (1000 chars, 200 overlap) using RecursiveCharacterTextSplitter
3. Embedded with Ollama `nomic-embed-text`
4. Indexed in FAISS (10,943 chunks)
5. Committed to Git for reproducibility

**Updating data:**

```bash
docker compose --env-file .env run --rm setup
```

**Frequency recommendation:** Weekly or when major LangGraph releases occur

**Extending with custom sources:**
If adding additional data (e.g., company docs):

1. Add text files to `data/` directory
2. Update `scripts/build_vectorstore.py` to include them
3. Document update process in this README
4. Rebuild and commit updated index

### Online Mode

**Services:**

- **Tavily Search API** - Real-time web search
  - Provides current information beyond static docs
  - Handles: Recent releases, blog posts, Stack Overflow discussions
  - Always up-to-date (no manual refresh needed)

**Why Tavily:**

- AI-optimized search (better than generic web scraping)
- Clean, structured results
- Free tier sufficient for development/testing

## Setup Instructions (Docker/Compose)

### 1. Install Ollama

```bash
# macOS
brew install ollama

# Linux/Windows: https://ollama.ai/download
```

### 2. Pull Required Models

```bash
ollama pull nomic-embed-text  # Embeddings (273MB)
ollama pull llama3.2:3b       # Chat model (2GB)
```

**Note:** The cross-encoder reranking model (`ms-marco-MiniLM-L-6-v2`, ~90MB) downloads automatically on first use inside the container.

### 3. Start Ollama Server

```bash
ollama serve  # Starts on http://localhost:11434
```

### 4. Clone and Configure

```bash
git clone https://github.com/APavlides/langgraph-helper-agent.git
cd langgraph-helper-agent

# Fill in .env (API keys, mode)
cp .env.example .env
```

### 5. Build Vector Store (First Time Only)

```bash
docker compose --env-file .env run --rm setup
```

### 6. Run Agent

**Offline mode (default):**

```bash
docker compose --env-file .env up agent-offline
```

**Online mode:**

```bash
docker compose --env-file .env up agent-online
```

## Example Run

```bash
$ python -m src.main "How do I use checkpointers?"

╭──────────────────────────── Answer ────────────────────────────╮
│ To use checkpointers in LangGraph:                             │
│                                                                 │
│ 1. Import a checkpointer:                                      │
│    from langgraph.checkpoint.memory import MemorySaver         │
│                                                                 │
│ 2. Add to your graph:                                          │
│    checkpointer = MemorySaver()                                │
│    graph = graph.compile(checkpointer=checkpointer)            │
│                                                                 │
│ 3. Use with thread_id for persistence:                         │
│    config = {"configurable": {"thread_id": "1"}}               │
│    graph.invoke({"messages": [...]}, config)                   │
│                                                                 │
│ Available checkpointers: MemorySaver, SqliteSaver,             │
│ PostgresSaver                                                   │
╰─────────────────────────────────────────────────────────────────╯

📊 Confidence: 0.92 | 📚 Sources: 3 docs | ⚡ Mode: offline
```

## Example Questions (Expected Coverage)

- How do I add persistence to a LangGraph agent?
- What's the difference between StateGraph and MessageGraph?
- Show me how to implement human-in-the-loop with LangGraph
- How do I handle errors and retries in LangGraph nodes?
- What are best practices for state management in LangGraph?

## Configuration

**Environment Variables:**

```bash
LLM_MODEL=llama3.2:3b            # Chat model
EMBEDDING_MODEL=nomic-embed-text # Embeddings
OLLAMA_BASE_URL=http://localhost:11434
AGENT_MODE=offline               # or 'online'
RERANK_THRESHOLD=0.0             # Trigger web search when score < threshold
TAVILY_API_KEY=your_key          # For online mode
```

**Config file:** `config.yaml` for advanced settings (see [docs/CONFIGURATION.md](docs/CONFIGURATION.md))

## Testing

```bash
# Install dev dependencies only
pip install -e ".[dev]"

# Run tests (33 tests)
pytest tests/ -v
```

**CI/CD:** Tests run automatically on every commit via GitHub Actions

## Docker

```bash
docker build -t langgraph-helper-agent .
docker run --rm langgraph-helper-agent "Your question"
```

See [docs/DOCKER.md](docs/DOCKER.md) for full containerization guide.

## Evaluation (RAGAS)

RAGAS evaluation uses Google Gemini and can exceed the free-tier request limits before completing the full dataset. Expect incomplete RAGAS metrics on the free tier unless you reduce dataset size or increase limits.

See [docs/RAGAS_EVALUATION.md](docs/RAGAS_EVALUATION.md) for setup and limits.

## Troubleshooting

**Ollama connection refused:**

```bash
curl http://localhost:11434/api/tags
ollama serve
```

**Out of memory:**

```bash
export LLM_MODEL=llama3.2:1b  # Smaller model
```

## Additional Documentation

- [docs/CONFIGURATION.md](docs/CONFIGURATION.md) - Advanced configuration options
- [docs/DOCKER.md](docs/DOCKER.md) - Docker deployment guide
- [docs/EVALUATION.md](docs/EVALUATION.md) - RAGAS evaluation setup
- [docs/RAGAS_EVALUATION.md](docs/RAGAS_EVALUATION.md) - Google Gemini free tier limits and troubleshooting

## License

MIT
