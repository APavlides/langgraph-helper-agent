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

# 4) Run agent (interactive chat - offline mode)
docker compose --env-file .env run --rm -e OLLAMA_BASE_URL=http://host.docker.internal:11434 agent-offline

# 5) Run agent (online mode with web search)
docker compose --env-file .env run --rm -e OLLAMA_BASE_URL=http://host.docker.internal:11434 agent-online
```

**Key Features:**

- Dual operating modes (offline/online)
- Local LLM via Ollama (no API costs for core functionality)
- RAG with FAISS vector store
- LangGraph state machine for routing and decision-making

> **For detailed architecture information**, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

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

# Create .env from example and configure for your system
cp .env.example .env
```

**Important `.env` settings:**

- `OLLAMA_BASE_URL` - Set to `http://host.docker.internal:11434` (macOS/Windows) or `http://localhost:11434` (Linux)
- `TAVILY_API_KEY` - Only needed for online mode (optional)
- `GOOGLE_API_KEY` - Only needed for RAGAS evaluation (optional)
- `AGENT_MODE` - Set to `offline` (default) or `online`

See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for all available options.

### 3. Build Vector Store (First Time Only)

```bash
docker compose --env-file .env run --rm setup
```

### 4. Run Agent

**Interactive chat (offline mode):**

```bash
docker compose --env-file .env run --rm agent-offline
```

**Ask a single question:**

```bash
docker compose --env-file .env run --rm agent-offline "How do I use checkpointers?"
```

**Online mode (with web search):**

```bash
docker compose --env-file .env run --rm agent-online
```

**Exit interactive mode:** Type `exit`, `quit`, or press Ctrl+C

## Runtime Mode Override

You can override the default mode at runtime even when using Docker Compose services:

```bash
# Run offline service but switch to online mode
docker compose --env-file .env run --rm agent-offline --mode online "Your question"

# Run online service but switch to offline mode
docker compose --env-file .env run --rm agent-online --mode offline "Your question"
```

## Example Run

```bash
$ docker compose --env-file .env run --rm agent-offline "How do I use checkpointers?"

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
# Run tests via Docker (recommended for reproducibility)
docker compose --env-file .env run --rm dev -c "pytest tests/ -v"

# Or locally if you have the environment set up
pip install -e ".[dev]"
pytest tests/ -v
```

**CI/CD:** Tests run automatically on every commit via GitHub Actions

## Docker

```bash
docker build -t langgraph-helper-agent .
docker run --rm langgraph-helper-agent "Your question"
```

See [docs/DOCKER.md](docs/DOCKER.md) for full containerization guide.

## Evaluation

The project includes two types of evaluation:

### 1. Custom Metrics (Default - No API Required)

Rule-based metrics that work offline without any LLM:

- **Topic Coverage** - Keyword/phrase matching against expected topics
- **Code Presence** - Regex detection of code blocks and inline code
- **Code Validity** - Python AST syntax validation (no execution)
- **Aggregate Score** - Weighted average of all metrics

**Run evaluation:**

```bash
docker compose --env-file .env run --rm dev -c "python -m evaluation.evaluate"
```

**Output:** JSON report with scores (0-1.0) for each metric per question.

### 2. RAGAS Metrics (Optional - Requires Google API)

LLM-as-judge metrics using Google Gemini for semantic evaluation:

- **Context Precision** - Quality of retrieved documents
- **Faithfulness** - Answer grounded in context (no hallucinations)
- **Answer Relevancy** - Answer addresses the question

**Limitations:**

- Free tier rate limits (RPM/RPD) typically insufficient for full 15-question dataset
- Local models (llama3.2:3b) produce invalid JSON for RAGAS on resource-constrained machines
- Requires paid Google tier for reliable evaluation

**Run with RAGAS:**

```bash
docker compose --env-file .env run --rm dev -c "python -m evaluation.evaluate --ragas"
```

See [docs/EVALUATION.md](docs/EVALUATION.md) and [docs/RAGAS_EVALUATION.md](docs/RAGAS_EVALUATION.md) for details.

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

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - System design, graph routing, and technology stack
- [docs/CONFIGURATION.md](docs/CONFIGURATION.md) - Advanced configuration options
- [docs/DOCKER.md](docs/DOCKER.md) - Docker deployment guide
- [docs/EVALUATION.md](docs/EVALUATION.md) - RAGAS evaluation setup
- [docs/RAGAS_EVALUATION.md](docs/RAGAS_EVALUATION.md) - Google Gemini free tier limits and troubleshooting

## License

MIT
