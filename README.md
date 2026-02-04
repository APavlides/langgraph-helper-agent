# LangGraph Helper Agent

**Summary:** A LangGraph-based RAG agent that answers LangGraph and LangChain questions using local documentation (offline) or web search (online).

## Quick Start (Essential Commands)

**Recommended (Docker/Compose for reproducibility):**

```bash
# 1) Install & start Ollama
brew install ollama
ollama pull nomic-embed-text
ollama pull llama3.2:3b
ollama serve

# 2) Clone repo
git clone https://github.com/APavlides/langgraph-helper-agent.git
cd langgraph-helper-agent

# 3) Build & prepare data
docker compose run --rm setup

# 4) Run (offline) — default
docker compose up agent-offline

# 5) Run (online)
export TAVILY_API_KEY=your_key
docker compose up agent-online
```

**Optional (Local Python install):**

```bash
python -m venv venv
source venv/bin/activate
pip install -e .

python scripts/refresh_data.py --full
python scripts/build_vectorstore.py

python -m src.main "How do I add persistence to a LangGraph agent?"

# Online mode (explicit)
export TAVILY_API_KEY=your_key
python -m src.main --mode online "What's new in LangGraph?"
```

**Key Features:**

- Dual operating modes (offline/online)
- Local LLM via Ollama (no API costs for core functionality)
- RAG with FAISS vector store
- LangGraph state machine for routing and decision-making

## Architecture Overview

### Graph Design

Built with **LangGraph** (≥ 0.2.0) as a state machine with the following nodes:

1. **retrieve** - Fetches relevant docs from FAISS vector store
2. **generate** - Creates answer using ChatOllama (llama3.2:3b)
3. **web_search** - Searches web via Tavily API (online mode only)
4. **regenerate** - Combines web results with docs for improved answer
5. **route** - Decides next action based on confidence score

**Routing Logic:**

```
Query → Retrieve → Generate → Route
                              ├─> [confidence < 0.7 & online mode] → Web Search → Regenerate → END
                              └─> [else] → END
```

### State Management

Uses TypedDict state schema with message history:

- `messages` - Conversation history (HumanMessage, AIMessage)
- `retrieved_contexts` - Documentation chunks from vector store
- `confidence_score` - Answer confidence (0-1 scale)
- `mode` - offline or online
- `web_search_results` - Search results (online mode only)

### Technology Stack

**Required (V1):**

- LangGraph ≥ 0.2.0 - State machine framework
- LangChain ≥ 0.3.0 - LLM abstractions
- langchain-ollama ≥ 0.1.0 - Ollama integration

**Core Components:**

- **Ollama** - Local LLM (llama3.2:3b) and embeddings (nomic-embed-text)
- **FAISS** - Vector store with 10,943 chunks from official docs
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

1. Same as offline mode initially (retrieve + generate)
2. If confidence < 0.7, triggers Tavily web search
3. Combines web results with docs for regenerated answer
4. Returns enriched response with current information

**Services used:**

- **Tavily Search API** - Web search with AI-optimized results
  - Why: Provides recent updates, blog posts, discussions
  - Free tier: 1,000 searches/month

**Use when:**

- Need current information beyond official docs
- Checking recent updates or community discussions
- Initial answer lacks confidence

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

**For online mode, set API key:**

```bash
export TAVILY_API_KEY=your_key
```

## Data Freshness Strategy

### Offline Mode

**Data preparation:**

1. Downloaded official llms.txt files from LangChain/LangGraph docs
2. Split into chunks (1000 chars, 200 overlap) using RecursiveCharacterTextSplitter
3. Embedded with Ollama `nomic-embed-text`
4. Indexed in FAISS (10,943 chunks)
5. Committed to Git for reproducibility

**Updating data:**

```bash
# Download latest docs
python scripts/refresh_data.py --full

# Rebuild vector store (~2 minutes)
python scripts/build_vectorstore.py

# Commit updated index
git add data/vectorstore/
git commit -m "chore: update documentation"
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

## Setup Instructions

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

### 3. Start Ollama Server

```bash
ollama serve  # Starts on http://localhost:11434
```

### 4. Clone and Install Project

```bash
git clone https://github.com/APavlides/langgraph-helper-agent.git
cd langgraph-helper-agent

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -e .
```

### 5. Build Vector Store (First Time Only)

```bash
# Download official docs
python scripts/refresh_data.py --full

# Build FAISS index (~2 minutes)
python scripts/build_vectorstore.py
```

### 6. Run Agent

**Offline mode (no API key needed):**

```bash
# Interactive chat
python -m src.main --interactive

# Single question
python -m src.main "How do I add persistence to a LangGraph agent?"
```

**Online mode (requires TAVILY_API_KEY):**

```bash
export TAVILY_API_KEY=your_key
python -m src.main --mode online "What's new in LangGraph 0.2?"
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

## Configuration

**Environment Variables:**

```bash
LLM_MODEL=llama3.2:3b           # Chat model
EMBEDDING_MODEL=nomic-embed-text # Embeddings
OLLAMA_BASE_URL=http://localhost:11434
AGENT_MODE=offline              # or 'online'
TAVILY_API_KEY=your_key         # For online mode
```

**Config file:** `config.yaml` for advanced settings (see [docs/CONFIGURATION.md](docs/CONFIGURATION.md))

## Testing

```bash
# Install dev dependencies
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

## License

MIT
