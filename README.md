# LangGraph Helper Agent

An AI agent for answering LangGraph and LangChain questions using local documentation and Ollama.

## Quick Start

### Requirements

- Python 3.10+
- Ollama (for local LLM and embeddings)
- LangGraph/LangChain V1

### 1. Install Ollama

```bash
# macOS
brew install ollama

# Or download from: https://ollama.ai/download
```

### 2. Pull Models

```bash
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

### 3. Start Ollama

```bash
ollama serve  # Runs in background on macOS
```

### 4. Setup Project

```bash
git clone https://github.com/yourusername/langgraph-helper-agent.git
cd langgraph-helper-agent

python -m venv venv
source venv/bin/activate

pip install -e .
```

### 5. Download Data & Build Vector Store

```bash
python scripts/refresh_data.py --full
python scripts/build_vectorstore.py
```

### 6. Run Agent

```bash
# Interactive mode
python -m src.main --interactive

# Single question
python -m src.main "How do I add persistence to a LangGraph agent?"

# Online mode (requires TAVILY_API_KEY)
python -m src.main --mode online "What's new in LangGraph?"
```

## Configuration

**Environment Variables:**

- `LLM_MODEL` - Model name (default: llama3.2:3b)
- `EMBEDDING_MODEL` - Embedding model (default: nomic-embed-text)
- `OLLAMA_BASE_URL` - Ollama server (default: http://localhost:11434)
- `AGENT_MODE` - offline or online (default: offline)
- `TAVILY_API_KEY` - For online mode (optional)

**File:** Edit `config.yaml` for advanced settings

## Troubleshooting

**Connection refused?**

```bash
curl http://localhost:11434/api/tags  # Check if Ollama is running
ollama serve  # Start if needed
```

**Model not found?**

```bash
ollama list
ollama pull nomic-embed-text
```

**Memory issues?**

```bash
export LLM_MODEL=llama3.2:1b  # Use smaller model
```

## Testing

Run the test suite locally to verify the agent and configuration:

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html
```

**What's Tested:**

- **Config Management** - Environment variables, YAML config, mode validation
- **State Management** - Message history, context persistence, online mode state

**Continuous Integration:**

Tests automatically run on every commit via GitHub Actions (`.github/workflows/ci.yml`). Push to trigger the test suite.

## Architecture

- **LLM:** ChatOllama (llama3.2:3b)
- **Embeddings:** OllamaEmbeddings (nomic-embed-text)
- **Vector Store:** FAISS
- **Framework:** LangGraph

**Flow:** Query → Retrieve docs → Generate answer

**Online mode:** Also triggers web search if confidence < 0.7

## Technology Stack

### Core Frameworks (V1)

- **LangGraph** >= 0.2.0 - State management and agentic workflows
- **LangChain** >= 0.3.0 - LLM abstractions and tools
- **langchain-ollama** >= 0.1.0 - Ollama integration

### LLM & Embeddings

- **Ollama** - Local LLM inference (no API costs)
- **llama3.2:3b** - Main chat model (2GB)
- **nomic-embed-text** - Embeddings model

### Data & Storage

- **FAISS** - Vector similarity search
- **RecursiveCharacterTextSplitter** - Smart document chunking

### Optional Features

- **Tavily Search** - Web search for online mode (free tier)
- **RAGAS** - Evaluation metrics (quality assessment)
- **Google Gemini** - LLM-as-judge for RAGAS (evaluation only)

## Data Freshness Strategy

### Current Data Sources

The agent uses **publicly available llms.txt documentation**:

- **LangGraph:** https://langchain-ai.github.io/langgraph/llms.txt
- **LangChain:** https://docs.langchain.com/llms.txt

These are maintained by the LangChain team and always reflect the latest API documentation.

### Keeping Data Fresh

#### For llms.txt Sources (Built-in)

Download fresh documentation and rebuild the vector store:

```bash
# Download latest docs
python scripts/refresh_data.py --full

# Rebuild vector store with Ollama
python scripts/build_vectorstore.py

# Commit and push
git add data/vectorstore/
git commit -m "chore: update documentation data"
git push
```

#### Automating Updates (Optional)

Set up a local cron job to refresh weekly:

```bash
# Edit crontab
crontab -e

# Add this line (runs Sunday 2 AM)
0 2 * * 0 cd /path/to/langgraph-helper-agent && python scripts/refresh_data.py --full && python scripts/build_vectorstore.py && git add data/vectorstore/ && git commit -m "chore: auto-update docs" && git push origin master
```

### Extending with Additional Data Sources

If you add custom data sources (e.g., blog posts, internal docs), document your maintenance strategy:

#### Example: Adding Blog Posts

1. **Source:** `https://blog.langchain.dev/`
2. **Update Frequency:** Monthly
3. **How to Maintain:**

```bash
python scripts/refresh_data.py --full
python scripts/build_vectorstore.py
git add data/vectorstore/
git commit -m "chore: monthly data refresh"
git push
```

#### Example: Adding Internal Documentation

1. **Source:** Your company's internal docs
2. **Update Frequency:** As needed
3. **How to Maintain:**

```bash
# Sync your internal docs
cp /path/to/internal/docs.txt data/internal_docs.txt

# Rebuild and commit
python scripts/build_vectorstore.py
git add data/vectorstore/
git commit -m "chore: update internal documentation"
git push
```

### Key Benefits

- ✅ **Llms.txt sources** - Automatically downloaded, easy to refresh
- ✅ **Custom sources** - Maintenance plan clearly documented
- ✅ **Version control** - All data changes tracked in Git
- ✅ **Reproducible** - Others can rebuild the exact same index
- ✅ **No API costs** - All processing happens locally with Ollama

## Project Structure

```
src/
  agent/          # LangGraph components
    graph.py      # State machine and routing logic
    nodes.py      # Node functions (retrieve, generate, search)
    state.py      # TypedDict state schema
  config.py       # Configuration management
  main.py         # CLI entry point

scripts/
  refresh_data.py         # Download docs from official sources
  build_vectorstore.py    # Build FAISS index from documentation

tests/
  unit/
    test_config.py        # Configuration tests
    test_state.py         # State management tests
    test_metrics.py       # Evaluation metrics tests

data/
  langchain_llms.txt      # LangChain API docs (text format)
  langgraph_llms.txt      # LangGraph API docs (text format)
  vectorstore/
    index.faiss           # Vector store for similarity search

evaluation/
  evaluate.py      # RAGAS evaluation with Google Gemini
  dataset.json     # 15 test questions across categories
  metrics.py       # Custom evaluation metrics

.github/workflows/
  ci.yml                 # Tests on every commit
  data-refresh.yml       # Data file validation
  evaluation.yml         # Vector store verification

config.yaml        # Configuration file
pyproject.toml     # Project dependencies and metadata
```

## Next Steps / Learning Opportunities

### 1. **Frontend Development** 🎨
Build a web interface to interact with the agent:
- **Backend**: FastAPI (Python web framework) or Flask
- **Frontend**: React, Vue, or Streamlit (simplest for ML projects)
- **Learning**: API design, websockets for streaming responses, state management

**Example with Streamlit (5 minutes to build):**
```bash
pip install streamlit
```

### 2. **Deployment & DevOps** 🚀
- **Docker**: Container the app for easy deployment
- **Cloud**: Deploy to Hugging Face Spaces, Railway, or AWS
- **Learning**: DevOps basics, containerization, CI/CD best practices

### 3. **Advanced LangGraph Features** 🔄
- Sub-graphs for complex workflows
- Conditional routing with more sophisticated logic
- Multi-turn conversation memory management
- Human-in-the-loop approval nodes

### 4. **RAG Optimization** 📊
- Experiment with different embedding models
- Implement query expansion or reranking
- Add semantic caching
- Compare RAGAS metrics across iterations

### 5. **Monitoring & Observability** 📈
- Add logging and tracing (LangSmith integration)
- Performance monitoring (response times, token usage)
- Error tracking and debugging

## License

MIT
