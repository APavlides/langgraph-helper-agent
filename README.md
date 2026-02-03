# LangGraph Helper Agent

An AI agent for answering LangGraph and LangChain questions using local documentation and Ollama.

## Quick Start

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

## Architecture

- **LLM:** ChatOllama (llama3.2:3b)
- **Embeddings:** OllamaEmbeddings (nomic-embed-text)
- **Vector Store:** FAISS
- **Framework:** LangGraph

**Flow:** Query → Retrieve docs → Generate answer

**Online mode:** Also triggers web search if confidence < 0.7

## Project Structure

```
src/
  agent/          # LangGraph components
  config.py       # Configuration
  main.py         # CLI entry point

scripts/
  refresh_data.py         # Download docs
  build_vectorstore.py    # Build FAISS index

data/              # Docs and vector store
config.yaml        # Configuration file
```

## License

MIT
