# Configuration System Explanation

## Overview

This project uses a **YAML-based configuration system** for managing Ollama models and agent settings. It provides a clean, maintainable way to configure the application without hardcoding values.

## Architecture

### 1. Configuration Flow

```
config.yaml
     ↓
Settings class (src/config.py)
     ↓
Agent components (graph.py, nodes.py)
     ↓
LangChain Ollama models
```

### 2. Configuration Priority

Settings are resolved in this order (highest to lowest priority):

1. **Direct arguments** to Settings class
2. **Environment variables** (from `.env` or shell)
3. **config.yaml** values
4. **Hardcoded defaults** in Settings class

Example:

```python
# config.yaml has: llm.model.name = "llama3.2:3b"
# .env has: LLM_MODEL="llama3.2:1b"
# Result: Uses "llama3.2:1b" (env var wins)
```

## How It Works

### Step 1: config.yaml Structure

The YAML file is organized into logical sections:

```yaml
llm: # LLM (chat model) settings
  model:
    name: "llama3.2:3b" # The Ollama model to use
  parameters:
    temperature: 0.1 # How creative/random (0.0-1.0)
    max_tokens: 2000 # Max response length
  ollama:
    base_url: "http://localhost:11434" # Ollama server
    timeout: 120 # Request timeout in seconds

embedding: # Embedding model settings
  model: "nomic-embed-text" # For vector store
  base_url: "http://localhost:11434"

agent: # Agent behavior settings
  mode: offline # offline or online
  retrieval_k: 5 # Number of docs to retrieve
  chunk_size: 1000 # Text chunk size
  chunk_overlap: 200 # Overlap between chunks
  confidence_threshold: 0.7 # When to trigger web search

data: # File paths
  dir: "data"
  vectorstore: "data/vectorstore"
```

### Step 2: Settings Class (src/config.py)

The `Settings` class loads and validates configuration:

```python
from src.config import Settings

# Loads config.yaml automatically
settings = Settings()

# Access settings
print(settings.llm_model)          # "llama3.2:3b"
print(settings.temperature)        # 0.1
print(settings.ollama_base_url)    # "http://localhost:11434"
```

**Key features:**

- Loads YAML on initialization
- Merges with environment variables
- Validates required settings
- Provides sensible defaults
- Creates directories if needed

### Step 3: LLM Creation (src/agent/graph.py)

The agent uses settings to create Ollama models:

```python
from langchain_ollama import ChatOllama, OllamaEmbeddings

def create_llm(settings: Settings):
    """Create Ollama chat model from settings."""
    return ChatOllama(
        model=settings.llm_model,           # From config.yaml
        temperature=settings.temperature,   # From config.yaml
        base_url=settings.ollama_base_url,  # From config.yaml
    )

def create_retriever(settings: Settings):
    """Create embeddings and vector store retriever."""
    embeddings = OllamaEmbeddings(
        model=settings.embedding_model,     # From config.yaml
        base_url=settings.ollama_base_url,  # From config.yaml
    )

    vectorstore = FAISS.load_local(
        str(settings.vectorstore_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    return vectorstore.as_retriever(
        search_kwargs={"k": settings.retrieval_k}  # From config.yaml
    )
```

## Usage Examples

### Example 1: Default Configuration

```bash
# Uses config.yaml as-is
python -m src.main --interactive
```

### Example 2: Override with Environment Variables

```bash
# Temporarily use a different model
LLM_MODEL=llama3.2:1b python -m src.main "What is LangGraph?"

# Or set in .env file
echo "LLM_MODEL=llama3.2:1b" >> .env
python -m src.main --interactive
```

### Example 3: Programmatic Configuration

```python
from src.config import Settings, AgentMode

# Override specific settings
settings = Settings(
    llm_model="llama3.2:7b",
    temperature=0.2,
    mode=AgentMode.ONLINE,
)

# Use in agent
from src.agent.graph import create_agent
agent = create_agent(settings)
```

### Example 4: Custom Config File

```python
# Load different config file
settings = Settings(config_path="config.production.yaml")
```

## Configuration Options Reference

### LLM Settings

| Setting                      | Type   | Default                    | Description                   |
| ---------------------------- | ------ | -------------------------- | ----------------------------- |
| `llm.model.name`             | string | `"llama3.2:3b"`            | Ollama model name             |
| `llm.parameters.temperature` | float  | `0.1`                      | Response randomness (0.0-1.0) |
| `llm.parameters.max_tokens`  | int    | `2000`                     | Max response length           |
| `llm.ollama.base_url`        | string | `"http://localhost:11434"` | Ollama server URL             |
| `llm.ollama.timeout`         | int    | `120`                      | Request timeout (seconds)     |

### Embedding Settings

| Setting              | Type   | Default                    | Description          |
| -------------------- | ------ | -------------------------- | -------------------- |
| `embedding.model`    | string | `"nomic-embed-text"`       | Embedding model name |
| `embedding.base_url` | string | `"http://localhost:11434"` | Ollama server URL    |

### Agent Settings

| Setting                      | Type   | Default     | Description                         |
| ---------------------------- | ------ | ----------- | ----------------------------------- |
| `agent.mode`                 | string | `"offline"` | Operating mode (`offline`/`online`) |
| `agent.retrieval_k`          | int    | `5`         | Number of docs to retrieve          |
| `agent.chunk_size`           | int    | `1000`      | Text chunk size (chars)             |
| `agent.chunk_overlap`        | int    | `200`       | Chunk overlap (chars)               |
| `agent.confidence_threshold` | float  | `0.7`       | Web search trigger (0.0-1.0)        |

### Data Paths

| Setting            | Type   | Default              | Description         |
| ------------------ | ------ | -------------------- | ------------------- |
| `data.dir`         | string | `"data"`             | Data directory path |
| `data.vectorstore` | string | `"data/vectorstore"` | Vector store path   |

## Environment Variable Reference

These override config.yaml values:

| Variable               | Maps To                      | Example                  |
| ---------------------- | ---------------------------- | ------------------------ |
| `LLM_MODEL`            | `llm.model.name`             | `llama3.2:1b`            |
| `EMBEDDING_MODEL`      | `embedding.model`            | `nomic-embed-text`       |
| `OLLAMA_BASE_URL`      | `llm.ollama.base_url`        | `http://localhost:11434` |
| `TEMPERATURE`          | `llm.parameters.temperature` | `0.2`                    |
| `MAX_TOKENS`           | `llm.parameters.max_tokens`  | `1500`                   |
| `AGENT_MODE`           | `agent.mode`                 | `offline` or `online`    |
| `RETRIEVAL_K`          | `agent.retrieval_k`          | `10`                     |
| `CHUNK_SIZE`           | `agent.chunk_size`           | `1500`                   |
| `CHUNK_OVERLAP`        | `agent.chunk_overlap`        | `300`                    |
| `CONFIDENCE_THRESHOLD` | `agent.confidence_threshold` | `0.8`                    |
| `DATA_DIR`             | `data.dir`                   | `./my_data`              |
| `VECTORSTORE_PATH`     | `data.vectorstore`           | `./my_data/vectors`      |
| `TAVILY_API_KEY`       | N/A (API key only)           | Required for online mode |

## Common Scenarios

### Switching Models

**Option 1: Edit config.yaml**

```yaml
llm:
  model:
    name: "llama3.2:1b" # Changed from 3b
```

**Option 2: Environment variable**

```bash
export LLM_MODEL=llama3.2:1b
python -m src.main --interactive
```

### Using Remote Ollama

If Ollama runs on a different machine:

```yaml
llm:
  ollama:
    base_url: "http://192.168.1.100:11434"

embedding:
  base_url: "http://192.168.1.100:11434"
```

Or:

```bash
export OLLAMA_BASE_URL=http://192.168.1.100:11434
```

### Tuning Performance

For faster responses (less accurate):

```yaml
llm:
  model:
    name: "llama3.2:1b" # Smaller model
  parameters:
    temperature: 0.3 # More creative
    max_tokens: 1000 # Shorter responses
```

For better quality (slower):

```yaml
llm:
  model:
    name: "llama3.2:7b" # Larger model
  parameters:
    temperature: 0.05 # Very deterministic
    max_tokens: 3000 # Longer responses

agent:
  retrieval_k: 10 # More context
```

### Development vs Production

**config.dev.yaml:**

```yaml
llm:
  model:
    name: "llama3.2:1b" # Fast for testing

agent:
  retrieval_k: 3 # Less context
```

**config.prod.yaml:**

```yaml
llm:
  model:
    name: "llama3.2:3b" # Better quality

agent:
  retrieval_k: 5 # More context
```

Use:

```python
settings = Settings(config_path="config.prod.yaml")
```

## Benefits of This Approach

1. **Single Source of Truth**: All config in one place
2. **Easy Experimentation**: Change models without code changes
3. **Environment Flexibility**: Different configs for dev/prod
4. **Version Control**: Track configuration changes in git
5. **Type Safety**: Pydantic validation ensures correct types
6. **Self-Documenting**: YAML is human-readable
7. **Testable**: Easy to inject test configurations

## Troubleshooting

### Config file not found

```python
# Settings class handles missing config.yaml gracefully
# Falls back to defaults
settings = Settings()  # Works even without config.yaml
```

### Wrong model name

```bash
# Check available models
ollama list

# Pull missing model
ollama pull llama3.2:3b
```

### Connection refused

```yaml
# Check Ollama is running
curl http://localhost:11434/api/tags

# Or update base_url
llm:
  ollama:
    base_url: "http://localhost:11434"
```

### Environment variables not working

```bash
# Check they're set
env | grep LLM_MODEL

# Ensure .env is in project root
cat .env

# Verify python-dotenv is installed
pip list | grep python-dotenv
```

## Implementation Details

This project uses **standard Python packages** for configuration:

- **PyYAML** (`pyyaml`): Parses YAML files
- **python-dotenv**: Loads `.env` files
- **dataclasses**: Type-safe settings (built-in Python 3.7+)
- **langchain-ollama**: Official Ollama integration

## Code Flow Diagram

```
┌─────────────────────────────────────────────────────────┐
│                  Application Startup                    │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │  Load config.yaml    │
          │  (PyYAML)            │
          └──────────┬───────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │  Load .env file      │
          │  (python-dotenv)     │
          └──────────┬───────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │  Settings.__init__   │
          │  - Merge configs     │
          │  - Apply priority    │
          │  - Validate          │
          └──────────┬───────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │  create_llm()        │
          │  creates ChatOllama  │
          └──────────┬───────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │  create_retriever()  │
          │  creates             │
          │  OllamaEmbeddings    │
          └──────────┬───────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │  create_agent()      │
          │  LangGraph agent     │
          └──────────────────────┘
```

This simple, direct approach gives you full control over Ollama configuration without unnecessary complexity.
