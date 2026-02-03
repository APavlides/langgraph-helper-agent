# Docker Setup Guide

This project includes a multi-stage Dockerfile for easy containerization and deployment.

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- Ollama running (for LLM and embeddings)
- Environment variables set (see below)

### 1. Build the Image

```bash
docker build -t langgraph-helper-agent:latest .
```

### 2. Setup (Download docs, build vector store)

```bash
docker compose run --rm setup
```

This will:
- Download LangGraph and LangChain documentation
- Build the FAISS vector store locally
- Save to `./data/vectorstore/`

### 3. Run Agent

**Offline mode (no API keys needed):**
```bash
docker compose up agent-offline
```

**Online mode (requires TAVILY_API_KEY):**
```bash
docker compose up agent-online
```

**Development shell:**
```bash
docker compose --profile dev up dev
```

## Environment Setup

Create a `.env` file in the project root:

```bash
# Required for online mode
TAVILY_API_KEY=your_tavily_api_key

# Optional - for evaluation
GOOGLE_API_KEY=your_google_api_key
```

Or set them in your shell:
```bash
export TAVILY_API_KEY=your_key
export GOOGLE_API_KEY=your_key
docker compose up agent-online
```

## Docker Compose Services

### `setup` (Profile: setup)
Downloads documentation and builds vector store.

```bash
docker compose --profile setup run --rm setup
```

### `agent-offline` (Default)
Run agent in offline mode with local documentation only.

```bash
docker compose up agent-offline
```

### `agent-online` (Default)
Run agent in online mode with web search capability.

```bash
docker compose up agent-online
```

### `evaluate` (Profile: eval)
Run RAGAS evaluation suite.

```bash
docker compose --profile eval run --rm evaluate
```

### `dev` (Profile: dev)
Development shell with live code reloading.

```bash
docker compose --profile dev up dev
```

## Architecture

### Multi-Stage Build

1. **Builder Stage**: 
   - Installs build dependencies
   - Compiles Python packages
   - Creates virtual environment
   - Final size: ~1GB (not included in final image)

2. **Runtime Stage**:
   - Copies only the virtual environment
   - Adds application code
   - Creates non-root user (`agent`)
   - Final size: ~500MB

### Security

- ✅ Non-root user (`agent`) runs the container
- ✅ Read-only data volumes (`:ro`)
- ✅ No unnecessary packages
- ✅ Health checks enabled

## Testing the Container

### Syntax Check
```bash
docker run --rm -i hadolint/hadolint < Dockerfile
```

### Manual Test (once Docker daemon is running)

```bash
# Build
docker build -t langgraph-helper:test .

# Test offline mode
docker run --rm \
  -v $(pwd)/data:/app/data:ro \
  langgraph-helper:test \
  "What is LangGraph?"

# Test with environment
docker run --rm \
  -e AGENT_MODE=offline \
  -v $(pwd)/data:/app/data:ro \
  langgraph-helper:test \
  "How do I add persistence?"
```

## Troubleshooting

### "Cannot connect to Ollama"
The container can't reach Ollama by default because Ollama runs on `localhost`. Fix it:

**macOS with Ollama Desktop:**
```bash
# Use host.docker.internal (macOS Docker Desktop specific)
export OLLAMA_BASE_URL=http://host.docker.internal:11434
docker compose up agent-offline
```

**Linux with Ollama:**
```bash
# Use --network host to share localhost
docker run --network host \
  -v $(pwd)/data:/app/data:ro \
  langgraph-helper-agent:latest
```

### "Vector store not found"
Run setup first:
```bash
docker compose run --rm setup
```

### Build taking too long
Docker build can be slow. Check if you need to clean up:
```bash
docker system prune  # Remove unused layers
```

## Production Deployment

### Push to Registry

```bash
docker tag langgraph-helper-agent:latest myregistry/langgraph-helper:1.0.0
docker push myregistry/langgraph-helper:1.0.0
```

### Deploy to Cloud

**Hugging Face Spaces (Free):**
```bash
# Create repo on Spaces, then push
docker tag langgraph-helper-agent yourusername/langgraph-helper
docker push yourusername/langgraph-helper
```

**AWS ECS:**
```bash
aws ecr create-repository --repository-name langgraph-helper
docker tag langgraph-helper-agent:latest <account>.dkr.ecr.<region>.amazonaws.com/langgraph-helper:latest
docker push <account>.dkr.ecr.<region>.amazonaws.com/langgraph-helper:latest
```

## Optimization Tips

### Reduce Image Size

Current: ~500MB (runtime only, not including base Python image)

To reduce further:
- Use `python:3.11-alpine` instead of `slim` (~80MB smaller, but slower builds)
- Remove optional dependencies: `pip install ".[all]"` → `pip install "."`

### Speed Up Builds

- Use BuildKit: `DOCKER_BUILDKIT=1 docker build .`
- Leverage layer caching: keep stable dependencies at top of Dockerfile

## Next Steps

1. **Test locally** once Docker daemon is available
2. **Add Ollama service** to docker-compose.yml (optional)
3. **Deploy** to Hugging Face Spaces or cloud provider
4. **Add frontend** (Streamlit or FastAPI) as another service

