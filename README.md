# LangGraph Helper Agent (Agentic RAG)

A production-ready **Self-Correcting RAG Agent** designed to answer questions about LangGraph and LangChain.

Unlike simple RAG pipelines, this system employs **Agentic Patterns** including query rewriting, self-grading, and retrieval loops to ensure high-quality answers.

## 🏗️ Architecture: The "Agentic Loop"

This project implements a cyclic graph (`StateGraph`) rather than a linear chain:

1.  **Query Rewriting:** An LLM node transforms ambiguous user input (e.g., "memory") into precise technical queries (e.g., "LangGraph persistence checkpointer").
2.  **Hybrid Search:** Retrieves documents using a combination of **Dense Vector Search** (FAISS) and **Sparse Keyword Search** (BM25) with Re-ranking.
3.  **Relevance Grading:** A "Grader" node evaluates retrieved documents.
    - _If irrelevant:_ The agent loops back to rewrite the query.
    - _If relevant:_ The agent proceeds to generation.
4.  **Generation:** An LLM synthesizes the answer from the verified context.

## 🚀 Quick Start (Docker)

No local setup required. Everything runs in isolated containers.

### 1. Build the Agent

```bash
docker compose build agent-offline
```

### 2. Run in Offline Mode (Local LLM)

Uses `llama3.2:3b` via Ollama and a pre-built FAISS index.

```bash
docker compose run --rm agent-offline
```

### 3. Run in Online Mode (Web Search)

_Requires `TAVILY_API_KEY` in `.env`_.

```bash
docker compose run --rm agent-online
```

## 🧪 Evaluation (LangSmith)

This repository includes a professional evaluation pipeline utilizing **LangSmith** to test not just answer correctness, but **Agent Trajectory**.

We verify that the agent _actually corrects itself_ when faced with ambiguous queries.

**Run the Evaluation:**

```bash
docker compose run --rm \
  -e LANGSMITH_API_KEY="your-key" \
  agent-offline scripts/run_langsmith_eval.py
```

**Evaluation Metrics:**

- **Correctness:** LLM-as-a-Judge comparison against gold-standard answers.
- **Trajectory Faithfulness:** Verifies the graph execution path (e.g., ensuring `rewrite` -> `retrieve` loop occurs for bad queries).

## 🛠️ Engineering Highlights

- **Strict Typing:** Full `TypedDict` state management (no loose dictionaries).
- **Modular Design:** Separate files for [`graph.py`](src/agent/graph.py), [`nodes.py`](src/agent/nodes.py), and [`state.py`](src/agent/state.py).
- **Dependency Management:** [`pyproject.toml`](pyproject.toml) with optional dependencies for evaluation.
- **Containerization:** Full Docker support for reproducible environments.

## 📂 Project Structure

```text
├── data/                   # ChromaDB/FAISS vector stores
├── src/
│   ├── agent/
│   │   ├── graph.py        # The StateGraph definition (Nodes + Edges)
│   │   ├── nodes.py        # The Node logic (Rewrite, Retrieve, Grade, Generate)
│   │   └── state.py        # Pydantic/TypedDict Schemas
│   └── config.py           # Configuration management
├── scripts/
│   ├── build_vectorstore.py # Hybrid search indexing script
│   └── run_langsmith_eval.py # Trajectory evaluation suite
├── docker-compose.yml
└── pyproject.toml
```
