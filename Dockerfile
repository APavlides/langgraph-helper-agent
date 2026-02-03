# LangGraph Helper Agent
# Multi-stage build for smaller final image
# 
# Build: docker build -t langgraph-helper-agent:latest .
# Run:   docker run --rm langgraph-helper-agent:latest "Your question"
# Size:  ~1.6GB (includes all dependencies)

# === Build stage ===
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files (README.md required by pyproject.toml)
COPY pyproject.toml README.md .env.example ./

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install the package with all dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir ".[all]"


# === Runtime stage ===
FROM python:3.11-slim as runtime

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY src/ src/
COPY scripts/ scripts/
COPY data/ data/
COPY evaluation/ evaluation/
COPY pyproject.toml .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash agent
RUN chown -R agent:agent /app
USER agent

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command
ENTRYPOINT ["python", "-m", "src.main"]
CMD ["--help"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from src.config import Settings; print('OK')" || exit 1

# Labels
LABEL org.opencontainers.image.title="LangGraph Helper Agent"
LABEL org.opencontainers.image.description="AI agent that helps developers work with LangGraph and LangChain"
LABEL org.opencontainers.image.source="https://github.com/yourusername/langgraph-helper-agent"
