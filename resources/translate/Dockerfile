# Translation Service Dockerfile
# Requires: NVIDIA GPU with nvidia-docker runtime, external Ollama service

# Build stage for dependencies
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir setuptools wheel && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir langchain langchain-community langchain-core langchain-experimental \
    langchain-text-splitters sentence-transformers transformers faiss-cpu \
    huggingface-hub pypdf pydantic pydantic-settings python-dotenv

# Runtime stage
FROM python:3.11-slim

# Labels for container metadata
LABEL maintainer="42 Project"
LABEL description="Translation service using LangChain and Ollama"
LABEL version="1.0"

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    OLLAMA_HOST=host.docker.internal:11434 \
    LANGUAGE=dutch \
    REGENERATE_FROM=0 \
    REGENERATE_UNTIL=999 \
    INPUT_PATH=/data/input \
    OUTPUT_PATH=/data/output

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY translate.py .

# Create directories for input/output
RUN mkdir -p /data/input /data/output

# Create entrypoint script
RUN echo '#!/bin/bash\n\
set -e\n\
echo "=== Translation Service ==="\n\
echo "Language: ${LANGUAGE}"\n\
echo "Ollama Host: ${OLLAMA_HOST}"\n\
echo "Input: ${INPUT_PATH}"\n\
echo "Output: ${OUTPUT_PATH}"\n\
echo "Processing: chapters ${REGENERATE_FROM} to ${REGENERATE_UNTIL}"\n\
echo "==========================="\n\
python translate.py\n\
' > /entrypoint.sh && chmod +x /entrypoint.sh

# Volume mount points
VOLUME ["/data/input", "/data/output"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default entrypoint
ENTRYPOINT ["/entrypoint.sh"]
