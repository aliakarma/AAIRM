FROM python:3.11-slim

LABEL maintainer="mnaqash@iu.edu.sa"
LABEL description="AAIRM: Agentic AI Inventory Replenishment and Management"
LABEL paper="Syed et al. (2025) Agentic Commerce, Frontiers in [Journal]"
LABEL version="0.1.0"

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install package in editable mode
RUN pip install --no-cache-dir -e .

# Create output directories
RUN mkdir -p \
    data/raw/m5 \
    data/raw/favorita \
    data/raw/instacart \
    data/processed \
    data/synthetic \
    experiments/results \
    checkpoints

# Non-root user for security
RUN useradd -m -u 1000 aairm && chown -R aairm:aairm /app
USER aairm

# Default: run smoke test to verify installation
CMD ["python", "-m", "pytest", "tests/smoke/", "-v", "--timeout=60"]
