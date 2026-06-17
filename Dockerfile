# ---------------------------------------------------------------------------
# Production Dockerfile — Flamingo/IDEFICS2 Radiology API
# ---------------------------------------------------------------------------
# Build:  docker build -t flamingo-radiology-api .
# Run:    docker run --gpus all -p 8080:8080 flamingo-radiology-api
# ---------------------------------------------------------------------------

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# System dependencies (opencv needs libgl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3-pip python3.11-dev \
    libgl1 libglib2.0-0 git curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Optional: Flash Attention (significant speedup on A100/H100)
# RUN pip install flash-attn --no-build-isolation

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/

# Create necessary directories
RUN mkdir -p /app/outputs /app/logs /app/data/few_shot_examples

# Non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
