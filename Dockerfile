# ─────────────────────────────────────────────────────────────────
# Schema Migration OpenEnv — Dockerfile
# Base: python:3.11-slim  (no GPU needed, SQLite is stdlib)
# Port: 7860 (HuggingFace Spaces default)
# ─────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Metadata
LABEL maintainer="Roshan Kumar Singh"
LABEL description="Schema Migration OpenEnv — database migration RL environment"
LABEL version="1.0.0"

# System deps (minimal — SQLite is built into Python stdlib)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Install Python deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=appuser:appuser . .

# Create __init__ files so Python treats dirs as packages
RUN touch env/__init__.py tasks/__init__.py graders/__init__.py

# Switch to non-root user
USER appuser

# Health check — ping /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Expose HuggingFace Spaces port
EXPOSE 7860

# Environment defaults (overridden by HF Spaces secrets)
ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Start the FastAPI server
CMD ["python", "server.py"]
