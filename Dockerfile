# Build stage — install dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies required by rdkit and common Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Runtime stage ─────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Runtime system deps (rdkit needs libxrender at runtime too)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application files
COPY api.py .
COPY models/ ./models/
COPY mobile_assets/ ./mobile_assets/
COPY dataset/ ./dataset/

# Hugging Face Spaces default port is 7860; Render injects $PORT
ENV PORT=7860
EXPOSE 7860

# Run as non-root user (HF Spaces requirement)
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Use api_light (fingerprint-only, no ONNX model loading) for minimal RAM usage
CMD ["sh", "-c", "uvicorn api_light:app --host 0.0.0.0 --port ${PORT:-7860}"]
