# CustomerSupportEnv — Dockerfile
# Compatible with Hugging Face Spaces (port 7860)
# Build: docker build -t customer-support-env .
# Run:   docker run -p 7860:7860 customer-support-env

FROM python:3.11-slim

LABEL maintainer="openenv-submission"
LABEL description="CustomerSupportEnv — OpenEnv-compatible customer support RL environment"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Create non-root user (HF Spaces requirement)
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

EXPOSE 7860

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
