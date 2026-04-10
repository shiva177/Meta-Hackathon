# Build v5 - reward clamping enforced at server + inference layers
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy package definition and install the package
# Use --no-deps so pip doesn't re-resolve already-installed requirements
COPY pyproject.toml .
COPY customer_support_env/ customer_support_env/
COPY server/ server/
RUN pip install --no-cache-dir --no-deps -e .

# Copy remaining application files
COPY server.py .
COPY inference.py .
COPY openenv.yaml .
COPY README.md .

# Non-root user for security
RUN adduser --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

# Environment defaults (can be overridden at runtime)
ENV PYTHONUNBUFFERED=1
ENV TASK_ID=easy

# Expose server port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the FastAPI server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
