# Dockerfile - Multi-stage build for lightweight footprint

# Stage 1: Builder
FROM python:3.10-slim AS builder

WORKDIR /app

# Install system dependencies needed for compiling python packages (if any)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Create wheels to avoid bringing build dependencies into the final image
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# Stage 2: Final runtime image
FROM python:3.10-slim

WORKDIR /app

# Copy wheels from builder and install
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache /wheels/* && rm -rf /wheels

# Copy application files
COPY ingest.py app.py ./

# Create directories for volumes
RUN mkdir -p /app/data /app/chroma_db

# Default command (can be overridden to run ingest.py instead)
CMD ["python", "app.py"]
