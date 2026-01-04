# Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml ./
COPY audiobook_mcp ./audiobook_mcp

# Install the package
RUN pip install --no-cache-dir .

# Production stage
FROM python:3.11-slim

# Install ffmpeg for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/audiobook-mcp /usr/local/bin/audiobook-mcp

# Copy application code
COPY audiobook_mcp ./audiobook_mcp

# Create a non-root user
RUN useradd -m -u 1001 mcp

# Create directory for project data (can be mounted as volume)
RUN mkdir -p /projects && chown -R mcp:mcp /projects

USER mcp

# MCP servers communicate via stdio
ENTRYPOINT ["audiobook-mcp"]
