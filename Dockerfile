# Multi-stage Ubuntu-optimized Mem0 AI MCP Server Dockerfile
# Stage 1: Build dependencies
FROM ubuntu:22.04 as builder

# Set environment variables for build
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    # Essential build tools
    build-essential \
    gcc \
    g++ \
    libc6-dev \
    # Python and development headers
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    # Network tools
    curl \
    wget \
    ca-certificates \
    # Database clients
    postgresql-client \
    libpq-dev \
    # Additional libraries for ML/vector operations
    libopenblas-dev \
    liblapack-dev \
    libblas-dev \
    gfortran \
    # Git for dependencies that might need it
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# Install uv (modern Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Set work directory
WORKDIR /app

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock* requirements*.txt ./

# Create virtual environment and install dependencies
RUN uv venv .venv --python 3.12
ENV PATH="/app/.venv/bin:$PATH"

# Install Python dependencies with optimizations
RUN uv pip install --no-cache-dir -r requirements.txt \
    && if [ -f requirements-dev.txt ]; then uv pip install --no-cache-dir -r requirements-dev.txt; fi \
    && python -m pip install --upgrade pip setuptools wheel

# Stage 2: Production runtime
FROM ubuntu:22.04 as runtime

# Set environment variables for runtime
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/app/.venv/bin:$PATH" \
    # Ubuntu/Linux optimizations
    PYTHONOPTIMIZE=1 \
    MALLOC_MMAP_THRESHOLD_=1024 \
    MALLOC_TRIM_THRESHOLD_=1024 \
    MALLOC_TOP_PAD_=1024 \
    MALLOC_MMAP_MAX_=65536

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    # Minimal Python runtime
    python3.12 \
    python3.12-venv \
    # Essential libraries
    libpq5 \
    libopenblas0 \
    liblapack3 \
    libblas3 \
    # Network and SSL
    curl \
    ca-certificates \
    # Monitoring tools
    htop \
    procps \
    # Database client
    postgresql-client \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# Create application user with minimal privileges
RUN groupadd -r mem0ai --gid=999 && \
    useradd -r -g mem0ai --uid=999 --home-dir=/app --shell=/bin/bash mem0ai

# Set work directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder --chown=mem0ai:mem0ai /app/.venv /app/.venv

# Copy application files
COPY --chown=mem0ai:mem0ai . .

# Create necessary directories with proper permissions
RUN mkdir -p /app/logs /app/plugins /app/ssl /app/backups /app/data \
    && chown -R mem0ai:mem0ai /app \
    && chmod -R 755 /app \
    && chmod +x /app/main.py

# Switch to non-root user
USER mem0ai

# Expose port
EXPOSE 8000

# Health check optimized for Ubuntu
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Volume for persistent data
VOLUME ["/app/data", "/app/logs"]

# Default command with optimizations
CMD ["python", "-O", "main.py"]