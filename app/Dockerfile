# Multi-stage build for Ubuntu production deployment
FROM ubuntu:22.04 as builder

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set Ubuntu-optimized environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONOPTIMIZE=1 \
    MALLOC_MMAP_THRESHOLD_=1024 \
    MALLOC_TRIM_THRESHOLD_=1024 \
    MALLOC_TOP_PAD_=1024 \
    MALLOC_MMAP_MAX_=65536

# Install build dependencies and Python
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    build-essential \
    gcc \
    g++ \
    curl \
    git \
    pkg-config \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install uv
RUN pip3 install uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock* ./

# Create virtual environment and install dependencies
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv
RUN uv sync --frozen --no-cache --no-dev

# Production stage
FROM ubuntu:22.04 as runtime

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Set Ubuntu-optimized runtime environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONOPTIMIZE=1 \
    MALLOC_MMAP_THRESHOLD_=1024 \
    MALLOC_TRIM_THRESHOLD_=1024 \
    MALLOC_TOP_PAD_=1024 \
    MALLOC_MMAP_MAX_=65536 \
    PATH="/app/.venv/bin:$PATH"

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r mem0ai && useradd -r -g mem0ai -d /app -s /bin/bash mem0ai

# Set working directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY . .

# Create necessary directories and set permissions
RUN mkdir -p /app/logs /app/uploads && \
    chown -R mem0ai:mem0ai /app

# Switch to non-root user
USER mem0ai

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the application
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--access-logfile", "-", "--error-logfile", "-", "main:app"]