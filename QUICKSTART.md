# Mem0 AI MCP Server - Quick Start Guide

## 🚀 Quick Setup

```bash
# Clone and setup
git clone https://github.com/Angleito/Claude-CursorMemoryMCP.git
cd Claude-CursorMemoryMCP

# Install dependencies with uv (recommended)
uv venv
source .venv/bin/activate
uv sync

# Configure environment
cp .env.example .env
# Edit .env with your database credentials
```

## 🏃 Running the Servers

### For Claude/Cursor Integration (MCP)
```bash
# Test with MCP inspector
mcp dev mcp_server.py

# Run MCP server
uv run mcp_server.py
```

### For REST API Access
```bash
# Run FastAPI server
uv run uvicorn app.main:app --reload
# API docs available at http://localhost:8000/docs
```

### Full Stack with Docker
```bash
docker-compose up -d
```

## 📁 Where to Find Things

### Core Logic
- Memory operations: `src/memory.py`
- Database queries: `src/database.py`
- MCP protocol: `src/mcp.py`
- Data models: `src/models.py`

### Advanced Features
- Backup system: `src/core/backup_recovery_system.py`
- Batch processing: `src/core/batch_processor.py`
- Embeddings: `src/core/embedding_pipeline.py`
- Deduplication: `src/core/memory_deduplicator.py`

### Performance
- Query optimization: `src/optimization/query_performance_tuner.py`
- Search optimization: `src/optimization/similarity_search_optimizer.py`
- Vector compression: `src/optimization/vector_compression.py`

### Configuration
- App settings: `config/settings.py`
- Database schema: `docs/sql/setup_pgvector.sql`
- Docker config: `docker-compose.yml`

## 🔧 Common Tasks

### Run Linting
```bash
# Check code quality
uv run ruff check .

# Auto-fix issues
uv run ruff check . --fix
```

### Run Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src
```

### Database Setup
```bash
# Create database and install pgvector
createdb mem0ai
psql mem0ai < docs/sql/setup_pgvector.sql
```

## 📚 Documentation

- Project structure: [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
- Full documentation: [README.md](README.md)
- Deployment guide: [docs/guides/DEPLOYMENT_GUIDE.md](docs/guides/DEPLOYMENT_GUIDE.md)
- Claude integration: [CLAUDE.md](CLAUDE.md)

## 🆘 Troubleshooting

### MCP Server Issues
- Check logs in `logs/` directory
- Verify database connection in `.env`
- Ensure pgvector extension is installed

### Import Errors
- Run `uv sync` to install dependencies
- Check Python version (3.11+ required)
- Verify `PYTHONPATH` includes project root

### Performance Issues
- Check index optimization: `src/optimization/vector_indexing_benchmark.py`
- Monitor with Prometheus metrics
- Review query performance logs