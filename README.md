# 🧠 Claude-Cursor Memory MCP Server

[![CI/CD Pipeline](https://github.com/Angleito/Claude-CursorMemoryMCP/actions/workflows/ci.yml/badge.svg)](https://github.com/Angleito/Claude-CursorMemoryMCP/actions/workflows/ci.yml)
[![Code Quality](https://github.com/Angleito/Claude-CursorMemoryMCP/actions/workflows/lint.yml/badge.svg)](https://github.com/Angleito/Claude-CursorMemoryMCP/actions/workflows/lint.yml)
[![Security](https://github.com/Angleito/Claude-CursorMemoryMCP/actions/workflows/security.yml/badge.svg)](https://github.com/Angleito/Claude-CursorMemoryMCP/actions/workflows/security.yml)
[![Coverage](https://codecov.io/gh/Angleito/Claude-CursorMemoryMCP/branch/main/graph/badge.svg)](https://codecov.io/gh/Angleito/Claude-CursorMemoryMCP)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linting: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

An intelligent memory persistence layer that integrates with Claude Desktop and Claude Code via the Model Context Protocol (MCP), providing long-term conversational memory and context across sessions.

## 🎯 Overview

The Claude-Cursor Memory MCP Server is a specialized memory system designed to enhance AI coding assistants by providing persistent memory capabilities. Built on PostgreSQL with pgvector, it allows Claude Desktop and Claude Code to remember conversations, code patterns, user preferences, and project context across sessions.

### 🚀 What is MCP?

The Model Context Protocol (MCP) is Anthropic's standardized way for AI assistants to connect to external data sources and tools. This memory server acts as an MCP server that:

- **Remembers Conversations**: Stores chat history and context across sessions
- **Learns Code Patterns**: Remembers coding preferences and patterns you use
- **Maintains Project Context**: Keeps track of project-specific information
- **Provides Smart Suggestions**: Uses memory to offer more relevant suggestions

### ⚡ Key Features

- **MCP Integration**: Native support for Claude Desktop and Claude Code
- **Intelligent Memory**: Vector-based similarity search for relevant context retrieval
- **Session Persistence**: Maintains conversation history across AI sessions
- **Code Pattern Learning**: Remembers your coding style and preferences
- **Project Memory**: Context-aware memory per project/workspace
- **Privacy-First**: Local deployment with GDPR compliance
- **Real-time Sync**: Instant memory updates across multiple AI sessions
- **Advanced Search**: Natural language queries to find relevant memories
- **Secure**: Enterprise-grade security with encryption and access controls

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.11+ (recommended: 3.13.3+)
- [uv](https://docs.astral.sh/uv/) package manager (recommended)
- PostgreSQL 13+ with pgvector extension
- Docker & Docker Compose (recommended)
- Claude Desktop or Claude Code for MCP integration

### Quick Start

1. **Clone and Setup**
   ```bash
   git clone https://github.com/Angleito/Claude-CursorMemoryMCP.git
   cd Claude-CursorMemoryMCP
   
   # Install using uv (recommended)
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv add "mcp[cli]"
   uv sync
   
   # Or use setup script
   python scripts/setup.py
   ```

2. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   nano .env
   ```

3. **Test the MCP Server**
   ```bash
   # Development mode with MCP Inspector
   mcp dev main.py
   
   # Or run directly
   uv run main.py
   ```

4. **Install in Claude Desktop/Claude Code**
   ```bash
   # Quick install for Claude Desktop
   mcp install main.py --name "Memory Server"
   
   # Or configure manually (see MCP Configuration section)
   ```

### Docker Deployment

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f
```

## 🔧 MCP Configuration

### Claude Desktop Integration

Add to your Claude Desktop configuration at `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "memory": {
      "command": "uv",
      "args": [
        "--directory",
        "/ABSOLUTE/PATH/TO/Claude-CursorMemoryMCP",
        "run",
        "main.py"
      ],
      "env": {
        "MEMORY_DATABASE_URL": "postgresql://localhost/memory",
        "MEMORY_REDIS_URL": "redis://localhost:6379"
      }
    }
  }
}
```

### Claude Code Integration

For Claude Code, add to your configuration file:

```json
{
  "mcpServers": {
    "memory": {
      "command": "uv",
      "args": [
        "--directory",
        "/ABSOLUTE/PATH/TO/Claude-CursorMemoryMCP",
        "run",
        "main.py"
      ],
      "env": {
        "MEMORY_DATABASE_URL": "postgresql://localhost/memory",
        "MEMORY_REDIS_URL": "redis://localhost:6379"
      }
    }
  }
}
```

### Alternative Python Installation

If you prefer using Python directly:

```json
{
  "mcpServers": {
    "memory": {
      "command": "python",
      "args": ["/ABSOLUTE/PATH/TO/Claude-CursorMemoryMCP/main.py"],
      "env": {
        "PYTHONPATH": "/ABSOLUTE/PATH/TO/Claude-CursorMemoryMCP",
        "MEMORY_DATABASE_URL": "postgresql://localhost/memory",
        "MEMORY_REDIS_URL": "redis://localhost:6379"
      }
    }
  }
}
```

## 📋 MCP Tools Available

### Memory Management
- `store_memory(content, context, tags)` - Store new memories
- `search_memory(query, limit)` - Search existing memories
- `get_recent_memories(limit)` - Get recent conversation history
- `delete_memory(memory_id)` - Remove specific memories

### Context Management
- `save_context(project, context)` - Save project context
- `load_context(project)` - Load project context
- `list_projects()` - List all projects with memory

### Code Pattern Learning
- `learn_pattern(code, description, language)` - Learn coding patterns
- `suggest_pattern(context)` - Get pattern suggestions
- `get_code_style(language)` - Get learned coding style

### Session Management
- `start_session(project, user)` - Start new coding session
- `end_session(session_id)` - End coding session
- `get_session_history(session_id)` - Get session history

## 🏗️ Core Architecture

### Memory Types

1. **Conversational Memory**
   - Chat history and context
   - Question-answer pairs
   - Problem-solution patterns

2. **Code Memory**
   - Code snippets and patterns
   - Function implementations
   - Architecture decisions

3. **Project Memory**
   - Project-specific context
   - File structures and patterns
   - Configuration preferences

4. **User Memory**
   - Coding style preferences
   - Frequently used patterns
   - Personal shortcuts and habits

### Vector Storage

Built on PostgreSQL with pgvector for:
- **Semantic Search**: Find relevant memories by meaning, not just keywords
- **Context Similarity**: Match similar coding scenarios
- **Pattern Recognition**: Identify recurring patterns and preferences
- **Intelligent Retrieval**: Surface the most relevant memories for current context

## 🔐 Security & Privacy

### Data Protection
- **Local Storage**: All data stays on your machine
- **Encryption**: Data encrypted at rest and in transit
- **Access Control**: Role-based permissions and API keys
- **GDPR Compliance**: Built-in privacy controls and data rights

### Security Features
- JWT authentication
- API key management
- Rate limiting and DDoS protection
- Audit logging
- Multi-factor authentication support

## 📊 Monitoring & Analytics

### Built-in Dashboards
- **Memory Usage**: Track memory growth and patterns
- **Search Performance**: Monitor query performance
- **User Activity**: Session and usage analytics
- **System Health**: Infrastructure monitoring

### Prometheus Metrics
```
memory_operations_total
memory_search_duration_seconds
active_sessions_gauge
memory_storage_bytes
```

## 🚀 Advanced Features

### Intelligent Memory Management
- **Automatic Cleanup**: Remove stale or irrelevant memories
- **Memory Consolidation**: Merge similar memories
- **Importance Scoring**: Prioritize valuable memories
- **Context Awareness**: Adapt memory relevance to current project

### Performance Optimization
- **Vector Compression**: Reduce storage requirements
- **Caching**: Redis-based result caching
- **Batch Processing**: Efficient bulk operations
- **Query Optimization**: Automatic PostgreSQL tuning

### Integration Ecosystem
- **API Access**: RESTful API for custom integrations
- **Webhook Support**: Real-time notifications
- **Plugin System**: Extend functionality
- **Export/Import**: Backup and migrate memories

## 📁 Project Structure

```
├── src/                    # Core MCP server implementation
│   ├── mcp.py             # MCP protocol implementation
│   ├── memory.py          # Memory storage and retrieval
│   ├── models.py          # Data models and schemas
│   └── websocket.py       # Real-time communication
├── auth/                  # Authentication and security
├── config/                # Configuration management
├── scripts/               # Setup and maintenance scripts
├── monitoring/            # Observability and metrics
├── examples/              # Client integration examples
└── docker-compose.yml     # Container orchestration
```

## 🤝 Usage Examples

### Storing Code Patterns

```python
# Via MCP in Claude Desktop/Claude Code
await store_memory(
    content="Always use TypeScript strict mode for new projects",
    context={"project": "web-app", "language": "typescript"},
    tags=["typescript", "configuration", "best-practice"]
)
```

### Searching Previous Solutions

```python
# Find similar problems you've solved before
memories = await search_memory(
    query="how to handle async errors in Python",
    limit=5
)
```

### Project Context Loading

```python
# Automatically load project context when switching projects
context = await load_context("my-react-app")
# Context includes: file patterns, coding style, dependencies, etc.
```

## 🧹 Code Quality & Linting

This project uses comprehensive linting and code quality tools to maintain high standards.

### Linting Tools Configured

- **Ruff**: Fast Python linter and formatter (replaces flake8, isort, and more)
- **Black**: Opinionated code formatter
- **MyPy**: Static type checking
- **Bandit**: Security vulnerability scanner
- **Safety**: Dependency vulnerability checker
- **Vulture**: Dead code detector
- **Pre-commit hooks**: Automated code quality checks

### Quick Linting Commands

```bash
# Run all linting checks
make lint

# Run linting with auto-fix
make lint-fix

# Run security checks only
make security

# Run type checking
make type-check

# Run pre-commit hooks
make pre-commit

# Quick lint (Ruff only)
make quick-lint

# Comprehensive linting script
./scripts/lint.sh
```

### Development Setup

```bash
# Install development dependencies
make install-dev

# Setup complete development environment
make setup-dev

# Setup linting infrastructure
./scripts/setup-linting.sh
```

### CI/CD Pipeline

The project includes a comprehensive GitHub Actions workflow (`.github/workflows/lint.yml`) that runs:

- Multi-version Python linting (3.13+ recommended, tested on 3.11-3.13)
- Security scanning with Bandit, Safety, and Semgrep
- Code quality analysis with Vulture, Radon, and Xenon
- Docker, Shell, YAML, and SQL linting
- Pre-commit hook validation

### Configuration Files

- `pyproject.toml`: Main configuration for Ruff, Black, MyPy, and more
- `.bandit`: Security scanning configuration
- `.pre-commit-config.yaml`: Pre-commit hooks setup
- `requirements-dev.txt`: Development dependencies
- `Makefile`: Convenient development commands

### Linting Standards

The project enforces:
- PEP 8 compliance with 88-character line length
- Google-style docstrings
- Type hints for all functions
- Security best practices
- Import sorting and formatting
- Dead code elimination

## 🤝 Contributing

1. Fork the repository
2. Setup development environment: `make setup-dev`
3. Create a feature branch
4. Write code following the linting standards
5. Run linting: `make lint-fix`
6. Add tests for new functionality
7. Run the test suite: `pytest`
8. Ensure CI passes: `make ci`
9. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
- GitHub Issues: [Report bugs and feature requests](https://github.com/Angleito/Claude-CursorMemoryMCP/issues)
- Documentation: [Full documentation](https://github.com/Angleito/Claude-CursorMemoryMCP/wiki)
- Discord: [Join our community](https://discord.gg/your-server)

## 🙏 Acknowledgments

- [Anthropic](https://anthropic.com) for the MCP protocol
- [Cursor](https://cursor.sh) for AI-powered coding
- [pgvector](https://github.com/pgvector/pgvector) for vector similarity search
- Open source community for the amazing tools and libraries