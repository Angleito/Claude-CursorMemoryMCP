# Linting and Code Quality Guide

This document provides comprehensive information about the linting and code quality infrastructure for the mem0ai project.

## 🎯 Overview

The project uses a multi-layered approach to code quality:

1. **Static Analysis**: Ruff, MyPy, Bandit for code analysis
2. **Formatting**: Black, Ruff for consistent code style
3. **Security**: Bandit, Safety, Semgrep for vulnerability detection
4. **Quality**: Vulture, Radon, Xenon for code quality metrics
5. **Automation**: Pre-commit hooks and GitHub Actions for CI/CD

## 🛠️ Tools Configuration

### Ruff (Primary Linter)

**File**: `pyproject.toml` → `[tool.ruff]`

Ruff is configured as the primary linting tool, replacing multiple tools:
- **Replaces**: flake8, isort, pyupgrade, autoflake
- **Rules enabled**: 200+ rules across multiple categories
- **Target Python**: 3.13+ (recommended, supports 3.11+)
- **Line length**: 88 characters
- **Performance**: ~10-100x faster than alternatives

Key rule categories:
```toml
select = [
    "E",      # pycodestyle errors
    "W",      # pycodestyle warnings  
    "F",      # Pyflakes
    "I",      # isort
    "B",      # flake8-bugbear
    "C4",     # flake8-comprehensions
    "UP",     # pyupgrade
    "N",      # pep8-naming
    "S",      # flake8-bandit
    "T20",    # flake8-print
    "SIM",    # flake8-simplify
    "RUF",    # Ruff-specific rules
    "D",      # pydocstyle
    "PL",     # Pylint
]
```

### Black (Code Formatter)

**File**: `pyproject.toml` → `[tool.black]`

Black ensures consistent code formatting:
- **Line length**: 88 characters
- **Target Python**: 3.13+ (recommended, supports 3.11-3.13)
- **Style**: Opinionated, minimal configuration
- **Integration**: Works seamlessly with Ruff

### MyPy (Type Checking)

**File**: `pyproject.toml` → `[tool.mypy]`

MyPy provides static type analysis:
- **Strict mode**: Enabled for better type safety
- **Target Python**: 3.13+ (recommended, supports 3.11+)
- **External libraries**: Type stubs for major dependencies
- **Configuration**: Comprehensive error reporting

Key settings:
```toml
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
no_implicit_optional = true
warn_redundant_casts = true
```

### Bandit (Security Scanner)

**File**: `.bandit`

Bandit scans for security vulnerabilities:
- **Confidence level**: Medium and above
- **Severity level**: Medium and above
- **Exclusions**: Test files, development scripts
- **Output format**: Text, JSON, XML, YAML supported

### Safety (Dependency Scanner)

Safety checks for known security vulnerabilities in dependencies:
- **Database**: PyUp.io vulnerability database
- **Scope**: All installed packages
- **Reporting**: JSON and text formats
- **Integration**: GitHub Actions and pre-commit

### Pre-commit Hooks

**File**: `.pre-commit-config.yaml`

Automated quality checks before commits:
- **Installation**: `pre-commit install`
- **Scope**: All staged files
- **Performance**: Parallel execution
- **Flexibility**: Can be bypassed with `--no-verify`

Configured hooks:
1. Ruff (linting and formatting)
2. Black (formatting)
3. MyPy (type checking)
4. Bandit (security)
5. Built-in hooks (trailing whitespace, YAML validation, etc.)
6. Detect-secrets (secret detection)
7. Shellcheck (shell script linting)
8. Hadolint (Dockerfile linting)

## 📋 Commands Reference

### Make Commands

```bash
# Essential commands
make lint              # Run all linting checks
make lint-fix          # Run linting with auto-fix
make format           # Alias for lint-fix
make security         # Run security scans only
make type-check       # Run MyPy type checking
make pre-commit       # Run pre-commit hooks

# Development setup
make install-dev      # Install development dependencies
make setup-dev        # Complete development environment setup

# Comprehensive checks
make ci               # Run all CI checks
make dev-check        # Development checks with auto-fix

# Quick commands
make quick-lint       # Fast linting (Ruff only)
make quick-test       # Quick test run

# Quality analysis
make quality          # Code quality checks
make reports-dir      # Create reports directory

# Configuration validation
make validate-config  # Validate all config files
```

### Script Commands

```bash
# Comprehensive linting script
./scripts/lint.sh                    # Run all checks
./scripts/lint.sh --quick            # Fast checks only
./scripts/lint.sh --security         # Security checks only
./scripts/lint.sh --format           # Formatting only
./scripts/lint.sh --type             # Type checking only

# Setup script
./scripts/setup-linting.sh           # Full setup
./scripts/setup-linting.sh --minimal # Minimal setup
```

### Direct Tool Commands

```bash
# Ruff
ruff check .                         # Check all files
ruff check . --fix                   # Fix issues automatically
ruff format .                        # Format all files
ruff check . --statistics            # Show statistics

# Black
black .                              # Format all files
black --check .                      # Check formatting
black --diff .                       # Show differences

# MyPy
mypy src/                            # Type check src directory
mypy . --ignore-missing-imports      # Ignore missing imports

# Bandit
bandit -r .                          # Scan all files
bandit -r . -f json                  # JSON output
bandit -r . -ll                      # Low confidence level

# Safety
safety check                         # Check dependencies
safety check --json                  # JSON output

# Pre-commit
pre-commit run --all-files           # Run on all files
pre-commit autoupdate                # Update hooks
```

## 🔧 Configuration Details

### File Structure

```
mem0ai/
├── .bandit                          # Bandit configuration
├── .pre-commit-config.yaml          # Pre-commit hooks
├── .secrets.baseline                # Secrets baseline
├── pyproject.toml                   # Main configuration
├── requirements-dev.txt             # Development dependencies
├── Makefile                         # Development commands
├── .github/workflows/lint.yml       # CI/CD pipeline
└── scripts/
    ├── lint.sh                      # Comprehensive linting
    └── setup-linting.sh             # Setup script
```

### IDE Integration

#### VS Code

Add to `.vscode/settings.json`:
```json
{
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.formatting.provider": "black",
    "python.linting.mypyEnabled": true,
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

#### PyCharm

1. Install Ruff plugin
2. Configure Black as external formatter
3. Enable MyPy integration
4. Setup pre-commit hook

### GitHub Actions

**File**: `.github/workflows/lint.yml`

The CI/CD pipeline includes:

1. **Multi-version testing**: Python 3.13+ (tested on 3.11-3.13)
2. **Parallel jobs**: Linting, security, quality analysis
3. **Artifact collection**: Reports and logs
4. **SARIF upload**: Security findings to GitHub Security
5. **Failure handling**: Continue on non-critical failures

Jobs breakdown:
- `lint`: Core linting across Python versions
- `security`: Security scanning with multiple tools
- `code-quality`: Quality metrics and dead code detection
- `docker-lint`: Dockerfile analysis
- `shell-lint`: Shell script validation
- `yaml-lint`: YAML file validation
- `sql-lint`: SQL formatting and validation
- `pre-commit`: Pre-commit hooks validation

## 📊 Quality Metrics

### Tracked Metrics

1. **Complexity**: Cyclomatic and cognitive complexity
2. **Coverage**: Code coverage percentage
3. **Security**: Vulnerability count and severity
4. **Maintainability**: Maintainability index
5. **Duplication**: Code duplication percentage
6. **Dependencies**: Outdated and vulnerable packages

### Thresholds

```python
# Complexity limits
max_complexity = 12          # McCabe complexity
max_cognitive_complexity = 15 # Cognitive complexity

# Quality gates
min_coverage = 80            # Minimum test coverage
max_duplication = 10         # Maximum code duplication %
min_maintainability = 70     # Minimum maintainability index
```

### Reports

Reports are generated in the `reports/` directory:
```
reports/
├── bandit-report.json       # Security scan results
├── coverage.xml             # Coverage report
├── mypy-report.txt          # Type checking report
└── quality-metrics.json    # Quality metrics
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Import Sorting Conflicts

**Problem**: Ruff and isort have different sorting
**Solution**: Disable isort or configure consistently

```toml
[tool.ruff.isort]
force-single-line = true
known-first-party = ["mem0ai", "src"]
```

#### 2. MyPy Import Errors

**Problem**: Missing type stubs
**Solution**: Add to `requirements-dev.txt`

```txt
types-redis>=4.6.0
types-requests>=2.31.0
```

#### 3. Pre-commit Hook Failures

**Problem**: Hooks fail on commit
**Solution**: Run manually and fix issues

```bash
pre-commit run --all-files
# Fix reported issues
git add .
git commit
```

#### 4. Performance Issues

**Problem**: Linting is slow
**Solution**: Use fast mode or selective linting

```bash
# Fast mode
make quick-lint

# Selective linting
ruff check src/ --fix
```

### Debugging

Enable verbose output:
```bash
# Ruff verbose mode
ruff check . --verbose

# MyPy verbose mode  
mypy . --verbose

# Pre-commit debug mode
pre-commit run --all-files --verbose
```

## 🔄 Maintenance

### Regular Tasks

1. **Weekly**: Update pre-commit hooks
   ```bash
   pre-commit autoupdate
   ```

2. **Monthly**: Update development dependencies
   ```bash
   pip-compile --upgrade requirements-dev.in
   ```

3. **Quarterly**: Review and update linting rules
   - Check for new Ruff rules
   - Update severity thresholds
   - Review exclusion patterns

### Version Updates

When updating tools:

1. **Test compatibility**: Run full test suite
2. **Update configurations**: Check for new options
3. **Update documentation**: Reflect changes
4. **Verify CI/CD**: Ensure pipeline works

### Custom Rules

To add custom linting rules:

1. **Ruff plugins**: Add to `pyproject.toml`
2. **Custom scripts**: Add to `scripts/`
3. **Pre-commit hooks**: Add to `.pre-commit-config.yaml`
4. **CI integration**: Update GitHub Actions

## 📚 References

- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Black Documentation](https://black.readthedocs.io/)
- [MyPy Documentation](https://mypy.readthedocs.io/)
- [Bandit Documentation](https://bandit.readthedocs.io/)
- [Pre-commit Documentation](https://pre-commit.com/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)

---

For questions or suggestions about the linting setup, please open an issue or contribute to the documentation.