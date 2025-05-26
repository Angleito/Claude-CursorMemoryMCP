# 🔧 Linting Infrastructure Documentation

This document provides a comprehensive overview of the linting and code quality infrastructure implemented for the mem0ai project.

## 📋 Overview

The project uses a multi-layered approach to code quality with automated linting, formatting, security scanning, and type checking. All tools are configured to work together seamlessly and run automatically via pre-commit hooks and CI/CD pipelines.

## 🛠️ Tools Configuration

### Primary Tools

| Tool | Purpose | Configuration File | Status |
|------|---------|-------------------|---------|
| **Ruff** | Fast Python linter & formatter | `pyproject.toml` | ✅ Configured |
| **Black** | Code formatter | `pyproject.toml` | ✅ Configured |
| **MyPy** | Static type checking | `mypy.ini` + `pyproject.toml` | ✅ Configured |
| **Bandit** | Security linter | `.bandit` + `pyproject.toml` | ✅ Configured |
| **isort** | Import sorting | `.isort.cfg` + `pyproject.toml` | ✅ Configured |

### Additional Quality Tools

| Tool | Purpose | Configuration | Status |
|------|---------|--------------|---------|
| **Safety** | Dependency vulnerability scanner | `pyproject.toml` | ✅ Configured |
| **Vulture** | Dead code detection | `pyproject.toml` | ✅ Configured |
| **Radon** | Code complexity analysis | `pyproject.toml` | ✅ Configured |
| **Xenon** | Complexity monitoring | `pyproject.toml` | ✅ Configured |
| **detect-secrets** | Secret detection | `.secrets.baseline` | ✅ Configured |
| **yamllint** | YAML linting | `.yamllint` | ✅ Configured |

## 📁 Configuration Files

### Main Configuration Files

```
📁 Project Root
├── pyproject.toml              # Main configuration hub
├── .pre-commit-config.yaml     # Pre-commit hooks configuration
├── .flake8                     # Flake8 configuration (backup)
├── .isort.cfg                  # Import sorting configuration
├── mypy.ini                    # Type checking configuration
├── .bandit                     # Security scanning configuration
├── .yamllint                   # YAML linting configuration
├── .secrets.baseline           # Secrets detection baseline
└── .gitignore                  # Git ignore patterns (updated)
```

### GitHub Actions Workflows

```
📁 .github/workflows/
└── ci.yml                      # Comprehensive CI/CD pipeline
```

### Scripts

```
📁 scripts/
├── setup-linting.sh           # Automated setup script
└── lint.sh                    # Quick linting script
```

## 🚀 Quick Start

### 1. Initial Setup

```bash
# Run the automated setup script
./scripts/setup-linting.sh

# Or manual setup
pip install -r requirements-dev.txt
pre-commit install
detect-secrets scan --baseline .secrets.baseline
```

### 2. Daily Usage

```bash
# Quick linting and auto-fix
make lint-fix

# Full quality check
make dev-check

# Run specific tools
make type-check    # MyPy type checking
make security      # Security scans
make quality       # Code quality analysis
```

### 3. Pre-commit Integration

All tools run automatically on commit:

```bash
git add .
git commit -m "Your commit message"
# Pre-commit hooks run automatically
```

## 📊 Code Quality Standards

### Formatting Standards

- **Line Length**: 88 characters (Black/Ruff standard)
- **Import Style**: Sorted by isort with Black profile
- **Quote Style**: Double quotes preferred
- **Trailing Commas**: Required for multi-line structures

### Type Checking Standards

- **Strict Mode**: Enabled with MyPy
- **Type Hints**: Required for all public functions
- **Return Types**: Must be explicitly typed
- **Imports**: Typed imports for all external libraries

### Security Standards

- **Bandit**: Security linting with medium confidence
- **Safety**: Dependency vulnerability scanning
- **Secrets**: Automated secret detection
- **Code Analysis**: Static analysis for security patterns

### Complexity Standards

- **Cyclomatic Complexity**: Maximum 12 (Ruff/McCabe)
- **Cognitive Complexity**: Maximum 12
- **Function Length**: Maximum 50 lines
- **Module Length**: Maximum 1000 lines

## 🔄 CI/CD Integration

### GitHub Actions Pipeline

The CI/CD pipeline runs multiple jobs in parallel:

1. **Lint Job**: Code quality and formatting checks
2. **Test Job**: Comprehensive testing with coverage
3. **Security Job**: Security scanning and analysis
4. **Docker Job**: Container building and security
5. **Performance Job**: Benchmark testing

### Quality Gates

- **Coverage**: Minimum 80% test coverage required
- **Linting**: All Ruff checks must pass
- **Type Checking**: MyPy strict mode must pass
- **Security**: No high-severity Bandit issues
- **Dependencies**: No known vulnerabilities in Safety

## 📈 Reports and Monitoring

### Report Locations

```
📁 reports/
├── ruff-report.json           # Ruff linting results
├── mypy-report.txt            # Type checking results
├── bandit-report.json         # Security scan results
├── safety-report.json         # Dependency vulnerability results
├── coverage.xml               # Test coverage report
└── htmlcov/                   # HTML coverage report
```

### Coverage Reports

- **Terminal**: Live coverage during test runs
- **HTML**: Detailed coverage report in `htmlcov/`
- **XML**: Machine-readable coverage for CI/CD
- **Codecov**: Online coverage tracking and badges

### Quality Metrics

- **Code Coverage**: Tracked via Codecov
- **Complexity**: Monitored via Radon/Xenon
- **Security**: Tracked via Bandit/Safety
- **Dead Code**: Detected via Vulture

## 🔧 Tool-Specific Configurations

### Ruff Configuration

```toml
[tool.ruff]
target-version = "py38"
line-length = 88
select = ["E", "W", "F", "I", "B", "C4", "UP", "N", "S", "T20", "SIM", "RUF", "D", "PL"]
ignore = ["E501", "D100", "D101", "D102", "D103", "D104", "D107", "S101"]
```

### MyPy Configuration

```ini
[mypy]
python_version = 3.11
strict = True
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
```

### Bandit Configuration

```ini
[bandit]
exclude_dirs = ["tests", "venv", "env"]
skips = ["B101", "B601"]
confidence = "MEDIUM"
severity = "MEDIUM"
```

## 🎯 Best Practices

### Code Quality Workflow

1. **Write Code**: Follow established patterns
2. **Run Tests**: Ensure functionality works
3. **Lint Locally**: `make lint-fix` before commit
4. **Commit**: Pre-commit hooks run automatically
5. **Push**: CI/CD pipeline validates everything
6. **Review**: Code review includes quality checks

### Handling Quality Issues

1. **Linting Errors**: Fix automatically with `make lint-fix`
2. **Type Errors**: Add type hints or use `# type: ignore`
3. **Security Issues**: Address or document why safe
4. **Complexity**: Refactor functions to reduce complexity
5. **Coverage**: Add tests for uncovered code

### Configuration Management

1. **Central Config**: Most settings in `pyproject.toml`
2. **Tool-Specific**: Separate files for complex configurations
3. **Environment**: Use `.env` files for secrets
4. **Documentation**: Keep this file updated with changes

## 🔄 Maintenance

### Regular Tasks

- **Weekly**: Update pre-commit hooks (`pre-commit autoupdate`)
- **Monthly**: Review and update tool versions
- **Quarterly**: Review quality standards and metrics
- **As Needed**: Add new rules or tools as project evolves

### Troubleshooting

#### Common Issues

1. **Pre-commit Failures**: Check tool versions and configuration
2. **Type Checking Errors**: Ensure all dependencies have type stubs
3. **Security False Positives**: Add to Bandit ignore list
4. **Performance Issues**: Consider disabling expensive checks locally

#### Getting Help

- **Tool Documentation**: Each tool has comprehensive docs
- **Community**: Most tools have active communities
- **Issues**: Report bugs to respective tool repositories
- **Internal**: Check project-specific documentation

## 🎊 Benefits

### Developer Experience

- **Consistency**: Automated formatting ensures consistent style
- **Quality**: Catches issues before they reach production
- **Learning**: Tools provide educational feedback
- **Efficiency**: Automated checks save manual review time

### Project Health

- **Maintainability**: Consistent code is easier to maintain
- **Security**: Automated security scanning prevents vulnerabilities
- **Reliability**: Type checking and testing reduce bugs
- **Performance**: Complexity monitoring prevents performance issues

### Team Collaboration

- **Standards**: Clear, enforced coding standards
- **Reviews**: Focus reviews on logic, not style
- **Onboarding**: New developers get immediate feedback
- **Documentation**: Self-documenting configuration

---

**Last Updated**: 2024-05-25
**Version**: 1.0.0
**Maintainer**: mem0ai Team