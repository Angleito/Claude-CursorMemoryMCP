#!/bin/bash
# Setup script for comprehensive linting and code quality tools
# Usage: ./scripts/setup-linting.sh

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
check_project_root() {
    if [[ ! -f "pyproject.toml" ]]; then
        log_error "pyproject.toml not found. Please run this script from the project root."
        exit 1
    fi
    log_success "Project root directory confirmed"
}

# Check Python version
check_python_version() {
    local python_version
    python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
    local major_version
    major_version=$(echo "$python_version" | cut -d'.' -f1)
    local minor_version
    minor_version=$(echo "$python_version" | cut -d'.' -f2)
    
    if [[ $major_version -lt 3 ]] || [[ $major_version -eq 3 && $minor_version -lt 8 ]]; then
        log_error "Python 3.8+ required. Found: $python_version"
        exit 1
    fi
    log_success "Python version check passed: $python_version"
}

# Create necessary directories
create_directories() {
    log_info "Creating necessary directories..."
    mkdir -p logs
    mkdir -p reports
    mkdir -p htmlcov
    mkdir -p .github/workflows
    log_success "Directories created"
}

# Install uv if not available
install_uv() {
    log_info "Checking for uv installation..."
    
    if command -v uv &> /dev/null; then
        log_success "uv is already installed"
        uv --version
        return 0
    fi
    
    log_info "Installing uv (modern Python package manager)..."
    
    # Install uv using the official installer
    if curl -LsSf https://astral.sh/uv/install.sh | sh; then
        # Add uv to PATH for current session
        export PATH="$HOME/.cargo/bin:$PATH"
        
        # Verify installation
        if command -v uv &> /dev/null; then
            log_success "uv installed successfully"
            uv --version
            return 0
        else
            log_warning "uv installation may have failed"
            return 1
        fi
    else
        log_warning "Failed to install uv"
        return 1
    fi
}

# Install development dependencies
install_dependencies() {
    log_info "Installing development dependencies..."
    
    # Try to install uv first
    if install_uv; then
        log_info "Using uv for package management"
        UV_AVAILABLE=true
        
        # Create virtual environment with uv if it doesn't exist
        if [[ ! -d ".venv" ]]; then
            log_info "Creating virtual environment with uv..."
            uv venv .venv
            log_success "Virtual environment created"
        fi
    else
        log_warning "uv not available, falling back to pip"
        UV_AVAILABLE=false
        python3 -m pip install --upgrade pip
    fi
    
    # Install requirements using modern uv approach
    if [[ -f "pyproject.toml" ]] && [[ "$UV_AVAILABLE" == "true" ]]; then
        log_info "Installing dependencies with uv sync..."
        uv sync
        log_success "Dependencies installed with uv sync"
    elif [[ -f "requirements-dev.txt" ]]; then
        if [[ "$UV_AVAILABLE" == "true" ]]; then
            uv pip install -r requirements-dev.txt
        else
            python3 -m pip install -r requirements-dev.txt
        fi
        log_success "Development dependencies installed"
    else
        log_warning "requirements-dev.txt not found, installing core linting tools..."
        if [[ "$UV_AVAILABLE" == "true" ]]; then
            uv pip install \
                ruff>=0.1.9 \
                black>=23.12.0 \
                isort>=5.13.0 \
                mypy>=1.8.0 \
                "bandit[toml]>=1.7.5" \
                safety>=2.3.0 \
                vulture>=2.10 \
                radon>=6.0.1 \
                xenon>=0.9.1 \
                pre-commit>=3.6.0 \
                detect-secrets>=1.4.0 \
                pytest>=7.4.0 \
                pytest-cov>=4.1.0 \
                yamllint>=1.33.0
        else
            python3 -m pip install \
                ruff>=0.1.9 \
                black>=23.12.0 \
                isort>=5.13.0 \
                mypy>=1.8.0 \
                bandit[toml]>=1.7.5 \
                safety>=2.3.0 \
                vulture>=2.10 \
                radon>=6.0.1 \
                xenon>=0.9.1 \
                pre-commit>=3.6.0 \
                detect-secrets>=1.4.0 \
                pytest>=7.4.0 \
                pytest-cov>=4.1.0 \
                yamllint>=1.33.0
        fi
        log_success "Core linting tools installed"
    fi
    
    # Install production dependencies if available
    if [[ -f "requirements.txt" ]]; then
        if [[ "$UV_AVAILABLE" == "true" ]]; then
            uv pip install -r requirements.txt
        else
            python3 -m pip install -r requirements.txt
        fi
        log_success "Production dependencies installed"
    fi
}

# Setup pre-commit hooks
setup_precommit() {
    log_info "Setting up pre-commit hooks..."
    
    if command -v pre-commit &> /dev/null; then
        # Install pre-commit hooks
        pre-commit install
        pre-commit install --hook-type commit-msg
        pre-commit install --hook-type pre-push
        
        # Update hooks to latest versions
        pre-commit autoupdate
        
        log_success "Pre-commit hooks installed and updated"
    else
        log_error "pre-commit not found. Installing..."
        if [[ "$UV_AVAILABLE" == "true" ]]; then
            uv pip install pre-commit
        else
            python3 -m pip install pre-commit
        fi
        pre-commit install
        log_success "Pre-commit installed and configured"
    fi
}

# Initialize detect-secrets baseline
setup_detect_secrets() {
    log_info "Setting up detect-secrets baseline..."
    
    if [[ ! -f ".secrets.baseline" ]]; then
        detect-secrets scan --baseline .secrets.baseline
        log_success "Secrets baseline created"
    else
        log_info "Updating existing secrets baseline..."
        detect-secrets scan --baseline .secrets.baseline --update
        log_success "Secrets baseline updated"
    fi
}

# Run initial linting check
run_initial_checks() {
    log_info "Running initial code quality checks..."
    
    # Create reports directory if it doesn't exist
    mkdir -p reports
    
    # Run Ruff check (non-blocking)
    log_info "Running Ruff linter..."
    if ruff check . --output-format=json > reports/ruff-report.json 2>/dev/null; then
        log_success "Ruff check completed successfully"
    else
        log_warning "Ruff found issues (see reports/ruff-report.json)"
    fi
    
    # Run Black check (non-blocking)
    log_info "Running Black formatter check..."
    if black --check . > reports/black-report.txt 2>&1; then
        log_success "Black formatting check passed"
    else
        log_warning "Black found formatting issues (see reports/black-report.txt)"
    fi
    
    # Run MyPy check (non-blocking)
    log_info "Running MyPy type checking..."
    if mypy src/ auth/ security/ monitoring/ --ignore-missing-imports > reports/mypy-report.txt 2>&1; then
        log_success "MyPy type checking passed"
    else
        log_warning "MyPy found type issues (see reports/mypy-report.txt)"
    fi
    
    # Run Bandit security scan (non-blocking)
    log_info "Running Bandit security scan..."
    if bandit -r . -f json -o reports/bandit-report.json 2>/dev/null; then
        log_success "Bandit security scan completed"
    else
        log_warning "Bandit found security issues (see reports/bandit-report.json)"
    fi
    
    # Run Safety dependency check (non-blocking)
    log_info "Running Safety dependency check..."
    if safety check --json --output reports/safety-report.json 2>/dev/null; then
        log_success "Safety dependency check passed"
    else
        log_warning "Safety found vulnerable dependencies (see reports/safety-report.json)"
    fi
    
    log_success "Initial code quality checks completed. See reports/ directory for details."
}

# Validate configuration files
validate_configs() {
    log_info "Validating configuration files..."
    
    # Validate YAML files
    if command -v yamllint &> /dev/null; then
        if yamllint . > reports/yamllint-report.txt 2>&1; then
            log_success "YAML files are valid"
        else
            log_warning "YAML validation issues found (see reports/yamllint-report.txt)"
        fi
    fi
    
    # Validate JSON files
    log_info "Validating JSON files..."
    json_valid=true
    while IFS= read -r -d '' file; do
        if ! python3 -m json.tool "$file" > /dev/null 2>&1; then
            log_warning "Invalid JSON file: $file"
            json_valid=false
        fi
    done < <(find . -name "*.json" -not -path "./node_modules/*" -not -path "./.git/*" -print0)
    
    if $json_valid; then
        log_success "All JSON files are valid"
    fi
    
    # Validate TOML files
    log_info "Validating TOML files..."
    if python3 -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb'))" 2>/dev/null; then
        log_success "TOML files are valid"
    else
        log_warning "TOML validation failed"
    fi
}

# Setup IDE configurations
setup_ide_configs() {
    log_info "Setting up IDE configurations..."
    
    # VS Code settings
    if [[ ! -d ".vscode" ]]; then
        mkdir -p .vscode
        cat > .vscode/settings.json << 'EOF'
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.linting.banditEnabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.args": ["--profile=black"],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "files.exclude": {
        "**/__pycache__": true,
        "**/.pytest_cache": true,
        "**/.mypy_cache": true,
        "**/.ruff_cache": true,
        "**/htmlcov": true,
        "**/reports": true
    }
}
EOF
        log_success "VS Code settings configured"
    fi
}

# Display usage instructions
show_usage() {
    log_info "Linting setup complete! Here are some useful commands:"
    echo ""
    
    # Show uv-specific commands if available
    if command -v uv &> /dev/null; then
        echo "Package management with uv:"
        echo "  uv add <package>    # Add a new dependency"
        echo "  uv sync             # Sync dependencies from lock file"
        echo "  uv run <command>    # Run command in virtual environment"
        echo "  uv pip install -r requirements.txt  # Install from requirements"
        echo ""
    fi
    
    echo "Linting commands:"
    echo "  make lint           # Run all linting checks"
    echo "  make lint-fix       # Run linting with auto-fix"
    echo "  make type-check     # Run type checking"
    echo "  make security       # Run security scans"
    echo "  make test-cov       # Run tests with coverage"
    echo "  make pre-commit     # Run pre-commit hooks"
    echo "  make ci             # Run full CI pipeline"
    echo ""
    echo "Configuration files:"
    echo "  pyproject.toml      # Main configuration"
    echo "  .pre-commit-config.yaml  # Pre-commit hooks"
    echo "  .flake8             # Flake8 configuration"
    echo "  .isort.cfg          # Import sorting"
    echo "  mypy.ini            # Type checking"
    echo "  .bandit             # Security scanning"
    echo ""
    
    # Show virtual environment info
    if [[ -d ".venv" ]]; then
        echo "Virtual environment:"
        echo "  .venv/              # Virtual environment (created with uv)"
        echo "  Activate with: source .venv/bin/activate"
        echo ""
    fi
    
    echo "Reports will be saved in the 'reports/' directory."
}

# Main execution
main() {
    log_info "Starting comprehensive linting setup for mem0ai..."
    
    check_project_root
    check_python_version
    create_directories
    install_dependencies
    setup_precommit
    setup_detect_secrets
    validate_configs
    setup_ide_configs
    run_initial_checks
    
    log_success "Linting setup completed successfully!"
    show_usage
}

# Execute main function
main "$@"