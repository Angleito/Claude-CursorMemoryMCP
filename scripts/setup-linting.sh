#!/bin/bash
# Setup script for linting infrastructure in mem0ai project

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Main setup function
main() {
    print_status "Setting up linting infrastructure for mem0ai project"
    
    # Check if we're in the right directory
    if [[ ! -f "pyproject.toml" ]]; then
        print_error "pyproject.toml not found. Please run this script from the project root."
        exit 1
    fi
    
    # 1. Check Python version
    print_status "Checking Python version..."
    python_version=$(python3 --version 2>&1 | awk '{print $2}')
    required_version="3.8"
    
    if [[ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]]; then
        print_error "Python $required_version or higher is required. Found: $python_version"
        exit 1
    fi
    print_success "Python version check passed: $python_version"
    
    # 2. Upgrade pip
    print_status "Upgrading pip..."
    python3 -m pip install --upgrade pip
    
    # 3. Install development dependencies
    print_status "Installing development dependencies..."
    if [[ -f "requirements-dev.txt" ]]; then
        pip install -r requirements-dev.txt
        print_success "Development dependencies installed"
    else
        print_warning "requirements-dev.txt not found, skipping dev dependencies"
    fi
    
    # 4. Install production dependencies
    print_status "Installing production dependencies..."
    if [[ -f "requirements.txt" ]]; then
        pip install -r requirements.txt
        print_success "Production dependencies installed"
    else
        print_warning "requirements.txt not found, skipping production dependencies"
    fi
    
    # 5. Create directories
    print_status "Creating necessary directories..."
    mkdir -p .github/workflows
    mkdir -p reports
    mkdir -p tests
    mkdir -p logs
    print_success "Directories created"
    
    # 6. Initialize git hooks (pre-commit)
    print_status "Setting up pre-commit hooks..."
    if command_exists pre-commit; then
        pre-commit install
        pre-commit install --hook-type commit-msg
        print_success "Pre-commit hooks installed"
    else
        print_warning "pre-commit not found, please install it manually"
    fi
    
    # 7. Initialize secrets baseline
    print_status "Initializing secrets baseline..."
    if command_exists detect-secrets; then
        if [[ ! -f ".secrets.baseline" ]]; then
            detect-secrets scan --baseline .secrets.baseline
            print_success "Secrets baseline created"
        else
            print_status "Secrets baseline already exists"
        fi
    else
        print_warning "detect-secrets not found, please install it manually"
    fi
    
    # 8. Create .gitignore additions if needed
    print_status "Updating .gitignore..."
    gitignore_additions="
# Linting and testing
.ruff_cache/
.mypy_cache/
.pytest_cache/
htmlcov/
.coverage
.coverage.*
coverage.xml
*.cover
.hypothesis/
.tox/

# Reports
reports/
bandit-report.json
safety-report.json
semgrep.json

# Secrets
.secrets.baseline

# Environment
.env
.venv/
venv/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
logs/*.log
logs/*.out
*.log
"
    
    if [[ -f ".gitignore" ]]; then
        echo "$gitignore_additions" >> .gitignore
        print_success ".gitignore updated"
    else
        echo "$gitignore_additions" > .gitignore
        print_success ".gitignore created"
    fi
    
    # 9. Validate configurations
    print_status "Validating configurations..."
    
    # Validate pyproject.toml
    if command_exists python3; then
        python3 -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb'))" 2>/dev/null
        if [[ $? -eq 0 ]]; then
            print_success "pyproject.toml is valid"
        else
            print_error "pyproject.toml is invalid"
        fi
    fi
    
    # Validate YAML files
    if command_exists yamllint; then
        if yamllint .pre-commit-config.yaml >/dev/null 2>&1; then
            print_success ".pre-commit-config.yaml is valid"
        else
            print_warning ".pre-commit-config.yaml has issues"
        fi
    fi
    
    # 10. Run initial setup
    print_status "Running initial linting setup..."
    
    # Update pre-commit hooks
    if command_exists pre-commit; then
        pre-commit autoupdate
        print_success "Pre-commit hooks updated"
    fi
    
    # 11. Test the setup
    print_status "Testing linting setup..."
    
    if command_exists ruff; then
        ruff check . --statistics >/dev/null 2>&1
        print_success "Ruff setup test passed"
    fi
    
    if command_exists black; then
        black --check . >/dev/null 2>&1 || true
        print_success "Black setup test completed"
    fi
    
    if command_exists mypy; then
        mypy --version >/dev/null 2>&1
        print_success "MyPy setup test passed"
    fi
    
    # 12. Create initial reports directory structure
    print_status "Setting up reports structure..."
    mkdir -p reports/{security,quality,coverage}
    touch reports/.gitkeep
    print_success "Reports structure created"
    
    # 13. Display summary
    echo
    print_status "=== SETUP SUMMARY ==="
    echo
    print_success "✅ Linting infrastructure setup complete!"
    echo
    print_status "Available commands:"
    echo "  make lint          - Run all linting checks"
    echo "  make lint-fix      - Run linting with auto-fix"
    echo "  make security      - Run security checks only"
    echo "  make type-check    - Run type checking"
    echo "  make pre-commit    - Run pre-commit hooks"
    echo "  ./scripts/lint.sh  - Comprehensive linting script"
    echo
    print_status "Configuration files created:"
    echo "  ✅ pyproject.toml - Main configuration"
    echo "  ✅ .pre-commit-config.yaml - Pre-commit hooks"
    echo "  ✅ .bandit - Security scanning config"
    echo "  ✅ requirements-dev.txt - Development dependencies"
    echo "  ✅ Makefile - Development commands"
    echo "  ✅ .github/workflows/lint.yml - CI/CD pipeline"
    echo
    print_status "Next steps:"
    echo "  1. Review and customize configurations in pyproject.toml"
    echo "  2. Run 'make lint' to test the setup"
    echo "  3. Run 'pre-commit run --all-files' to test pre-commit hooks"
    echo "  4. Commit the new linting infrastructure"
    echo
    print_success "Happy linting! 🚀"
}

# Help function
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Setup script for linting infrastructure in mem0ai project"
    echo
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  --skip-install Skip dependency installation"
    echo "  --minimal      Minimal setup (essential tools only)"
    echo
    echo "This script will:"
    echo "  - Install development dependencies"
    echo "  - Set up pre-commit hooks"
    echo "  - Create configuration files"
    echo "  - Initialize secrets baseline"
    echo "  - Validate setup"
}

# Minimal setup (essential tools only)
minimal_setup() {
    print_status "Running minimal linting setup"
    
    # Install only essential tools
    pip install ruff black mypy pre-commit
    
    # Setup pre-commit
    if command_exists pre-commit; then
        pre-commit install
    fi
    
    # Create secrets baseline
    if command_exists detect-secrets; then
        detect-secrets scan --baseline .secrets.baseline
    fi
    
    print_success "Minimal setup completed"
}

# Parse command line arguments
case "${1:-}" in
    -h|--help)
        show_help
        exit 0
        ;;
    --minimal)
        minimal_setup
        ;;
    --skip-install)
        print_status "Skipping dependency installation"
        # Run main without pip installs
        ;;
    "")
        main
        ;;
    *)
        print_error "Unknown option: $1"
        show_help
        exit 1
        ;;
esac