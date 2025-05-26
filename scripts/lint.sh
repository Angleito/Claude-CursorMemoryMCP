#!/bin/bash
# Comprehensive linting script for mem0ai project

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

# Function to run a linting tool
run_lint_tool() {
    local tool_name="$1"
    local command="$2"
    local description="$3"
    
    print_status "Running $tool_name - $description"
    
    if command_exists "${tool_name%% *}"; then
        if eval "$command"; then
            print_success "$tool_name completed successfully"
        else
            print_error "$tool_name found issues"
            return 1
        fi
    else
        print_warning "$tool_name not found, skipping..."
    fi
}

# Main linting function
main() {
    print_status "Starting comprehensive linting for mem0ai project"
    
    # Check if we're in the right directory
    if [[ ! -f "pyproject.toml" ]]; then
        print_error "pyproject.toml not found. Please run this script from the project root."
        exit 1
    fi
    
    # Track overall success
    overall_success=true
    
    # 1. Ruff (fast linting and formatting)
    if ! run_lint_tool "ruff" "ruff check . --fix" "Fast Python linting with auto-fix"; then
        overall_success=false
    fi
    
    if ! run_lint_tool "ruff" "ruff format ." "Code formatting"; then
        overall_success=false
    fi
    
    # 2. Black (code formatting backup)
    if ! run_lint_tool "black" "black --check ." "Code formatting check"; then
        overall_success=false
    fi
    
    # 3. isort (import sorting)
    if ! run_lint_tool "isort" "isort --check-only --diff ." "Import sorting check"; then
        overall_success=false
    fi
    
    # 4. MyPy (type checking)
    if ! run_lint_tool "mypy" "mypy src/ auth/ security/ monitoring/ --ignore-missing-imports" "Static type checking"; then
        overall_success=false
    fi
    
    # 5. Bandit (security linting)
    if ! run_lint_tool "bandit" "bandit -r . -f txt" "Security vulnerability scanning"; then
        overall_success=false
    fi
    
    # 6. Safety (dependency vulnerabilities)
    if ! run_lint_tool "safety" "safety check" "Dependency vulnerability check"; then
        overall_success=false
    fi
    
    # 7. Vulture (dead code detection)
    if ! run_lint_tool "vulture" "vulture src/ auth/ security/ monitoring/ --min-confidence 80" "Dead code detection"; then
        overall_success=false
    fi
    
    # 8. Radon (complexity analysis)
    if ! run_lint_tool "radon" "radon cc src/ auth/ security/ monitoring/ -a" "Cyclomatic complexity analysis"; then
        overall_success=false
    fi
    
    # 9. YAML linting
    if ! run_lint_tool "yamllint" "yamllint ." "YAML file linting"; then
        overall_success=false
    fi
    
    # 10. SQL linting
    if ! run_lint_tool "sqlfluff" "sqlfluff lint --dialect postgres *.sql" "SQL file linting"; then
        overall_success=false
    fi
    
    # 11. Dockerfile linting
    if ! run_lint_tool "hadolint" "hadolint Dockerfile" "Dockerfile linting"; then
        overall_success=false
    fi
    
    if [[ -f "app/Dockerfile" ]]; then
        if ! run_lint_tool "hadolint" "hadolint app/Dockerfile" "App Dockerfile linting"; then
            overall_success=false
        fi
    fi
    
    # 12. Shell script linting
    if ! run_lint_tool "shellcheck" "find . -name '*.sh' -exec shellcheck {} +" "Shell script linting"; then
        overall_success=false
    fi
    
    # 13. Secret detection
    if ! run_lint_tool "detect-secrets" "detect-secrets scan --baseline .secrets.baseline" "Secret detection"; then
        overall_success=false
    fi
    
    # Summary
    echo
    print_status "Linting summary:"
    
    if $overall_success; then
        print_success "All linting checks passed! 🎉"
        exit 0
    else
        print_error "Some linting checks failed. Please review the output above."
        exit 1
    fi
}

# Help function
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Comprehensive linting script for mem0ai project"
    echo
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -q, --quick    Run only fast linting tools"
    echo "  -f, --fix      Run linting with auto-fix where possible"
    echo "  --security     Run only security-related checks"
    echo "  --format       Run only formatting tools"
    echo "  --type         Run only type checking"
    echo
    echo "Examples:"
    echo "  $0                 # Run all linting checks"
    echo "  $0 --quick         # Run fast checks only"
    echo "  $0 --fix           # Run with auto-fix"
    echo "  $0 --security      # Security checks only"
}

# Quick linting (fast tools only)
quick_lint() {
    print_status "Running quick linting checks"
    
    overall_success=true
    
    if ! run_lint_tool "ruff" "ruff check . --fix" "Fast linting"; then
        overall_success=false
    fi
    
    if ! run_lint_tool "ruff" "ruff format ." "Fast formatting"; then
        overall_success=false
    fi
    
    if $overall_success; then
        print_success "Quick linting completed successfully! ⚡"
    else
        print_error "Quick linting found issues."
        exit 1
    fi
}

# Security-only checks
security_lint() {
    print_status "Running security-focused linting"
    
    overall_success=true
    
    if ! run_lint_tool "bandit" "bandit -r . -f txt" "Security scanning"; then
        overall_success=false
    fi
    
    if ! run_lint_tool "safety" "safety check" "Dependency vulnerabilities"; then
        overall_success=false
    fi
    
    if ! run_lint_tool "detect-secrets" "detect-secrets scan --baseline .secrets.baseline" "Secret detection"; then
        overall_success=false
    fi
    
    if $overall_success; then
        print_success "Security linting completed successfully! 🔒"
    else
        print_error "Security issues found."
        exit 1
    fi
}

# Format-only checks
format_lint() {
    print_status "Running formatting tools"
    
    overall_success=true
    
    if ! run_lint_tool "ruff" "ruff format ." "Ruff formatting"; then
        overall_success=false
    fi
    
    if ! run_lint_tool "black" "black ." "Black formatting"; then
        overall_success=false
    fi
    
    if ! run_lint_tool "isort" "isort ." "Import sorting"; then
        overall_success=false
    fi
    
    if $overall_success; then
        print_success "Code formatting completed successfully! ✨"
    else
        print_error "Formatting issues found."
        exit 1
    fi
}

# Type checking only
type_lint() {
    print_status "Running type checking"
    
    if ! run_lint_tool "mypy" "mypy src/ auth/ security/ monitoring/ --ignore-missing-imports" "Type checking"; then
        print_error "Type checking found issues."
        exit 1
    else
        print_success "Type checking completed successfully! 🔍"
    fi
}

# Parse command line arguments
case "${1:-}" in
    -h|--help)
        show_help
        exit 0
        ;;
    -q|--quick)
        quick_lint
        ;;
    --security)
        security_lint
        ;;
    --format|-f|--fix)
        format_lint
        ;;
    --type)
        type_lint
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