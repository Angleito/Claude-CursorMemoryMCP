#!/bin/bash
set -euo pipefail

# Configuration Validation Script for Mem0AI
# This script validates all configuration files, deployment scripts, and Docker configurations

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Error tracking
ERRORS=0
WARNINGS=0

# Function to log results
log_info() {
    echo -e "${BLUE}[INFO] $1${NC}"
}

log_success() {
    echo -e "${GREEN}[✓] $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}[⚠] $1${NC}"
    ((WARNINGS++))
}

log_error() {
    echo -e "${RED}[✗] $1${NC}"
    ((ERRORS++))
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to validate JSON files
validate_json() {
    local file="$1"
    if command_exists jq; then
        if jq empty "$file" 2>/dev/null; then
            log_success "JSON syntax valid: $file"
        else
            log_error "JSON syntax invalid: $file"
            return 1
        fi
    elif command_exists python3; then
        if python3 -m json.tool "$file" >/dev/null 2>&1; then
            log_success "JSON syntax valid: $file"
        else
            log_error "JSON syntax invalid: $file"
            return 1
        fi
    else
        log_warning "No JSON validator available for: $file"
    fi
}

# Function to validate YAML files
validate_yaml() {
    local file="$1"
    if command_exists yamllint; then
        if yamllint -d relaxed "$file" >/dev/null 2>&1; then
            log_success "YAML syntax valid: $file"
        else
            log_error "YAML syntax invalid: $file"
            yamllint -d relaxed "$file"
            return 1
        fi
    elif command_exists python3; then
        if python3 -c "import yaml; yaml.safe_load(open('$file'))" 2>/dev/null; then
            log_success "YAML syntax valid: $file"
        else
            log_error "YAML syntax invalid: $file"
            return 1
        fi
    else
        log_warning "No YAML validator available for: $file"
    fi
}

# Function to validate shell scripts
validate_shell_script() {
    local file="$1"
    if command_exists shellcheck; then
        if shellcheck "$file" >/dev/null 2>&1; then
            log_success "Shell script syntax valid: $file"
        else
            log_error "Shell script issues found: $file"
            shellcheck "$file"
            return 1
        fi
    elif command_exists bash; then
        if bash -n "$file" 2>/dev/null; then
            log_success "Shell script syntax valid: $file"
        else
            log_error "Shell script syntax invalid: $file"
            return 1
        fi
    else
        log_warning "No shell script validator available for: $file"
    fi
}

# Function to validate Nginx configuration
validate_nginx_config() {
    local file="$1"
    if command_exists nginx; then
        if nginx -t -c "$file" 2>/dev/null; then
            log_success "Nginx configuration valid: $file"
        else
            log_error "Nginx configuration invalid: $file"
            nginx -t -c "$file"
            return 1
        fi
    else
        log_warning "Nginx not available to validate: $file"
    fi
}

# Function to validate Docker Compose files
validate_docker_compose() {
    local file="$1"
    if command_exists docker-compose; then
        if docker-compose -f "$file" config >/dev/null 2>&1; then
            log_success "Docker Compose syntax valid: $file"
        else
            log_error "Docker Compose syntax invalid: $file"
            docker-compose -f "$file" config
            return 1
        fi
    else
        log_warning "Docker Compose not available to validate: $file"
    fi
}

# Function to validate Dockerfile
validate_dockerfile() {
    local file="$1"
    if command_exists hadolint; then
        if hadolint "$file" >/dev/null 2>&1; then
            log_success "Dockerfile best practices validated: $file"
        else
            log_warning "Dockerfile issues found: $file"
            hadolint "$file"
        fi
    else
        log_warning "hadolint not available to validate: $file"
    fi
    
    # Basic syntax check
    if grep -q "^FROM " "$file"; then
        log_success "Dockerfile has valid FROM instruction: $file"
    else
        log_error "Dockerfile missing FROM instruction: $file"
    fi
}

# Function to check environment variables
validate_env_vars() {
    local env_file="$1"
    if [[ -f "$env_file" ]]; then
        log_info "Checking environment variables in: $env_file"
        
        # Check for required variables
        local required_vars=(
            "POSTGRES_PASSWORD"
            "REDIS_PASSWORD"
            "JWT_SECRET"
            "ENCRYPTION_KEY"
        )
        
        for var in "${required_vars[@]}"; do
            if grep -q "^${var}=" "$env_file"; then
                if grep -q "^${var}=your-.*-here" "$env_file"; then
                    log_warning "Environment variable $var still has placeholder value"
                else
                    log_success "Environment variable $var is configured"
                fi
            else
                log_error "Required environment variable $var missing"
            fi
        done
    else
        log_warning "Environment file not found: $env_file"
    fi
}

# Function to check security configurations
validate_security() {
    log_info "Checking security configurations..."
    
    # Check for hardcoded passwords
    if grep -r "password.*=.*admin" "$PROJECT_ROOT" --exclude-dir=.git --exclude="*.log" 2>/dev/null; then
        log_error "Hardcoded admin passwords found"
    else
        log_success "No hardcoded admin passwords found"
    fi
    
    # Check for debug settings in production
    if grep -r "DEBUG.*=.*true" "$PROJECT_ROOT" --exclude-dir=.git --exclude="*.log" 2>/dev/null; then
        log_warning "Debug mode enabled (ensure this is intentional for production)"
    else
        log_success "Debug mode appropriately configured"
    fi
    
    # Check CORS configuration
    if grep -r '"origins": \["\*"\]' "$PROJECT_ROOT" --exclude-dir=.git 2>/dev/null; then
        log_warning "CORS allows all origins (potential security risk)"
    else
        log_success "CORS appropriately configured"
    fi
}

# Function to check file permissions
validate_permissions() {
    log_info "Checking file permissions..."
    
    # Check script permissions
    find "$PROJECT_ROOT/scripts" -name "*.sh" -type f | while read -r script; do
        if [[ -x "$script" ]]; then
            log_success "Script is executable: $script"
        else
            log_warning "Script not executable: $script"
        fi
    done
    
    # Check sensitive file permissions
    if [[ -f "$PROJECT_ROOT/.env" ]]; then
        if [[ $(stat -c "%a" "$PROJECT_ROOT/.env") == "600" ]]; then
            log_success ".env file has secure permissions (600)"
        else
            log_warning ".env file permissions should be 600"
        fi
    fi
}

# Main validation function
main() {
    echo -e "${GREEN}🔍 Mem0AI Configuration Validation${NC}"
    echo "===================================="
    echo ""
    
    cd "$PROJECT_ROOT"
    
    # Validate Docker configurations
    log_info "Validating Docker configurations..."
    find . -name "Dockerfile*" -type f | while read -r dockerfile; do
        validate_dockerfile "$dockerfile"
    done
    
    find . -name "docker-compose*.yml" -type f | while read -r compose_file; do
        validate_docker_compose "$compose_file"
    done
    
    # Validate JSON files
    log_info "Validating JSON configurations..."
    find . -name "*.json" -type f | while read -r json_file; do
        validate_json "$json_file"
    done
    
    # Validate YAML files
    log_info "Validating YAML configurations..."
    find . -name "*.yml" -o -name "*.yaml" -type f | while read -r yaml_file; do
        validate_yaml "$yaml_file"
    done
    
    # Validate shell scripts
    log_info "Validating shell scripts..."
    find . -name "*.sh" -type f | while read -r script; do
        validate_shell_script "$script"
    done
    
    # Validate Nginx configurations
    log_info "Validating Nginx configurations..."
    find . -path "*/nginx*" -name "*.conf" -type f | while read -r nginx_conf; do
        validate_nginx_config "$nginx_conf"
    done
    
    # Validate environment configuration
    validate_env_vars ".env"
    validate_env_vars ".env.example"
    
    # Validate security configurations
    validate_security
    
    # Validate file permissions
    validate_permissions
    
    # Summary
    echo ""
    echo "===================================="
    echo -e "${GREEN}Validation Summary${NC}"
    echo "===================================="
    
    if [[ $ERRORS -eq 0 && $WARNINGS -eq 0 ]]; then
        echo -e "${GREEN}✅ All configurations are valid!${NC}"
        exit 0
    elif [[ $ERRORS -eq 0 ]]; then
        echo -e "${YELLOW}⚠️  Validation completed with $WARNINGS warnings${NC}"
        exit 0
    else
        echo -e "${RED}❌ Validation failed with $ERRORS errors and $WARNINGS warnings${NC}"
        exit 1
    fi
}

# Install validation tools if running with --install-tools
if [[ "${1:-}" == "--install-tools" ]]; then
    log_info "Installing validation tools..."
    
    # Install common tools
    if command_exists apt-get; then
        sudo apt-get update
        sudo apt-get install -y jq yamllint shellcheck nginx
    elif command_exists yum; then
        sudo yum install -y jq yamllint ShellCheck nginx
    elif command_exists brew; then
        brew install jq yamllint shellcheck nginx hadolint
    else
        log_warning "Package manager not supported for automatic tool installation"
    fi
    
    # Install hadolint for Dockerfile validation
    if ! command_exists hadolint; then
        log_info "Installing hadolint..."
        curl -sL -o hadolint "https://github.com/hadolint/hadolint/releases/latest/download/hadolint-$(uname -s)-$(uname -m)"
        chmod +x hadolint
        sudo mv hadolint /usr/local/bin/
    fi
    
    log_success "Validation tools installed"
    echo ""
fi

# Run main validation
main "$@"