#!/bin/bash
set -euo pipefail

# Generate secure secrets for Mem0AI deployment
# This script creates a .env file with randomly generated secrets
# shellcheck disable=SC1091

# Script metadata
readonly SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration
readonly ENV_FILE="${PROJECT_ROOT}/.env"
readonly ENV_EXAMPLE="${PROJECT_ROOT}/.env.example"
readonly ENV_BACKUP="${ENV_FILE}.backup.$(date +%Y%m%d_%H%M%S)"

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Default options
FORCE_OVERWRITE=false
QUIET_MODE=false

# Show usage information
show_usage() {
    cat << EOF
Usage: $0 [options]

Options:
  -f, --force        Force overwrite existing .env file without prompt
  -q, --quiet        Suppress non-essential output
  -h, --help         Show this help message
  
This script generates cryptographically secure secrets for Mem0AI deployment.
It creates a .env file with randomly generated passwords and keys.

Security features:
  - Uses /dev/urandom when available for better entropy
  - Generates high-entropy secrets with appropriate lengths
  - Validates all generated secrets
  - Sets secure file permissions (600)
  - Creates backup of existing .env file
  - Comprehensive error handling and rollback

Generated secrets:
  - PostgreSQL password (32 chars)
  - Redis password (32 chars) 
  - JWT secret (base64, 512 bits)
  - Encryption key (base64, 256 bits)
  - Backup encryption key (base64, 256 bits)
  - Grafana admin password (20 chars)
  - Session secret (base64, 256 bits)
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--force)
            FORCE_OVERWRITE=true
            shift
            ;;
        -q|--quiet)
            QUIET_MODE=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo -e "${RED}[ERROR]${NC} Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Logging functions
log_info() { 
    if [[ "$QUIET_MODE" != "true" ]]; then
        echo -e "${GREEN}[INFO]${NC} $*"
    fi
}
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }
log_debug() { 
    if [[ "$QUIET_MODE" != "true" ]]; then
        echo -e "${BLUE}[DEBUG]${NC} $*"
    fi
}

# Error handling
handle_error() {
    local exit_code=$?
    local line_no=$1
    log_error "Script failed at line $line_no with exit code $exit_code"
    
    # Restore backup if it exists
    if [[ -f "$ENV_BACKUP" ]] && [[ -f "$ENV_FILE" ]]; then
        log_warn "Restoring original .env file..."
        mv "$ENV_BACKUP" "$ENV_FILE"
    fi
    
    exit $exit_code
}

# Set error trap
trap 'handle_error ${LINENO}' ERR

# Validate system requirements
validate_requirements() {
    log_debug "Validating system requirements..."
    
    # Check for required commands
    local required_commands=("openssl" "date" "head" "tr")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            log_error "Required command '$cmd' is not available"
            exit 1
        fi
    done
    
    # Check OpenSSL version and capabilities
    if ! openssl version &> /dev/null; then
        log_error "OpenSSL is not functioning properly"
        exit 1
    fi
    
    # Test random number generation
    if ! openssl rand -hex 1 &> /dev/null; then
        log_error "OpenSSL random number generation is not working"
        exit 1
    fi
    
    log_debug "System requirements validated"
}

# Function to get user confirmation
get_confirmation() {
    local prompt="$1"
    local default="${2:-N}"
    local response
    
    if [[ "$FORCE_OVERWRITE" == "true" ]]; then
        return 0
    fi
    
    while true; do
        read -p "$prompt [${default}]: " -r response
        response=${response:-$default}
        
        case "$response" in
            [Yy]|[Yy][Ee][Ss])
                return 0
                ;;
            [Nn]|[Nn][Oo])
                return 1
                ;;
            *)
                log_warn "Please answer yes or no"
                ;;
        esac
    done
}

# Validate entropy before generating secrets
validate_entropy() {
    # Check available entropy (Linux only)
    if [[ -r /proc/sys/kernel/random/entropy_avail ]]; then
        local entropy
        entropy=$(cat /proc/sys/kernel/random/entropy_avail)
        if [[ $entropy -lt 1000 ]]; then
            log_warn "Low system entropy ($entropy). Secrets may take longer to generate."
            log_warn "Consider installing haveged or rng-tools for better entropy."
        else
            log_debug "System entropy is adequate ($entropy)"
        fi
    fi
}

# Function to generate cryptographically secure random password
generate_password() {
    local length=${1:-32}
    local min_length=8
    local max_length=128
    
    # Validate length parameter
    if [[ ! $length =~ ^[0-9]+$ ]] || [[ $length -lt $min_length ]] || [[ $length -gt $max_length ]]; then
        log_error "Invalid password length: $length (must be between $min_length and $max_length)"
        return 1
    fi
    
    # Generate password with mixed character set for better security
    # Use /dev/urandom for better randomness if available
    if [[ -c /dev/urandom ]]; then
        # Use urandom with base64 encoding and character filtering
        local password
        while true; do
            password=$(head -c $((length * 2)) /dev/urandom | base64 | tr -d "=+/\n" | head -c "$length")
            
            # Ensure we have the exact length
            if [[ ${#password} -eq $length ]]; then
                echo "$password"
                return 0
            fi
        done
    else
        # Fallback to openssl
        local password
        while true; do
            password=$(openssl rand -base64 $((length * 2)) | tr -d "=+/\n" | head -c "$length")
            
            if [[ ${#password} -eq $length ]]; then
                echo "$password"
                return 0
            fi
        done
    fi
}

# Function to generate JWT secret (base64 encoded, URL-safe)
generate_jwt_secret() {
    local jwt_bytes=64  # 512 bits
    
    # Generate high-entropy JWT secret
    if [[ -c /dev/urandom ]]; then
        head -c $jwt_bytes /dev/urandom | base64 | tr -d "\n"
    else
        openssl rand -base64 $jwt_bytes | tr -d "\n"
    fi
}

# Function to generate encryption key (32 bytes = 256 bits, base64 encoded)
generate_encryption_key() {
    local key_bytes=32  # 256 bits
    
    # Generate cryptographically secure encryption key
    if [[ -c /dev/urandom ]]; then
        head -c $key_bytes /dev/urandom | base64 | tr -d "\n"
    else
        openssl rand -base64 $key_bytes | tr -d "\n"
    fi
}

# Function to generate secure API key
generate_api_key() {
    local prefix="${1:-sk}"
    local entropy_bytes=32
    
    local random_part
    if [[ -c /dev/urandom ]]; then
        random_part=$(head -c $entropy_bytes /dev/urandom | base64 | tr -d "=+/\n")
    else
        random_part=$(openssl rand -base64 $entropy_bytes | tr -d "=+/\n")
    fi
    
    echo "${prefix}_$(echo "$random_part" | head -c 48)"
}

# Function to validate generated secret
validate_secret() {
    local secret="$1"
    local min_length="${2:-16}"
    local name="${3:-secret}"
    
    if [[ -z "$secret" ]]; then
        log_error "Generated $name is empty"
        return 1
    fi
    
    if [[ ${#secret} -lt $min_length ]]; then
        log_error "Generated $name is too short: ${#secret} < $min_length"
        return 1
    fi
    
    # Check for obvious patterns (repeated characters)
    if [[ "$secret" =~ ^(..)\\1{8,}$ ]]; then
        log_error "Generated $name contains suspicious patterns"
        return 1
    fi
    
    return 0
}

# Function to safely replace secrets in .env file
replace_secret() {
    local placeholder="$1"
    local secret="$2"
    local description="$3"
    
    log_debug "Replacing $description..."
    
    # Escape special characters for sed
    local escaped_secret
    escaped_secret=$(printf '%s\n' "$secret" | sed 's/[[$.*^$()+?{|]/\\&/g')
    
    # Replace placeholder with secret
    if ! sed -i.tmp "s|$placeholder|$escaped_secret|g" "$ENV_FILE"; then
        log_error "Failed to replace $description in .env file"
        return 1
    fi
    
    # Verify replacement was successful
    if grep -q "$placeholder" "$ENV_FILE"; then
        log_warn "Placeholder '$placeholder' still exists in .env file"
    fi
    
    return 0
}

# Validate final .env file
validate_env_file() {
    log_debug "Validating generated .env file..."
    
    # Check if file exists and is readable
    if [[ ! -r "$ENV_FILE" ]]; then
        log_error "Generated .env file is not readable"
        return 1
    fi
    
    # Check for remaining placeholders
    local remaining_placeholders
    remaining_placeholders=$(grep -E "(your-.*-here|CHANGE_ME_)" "$ENV_FILE" || true)
    
    if [[ -n "$remaining_placeholders" ]]; then
        log_warn "Some placeholders may not have been replaced:"
        echo "$remaining_placeholders" | while IFS= read -r line; do
            log_warn "  $line"
        done
    fi
    
    # Check for empty critical values
    local critical_vars=("POSTGRES_PASSWORD" "JWT_SECRET" "ENCRYPTION_KEY")
    for var in "${critical_vars[@]}"; do
        if ! grep -q "^${var}=.\\+" "$ENV_FILE"; then
            log_error "Critical variable $var is empty or missing"
            return 1
        fi
    done
    
    log_debug ".env file validation passed"
    return 0
}

# Set secure permissions on .env file
secure_env_file() {
    log_info "🔒 Setting secure permissions on .env file..."
    
    # Set restrictive permissions (readable/writable by owner only)
    if ! chmod 600 "$ENV_FILE"; then
        log_error "Failed to set secure permissions on .env file"
        return 1
    fi
    
    # Verify permissions
    local perms
    perms=$(stat -c "%a" "$ENV_FILE" 2>/dev/null || stat -f "%A" "$ENV_FILE" 2>/dev/null || echo "unknown")
    
    if [[ "$perms" == "600" ]]; then
        log_debug "File permissions set to 600 (secure)"
    else
        log_warn "File permissions are $perms (expected 600)"
    fi
    
    return 0
}

# Main execution
main() {
    log_info "🔐 Mem0AI Secrets Generator"
    echo "=============================="

    # Validate requirements first
    validate_requirements

    # Check if .env already exists
    if [[ -f "$ENV_FILE" ]]; then
        log_warn "⚠️  .env file already exists!"
        
        # Create backup automatically
        cp "$ENV_FILE" "$ENV_BACKUP"
        log_info "Created backup: $(basename "$ENV_BACKUP")"
        
        if ! get_confirmation "Do you want to overwrite it?"; then
            log_info "Operation cancelled."
            rm -f "$ENV_BACKUP"
            exit 0
        fi
    fi

    # Copy from example with validation
    if [[ ! -f "$ENV_EXAMPLE" ]]; then
        log_error "❌ .env.example not found at: $ENV_EXAMPLE"
        exit 1
    fi

    # Validate .env.example file
    if [[ ! -r "$ENV_EXAMPLE" ]]; then
        log_error "Cannot read .env.example file"
        exit 1
    fi

    # Check if .env.example is not empty
    if [[ ! -s "$ENV_EXAMPLE" ]]; then
        log_error ".env.example file is empty"
        exit 1
    fi

    # Copy example to .env
    if ! cp "$ENV_EXAMPLE" "$ENV_FILE"; then
        log_error "Failed to copy .env.example to .env"
        exit 1
    fi

    log_debug "Copied .env.example to .env"

    log_info "📝 Generating secure secrets..."

    validate_entropy

    # Generate secrets with validation
    log_debug "Generating PostgreSQL password..."
    POSTGRES_PASSWORD=$(generate_password 32)
    validate_secret "$POSTGRES_PASSWORD" 24 "PostgreSQL password"

    log_debug "Generating Redis password..."
    REDIS_PASSWORD=$(generate_password 32)
    validate_secret "$REDIS_PASSWORD" 24 "Redis password"

    log_debug "Generating JWT secret..."
    JWT_SECRET=$(generate_jwt_secret)
    validate_secret "$JWT_SECRET" 64 "JWT secret"

    log_debug "Generating encryption key..."
    ENCRYPTION_KEY=$(generate_encryption_key)
    validate_secret "$ENCRYPTION_KEY" 32 "encryption key"

    log_debug "Generating backup encryption key..."
    BACKUP_ENCRYPTION_KEY=$(generate_encryption_key)
    validate_secret "$BACKUP_ENCRYPTION_KEY" 32 "backup encryption key"

    log_debug "Generating Grafana password..."
    GRAFANA_PASSWORD=$(generate_password 20)
    validate_secret "$GRAFANA_PASSWORD" 16 "Grafana password"

    log_debug "Generating session secret..."
    SESSION_SECRET=$(generate_encryption_key)
    validate_secret "$SESSION_SECRET" 32 "session secret"

    # Replace placeholders in .env file with proper error handling
    log_info "📝 Updating .env file with generated secrets..."

    # Common placeholder patterns
    replace_secret "your-strong-postgres-password-here" "$POSTGRES_PASSWORD" "PostgreSQL password"
    replace_secret "your-strong-redis-password-here" "$REDIS_PASSWORD" "Redis password"
    replace_secret "your-jwt-secret-key-here-at-least-32-characters-long" "$JWT_SECRET" "JWT secret"
    replace_secret "your-encryption-key-here-32-bytes-long" "$ENCRYPTION_KEY" "encryption key"
    replace_secret "your-backup-encryption-key-here" "$BACKUP_ENCRYPTION_KEY" "backup encryption key"
    replace_secret "your-grafana-admin-password-here" "$GRAFANA_PASSWORD" "Grafana password"
    replace_secret "your-session-secret-here" "$SESSION_SECRET" "session secret"

    # Alternative placeholder patterns (fallback)
    replace_secret "CHANGE_ME_POSTGRES_PASSWORD" "$POSTGRES_PASSWORD" "PostgreSQL password (alt)"
    replace_secret "CHANGE_ME_REDIS_PASSWORD" "$REDIS_PASSWORD" "Redis password (alt)"
    replace_secret "CHANGE_ME_JWT_SECRET" "$JWT_SECRET" "JWT secret (alt)"
    replace_secret "CHANGE_ME_ENCRYPTION_KEY" "$ENCRYPTION_KEY" "encryption key (alt)"
    replace_secret "CHANGE_ME_BACKUP_KEY" "$BACKUP_ENCRYPTION_KEY" "backup encryption key (alt)"
    replace_secret "CHANGE_ME_GRAFANA_PASSWORD" "$GRAFANA_PASSWORD" "Grafana password (alt)"

    # Remove temporary backup file
    if [[ -f "${ENV_FILE}.tmp" ]]; then
        rm "${ENV_FILE}.tmp"
    fi

    # Validate and secure the .env file
    validate_env_file
    secure_env_file

    # Clean up backup if everything was successful
    if [[ -f "$ENV_BACKUP" ]]; then
        rm -f "$ENV_BACKUP"
        log_debug "Removed backup file"
    fi

    log_info "✅ Secrets generated successfully!"
    echo ""
    log_info "📋 Next steps:"
    echo "1. Edit .env file and configure:"
    echo "   - DOMAIN (your actual domain)"
    echo "   - ADMIN_EMAIL (your email)"
    echo "   - SSL_EMAIL (SSL certificate email)"
    echo "   - OPENAI_API_KEY (your OpenAI API key)"
    echo "   - CORS_ORIGIN (allowed origins)"
    echo "   - AWS credentials (if using S3 backups)"
    echo ""
    echo "2. The .env file has been automatically secured with 600 permissions"
    echo ""
    log_info "🔑 Generated credential summary:"
    echo "   - PostgreSQL Password: ${#POSTGRES_PASSWORD} characters"
    echo "   - Redis Password: ${#REDIS_PASSWORD} characters"
    echo "   - JWT Secret: ${#JWT_SECRET} characters (base64)"
    echo "   - Encryption Key: ${#ENCRYPTION_KEY} characters (base64)"
    echo "   - Backup Encryption Key: ${#BACKUP_ENCRYPTION_KEY} characters (base64)"
    echo "   - Grafana Password: ${#GRAFANA_PASSWORD} characters"
    echo "   - Session Secret: ${#SESSION_SECRET} characters (base64)"
    echo ""
    log_warn "⚠️  Keep these credentials secure!"
    echo "   - Store them in a password manager"
    echo "   - Never commit .env to version control"
    echo "   - Backup your .env file securely"
    echo "   - Rotate secrets regularly"

    # Final security reminder
    echo ""
    log_info "🛡️  Security reminders:"
    echo "   - .env file location: $ENV_FILE"
    echo "   - File permissions: 600 (owner read/write only)"
    echo "   - Generated secrets use cryptographically secure random sources"
    echo "   - All secrets are unique and high-entropy"
    echo ""
    log_info "🎉 Secret generation completed successfully!"
}

# Run main function
main "$@"