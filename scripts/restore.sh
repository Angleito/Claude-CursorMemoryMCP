#!/bin/bash
set -euo pipefail

# Mem0AI Restore Script
# Restores from encrypted backups with comprehensive safety checks and rollback capabilities
# shellcheck disable=SC1091

# Script metadata
readonly SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Load environment variables
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    # shellcheck source=/dev/null
    source "${PROJECT_ROOT}/.env"
else
    echo "❌ .env file not found in ${PROJECT_ROOT}!"
    exit 1
fi

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

# Configuration
readonly BACKUP_DIR="${PROJECT_ROOT}/backups"
readonly RESTORE_LOG="${BACKUP_DIR}/restore_$(date +%Y%m%d_%H%M%S).log"
readonly SAFETY_BACKUP_PREFIX="safety_backup_$(date +%Y%m%d_%H%M%S)"
readonly RESTORE_TIMEOUT=${RESTORE_TIMEOUT:-7200}  # 2 hours default
readonly MAX_PARALLEL_RESTORES=1

# Global variables
RESTORE_IN_PROGRESS=""
SAFETY_BACKUP_CREATED=""
ORIGINAL_SERVICES_STATE=""

# Logging functions
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "[$timestamp] [$level] $message" | tee -a "$RESTORE_LOG"
}

log_info() { log "INFO" "$@"; }
log_warn() { log "WARN" "$@"; }
log_error() { log "ERROR" "$@"; }
log_debug() { log "DEBUG" "$@"; }

# Error handling and cleanup
handle_error() {
    local exit_code=$?
    local line_no=$1
    log_error "Restore failed at line $line_no with exit code $exit_code"
    
    if [[ -n "$RESTORE_IN_PROGRESS" ]]; then
        log_error "Initiating emergency rollback..."
        emergency_rollback
    fi
    
    cleanup_restore_workspace
    exit $exit_code
}

# Set error trap
trap 'handle_error ${LINENO}' ERR

# Cleanup function
cleanup_restore_workspace() {
    log_info "Cleaning up restore workspace..."
    
    # Remove temporary files
    if [[ -n "${RESTORE_DIR:-}" ]] && [[ -d "$RESTORE_DIR" ]]; then
        rm -rf "$RESTORE_DIR"
        log_debug "Removed restore workspace: $RESTORE_DIR"
    fi
    
    # Clear global state
    RESTORE_IN_PROGRESS=""
}

# Signal handler for graceful shutdown
signal_handler() {
    local signal=$1
    log_warn "Received signal $signal, initiating graceful shutdown..."
    
    if [[ -n "$RESTORE_IN_PROGRESS" ]]; then
        log_warn "Restore in progress, performing rollback..."
        emergency_rollback
    fi
    
    cleanup_restore_workspace
    exit 130
}

# Setup signal handlers
trap 'signal_handler SIGINT' SIGINT
trap 'signal_handler SIGTERM' SIGTERM

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [options]

Options:
  -f, --file BACKUP_FILE    Specify backup file to restore
  -l, --list               List available backups
  -v, --verify BACKUP_FILE Verify backup integrity without restoring
  -r, --rollback           Rollback to last safety backup
  -s, --skip-safety        Skip safety backup creation (dangerous!)
  -y, --yes                Skip confirmation prompts (automation mode)
  -h, --help               Show this help message
  
Safety Features:
  - Automatic safety backup creation before restore
  - Service state preservation and restoration
  - Integrity verification of backup files
  - Automatic rollback on failure
  - Progress monitoring and logging

Examples:
  $0 -l                                   # List backups
  $0 -f mem0ai_backup_20231201_120000     # Restore specific backup
  $0 -v mem0ai_backup_20231201_120000     # Verify backup integrity
  $0 -r                                   # Rollback to safety backup
EOF
}

# Check if another restore is running
check_restore_lock() {
    local lock_file="/tmp/mem0ai_restore.lock"
    
    if [[ -f "$lock_file" ]]; then
        local lock_pid
        lock_pid=$(cat "$lock_file" 2>/dev/null || echo "")
        
        if [[ -n "$lock_pid" ]] && kill -0 "$lock_pid" 2>/dev/null; then
            log_error "Another restore process is already running (PID: $lock_pid)"
            exit 1
        else
            log_warn "Stale lock file found, removing..."
            rm -f "$lock_file"
        fi
    fi
    
    # Create lock file
    echo $$ > "$lock_file"
    
    # Remove lock on exit
    trap "rm -f '$lock_file'" EXIT
}

# Validate system state before restore
validate_system_state() {
    log_info "Validating system state..."
    
    # Check if running as correct user
    if [[ $EUID -eq 0 ]]; then
        log_warn "Running as root - this may cause permission issues"
    fi
    
    # Check Docker and docker-compose availability
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        return 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "docker-compose is not installed or not in PATH"
        return 1
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running or accessible"
        return 1
    fi
    
    # Check available disk space (need at least 5GB)
    local available_kb
    available_kb=$(df "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
    local available_gb=$((available_kb / 1024 / 1024))
    
    if [[ $available_gb -lt 5 ]]; then
        log_error "Insufficient disk space. Available: ${available_gb}GB, Required: 5GB minimum"
        return 1
    fi
    
    log_info "System validation passed. Available space: ${available_gb}GB"
    return 0
}

# Get current services state
get_services_state() {
    log_debug "Capturing current services state..."
    
    local services_state=""
    
    # Get docker-compose services
    if [[ -f "${PROJECT_ROOT}/docker-compose.yml" ]]; then
        cd "$PROJECT_ROOT"
        services_state=$(docker-compose ps --format json 2>/dev/null || echo "[]")
    fi
    
    echo "$services_state"
}

# Stop all services gracefully
stop_services() {
    log_info "🛑 Stopping services gracefully..."
    
    # Save current state
    ORIGINAL_SERVICES_STATE=$(get_services_state)
    
    cd "$PROJECT_ROOT"
    
    # Stop services with timeout
    if timeout 120 docker-compose down; then
        log_info "Services stopped successfully"
        
        # Wait for containers to fully stop
        local wait_count=0
        while [[ $wait_count -lt 30 ]]; do
            if ! docker-compose ps -q | grep -q .; then
                log_debug "All containers stopped"
                break
            fi
            sleep 2
            ((wait_count++))
        done
        
        if [[ $wait_count -eq 30 ]]; then
            log_warn "Some containers may still be running"
        fi
        
        return 0
    else
        log_error "Failed to stop services gracefully"
        return 1
    fi
}

# Start services with health checks
start_services() {
    log_info "🚀 Starting services..."
    
    cd "$PROJECT_ROOT"
    
    # Start services
    if docker-compose up -d; then
        log_info "Services startup initiated"
        
        # Wait for services to be healthy
        wait_for_services_healthy
        return $?
    else
        log_error "Failed to start services"
        return 1
    fi
}

# Wait for services to be healthy
wait_for_services_healthy() {
    log_info "⏳ Waiting for services to be healthy..."
    
    local max_wait=300  # 5 minutes
    local wait_count=0
    
    while [[ $wait_count -lt $max_wait ]]; do
        local all_healthy=true
        
        # Check if application endpoint is responding
        if curl -sf "http://localhost/health" > /dev/null 2>&1; then
            log_info "Application health check passed"
        else
            all_healthy=false
        fi
        
        # Check database connectivity
        if docker-compose exec -T postgres pg_isready -U "${POSTGRES_USER}" > /dev/null 2>&1; then
            log_debug "PostgreSQL health check passed"
        else
            all_healthy=false
        fi
        
        if $all_healthy; then
            log_info "✅ All services are healthy"
            return 0
        fi
        
        sleep 5
        ((wait_count += 5))
        
        if [[ $((wait_count % 30)) -eq 0 ]]; then
            log_info "Still waiting for services... (${wait_count}s/${max_wait}s)"
        fi
    done
    
    log_warn "⚠️  Services health check timeout after ${max_wait}s"
    return 1
}

# List available backups with detailed information
list_backups() {
    echo -e "${GREEN}📋 Available backups:${NC}"
    echo "===================="
    
    if [[ ! -d "$BACKUP_DIR" ]]; then
        echo "No backup directory found."
        return 0
    fi
    
    local backup_count=0
    
    cd "$BACKUP_DIR"
    for backup in mem0ai_backup_*.tar.gz*; do
        if [[ -f "$backup" ]]; then
            local size
            size=$(du -h "$backup" | cut -f1)
            local date_part
            date_part=$(echo "$backup" | sed 's/mem0ai_backup_\([0-9]\{8\}_[0-9]\{6\}\).*/\1/' | sed 's/_/ /')
            local creation_time
            creation_time=$(stat -c %y "$backup" 2>/dev/null | cut -d' ' -f1-2 || echo "unknown")
            
            # Check if backup is encrypted
            local encryption_status="🔓 Unencrypted"
            if [[ "$backup" == *.enc ]]; then
                encryption_status="🔐 Encrypted"
            fi
            
            echo "  📦 $backup"
            echo "      Size: $size"
            echo "      Date: $date_part"
            echo "      Created: $creation_time"
            echo "      Status: $encryption_status"
            echo ""
            
            ((backup_count++))
        fi
    done
    
    if [[ $backup_count -eq 0 ]]; then
        echo "No backup files found."
    else
        echo "Total backups found: $backup_count"
    fi
    
    cd - > /dev/null
}

# Verify backup integrity
verify_backup() {
    local backup_file="$1"
    local full_path="${BACKUP_DIR}/${backup_file}"
    
    log_info "🔍 Verifying backup integrity: $backup_file"
    
    # Add extension if not provided
    if [[ ! -f "$full_path" ]]; then
        if [[ -f "${full_path}.tar.gz.enc" ]]; then
            full_path="${full_path}.tar.gz.enc"
            backup_file="${backup_file}.tar.gz.enc"
        elif [[ -f "${full_path}.tar.gz" ]]; then
            full_path="${full_path}.tar.gz"
            backup_file="${backup_file}.tar.gz"
        else
            log_error "Backup file not found: $backup_file"
            return 1
        fi
    fi
    
    # Check file exists and is readable
    if [[ ! -r "$full_path" ]]; then
        log_error "Backup file is not readable: $full_path"
        return 1
    fi
    
    # Check file size
    local file_size
    file_size=$(stat -c%s "$full_path")
    if [[ $file_size -lt 1024 ]]; then  # Less than 1KB is suspicious
        log_error "Backup file is too small: ${file_size} bytes"
        return 1
    fi
    
    log_info "File size: $(du -h "$full_path" | cut -f1)"
    
    # Test archive integrity
    local temp_test_dir
    temp_test_dir=$(mktemp -d "/tmp/backup_verify_XXXXXX")
    
    local verification_result=0
    
    if [[ "$full_path" == *.enc ]]; then
        log_info "Testing encrypted backup integrity..."
        
        if [[ -z "${BACKUP_ENCRYPTION_KEY:-}" ]]; then
            log_error "BACKUP_ENCRYPTION_KEY not set for encrypted backup"
            rm -rf "$temp_test_dir"
            return 1
        fi
        
        # Test decryption and extraction
        if openssl enc -aes-256-cbc -d -salt -k "$BACKUP_ENCRYPTION_KEY" \
            -in "$full_path" | tar -tzf - > /dev/null 2>&1; then
            log_info "✅ Encrypted backup integrity verified"
        else
            log_error "❌ Encrypted backup integrity check failed"
            verification_result=1
        fi
    else
        log_info "Testing unencrypted backup integrity..."
        
        # Test tar file integrity
        if tar -tzf "$full_path" > /dev/null 2>&1; then
            log_info "✅ Backup integrity verified"
        else
            log_error "❌ Backup integrity check failed"
            verification_result=1
        fi
    fi
    
    # Test partial extraction
    if [[ $verification_result -eq 0 ]]; then
        log_info "Testing partial extraction..."
        
        if [[ "$full_path" == *.enc ]]; then
            if openssl enc -aes-256-cbc -d -salt -k "$BACKUP_ENCRYPTION_KEY" \
                -in "$full_path" | tar -xzf - -C "$temp_test_dir" --wildcards "*/backup_info.json" 2>/dev/null; then
                
                local backup_info_file
                backup_info_file=$(find "$temp_test_dir" -name "backup_info.json" | head -1)
                
                if [[ -f "$backup_info_file" ]] && jq . "$backup_info_file" > /dev/null 2>&1; then
                    log_info "✅ Backup metadata is valid JSON"
                    
                    # Display backup information
                    local backup_name version timestamp
                    backup_name=$(jq -r '.backup_name // "unknown"' "$backup_info_file")
                    version=$(jq -r '.version // "unknown"' "$backup_info_file")
                    timestamp=$(jq -r '.timestamp // "unknown"' "$backup_info_file")
                    
                    log_info "Backup name: $backup_name"
                    log_info "Version: $version"
                    log_info "Timestamp: $timestamp"
                else
                    log_warn "Backup metadata not found or invalid"
                fi
            else
                log_warn "Could not extract backup metadata for verification"
            fi
        else
            if tar -xzf "$full_path" -C "$temp_test_dir" --wildcards "*/backup_info.json" 2>/dev/null; then
                local backup_info_file
                backup_info_file=$(find "$temp_test_dir" -name "backup_info.json" | head -1)
                
                if [[ -f "$backup_info_file" ]] && jq . "$backup_info_file" > /dev/null 2>&1; then
                    log_info "✅ Backup metadata is valid JSON"
                else
                    log_warn "Backup metadata not found or invalid"
                fi
            else
                log_warn "Could not extract backup metadata for verification"
            fi
        fi
    fi
    
    rm -rf "$temp_test_dir"
    return $verification_result
}

# Create safety backup
create_safety_backup() {
    log_info "💾 Creating safety backup of current data..."
    
    local safety_backup_dir="${BACKUP_DIR}/${SAFETY_BACKUP_PREFIX}"
    mkdir -p "$safety_backup_dir"
    
    # Save current docker-compose state
    echo "$ORIGINAL_SERVICES_STATE" > "${safety_backup_dir}/services_state.json"
    
    # Backup current volumes if they exist
    local volumes_backed_up=0
    
    for volume in $(docker volume ls --format "{{.Name}}" | grep -E "(mem0ai|postgres|redis|qdrant)" || true); do
        log_info "Backing up volume: $volume"
        
        if docker run --rm \
            -v "${volume}:/source:ro" \
            -v "${safety_backup_dir}:/backup" \
            alpine tar -czf "/backup/${volume}.tar.gz" -C /source .; then
            
            log_debug "Volume $volume backed up successfully"
            ((volumes_backed_up++))
        else
            log_warn "Failed to backup volume: $volume"
        fi
    done
    
    # Backup configuration files
    if [[ -d "${PROJECT_ROOT}/config" ]]; then
        cp -r "${PROJECT_ROOT}/config" "${safety_backup_dir}/" 2>/dev/null || true
    fi
    
    if [[ -f "${PROJECT_ROOT}/.env" ]]; then
        cp "${PROJECT_ROOT}/.env" "${safety_backup_dir}/" 2>/dev/null || true
    fi
    
    # Create safety backup metadata
    cat > "${safety_backup_dir}/safety_backup_info.json" << EOF
{
    "safety_backup_name": "${SAFETY_BACKUP_PREFIX}",
    "created_at": "$(date -Iseconds)",
    "original_services_state": $(echo "$ORIGINAL_SERVICES_STATE"),
    "volumes_backed_up": $volumes_backed_up,
    "purpose": "Pre-restore safety backup"
}
EOF
    
    SAFETY_BACKUP_CREATED="$safety_backup_dir"
    log_info "✅ Safety backup created: $SAFETY_BACKUP_PREFIX ($volumes_backed_up volumes)"
}

# Emergency rollback function
emergency_rollback() {
    log_error "🚨 EMERGENCY ROLLBACK INITIATED"
    
    if [[ -z "$SAFETY_BACKUP_CREATED" ]] || [[ ! -d "$SAFETY_BACKUP_CREATED" ]]; then
        log_error "No safety backup available for rollback!"
        return 1
    fi
    
    log_info "Rolling back from safety backup: $SAFETY_BACKUP_CREATED"
    
    # Stop current services
    cd "$PROJECT_ROOT"
    docker-compose down --timeout 30 || true
    
    # Restore volumes from safety backup
    for volume_backup in "${SAFETY_BACKUP_CREATED}"/*.tar.gz; do
        if [[ -f "$volume_backup" ]]; then
            local volume_name
            volume_name=$(basename "$volume_backup" .tar.gz)
            
            log_info "Restoring volume: $volume_name"
            
            # Remove existing volume and recreate
            docker volume rm "$volume_name" 2>/dev/null || true
            docker volume create "$volume_name"
            
            # Restore volume data
            docker run --rm \
                -v "${volume_name}:/target" \
                -v "${SAFETY_BACKUP_CREATED}:/backup:ro" \
                alpine tar -xzf "/backup/${volume_name}.tar.gz" -C /target
        fi
    done
    
    # Restore configuration files
    if [[ -d "${SAFETY_BACKUP_CREATED}/config" ]]; then
        rm -rf "${PROJECT_ROOT}/config"
        cp -r "${SAFETY_BACKUP_CREATED}/config" "${PROJECT_ROOT}/"
    fi
    
    if [[ -f "${SAFETY_BACKUP_CREATED}/.env" ]]; then
        cp "${SAFETY_BACKUP_CREATED}/.env" "${PROJECT_ROOT}/"
    fi
    
    # Start services
    start_services
    
    log_info "✅ Emergency rollback completed"
}

# Manual rollback to last safety backup
rollback_to_safety() {
    log_info "🔄 Rolling back to last safety backup..."
    
    # Find most recent safety backup
    local latest_safety_backup
    latest_safety_backup=$(find "$BACKUP_DIR" -name "safety_backup_*" -type d | sort -r | head -1)
    
    if [[ -z "$latest_safety_backup" ]]; then
        log_error "No safety backup found for rollback"
        return 1
    fi
    
    log_info "Found safety backup: $(basename "$latest_safety_backup")"
    
    # Confirm rollback
    echo -e "${YELLOW}⚠️  This will restore your system to the state before the last restore operation.${NC}"
    read -p "Are you sure you want to proceed? (yes/no): " -r
    if [[ ! $REPLY =~ ^yes$ ]]; then
        echo "Rollback cancelled."
        return 0
    fi
    
    SAFETY_BACKUP_CREATED="$latest_safety_backup"
    emergency_rollback
}

# Main restore function
restore_backup() {
    local backup_file="$1"
    local skip_safety="${2:-false}"
    
    local full_path="${BACKUP_DIR}/${backup_file}"
    
    # Add extension if not provided
    if [[ ! -f "$full_path" ]]; then
        if [[ -f "${full_path}.tar.gz.enc" ]]; then
            full_path="${full_path}.tar.gz.enc"
        elif [[ -f "${full_path}.tar.gz" ]]; then
            full_path="${full_path}.tar.gz"
        else
            log_error "Backup file not found: $backup_file"
            return 1
        fi
    fi
    
    log_info "📦 Starting restore from backup: $(basename "$full_path")"
    
    # Verify backup integrity first
    if ! verify_backup "$(basename "$full_path")"; then
        log_error "Backup integrity verification failed"
        return 1
    fi
    
    # Mark restore as in progress
    RESTORE_IN_PROGRESS="$backup_file"
    
    # Create restore workspace
    RESTORE_DIR=$(mktemp -d "/tmp/mem0ai_restore_XXXXXX")
    log_debug "Created restore workspace: $RESTORE_DIR"
    
    # Extract backup
    log_info "📂 Extracting backup..."
    
    if [[ "$full_path" == *.enc ]]; then
        # Decrypt and extract encrypted backup
        if [[ -z "${BACKUP_ENCRYPTION_KEY:-}" ]]; then
            log_error "BACKUP_ENCRYPTION_KEY not set for encrypted backup"
            return 1
        fi
        
        if ! openssl enc -aes-256-cbc -d -salt -k "$BACKUP_ENCRYPTION_KEY" \
            -in "$full_path" | tar -xzf - -C "$RESTORE_DIR"; then
            log_error "Failed to decrypt and extract backup"
            return 1
        fi
    else
        # Extract unencrypted backup
        if ! tar -xzf "$full_path" -C "$RESTORE_DIR"; then
            log_error "Failed to extract backup"
            return 1
        fi
    fi
    
    # Find backup directory
    local backup_content
    backup_content=$(ls "$RESTORE_DIR" | head -1)
    local backup_path="${RESTORE_DIR}/${backup_content}"
    
    # Verify backup structure
    if [[ ! -f "${backup_path}/backup_info.json" ]]; then
        log_error "Invalid backup: backup_info.json not found"
        return 1
    fi
    
    # Display backup information
    log_info "📊 Backup information:"
    if command -v jq > /dev/null; then
        jq . "${backup_path}/backup_info.json" | while IFS= read -r line; do
            log_info "  $line"
        done
    else
        cat "${backup_path}/backup_info.json"
    fi
    echo ""
    
    # Create safety backup unless skipped
    if [[ "$skip_safety" != "true" ]]; then
        create_safety_backup
    else
        log_warn "⚠️  Safety backup creation skipped - no rollback possible!"
    fi
    
    # Stop services
    stop_services
    
    # Restore databases
    log_info "🗄️  Restoring databases..."
    
    # Start only database services for restore
    docker-compose up -d postgres redis qdrant
    sleep 15
    
    # Restore PostgreSQL
    if [[ -f "${backup_path}/postgres_backup.sql" ]]; then
        restore_postgresql "${backup_path}/postgres_backup.sql"
    else
        log_warn "PostgreSQL backup not found, skipping"
    fi
    
    # Restore Redis
    if [[ -f "${backup_path}/redis_dump.rdb" ]]; then
        restore_redis "${backup_path}/redis_dump.rdb"
    else
        log_warn "Redis backup not found, skipping"
    fi
    
    # Restore Qdrant
    if [[ -d "${backup_path}/qdrant" ]]; then
        restore_qdrant "${backup_path}/qdrant"
    else
        log_warn "Qdrant backup not found, skipping"
    fi
    
    # Restore configuration files
    restore_config_files "$backup_path"
    
    # Start all services
    start_services
    
    # Wait for services to be ready and verify
    if wait_for_services_healthy; then
        log_info "✅ Restore completed successfully!"
        log_info "=============================="
        log_info "Backup restored: $(basename "$full_path")"
        if [[ -n "$SAFETY_BACKUP_CREATED" ]]; then
            log_info "Safety backup: $(basename "$SAFETY_BACKUP_CREATED")"
        fi
        log_info "Log file: $RESTORE_LOG"
        
        # Clear restore state
        RESTORE_IN_PROGRESS=""
        
        return 0
    else
        log_error "Service health check failed after restore"
        return 1
    fi
}

# Restore PostgreSQL database
restore_postgresql() {
    local backup_file="$1"
    
    log_info "- Restoring PostgreSQL database..."
    
    # Validate backup file
    if [[ ! -s "$backup_file" ]]; then
        log_error "PostgreSQL backup file is empty or missing"
        return 1
    fi
    
    # Drop and recreate database with error handling
    if ! docker-compose exec -T postgres psql -U "${POSTGRES_USER}" -d postgres \
        -c "DROP DATABASE IF EXISTS ${POSTGRES_DB};"; then
        log_error "Failed to drop existing database"
        return 1
    fi
    
    if ! docker-compose exec -T postgres psql -U "${POSTGRES_USER}" -d postgres \
        -c "CREATE DATABASE ${POSTGRES_DB};"; then
        log_error "Failed to create database"
        return 1
    fi
    
    # Restore data with timeout
    if timeout "$RESTORE_TIMEOUT" docker-compose exec -T postgres psql \
        -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" \
        < "$backup_file" > "${RESTORE_LOG}.postgres" 2>&1; then
        
        log_info "✅ PostgreSQL database restored successfully"
        return 0
    else
        log_error "PostgreSQL restore failed or timed out"
        return 1
    fi
}

# Restore Redis data
restore_redis() {
    local backup_file="$1"
    
    log_info "- Restoring Redis data..."
    
    if [[ ! -s "$backup_file" ]]; then
        log_error "Redis backup file is empty or missing"
        return 1
    fi
    
    # Stop Redis, replace dump, start Redis
    docker-compose stop redis
    
    local redis_container
    redis_container=$(docker-compose ps -q redis)
    
    if [[ -n "$redis_container" ]] && \
       docker cp "$backup_file" "${redis_container}:/data/dump.rdb"; then
        
        docker-compose start redis
        
        # Wait for Redis to be ready
        local wait_count=0
        while [[ $wait_count -lt 30 ]]; do
            if docker-compose exec -T redis redis-cli ping | grep -q "PONG"; then
                log_info "✅ Redis data restored successfully"
                return 0
            fi
            sleep 2
            ((wait_count++))
        done
        
        log_error "Redis failed to start after restore"
        return 1
    else
        log_error "Failed to copy Redis backup to container"
        return 1
    fi
}

# Restore Qdrant data
restore_qdrant() {
    local backup_dir="$1"
    
    log_info "- Restoring Qdrant data..."
    
    if [[ ! -d "$backup_dir" ]]; then
        log_error "Qdrant backup directory not found"
        return 1
    fi
    
    # Stop Qdrant
    docker-compose stop qdrant
    
    # Copy snapshot files
    local qdrant_container
    qdrant_container=$(docker-compose ps -q qdrant)
    
    if [[ -n "$qdrant_container" ]] && \
       docker cp "${backup_dir}/." "${qdrant_container}:/qdrant/snapshots/"; then
        
        # Start Qdrant
        docker-compose start qdrant
        sleep 10
        
        # TODO: Implement Qdrant snapshot restoration via API
        log_info "✅ Qdrant snapshots copied (manual restoration may be required)"
        return 0
    else
        log_error "Failed to copy Qdrant snapshots"
        return 1
    fi
}

# Restore configuration files
restore_config_files() {
    local backup_path="$1"
    
    log_info "📁 Restoring configuration files..."
    
    # Restore configuration
    if [[ -d "${backup_path}/config" ]]; then
        log_info "- Restoring configuration files..."
        rm -rf "${PROJECT_ROOT}/config"
        cp -r "${backup_path}/config" "${PROJECT_ROOT}/"
        log_info "✅ Configuration files restored"
    fi
    
    # Restore SSL certificates
    if [[ -d "${backup_path}/ssl" ]]; then
        log_info "- Restoring SSL certificates..."
        rm -rf "${PROJECT_ROOT}/ssl"
        cp -r "${backup_path}/ssl" "${PROJECT_ROOT}/"
        log_info "✅ SSL certificates restored"
    fi
    
    # Restore uploads
    if [[ -d "${backup_path}/uploads" ]]; then
        log_info "- Restoring uploaded files..."
        rm -rf "${PROJECT_ROOT}/uploads"
        cp -r "${backup_path}/uploads" "${PROJECT_ROOT}/"
        log_info "✅ Uploaded files restored"
    fi
}

# Main function
main() {
    local backup_file=""
    local verify_only=false
    local skip_safety=false
    local auto_confirm=false
    local do_rollback=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -f|--file)
                backup_file="$2"
                shift 2
                ;;
            -l|--list)
                list_backups
                exit 0
                ;;
            -v|--verify)
                backup_file="$2"
                verify_only=true
                shift 2
                ;;
            -r|--rollback)
                do_rollback=true
                shift
                ;;
            -s|--skip-safety)
                skip_safety=true
                shift
                ;;
            -y|--yes)
                auto_confirm=true
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Initialize logging
    mkdir -p "$BACKUP_DIR"
    log_info "🔄 Mem0AI Restore Process Started"
    echo "=========================="
    
    # Check for restore lock
    check_restore_lock
    
    # Validate system state
    if ! validate_system_state; then
        exit 1
    fi
    
    # Handle rollback request
    if [[ "$do_rollback" == "true" ]]; then
        rollback_to_safety
        exit $?
    fi
    
    # If no backup file specified, show list and prompt
    if [[ -z "$backup_file" ]]; then
        list_backups
        echo ""
        read -p "Enter backup name to restore (without .tar.gz extension): " backup_file
        
        if [[ -z "$backup_file" ]]; then
            echo "No backup specified."
            exit 1
        fi
    fi
    
    # Handle verify-only mode
    if [[ "$verify_only" == "true" ]]; then
        verify_backup "$backup_file"
        exit $?
    fi
    
    # Confirm destructive operation unless auto-confirmed
    if [[ "$auto_confirm" != "true" ]]; then
        echo ""
        echo -e "${RED}⚠️  WARNING: This will replace all current data!${NC}"
        echo -e "${YELLOW}A safety backup will be created automatically.${NC}"
        read -p "Are you sure you want to continue? (yes/no): " -r
        if [[ ! $REPLY =~ ^yes$ ]]; then
            echo "Restore cancelled."
            exit 0
        fi
    fi
    
    # Start restore process
    if restore_backup "$backup_file" "$skip_safety"; then
        log_info "🎉 Restore completed successfully!"
        exit 0
    else
        log_error "Restore failed!"
        exit 1
    fi
}

# Run main function with all arguments
main "$@"