#!/bin/bash
set -euo pipefail

# Mem0AI Backup Script
# Creates encrypted backups of all data and uploads to S3 (optional)
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
readonly NC='\033[0m'

# Configuration with validation
readonly BACKUP_DIR="${PROJECT_ROOT}/backups"
readonly DATE=$(date +%Y%m%d_%H%M%S)
readonly BACKUP_NAME="mem0ai_backup_${DATE}"
readonly BACKUP_PATH="${BACKUP_DIR}/${BACKUP_NAME}"
readonly LOG_FILE="${BACKUP_DIR}/backup_${DATE}.log"
readonly MAX_BACKUP_SIZE_GB=${MAX_BACKUP_SIZE_GB:-100}
readonly BACKUP_TIMEOUT=${BACKUP_TIMEOUT:-3600}  # 1 hour default

# Validate required environment variables
required_vars=("POSTGRES_USER" "POSTGRES_DB")
for var in "${required_vars[@]}"; do
    if [[ -z "${!var:-}" ]]; then
        echo "❌ Required environment variable $var is not set!"
        exit 1
    fi
done

# Logging function
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
}

log_info() { log "INFO" "$@"; }
log_warn() { log "WARN" "$@"; }
log_error() { log "ERROR" "$@"; }

# Error handling function
handle_error() {
    local exit_code=$?
    local line_no=$1
    log_error "Script failed at line $line_no with exit code $exit_code"
    cleanup_on_error
    exit $exit_code
}

# Cleanup function for error cases
cleanup_on_error() {
    log_warn "Cleaning up due to error..."
    if [[ -d "$BACKUP_PATH" ]]; then
        rm -rf "$BACKUP_PATH"
    fi
}

# Set error trap
trap 'handle_error ${LINENO}' ERR

# Check disk space before starting
check_disk_space() {
    local available_kb
    available_kb=$(df "$BACKUP_DIR" | awk 'NR==2 {print $4}')
    local available_gb=$((available_kb / 1024 / 1024))
    
    if [[ $available_gb -lt $MAX_BACKUP_SIZE_GB ]]; then
        log_error "Insufficient disk space. Available: ${available_gb}GB, Required: ${MAX_BACKUP_SIZE_GB}GB"
        exit 1
    fi
    
    log_info "Disk space check passed. Available: ${available_gb}GB"
}

# Function to check if docker-compose service is running
check_service_running() {
    local service="$1"
    if ! docker-compose ps "$service" | grep -q "Up"; then
        log_error "Service $service is not running"
        return 1
    fi
    return 0
}

# Backup PostgreSQL with timeout and validation
backup_postgresql() {
    log_info "- Backing up PostgreSQL database..."
    
    if ! check_service_running "postgres"; then
        log_error "PostgreSQL service is not running"
        return 1
    fi
    
    local postgres_backup="${BACKUP_PATH}/postgres_backup.sql"
    
    # Use timeout to prevent hanging
    if timeout "$BACKUP_TIMEOUT" docker-compose exec -T postgres pg_dump \
        -U "${POSTGRES_USER}" \
        -d "${POSTGRES_DB}" \
        --no-password \
        --clean \
        --if-exists \
        --verbose 2>"${LOG_FILE}.postgres" > "$postgres_backup"; then
        
        # Validate backup file
        if [[ -s "$postgres_backup" ]] && grep -q "PostgreSQL database dump" "$postgres_backup"; then
            local size
            size=$(du -h "$postgres_backup" | cut -f1)
            log_info "✅ PostgreSQL backup completed ($size)"
            return 0
        else
            log_error "PostgreSQL backup file appears to be invalid or empty"
            return 1
        fi
    else
        log_error "PostgreSQL backup failed or timed out"
        return 1
    fi
}

# Backup Qdrant collections with improved error handling
backup_qdrant() {
    log_info "- Backing up Qdrant vector database..."
    
    if ! check_service_running "qdrant"; then
        log_warn "Qdrant service is not running, skipping backup"
        return 0
    fi
    
    local qdrant_backup_dir="${BACKUP_PATH}/qdrant"
    mkdir -p "$qdrant_backup_dir"
    
    # Create Qdrant snapshot with better error handling
    local snapshot_name="snapshot_${DATE}"
    local snapshot_response
    
    log_info "Creating Qdrant snapshot: $snapshot_name"
    
    if snapshot_response=$(docker-compose exec -T qdrant \
        curl -s -f -X POST "http://localhost:6333/snapshots" \
        -H "Content-Type: application/json" \
        -d "{\"snapshot_name\": \"${snapshot_name}\"}" 2>&1); then
        
        log_info "Snapshot creation response: $snapshot_response"
        
        # Wait for snapshot to be created with timeout
        local wait_count=0
        local max_wait=30
        
        while [[ $wait_count -lt $max_wait ]]; do
            if docker-compose exec -T qdrant \
                find /qdrant/storage -name "*${snapshot_name}*" -type f | grep -q "."; then
                log_info "Snapshot files found after ${wait_count}s"
                break
            fi
            sleep 1
            ((wait_count++))
        done
        
        if [[ $wait_count -eq $max_wait ]]; then
            log_error "Timeout waiting for Qdrant snapshot creation"
            return 1
        fi
        
        # Copy snapshot files
        if docker-compose exec -T qdrant \
            find /qdrant/storage -name "*${snapshot_name}*" \
            -exec cp {} /qdrant/snapshots/ \; 2>"${LOG_FILE}.qdrant"; then
            
            # Export snapshot to backup directory
            local qdrant_container
            qdrant_container=$(docker-compose ps -q qdrant)
            
            if [[ -n "$qdrant_container" ]] && \
               docker cp "${qdrant_container}:/qdrant/snapshots" "$qdrant_backup_dir/"; then
                
                # Validate backup
                if [[ -d "${qdrant_backup_dir}/snapshots" ]] && \
                   [[ -n "$(find "${qdrant_backup_dir}/snapshots" -name "*${snapshot_name}*" -type f)" ]]; then
                    local size
                    size=$(du -sh "$qdrant_backup_dir" | cut -f1)
                    log_info "✅ Qdrant backup completed ($size)"
                    return 0
                else
                    log_error "Qdrant backup validation failed"
                    return 1
                fi
            else
                log_error "Failed to copy Qdrant snapshots"
                return 1
            fi
        else
            log_error "Failed to copy snapshot files within Qdrant container"
            return 1
        fi
    else
        log_error "Failed to create Qdrant snapshot: $snapshot_response"
        return 1
    fi
}

# Backup Redis data with improved error handling
backup_redis() {
    log_info "- Backing up Redis cache..."
    
    if ! check_service_running "redis"; then
        log_warn "Redis service is not running, skipping backup"
        return 0
    fi
    
    local redis_backup="${BACKUP_PATH}/redis_dump.rdb"
    
    # Trigger Redis background save
    if docker-compose exec -T redis redis-cli BGSAVE 2>"${LOG_FILE}.redis"; then
        log_info "Redis BGSAVE initiated"
        
        # Wait for save to complete with timeout
        local wait_count=0
        local max_wait=60
        
        while [[ $wait_count -lt $max_wait ]]; do
            local save_status
            save_status=$(docker-compose exec -T redis redis-cli LASTSAVE 2>/dev/null || echo "error")
            
            if [[ "$save_status" != "error" ]]; then
                # Check if save is complete by comparing with previous timestamp
                sleep 2
                local new_save_status
                new_save_status=$(docker-compose exec -T redis redis-cli LASTSAVE 2>/dev/null || echo "error")
                
                if [[ "$save_status" != "$new_save_status" ]]; then
                    log_info "Redis save completed after ${wait_count}s"
                    break
                fi
            fi
            
            sleep 1
            ((wait_count++))
        done
        
        if [[ $wait_count -eq $max_wait ]]; then
            log_warn "Timeout waiting for Redis save, proceeding anyway"
        fi
        
        # Copy Redis dump
        local redis_container
        redis_container=$(docker-compose ps -q redis)
        
        if [[ -n "$redis_container" ]] && \
           docker cp "${redis_container}:/data/dump.rdb" "$redis_backup"; then
            
            # Validate backup
            if [[ -s "$redis_backup" ]]; then
                local size
                size=$(du -h "$redis_backup" | cut -f1)
                log_info "✅ Redis backup completed ($size)"
                return 0
            else
                log_error "Redis backup file is empty or invalid"
                return 1
            fi
        else
            log_error "Failed to copy Redis dump"
            return 1
        fi
    else
        log_error "Failed to initiate Redis BGSAVE"
        return 1
    fi
}

# Backup configuration files safely
backup_config_files() {
    log_info "📁 Creating file backups..."
    
    # Backup configuration files
    log_info "- Configuration files..."
    mkdir -p "${BACKUP_PATH}/config"
    if [[ -d "${PROJECT_ROOT}/config" ]]; then
        cp -r "${PROJECT_ROOT}/config/"* "${BACKUP_PATH}/config/" 2>/dev/null || true
    fi
    
    # Backup SSL certificates
    log_info "- SSL certificates..."
    if [[ -d "${PROJECT_ROOT}/ssl" ]]; then
        mkdir -p "${BACKUP_PATH}/ssl"
        cp -r "${PROJECT_ROOT}/ssl/"* "${BACKUP_PATH}/ssl/" 2>/dev/null || true
    fi
    
    # Backup application logs (last 7 days)
    log_info "- Application logs..."
    if [[ -d "${PROJECT_ROOT}/logs" ]]; then
        mkdir -p "${BACKUP_PATH}/logs"
        find "${PROJECT_ROOT}/logs" -name "*.log" -mtime -7 -exec cp {} "${BACKUP_PATH}/logs/" \; 2>/dev/null || true
    fi
    
    # Backup uploads directory
    log_info "- Uploaded files..."
    if [[ -d "${PROJECT_ROOT}/uploads" ]]; then
        mkdir -p "${BACKUP_PATH}/uploads"
        cp -r "${PROJECT_ROOT}/uploads/"* "${BACKUP_PATH}/uploads/" 2>/dev/null || true
    fi
}

# Create backup metadata with comprehensive information
create_backup_metadata() {
    log_info "📝 Creating backup metadata..."
    
    local backup_size
    backup_size=$(du -sb "${BACKUP_PATH}" | cut -f1)
    
    local docker_compose_version
    docker_compose_version=$(docker-compose --version 2>/dev/null || echo "unknown")
    
    cat > "${BACKUP_PATH}/backup_info.json" << EOF
{
    "backup_name": "${BACKUP_NAME}",
    "timestamp": "${DATE}",
    "version": "2.0.0",
    "hostname": "$(hostname)",
    "script_version": "$(git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || echo "unknown")",
    "components": [
        "postgresql",
        "qdrant", 
        "redis",
        "config",
        "ssl",
        "logs",
        "uploads"
    ],
    "size_bytes": $backup_size,
    "docker_compose_version": "$docker_compose_version",
    "environment": {
        "postgres_db": "${POSTGRES_DB}",
        "backup_dir": "${BACKUP_DIR}",
        "encryption_enabled": $([ -n "${BACKUP_ENCRYPTION_KEY:-}" ] && echo "true" || echo "false")
    },
    "checksums": {
        "postgres": "$(sha256sum "${BACKUP_PATH}/postgres_backup.sql" 2>/dev/null | cut -d' ' -f1 || echo "n/a")",
        "redis": "$(sha256sum "${BACKUP_PATH}/redis_dump.rdb" 2>/dev/null | cut -d' ' -f1 || echo "n/a")"
    }
}
EOF
}

# Create encrypted archive with validation
create_encrypted_archive() {
    log_info "🔐 Creating backup archive..."
    
    local final_backup
    
    if [[ -n "${BACKUP_ENCRYPTION_KEY:-}" ]]; then
        cd "$BACKUP_DIR"
        
        # Create tar archive and encrypt
        if tar -czf - "$BACKUP_NAME" | \
           openssl enc -aes-256-cbc -salt -k "$BACKUP_ENCRYPTION_KEY" \
           > "${BACKUP_NAME}.tar.gz.enc"; then
            
            # Validate encrypted file
            if [[ -s "${BACKUP_NAME}.tar.gz.enc" ]]; then
                # Remove unencrypted backup
                rm -rf "$BACKUP_NAME"
                final_backup="${BACKUP_NAME}.tar.gz.enc"
                log_info "✅ Backup encrypted: ${final_backup}"
            else
                log_error "Encrypted backup file is empty"
                return 1
            fi
        else
            log_error "Failed to create encrypted backup"
            return 1
        fi
    else
        # Create unencrypted archive
        cd "$BACKUP_DIR"
        if tar -czf "${BACKUP_NAME}.tar.gz" "$BACKUP_NAME"; then
            if [[ -s "${BACKUP_NAME}.tar.gz" ]]; then
                rm -rf "$BACKUP_NAME"
                final_backup="${BACKUP_NAME}.tar.gz"
                log_warn "⚠️  Backup not encrypted (BACKUP_ENCRYPTION_KEY not set)"
            else
                log_error "Backup archive is empty"
                return 1
            fi
        else
            log_error "Failed to create backup archive"
            return 1
        fi
    fi
    
    cd - > /dev/null
    echo "$final_backup"
}

# Upload to S3 with retry logic
upload_to_s3() {
    local backup_file="$1"
    
    if [[ -n "${S3_BUCKET_NAME:-}" && -n "${AWS_ACCESS_KEY_ID:-}" ]]; then
        log_info "☁️  Uploading to S3..."
        
        # Check if AWS CLI is available
        if command -v aws > /dev/null; then
            local s3_path="s3://${S3_BUCKET_NAME}/mem0ai-backups/${backup_file}"
            local retry_count=0
            local max_retries=3
            
            while [[ $retry_count -lt $max_retries ]]; do
                if aws s3 cp "${BACKUP_DIR}/${backup_file}" "$s3_path" \
                    --region "${AWS_REGION:-us-east-1}" \
                    --storage-class "${S3_STORAGE_CLASS:-STANDARD_IA}"; then
                    
                    # Verify upload
                    if aws s3 ls "$s3_path" > /dev/null; then
                        log_info "✅ Backup uploaded to S3: $s3_path"
                        return 0
                    else
                        log_warn "Upload verification failed, retrying..."
                    fi
                else
                    log_warn "S3 upload attempt $((retry_count + 1)) failed"
                fi
                
                ((retry_count++))
                if [[ $retry_count -lt $max_retries ]]; then
                    sleep $((retry_count * 5))  # Exponential backoff
                fi
            done
            
            log_error "S3 upload failed after $max_retries attempts"
            return 1
        else
            log_warn "⚠️  AWS CLI not found, skipping S3 upload"
            return 0
        fi
    else
        log_info "S3 upload skipped (credentials not configured)"
        return 0
    fi
}

# Cleanup old backups with safety checks
cleanup_old_backups() {
    log_info "🧹 Cleaning up old backups..."
    
    local retention_days=${BACKUP_RETENTION_DAYS:-7}
    local deleted_count=0
    
    # Find and delete old backup files
    while IFS= read -r -d '' file; do
        if rm "$file"; then
            log_info "Deleted old backup: $(basename "$file")"
            ((deleted_count++))
        else
            log_warn "Failed to delete: $file"
        fi
    done < <(find "$BACKUP_DIR" -name "mem0ai_backup_*" -mtime +$retention_days -print0)
    
    log_info "Cleaned up $deleted_count old backup files"
}

# Update monitoring metrics
update_metrics() {
    local backup_file="$1"
    local status="$2"
    
    if command -v curl > /dev/null; then
        local timestamp
        timestamp=$(date +%s)
        local size=0
        
        if [[ -f "${BACKUP_DIR}/${backup_file}" ]]; then
            size=$(stat -c%s "${BACKUP_DIR}/${backup_file}" 2>/dev/null || echo 0)
        fi
        
        # Update Prometheus metrics
        {
            echo "backup_last_success_timestamp_seconds $timestamp"
            echo "backup_size_bytes $size"
            echo "backup_status{status=\"$status\"} 1"
        } | curl -X POST "http://localhost:9091/metrics/job/backup" \
             --data-binary @- 2>/dev/null || true
        
        log_info "📊 Backup metrics updated"
    fi
}

# Main execution
main() {
    log_info "🔄 Mem0AI Backup Process Started"
    echo "=================================="
    echo "Backup: $BACKUP_NAME"
    echo "Path: $BACKUP_PATH"
    echo "Log: $LOG_FILE"
    echo ""
    
    # Pre-flight checks
    check_disk_space
    
    # Create backup directory with proper permissions
    if ! mkdir -p "$BACKUP_DIR"; then
        log_error "Failed to create backup directory: $BACKUP_DIR"
        exit 1
    fi
    
    # Create backup workspace
    if ! mkdir -p "$BACKUP_PATH"; then
        log_error "Failed to create backup workspace: $BACKUP_PATH"
        exit 1
    fi
    
    log_info "Created backup workspace: $BACKUP_PATH"
    
    # Perform backups
    backup_postgresql
    backup_qdrant
    backup_redis
    backup_config_files
    
    # Create metadata
    create_backup_metadata
    
    # Create archive
    local final_backup
    final_backup=$(create_encrypted_archive)
    
    # Upload to S3 if configured
    upload_to_s3 "$final_backup"
    
    # Cleanup old backups
    cleanup_old_backups
    
    # Calculate final size and update metrics
    local backup_size
    backup_size=$(du -h "${BACKUP_DIR}/${final_backup}" | cut -f1)
    
    update_metrics "$final_backup" "success"
    
    echo ""
    log_info "✅ Backup completed successfully!"
    echo "=================================="
    echo "Backup file: ${final_backup}"
    echo "Size: ${backup_size}"
    echo "Location: ${BACKUP_DIR}/${final_backup}"
    echo "Log: ${LOG_FILE}"
}

# Run main function
main "$@"