#!/bin/bash
set -euo pipefail

# Mem0AI Production Deployment Script
# This script automates the complete deployment process on Ubuntu VPS

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="/opt/mem0ai"
USER="mem0ai"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        print_error "This script must be run as root (use sudo)"
        exit 1
    fi
}

# Function to check system requirements
check_requirements() {
    print_status "Checking system requirements..."
    
    # Check Ubuntu version
    if ! grep -q "Ubuntu" /etc/os-release; then
        print_error "This script is designed for Ubuntu. Other distributions may not be supported."
        exit 1
    fi
    
    # Check minimum RAM (2GB)
    total_ram=$(free -m | awk 'NR==2{printf "%.0f", $2}')
    if [[ $total_ram -lt 1800 ]]; then
        print_warning "Less than 2GB RAM detected. Performance may be limited."
    fi
    
    # Check disk space (minimum 10GB)
    available_space=$(df / | awk 'NR==2{print $4}')
    if [[ $available_space -lt 10485760 ]]; then  # 10GB in KB
        print_error "Less than 10GB disk space available. Deployment may fail."
        exit 1
    fi
    
    print_success "System requirements check passed"
}

# Function to install dependencies
install_dependencies() {
    print_status "Installing system dependencies..."
    
    # Update package list
    apt update
    
    # Install essential packages
    apt install -y \
        curl \
        wget \
        git \
        ufw \
        htop \
        unzip \
        ca-certificates \
        gnupg \
        lsb-release \
        software-properties-common \
        apt-transport-https \
        python3 \
        python3-pip \
        python3-venv
    
    # Install uv (modern Python package manager)
    install_uv
    
    # Install Docker
    if ! command -v docker &> /dev/null; then
        print_status "Installing Docker..."
        curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
        echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
        apt update
        apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
        systemctl enable docker
        systemctl start docker
    fi
    
    # Install Docker Compose (standalone)
    if ! command -v docker-compose &> /dev/null; then
        print_status "Installing Docker Compose..."
        curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        chmod +x /usr/local/bin/docker-compose
    fi
    
    print_success "Dependencies installed successfully"
}

# Function to install uv
install_uv() {
    print_status "Installing uv (modern Python package manager)..."
    
    if ! command -v uv &> /dev/null; then
        # Install uv using the official installer
        curl -LsSf https://astral.sh/uv/install.sh | sh
        
        # Add uv to PATH for current session
        export PATH="$HOME/.cargo/bin:$PATH"
        
        # Add uv to system PATH
        echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> /etc/environment
        
        # Verify installation
        if command -v uv &> /dev/null; then
            print_success "uv installed successfully"
            uv --version
        else
            print_warning "uv installation may have failed, falling back to pip"
        fi
    else
        print_success "uv is already installed"
        uv --version
    fi
}

# Function to create system user
create_user() {
    print_status "Creating system user..."
    
    if ! id "$USER" &>/dev/null; then
        useradd -r -s /bin/bash -d "$INSTALL_DIR" -m "$USER"
        usermod -aG docker "$USER"
        print_success "User $USER created"
    else
        print_warning "User $USER already exists"
    fi
}

# Function to setup directory structure
setup_directories() {
    print_status "Setting up directory structure..."
    
    # Create main directory
    mkdir -p "$INSTALL_DIR"
    
    # Copy files
    cp -r "$SCRIPT_DIR"/* "$INSTALL_DIR/"
    
    # Create additional directories
    mkdir -p "$INSTALL_DIR"/{logs,uploads,backups,ssl,ssl-challenge}
    
    # Set permissions
    chown -R "$USER:$USER" "$INSTALL_DIR"
    chmod +x "$INSTALL_DIR"/scripts/*.sh
    
    print_success "Directory structure created"
}

# Function to configure environment
configure_environment() {
    print_status "Configuring environment..."
    
    # Check if .env exists
    if [[ ! -f "$INSTALL_DIR/.env" ]]; then
        print_warning ".env file not found. You need to configure it manually."
        print_status "Copying .env.example to .env..."
        cp "$INSTALL_DIR/.env.example" "$INSTALL_DIR/.env"
        chown "$USER:$USER" "$INSTALL_DIR/.env"
        chmod 600 "$INSTALL_DIR/.env"
        
        echo ""
        print_warning "IMPORTANT: Edit $INSTALL_DIR/.env and configure:"
        echo "  - DOMAIN (your domain name)"
        echo "  - ADMIN_EMAIL (your email)"
        echo "  - SSL_EMAIL (SSL certificate email)"
        echo "  - OPENAI_API_KEY (your OpenAI API key)"
        echo "  - Generate secrets with: cd $INSTALL_DIR && sudo -u $USER ./scripts/generate-secrets.sh"
        echo ""
        read -p "Press Enter when you have configured the .env file..."
    fi
    
    print_success "Environment configuration ready"
}

# Function to setup systemd services
setup_systemd() {
    print_status "Setting up systemd services..."
    
    # Copy service files
    cp "$INSTALL_DIR/config/systemd/"*.service /etc/systemd/system/
    cp "$INSTALL_DIR/config/systemd/"*.timer /etc/systemd/system/
    
    # Update service files with correct paths
    sed -i "s|/opt/mem0ai|$INSTALL_DIR|g" /etc/systemd/system/mem0ai*.service
    sed -i "s|/opt/mem0ai|$INSTALL_DIR|g" /etc/systemd/system/mem0ai*.timer
    
    # Reload systemd
    systemctl daemon-reload
    
    # Enable services
    systemctl enable mem0ai.service
    systemctl enable mem0ai-backup.timer
    
    print_success "Systemd services configured"
}

# Function to configure nginx
configure_nginx() {
    print_status "Configuring nginx sites..."
    
    # Create nginx auth file for monitoring
    if ! command -v htpasswd &> /dev/null; then
        apt install -y apache2-utils
    fi
    
    # Create basic auth for monitoring with strong password
    MONITORING_PASSWORD=$(openssl rand -base64 16)
    echo "admin:$(openssl passwd -apr1 "$MONITORING_PASSWORD")" > "$INSTALL_DIR/config/nginx/.htpasswd"
    echo -e "${YELLOW}⚠️  Monitoring credentials: admin / $MONITORING_PASSWORD${NC}"
    echo "Store these credentials securely!"
    
    # Create sites-enabled directory
    mkdir -p "$INSTALL_DIR/config/nginx/sites-enabled"
    
    print_success "Nginx configuration prepared"
}

# Function to run security hardening
run_security_hardening() {
    print_status "Running security hardening..."
    
    if [[ -f "$INSTALL_DIR/scripts/security-hardening.sh" ]]; then
        bash "$INSTALL_DIR/scripts/security-hardening.sh"
        print_success "Security hardening completed"
    else
        print_warning "Security hardening script not found"
    fi
}

# Function to start services
start_services() {
    print_status "Starting Mem0AI services..."
    
    # Switch to mem0ai user and start services
    cd "$INSTALL_DIR"
    sudo -u "$USER" docker-compose up -d
    
    # Start systemd service
    systemctl start mem0ai.service
    systemctl start mem0ai-backup.timer
    
    print_success "Services started"
}

# Function to setup SSL certificates
setup_ssl() {
    print_status "Setting up SSL certificates..."
    
    # Check if domain is configured
    if grep -q "your-domain.com" "$INSTALL_DIR/.env"; then
        print_warning "Domain not configured in .env file. Skipping SSL setup."
        print_warning "Configure your domain and run: cd $INSTALL_DIR && sudo -u $USER ./scripts/ssl-setup.sh"
        return
    fi
    
    # Run SSL setup
    cd "$INSTALL_DIR"
    sudo -u "$USER" ./scripts/ssl-setup.sh
    
    print_success "SSL certificates configured"
}

# Function to run health checks
run_health_checks() {
    print_status "Running health checks..."
    
    # Wait for services to start
    sleep 30
    
    # Check Docker containers
    if ! docker-compose -f "$INSTALL_DIR/docker-compose.yml" ps | grep -q "Up"; then
        print_error "Some containers are not running"
        docker-compose -f "$INSTALL_DIR/docker-compose.yml" ps
        return 1
    fi
    
    # Check HTTP endpoint
    if curl -sf http://localhost/health > /dev/null; then
        print_success "HTTP health check passed"
    else
        print_warning "HTTP health check failed"
    fi
    
    # Check systemd service
    if systemctl is-active --quiet mem0ai.service; then
        print_success "Systemd service is active"
    else
        print_warning "Systemd service is not active"
    fi
    
    print_success "Health checks completed"
}

# Function to display deployment summary
show_summary() {
    echo ""
    echo "==============================================="
    echo -e "${GREEN}🎉 Mem0AI Deployment Summary${NC}"
    echo "==============================================="
    echo ""
    echo -e "${YELLOW}📁 Installation Directory:${NC} $INSTALL_DIR"
    echo -e "${YELLOW}👤 System User:${NC} $USER"
    echo -e "${YELLOW}🐳 Docker Status:${NC} $(systemctl is-active docker)"
    echo -e "${YELLOW}🚀 Service Status:${NC} $(systemctl is-active mem0ai.service)"
    echo ""
    echo -e "${YELLOW}🌐 Access URLs:${NC}"
    echo "   - Application: http://$(hostname -I | awk '{print $1}')"
    echo "   - Health Check: http://$(hostname -I | awk '{print $1}')/health"
    echo "   - Monitoring: http://$(hostname -I | awk '{print $1}'):3000 (admin/admin123)"
    echo ""
    echo -e "${YELLOW}🔧 Management Commands:${NC}"
    echo "   - Start services: systemctl start mem0ai.service"
    echo "   - Stop services: systemctl stop mem0ai.service"
    echo "   - View logs: journalctl -u mem0ai.service -f"
    echo "   - Manual backup: cd $INSTALL_DIR && sudo -u $USER ./scripts/backup.sh"
    echo "   - Security audit: /usr/local/bin/security-audit.sh"
    echo ""
    echo -e "${YELLOW}📚 Important Files:${NC}"
    echo "   - Environment: $INSTALL_DIR/.env"
    echo "   - Docker Compose: $INSTALL_DIR/docker-compose.yml"
    echo "   - Nginx Config: $INSTALL_DIR/config/nginx/"
    echo "   - Logs: $INSTALL_DIR/logs/"
    echo "   - Backups: $INSTALL_DIR/backups/"
    echo ""
    echo -e "${YELLOW}⚠️  Next Steps:${NC}"
    echo "1. Configure your domain in .env file"
    echo "2. Run SSL setup: cd $INSTALL_DIR && sudo -u $USER ./scripts/ssl-setup.sh"
    echo "3. Configure monitoring alerts"
    echo "4. Test backup and restore procedures"
    echo "5. Review security settings"
    echo ""
    echo -e "${GREEN}✅ Deployment completed successfully!${NC}"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --skip-security    Skip security hardening"
    echo "  --skip-ssl         Skip SSL certificate setup"
    echo "  --help             Show this help message"
    echo ""
    echo "This script deploys Mem0AI on Ubuntu VPS with:"
    echo "  - Docker and Docker Compose"
    echo "  - Nginx reverse proxy"
    echo "  - PostgreSQL, Redis, and Qdrant databases"
    echo "  - Prometheus and Grafana monitoring"
    echo "  - Automated backups"
    echo "  - Security hardening"
    echo "  - SSL certificates (Let's Encrypt)"
}

# Main deployment function
main() {
    local skip_security=false
    local skip_ssl=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-security)
                skip_security=true
                shift
                ;;
            --skip-ssl)
                skip_ssl=true
                shift
                ;;
            --help)
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
    
    echo -e "${GREEN}🚀 Mem0AI Production Deployment${NC}"
    echo "=================================="
    echo ""
    
    # Run deployment steps
    check_root
    check_requirements
    install_dependencies
    create_user
    setup_directories
    configure_environment
    setup_systemd
    configure_nginx
    
    if [[ "$skip_security" != true ]]; then
        run_security_hardening
    fi
    
    start_services
    
    if [[ "$skip_ssl" != true ]]; then
        setup_ssl
    fi
    
    run_health_checks
    show_summary
}

# Run main function with all arguments
main "$@"