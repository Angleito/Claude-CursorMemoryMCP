# VPS Security Hardening Guide for Memory Vector Database

## Executive Summary

This comprehensive guide provides enterprise-grade security hardening procedures for Ubuntu VPS hosting the self-hosted memory vector database. Follow these steps to ensure maximum security for sensitive code memories.

## Table of Contents

1. [Initial Server Setup](#initial-server-setup)
2. [User Management and SSH Security](#user-management-and-ssh-security)
3. [Firewall Configuration](#firewall-configuration)
4. [System Updates and Package Management](#system-updates-and-package-management)
5. [File System Security](#file-system-security)
6. [Network Security](#network-security)
7. [Application Security](#application-security)
8. [Monitoring and Logging](#monitoring-and-logging)
9. [Backup and Recovery](#backup-and-recovery)
10. [Compliance and Auditing](#compliance-and-auditing)

## Initial Server Setup

### 1. Update System Packages

```bash
# Update package lists and upgrade system
sudo apt update && sudo apt upgrade -y

# Install essential security packages
sudo apt install -y fail2ban ufw unattended-upgrades apt-listchanges
sudo apt install -y lynis rkhunter chkrootkit aide
sudo apt install -y htop iotop nethogs vnstat
```

### 2. Configure Automatic Security Updates

```bash
# Configure unattended upgrades
sudo dpkg-reconfigure -plow unattended-upgrades

# Edit configuration
sudo nano /etc/apt/apt.conf.d/50unattended-upgrades
```

Add these configurations:
```
Unattended-Upgrade::Automatic-Reboot "true";
Unattended-Upgrade::Automatic-Reboot-Time "02:00";
Unattended-Upgrade::Remove-Unused-Dependencies "true";
Unattended-Upgrade::Remove-New-Unused-Dependencies "true";
```

## User Management and SSH Security

### 1. Create Non-Root User

```bash
# Create deployment user
sudo adduser memdb
sudo usermod -aG sudo memdb

# Create application user (no shell access)
sudo useradd -r -s /bin/false memdb-app
```

### 2. SSH Hardening

Create SSH configuration:
```bash
sudo cp /etc/ssh/sshd_config /etc/ssh/sshd_config.backup
sudo nano /etc/ssh/sshd_config
```

SSH Configuration (`/etc/ssh/sshd_config`):
```
# Basic Settings
Port 2222                          # Change default port
Protocol 2
AddressFamily inet

# Authentication
PermitRootLogin no
PasswordAuthentication no
PubkeyAuthentication yes
AuthorizedKeysFile .ssh/authorized_keys
PermitEmptyPasswords no
ChallengeResponseAuthentication no
UsePAM yes

# User Restrictions
AllowUsers memdb
DenyUsers root

# Connection Settings
MaxAuthTries 3
MaxSessions 2
ClientAliveInterval 300
ClientAliveCountMax 2
LoginGraceTime 60

# Security Features
X11Forwarding no
AllowAgentForwarding no
AllowTcpForwarding no
PermitTunnel no
PermitUserEnvironment no
```

### 3. SSH Key Management

```bash
# Generate SSH key pair (on local machine)
ssh-keygen -t ed25519 -C "memdb-server-key" -f ~/.ssh/memdb_server

# Copy public key to server
ssh-copy-id -i ~/.ssh/memdb_server.pub -p 2222 memdb@your-server-ip

# Set proper permissions
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
```

### 4. Two-Factor Authentication

```bash
# Install Google Authenticator
sudo apt install libpam-google-authenticator

# Configure for user
google-authenticator

# Update SSH PAM configuration
sudo nano /etc/pam.d/sshd
```

Add to `/etc/pam.d/sshd`:
```
auth required pam_google_authenticator.so
```

Update SSH config:
```
ChallengeResponseAuthentication yes
AuthenticationMethods publickey,keyboard-interactive
```

## Firewall Configuration

### 1. UFW (Uncomplicated Firewall) Setup

```bash
# Reset firewall
sudo ufw --force reset

# Default policies
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow SSH (custom port)
sudo ufw allow 2222/tcp comment 'SSH'

# Allow HTTP/HTTPS
sudo ufw allow 80/tcp comment 'HTTP'
sudo ufw allow 443/tcp comment 'HTTPS'

# Allow application port (if needed)
sudo ufw allow 8000/tcp comment 'MemDB API'

# Allow monitoring
sudo ufw allow 8080/tcp comment 'Prometheus'

# Rate limiting for SSH
sudo ufw limit 2222/tcp

# Enable firewall
sudo ufw enable
```

### 2. Advanced IPTables Rules

Create script `/etc/iptables/rules.v4`:
```bash
#!/bin/bash
# Advanced firewall rules

# Flush existing rules
iptables -F
iptables -X
iptables -Z

# Default policies
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT ACCEPT

# Allow loopback
iptables -A INPUT -i lo -j ACCEPT
iptables -A OUTPUT -o lo -j ACCEPT

# Allow established connections
iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT

# Rate limiting for SSH
iptables -A INPUT -p tcp --dport 2222 -m conntrack --ctstate NEW -m recent --set --name SSH
iptables -A INPUT -p tcp --dport 2222 -m conntrack --ctstate NEW -m recent --update --seconds 60 --hitcount 4 --name SSH -j DROP
iptables -A INPUT -p tcp --dport 2222 -j ACCEPT

# HTTP/HTTPS with rate limiting
iptables -A INPUT -p tcp --dport 80 -m connlimit --connlimit-above 20 -j DROP
iptables -A INPUT -p tcp --dport 443 -m connlimit --connlimit-above 20 -j DROP
iptables -A INPUT -p tcp --dport 80 -j ACCEPT
iptables -A INPUT -p tcp --dport 443 -j ACCEPT

# Application port with rate limiting
iptables -A INPUT -p tcp --dport 8000 -m connlimit --connlimit-above 10 -j DROP
iptables -A INPUT -p tcp --dport 8000 -j ACCEPT

# Drop invalid packets
iptables -A INPUT -m conntrack --ctstate INVALID -j DROP

# Protection against common attacks
iptables -A INPUT -p tcp ! --syn -m conntrack --ctstate NEW -j DROP
iptables -A INPUT -p tcp --tcp-flags ALL ALL -j DROP
iptables -A INPUT -p tcp --tcp-flags ALL NONE -j DROP

# Log dropped packets
iptables -A INPUT -j LOG --log-prefix "IPTABLES-DROPPED: "
iptables -A INPUT -j DROP
```

## System Updates and Package Management

### 1. Package Security

```bash
# Install apt-listbugs for bug checking
sudo apt install apt-listbugs

# Configure APT security settings
sudo nano /etc/apt/apt.conf.d/99security
```

Add to security configuration:
```
APT::Get::AllowUnauthenticated "false";
APT::Get::AutomaticRemove "true";
Acquire::AllowInsecureRepositories "false";
Acquire::AllowDowngradeToInsecureRepositories "false";
```

### 2. Kernel Security

```bash
# Install and configure AppArmor
sudo apt install apparmor apparmor-utils
sudo systemctl enable apparmor
sudo systemctl start apparmor

# Check AppArmor status
sudo aa-status
```

## File System Security

### 1. Mount Options

Update `/etc/fstab` with secure mount options:
```
/dev/sda1 / ext4 defaults,nodev,nosuid,noexec 0 1
/dev/sda2 /tmp ext4 defaults,nodev,nosuid,noexec 0 2
/dev/sda3 /var ext4 defaults,nodev,nosuid 0 2
/dev/sda4 /home ext4 defaults,nodev,nosuid 0 2
```

### 2. File Permissions

```bash
# Secure important directories
sudo chmod 750 /etc/ssh
sudo chmod 600 /etc/ssh/ssh_host_*
sudo chmod 644 /etc/ssh/*.pub
sudo chmod 600 /etc/shadow
sudo chmod 644 /etc/passwd

# Create secure directories for application
sudo mkdir -p /opt/memdb/{data,logs,config,backups}
sudo chown -R memdb-app:memdb-app /opt/memdb
sudo chmod -R 750 /opt/memdb
sudo chmod 700 /opt/memdb/data
sudo chmod 700 /opt/memdb/config
```

### 3. File Integrity Monitoring

```bash
# Install and configure AIDE
sudo apt install aide aide-common
sudo aideinit
sudo mv /var/lib/aide/aide.db.new /var/lib/aide/aide.db

# Create daily check script
sudo nano /etc/cron.daily/aide-check
```

AIDE check script:
```bash
#!/bin/bash
/usr/bin/aide --check --config=/etc/aide/aide.conf | mail -s "AIDE Check Report" admin@yourdomain.com
```

## Network Security

### 1. Disable Unused Services

```bash
# List all services
systemctl list-unit-files --type=service

# Disable unnecessary services
sudo systemctl disable bluetooth
sudo systemctl disable cups
sudo systemctl disable avahi-daemon
sudo systemctl disable whoopsie
```

### 2. Network Parameter Tuning

Create `/etc/sysctl.d/99-security.conf`:
```
# IP Spoofing protection
net.ipv4.conf.default.rp_filter = 1
net.ipv4.conf.all.rp_filter = 1

# Ignore ICMP ping requests
net.ipv4.icmp_echo_ignore_all = 1

# Ignore ICMP redirects
net.ipv4.conf.all.accept_redirects = 0
net.ipv6.conf.all.accept_redirects = 0

# Ignore source-routed packets
net.ipv4.conf.all.accept_source_route = 0
net.ipv6.conf.all.accept_source_route = 0

# Log Martian packets
net.ipv4.conf.all.log_martians = 1

# Ignore broadcast requests
net.ipv4.icmp_echo_ignore_broadcasts = 1

# Bad error message protection
net.ipv4.icmp_ignore_bogus_error_responses = 1

# SYN flood protection
net.ipv4.tcp_syncookies = 1
net.ipv4.tcp_max_syn_backlog = 2048
net.ipv4.tcp_synack_retries = 2
net.ipv4.tcp_syn_retries = 5

# TCP hardening
net.ipv4.tcp_timestamps = 0
net.ipv4.tcp_sack = 0

# Kernel hardening
kernel.exec-shield = 1
kernel.randomize_va_space = 2
```

Apply settings:
```bash
sudo sysctl -p /etc/sysctl.d/99-security.conf
```

## Application Security

### 1. Application User Security

```bash
# Create systemd service for the application
sudo nano /etc/systemd/system/memdb.service
```

Systemd service file:
```ini
[Unit]
Description=Memory Vector Database
After=network.target postgresql.service redis.service
Wants=postgresql.service redis.service

[Service]
Type=exec
User=memdb-app
Group=memdb-app
WorkingDirectory=/opt/memdb
Environment=PATH=/opt/memdb/venv/bin
Environment="PYTHONPATH=/opt/memdb"
ExecStart=/opt/memdb/venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=3

# Security settings
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/opt/memdb/data /opt/memdb/logs
CapabilityBoundingSet=
AmbientCapabilities=
SystemCallFilter=~@clock @debug @module @mount @obsolete @reboot @swap @cpu-emulation @privileged

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096

[Install]
WantedBy=multi-user.target
```

### 2. SSL/TLS Configuration

```bash
# Install certbot for Let's Encrypt
sudo apt install certbot python3-certbot-nginx

# Generate SSL certificate
sudo certbot certonly --standalone -d yourdomain.com

# Set up certificate renewal
sudo crontab -e
```

Add to crontab:
```
0 12 * * * /usr/bin/certbot renew --quiet
```

### 3. Nginx Reverse Proxy Security

Install and configure Nginx:
```bash
sudo apt install nginx
sudo nano /etc/nginx/sites-available/memdb
```

Nginx configuration:
```nginx
server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
    ssl_session_timeout 1d;
    ssl_session_cache shared:MozTLS:10m;
    ssl_session_tickets off;

    # Modern configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305:DHE-RSA-AES128-GCM-SHA256:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;

    # HSTS
    add_header Strict-Transport-Security "max-age=63072000" always;

    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Referrer-Policy "strict-origin-when-cross-origin";
    add_header Content-Security-Policy "default-src 'self'";

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;

    # Proxy settings
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 5s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        # Buffer settings
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
    }

    # Monitoring endpoint
    location /metrics {
        proxy_pass http://127.0.0.1:8080;
        allow 127.0.0.1;
        deny all;
    }
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name yourdomain.com;
    return 301 https://$server_name$request_uri;
}
```

## Monitoring and Logging

### 1. Log Management

```bash
# Configure rsyslog for centralized logging
sudo nano /etc/rsyslog.d/49-memdb.conf
```

Rsyslog configuration:
```
# Memory DB application logs
if $programname == 'memdb' then /var/log/memdb/application.log
& stop

# Security logs
auth,authpriv.*                 /var/log/auth.log
kern.*                          /var/log/kern.log
mail.*                          /var/log/mail.log
```

### 2. Fail2ban Configuration

```bash
sudo nano /etc/fail2ban/jail.local
```

Fail2ban configuration:
```ini
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 3
ignoreip = 127.0.0.1/8

[sshd]
enabled = true
port = 2222
filter = sshd
logpath = /var/log/auth.log
maxretry = 3

[nginx-http-auth]
enabled = true
filter = nginx-http-auth
logpath = /var/log/nginx/error.log

[memdb-api]
enabled = true
filter = memdb-api
logpath = /var/log/memdb/application.log
maxretry = 5
```

Create custom filter `/etc/fail2ban/filter.d/memdb-api.conf`:
```ini
[Definition]
failregex = ^.*rate_limit_exceeded.*client_ip:<HOST>.*$
            ^.*authentication_failed.*client_ip:<HOST>.*$
ignoreregex =
```

## Backup and Recovery

### 1. Automated Backup Script

Create `/opt/memdb/scripts/backup.sh`:
```bash
#!/bin/bash

# Configuration
BACKUP_DIR="/opt/memdb/backups"
DB_NAME="memdb"
RETENTION_DAYS=30
ENCRYPTION_KEY="/opt/memdb/config/backup.key"

# Create timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="${BACKUP_DIR}/memdb_backup_${TIMESTAMP}.sql.gpg"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Database backup
pg_dump "$DB_NAME" | gpg --symmetric --cipher-algo AES256 --compress-algo 1 --s2k-mode 3 --s2k-digest-algo SHA512 --s2k-count 65536 --quiet --output "$BACKUP_FILE" --passphrase-file "$ENCRYPTION_KEY"

# Application data backup
tar -czf "${BACKUP_DIR}/memdb_data_${TIMESTAMP}.tar.gz" /opt/memdb/data/

# Cleanup old backups
find "$BACKUP_DIR" -name "memdb_backup_*.sql.gpg" -mtime +$RETENTION_DAYS -delete
find "$BACKUP_DIR" -name "memdb_data_*.tar.gz" -mtime +$RETENTION_DAYS -delete

# Log backup completion
logger "MemDB backup completed: $BACKUP_FILE"
```

### 2. Backup Cron Job

```bash
sudo crontab -u memdb-app -e
```

Add to crontab:
```
# Daily backup at 2 AM
0 2 * * * /opt/memdb/scripts/backup.sh

# Weekly full system backup
0 3 * * 0 /opt/memdb/scripts/full_backup.sh
```

## Security Maintenance Checklist

### Daily Tasks
- [ ] Check system logs for anomalies
- [ ] Monitor resource usage
- [ ] Verify backup completion
- [ ] Check fail2ban status

### Weekly Tasks
- [ ] Review security logs
- [ ] Update system packages
- [ ] Check SSL certificate status
- [ ] Run security scans

### Monthly Tasks
- [ ] Full security audit with Lynis
- [ ] Review and rotate API keys
- [ ] Update firewall rules if needed
- [ ] Test backup restoration

### Quarterly Tasks
- [ ] Penetration testing
- [ ] Review and update security policies
- [ ] Audit user access and permissions
- [ ] Update incident response procedures

## Emergency Response Procedures

### 1. Security Incident Response

```bash
# Immediate containment
sudo ufw enable
sudo ufw default deny incoming
sudo systemctl stop memdb
sudo systemctl stop nginx

# Collect evidence
sudo tar -czf /tmp/incident_$(date +%Y%m%d_%H%M%S).tar.gz /var/log/ /opt/memdb/logs/

# Network isolation (if needed)
sudo iptables -A INPUT -j DROP
sudo iptables -A OUTPUT -j DROP
```

### 2. Recovery Procedures

```bash
# System recovery from backup
sudo systemctl stop memdb
sudo systemctl stop postgresql

# Restore database
gpg --decrypt --quiet --passphrase-file /opt/memdb/config/backup.key backup_file.sql.gpg | psql memdb

# Restore application data
sudo tar -xzf memdb_data_backup.tar.gz -C /

# Restart services
sudo systemctl start postgresql
sudo systemctl start memdb
sudo systemctl start nginx
```

## Compliance and Auditing

### GDPR Compliance Configuration

```bash
# Data retention policy enforcement
sudo nano /opt/memdb/scripts/gdpr_cleanup.sh
```

GDPR cleanup script:
```bash
#!/bin/bash
# GDPR data retention cleanup

# Remove data older than retention period
python3 /opt/memdb/scripts/gdpr_cleanup.py

# Log cleanup action
logger "GDPR data retention cleanup completed"
```

### Security Audit Commands

```bash
# Run comprehensive security audit
sudo lynis audit system

# Check for rootkits
sudo rkhunter --check

# File integrity check
sudo aide --check

# Network security scan
nmap -sS -sV -O localhost
```

This comprehensive security hardening guide ensures enterprise-grade protection for your memory vector database deployment.