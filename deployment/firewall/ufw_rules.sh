#!/bin/bash
# UFW Firewall Configuration for Memory Vector Database
# Run with sudo privileges

set -e

echo "Configuring UFW firewall for Memory Vector Database..."

# Reset UFW to defaults
ufw --force reset

# Set default policies
ufw default deny incoming
ufw default allow outgoing

# Allow SSH on custom port (change 2222 to your SSH port)
ufw allow 2222/tcp comment 'SSH Access'

# Rate limit SSH to prevent brute force
ufw limit 2222/tcp

# Allow HTTP and HTTPS
ufw allow 80/tcp comment 'HTTP'
ufw allow 443/tcp comment 'HTTPS'

# Allow application API port (adjust as needed)
ufw allow 8000/tcp comment 'MemDB API'

# Allow monitoring port (Prometheus)
ufw allow from 127.0.0.1 to any port 8080 comment 'Prometheus Metrics'

# Allow PostgreSQL only from localhost
ufw allow from 127.0.0.1 to any port 5432 comment 'PostgreSQL Local'

# Allow Redis only from localhost  
ufw allow from 127.0.0.1 to any port 6379 comment 'Redis Local'

# Allow specific monitoring services (adjust IPs as needed)
# ufw allow from MONITORING_SERVER_IP to any port 9100 comment 'Node Exporter'

# Deny all other traffic on database ports
ufw deny 5432 comment 'Block External PostgreSQL'
ufw deny 6379 comment 'Block External Redis'

# Advanced rules for DDoS protection
# These require iptables modules to be loaded

# Enable logging
ufw logging on

# Enable firewall
ufw --force enable

echo "UFW firewall configuration completed!"
echo "Current UFW status:"
ufw status verbose

# Create backup of current rules
echo "Creating backup of UFW rules..."
cp /etc/ufw/user.rules /etc/ufw/user.rules.backup.$(date +%Y%m%d_%H%M%S)
cp /etc/ufw/user6.rules /etc/ufw/user6.rules.backup.$(date +%Y%m%d_%H%M%S)

echo "Firewall setup complete!"
echo "Remember to:"
echo "1. Test SSH access before closing current session"
echo "2. Adjust SSH port in the script if different from 2222"
echo "3. Add specific IP ranges for administrative access if needed"