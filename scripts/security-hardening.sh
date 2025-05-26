#!/bin/bash
set -euo pipefail

# Ubuntu VPS Security Hardening Script for Mem0AI
# This script implements security best practices for production deployment

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}❌ This script must be run as root${NC}"
   exit 1
fi

echo -e "${GREEN}🔒 Ubuntu VPS Security Hardening for Mem0AI${NC}"
echo "=============================================="

# Function to log actions
log_action() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

# Update system packages
log_action "📦 Updating system packages..."
apt update && apt upgrade -y

# Install essential security packages
log_action "🛠️  Installing security packages..."
apt install -y \
    ufw \
    fail2ban \
    unattended-upgrades \
    apt-listchanges \
    logwatch \
    rkhunter \
    chkrootkit \
    lynis \
    aide \
    clamav \
    clamav-daemon

# Configure automatic security updates
log_action "🔄 Configuring automatic security updates..."
cat > /etc/apt/apt.conf.d/50unattended-upgrades << 'EOF'
Unattended-Upgrade::Allowed-Origins {
    "${distro_id}:${distro_codename}-security";
    "${distro_id}ESMApps:${distro_codename}-apps-security";
    "${distro_id}ESM:${distro_codename}-infra-security";
};
Unattended-Upgrade::Package-Blacklist {
    // "vim";
    // "libc6-dev";
    "linux-image*";
    "linux-headers*";
    "linux-*";
};
Unattended-Upgrade::DevRelease "false";
Unattended-Upgrade::Remove-Unused-Dependencies "true";
Unattended-Upgrade::Remove-New-Unused-Dependencies "true";
Unattended-Upgrade::Automatic-Reboot "false";
Unattended-Upgrade::Automatic-Reboot-Time "02:00";
EOF

cat > /etc/apt/apt.conf.d/20auto-upgrades << 'EOF'
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Download-Upgradeable-Packages "1";
APT::Periodic::AutocleanInterval "7";
APT::Periodic::Unattended-Upgrade "1";
EOF

# Configure UFW firewall
log_action "🔥 Configuring UFW firewall..."
ufw --force reset
ufw default deny incoming
ufw default allow outgoing

# Allow SSH (change port if needed)
ufw allow 22/tcp

# Allow HTTP and HTTPS
ufw allow 80/tcp
ufw allow 443/tcp

# Allow specific application ports (restrict by IP if needed)
# Uncomment and modify as needed:
# ufw allow from 10.0.0.0/8 to any port 3000  # Grafana
# ufw allow from 10.0.0.0/8 to any port 9090  # Prometheus

ufw --force enable

# Configure fail2ban
log_action "🚫 Configuring fail2ban..."
cat > /etc/fail2ban/jail.local << 'EOF'
[DEFAULT]
# Ban time in seconds (24 hours)
bantime = 86400

# Find time window
findtime = 600

# Number of failures before ban
maxretry = 3

# Ignore local IPs
ignoreip = 127.0.0.1/8 ::1

# Email notifications
destemail = root@localhost
sendername = Fail2Ban
mta = sendmail
action = %(action_mwl)s

[sshd]
enabled = true
port = ssh
filter = sshd
logpath = /var/log/auth.log
maxretry = 3

[nginx-http-auth]
enabled = true
filter = nginx-http-auth
port = http,https
logpath = /var/log/nginx/error.log

[nginx-limit-req]
enabled = true
filter = nginx-limit-req
port = http,https
logpath = /var/log/nginx/error.log
maxretry = 10
findtime = 600
bantime = 7200

[nginx-botsearch]
enabled = true
filter = nginx-botsearch
port = http,https
logpath = /var/log/nginx/access.log
maxretry = 2
EOF

# Create custom fail2ban filters
mkdir -p /etc/fail2ban/filter.d

cat > /etc/fail2ban/filter.d/nginx-botsearch.conf << 'EOF'
[Definition]
failregex = ^<HOST>.*GET.*(\.php|\.asp|\.exe|\.pl|\.cgi|\.scgi)
ignoreregex =
EOF

# SSH hardening
log_action "🔐 Hardening SSH configuration..."
cp /etc/ssh/sshd_config /etc/ssh/sshd_config.backup

cat > /etc/ssh/sshd_config << 'EOF'
# SSH Configuration for Mem0AI Production Server

# Protocol and Port
Protocol 2
Port 22

# Authentication
PermitRootLogin no
PasswordAuthentication no
ChallengeResponseAuthentication no
PubkeyAuthentication yes
AuthorizedKeysFile .ssh/authorized_keys

# Session settings
LoginGraceTime 30
MaxAuthTries 3
MaxSessions 3
ClientAliveInterval 300
ClientAliveCountMax 2

# Disable unused features
X11Forwarding no
AllowTcpForwarding no
GatewayPorts no
PermitTunnel no
PermitUserEnvironment no

# Logging
SyslogFacility AUTH
LogLevel VERBOSE

# Cryptography
Ciphers chacha20-poly1305@openssh.com,aes256-gcm@openssh.com,aes128-gcm@openssh.com,aes256-ctr,aes192-ctr,aes128-ctr
MACs hmac-sha2-256-etm@openssh.com,hmac-sha2-512-etm@openssh.com,hmac-sha2-256,hmac-sha2-512
KexAlgorithms curve25519-sha256@libssh.org,diffie-hellman-group16-sha512,diffie-hellman-group18-sha512

# Misc
UseDNS no
PermitEmptyPasswords no
EOF

# System hardening
log_action "🛡️  Applying system hardening..."

# Kernel parameter hardening
cat > /etc/sysctl.d/99-security.conf << 'EOF'
# IP Spoofing protection
net.ipv4.conf.default.rp_filter = 1
net.ipv4.conf.all.rp_filter = 1

# Ignore ICMP redirects
net.ipv4.conf.all.accept_redirects = 0
net.ipv6.conf.all.accept_redirects = 0
net.ipv4.conf.default.accept_redirects = 0
net.ipv6.conf.default.accept_redirects = 0

# Ignore send redirects
net.ipv4.conf.all.send_redirects = 0
net.ipv4.conf.default.send_redirects = 0

# Disable source packet routing
net.ipv4.conf.all.accept_source_route = 0
net.ipv6.conf.all.accept_source_route = 0
net.ipv4.conf.default.accept_source_route = 0
net.ipv6.conf.default.accept_source_route = 0

# Log Martians
net.ipv4.conf.all.log_martians = 1
net.ipv4.conf.default.log_martians = 1

# Ignore ICMP ping requests
net.ipv4.icmp_echo_ignore_all = 1

# Ignore Directed pings
net.ipv4.icmp_echo_ignore_broadcasts = 1

# Disable IPv6 if not needed
net.ipv6.conf.all.disable_ipv6 = 1
net.ipv6.conf.default.disable_ipv6 = 1

# TCP SYN flood protection
net.ipv4.tcp_syncookies = 1
net.ipv4.tcp_max_syn_backlog = 2048
net.ipv4.tcp_synack_retries = 2
net.ipv4.tcp_syn_retries = 5

# Memory protection
kernel.dmesg_restrict = 1
kernel.kptr_restrict = 2
kernel.yama.ptrace_scope = 1

# File system hardening
fs.protected_hardlinks = 1
fs.protected_symlinks = 1
fs.suid_dumpable = 0
EOF

sysctl -p /etc/sysctl.d/99-security.conf

# Secure shared memory
log_action "🔒 Securing shared memory..."
if ! grep -q "tmpfs /run/shm" /etc/fstab; then
    echo "tmpfs /run/shm tmpfs defaults,noexec,nosuid 0 0" >> /etc/fstab
fi

# Network security
log_action "🌐 Configuring network security..."

# Disable unused network protocols
cat > /etc/modprobe.d/blacklist-network.conf << 'EOF'
# Disable unused network protocols
blacklist dccp
blacklist sctp
blacklist rds
blacklist tipc
EOF

# File permissions hardening
log_action "📁 Hardening file permissions..."
chmod 644 /etc/passwd
chmod 600 /etc/shadow
chmod 644 /etc/group
chmod 600 /boot/grub/grub.cfg
chmod 600 /etc/ssh/sshd_config

# Remove unnecessary packages and services
log_action "🧹 Removing unnecessary packages..."
apt autoremove -y
apt autoclean

# Disable unnecessary services
log_action "🔇 Disabling unnecessary services..."
services_to_disable=(
    "bluetooth"
    "cups"
    "avahi-daemon"
    "whoopsie"
)

for service in "${services_to_disable[@]}"; do
    if systemctl is-enabled "$service" &>/dev/null; then
        systemctl disable "$service"
        systemctl stop "$service"
        echo "Disabled $service"
    fi
done

# Configure log rotation
log_action "📝 Configuring log rotation..."
cat > /etc/logrotate.d/mem0ai << 'EOF'
/opt/mem0ai/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    notifempty
    copytruncate
    create 644 mem0ai mem0ai
}
EOF

# Setup intrusion detection
log_action "🕵️  Setting up intrusion detection..."

# Initialize AIDE database
aide --init
mv /var/lib/aide/aide.db.new /var/lib/aide/aide.db

# Configure rkhunter
rkhunter --update
rkhunter --propupd

# Create security monitoring script
log_action "📊 Creating security monitoring script..."
cat > /usr/local/bin/security-check.sh << 'EOF'
#!/bin/bash
# Daily security check script

DATE=$(date '+%Y-%m-%d %H:%M:%S')
LOGFILE="/var/log/security-check.log"

echo "[$DATE] Starting security check..." >> $LOGFILE

# Check for rootkits
echo "[$DATE] Running rkhunter..." >> $LOGFILE
rkhunter --check --skip-keypress --report-warnings-only >> $LOGFILE 2>&1

# Check file integrity
echo "[$DATE] Running AIDE..." >> $LOGFILE
aide --check >> $LOGFILE 2>&1

# Check for failed login attempts
echo "[$DATE] Checking failed logins..." >> $LOGFILE
grep "Failed password" /var/log/auth.log | tail -10 >> $LOGFILE

# Check listening ports
echo "[$DATE] Checking listening ports..." >> $LOGFILE
ss -tuln >> $LOGFILE

echo "[$DATE] Security check completed." >> $LOGFILE
echo "----------------------------------------" >> $LOGFILE
EOF

chmod +x /usr/local/bin/security-check.sh

# Setup cron jobs for security tasks
log_action "⏰ Setting up security cron jobs..."
cat > /etc/cron.d/security << 'EOF'
# Daily security checks
0 2 * * * root /usr/local/bin/security-check.sh
0 3 * * * root /usr/bin/rkhunter --update && /usr/bin/rkhunter --check --skip-keypress --report-warnings-only
0 4 * * * root /usr/bin/freshclam && /usr/bin/clamscan -r /home /var/www /opt --quiet --infected --remove
EOF

# Create security audit script
log_action "🔍 Creating security audit script..."
cat > /usr/local/bin/security-audit.sh << 'EOF'
#!/bin/bash
# Security audit script

echo "=== Mem0AI Security Audit Report ==="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo ""

echo "=== System Information ==="
lsb_release -a
echo ""

echo "=== Kernel Version ==="
uname -r
echo ""

echo "=== Listening Ports ==="
ss -tuln
echo ""

echo "=== UFW Status ==="
ufw status verbose
echo ""

echo "=== Fail2ban Status ==="
fail2ban-client status
echo ""

echo "=== Recent Failed Logins ==="
grep "Failed password" /var/log/auth.log | tail -10
echo ""

echo "=== System Users ==="
awk -F: '($3 >= 1000) {print $1}' /etc/passwd
echo ""

echo "=== Docker Security ==="
docker version
docker info | grep -E "(Security|Logging|Storage)"
echo ""

echo "=== SSL Certificate Status ==="
if [[ -f /opt/mem0ai/ssl/live/*/cert.pem ]]; then
    openssl x509 -in /opt/mem0ai/ssl/live/*/cert.pem -text -noout | grep -E "(Not Before|Not After)"
fi
echo ""

echo "=== Disk Usage ==="
df -h
echo ""

echo "=== Memory Usage ==="
free -h
echo ""

echo "=== Load Average ==="
uptime
echo ""

echo "=== End of Report ==="
EOF

chmod +x /usr/local/bin/security-audit.sh

# Create admin user for Mem0AI (non-root)
log_action "👤 Creating mem0ai system user..."
if ! id -u mem0ai &>/dev/null; then
    useradd -r -s /bin/bash -d /opt/mem0ai -m mem0ai
    usermod -aG docker mem0ai
fi

# Set up directory structure
mkdir -p /opt/mem0ai
chown -R mem0ai:mem0ai /opt/mem0ai

# Restart services
log_action "🔄 Restarting security services..."
systemctl restart fail2ban
systemctl restart ssh
systemctl enable fail2ban
systemctl enable ufw

echo ""
echo -e "${GREEN}✅ Security hardening completed!${NC}"
echo "=================================="
echo ""
echo -e "${YELLOW}📋 Security Checklist:${NC}"
echo "✅ System packages updated"
echo "✅ UFW firewall configured and enabled"
echo "✅ Fail2ban configured for intrusion prevention"
echo "✅ SSH hardened (disable password auth, configure keys)"
echo "✅ Kernel parameters hardened"
echo "✅ File permissions secured"
echo "✅ Unnecessary services disabled"
echo "✅ Intrusion detection tools installed"
echo "✅ Automatic security updates enabled"
echo "✅ Security monitoring scheduled"
echo ""
echo -e "${YELLOW}🔑 Next Steps:${NC}"
echo "1. Set up SSH key authentication for your user"
echo "2. Test SSH access before logging out"
echo "3. Configure backup encryption keys"
echo "4. Review firewall rules for your specific needs"
echo "5. Set up monitoring alerts"
echo ""
echo -e "${YELLOW}📊 Security Tools:${NC}"
echo "- Run security audit: /usr/local/bin/security-audit.sh"
echo "- Manual security check: /usr/local/bin/security-check.sh"
echo "- Check fail2ban status: fail2ban-client status"
echo "- View firewall status: ufw status verbose"
echo ""
echo -e "${RED}⚠️  Important:${NC}"
echo "- Backup your SSH keys"
echo "- Test all functionality after hardening"
echo "- Keep security tools updated"
echo "- Regularly review logs in /var/log/"