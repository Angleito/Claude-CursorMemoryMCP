#!/bin/bash
# Advanced IPTables Configuration for Enterprise Security
# Memory Vector Database Firewall Rules

set -e

# Configuration variables
SSH_PORT="2222"
HTTP_PORT="80"
HTTPS_PORT="443"
API_PORT="8000"
METRICS_PORT="8080"
POSTGRES_PORT="5432"
REDIS_PORT="6379"

# Network ranges (customize as needed)
ADMIN_NETWORK="10.0.0.0/8"        # Admin access network
MONITORING_NETWORK="172.16.0.0/12" # Monitoring network
LOCAL_NETWORK="192.168.0.0/16"     # Local network

echo "Applying advanced IPTables rules for Memory Vector Database..."

# Flush existing rules
iptables -F
iptables -X
iptables -t nat -F
iptables -t nat -X
iptables -t mangle -F
iptables -t mangle -X
iptables -t raw -F
iptables -t raw -X

# Set default policies
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT ACCEPT

# Allow loopback traffic
iptables -A INPUT -i lo -j ACCEPT
iptables -A OUTPUT -o lo -j ACCEPT

# Allow established and related connections
iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT

# Drop invalid packets
iptables -A INPUT -m conntrack --ctstate INVALID -j DROP

# Protection against common attacks
echo "Setting up attack protection..."

# SYN flood protection
iptables -A INPUT -p tcp --syn -m limit --limit 1/s --limit-burst 3 -j ACCEPT
iptables -A INPUT -p tcp --syn -j DROP

# Port scan protection
iptables -A INPUT -p tcp --tcp-flags ALL ALL -j DROP
iptables -A INPUT -p tcp --tcp-flags ALL NONE -j DROP
iptables -A INPUT -p tcp --tcp-flags ALL FIN,PSH,URG -j DROP
iptables -A INPUT -p tcp --tcp-flags ALL SYN,RST,ACK,FIN,URG -j DROP
iptables -A INPUT -p tcp --tcp-flags SYN,RST SYN,RST -j DROP
iptables -A INPUT -p tcp --tcp-flags SYN,FIN SYN,FIN -j DROP

# XMAS scan protection
iptables -A INPUT -p tcp --tcp-flags ALL FIN,PSH,URG -j DROP

# NULL scan protection
iptables -A INPUT -p tcp --tcp-flags ALL NONE -j DROP

# Ping of death protection
iptables -A INPUT -p icmp --icmp-type echo-request -m limit --limit 1/second -j ACCEPT
iptables -A INPUT -p icmp --icmp-type echo-request -j DROP

# SSH Protection with rate limiting
echo "Setting up SSH protection..."
iptables -N SSH_CHECK
iptables -A SSH_CHECK -m recent --set --name SSH_ATTEMPT
iptables -A SSH_CHECK -m recent --update --seconds 60 --hitcount 4 --name SSH_ATTEMPT -j LOG --log-prefix "SSH_BRUTE_FORCE: "
iptables -A SSH_CHECK -m recent --update --seconds 60 --hitcount 4 --name SSH_ATTEMPT -j DROP
iptables -A SSH_CHECK -p tcp --dport $SSH_PORT -j ACCEPT

# Allow SSH from admin networks only
iptables -A INPUT -p tcp -s $ADMIN_NETWORK --dport $SSH_PORT -j SSH_CHECK
iptables -A INPUT -p tcp --dport $SSH_PORT -j LOG --log-prefix "SSH_UNAUTHORIZED: "
iptables -A INPUT -p tcp --dport $SSH_PORT -j DROP

# HTTP/HTTPS with DDoS protection
echo "Setting up web server protection..."
iptables -N HTTP_CHECK
iptables -A HTTP_CHECK -p tcp --dport $HTTP_PORT -m connlimit --connlimit-above 50 --connlimit-mask 24 -j LOG --log-prefix "HTTP_DDOS: "
iptables -A HTTP_CHECK -p tcp --dport $HTTP_PORT -m connlimit --connlimit-above 50 --connlimit-mask 24 -j DROP
iptables -A HTTP_CHECK -p tcp --dport $HTTP_PORT -m limit --limit 25/minute --limit-burst 100 -j ACCEPT

iptables -N HTTPS_CHECK
iptables -A HTTPS_CHECK -p tcp --dport $HTTPS_PORT -m connlimit --connlimit-above 50 --connlimit-mask 24 -j LOG --log-prefix "HTTPS_DDOS: "
iptables -A HTTPS_CHECK -p tcp --dport $HTTPS_PORT -m connlimit --connlimit-above 50 --connlimit-mask 24 -j DROP
iptables -A HTTPS_CHECK -p tcp --dport $HTTPS_PORT -m limit --limit 25/minute --limit-burst 100 -j ACCEPT

iptables -A INPUT -p tcp --dport $HTTP_PORT -j HTTP_CHECK
iptables -A INPUT -p tcp --dport $HTTPS_PORT -j HTTPS_CHECK

# API endpoint protection
echo "Setting up API protection..."
iptables -N API_CHECK
iptables -A API_CHECK -p tcp --dport $API_PORT -m connlimit --connlimit-above 20 --connlimit-mask 24 -j LOG --log-prefix "API_ABUSE: "
iptables -A API_CHECK -p tcp --dport $API_PORT -m connlimit --connlimit-above 20 --connlimit-mask 24 -j DROP
iptables -A API_CHECK -p tcp --dport $API_PORT -m limit --limit 10/minute --limit-burst 20 -j ACCEPT

iptables -A INPUT -p tcp --dport $API_PORT -j API_CHECK

# Database protection (localhost only)
echo "Setting up database protection..."
iptables -A INPUT -p tcp -s 127.0.0.1 --dport $POSTGRES_PORT -j ACCEPT
iptables -A INPUT -p tcp --dport $POSTGRES_PORT -j LOG --log-prefix "POSTGRES_UNAUTHORIZED: "
iptables -A INPUT -p tcp --dport $POSTGRES_PORT -j DROP

iptables -A INPUT -p tcp -s 127.0.0.1 --dport $REDIS_PORT -j ACCEPT
iptables -A INPUT -p tcp --dport $REDIS_PORT -j LOG --log-prefix "REDIS_UNAUTHORIZED: "
iptables -A INPUT -p tcp --dport $REDIS_PORT -j DROP

# Monitoring access (restricted)
echo "Setting up monitoring access..."
iptables -A INPUT -p tcp -s 127.0.0.1 --dport $METRICS_PORT -j ACCEPT
iptables -A INPUT -p tcp -s $MONITORING_NETWORK --dport $METRICS_PORT -j ACCEPT
iptables -A INPUT -p tcp --dport $METRICS_PORT -j LOG --log-prefix "METRICS_UNAUTHORIZED: "
iptables -A INPUT -p tcp --dport $METRICS_PORT -j DROP

# Geo-blocking (example for common attack sources)
echo "Setting up geo-blocking..."
# You would need to install ipset and geoip data for this to work
# iptables -A INPUT -m geoip --src-cc CN,RU,KP -j LOG --log-prefix "GEOBLOCK: "
# iptables -A INPUT -m geoip --src-cc CN,RU,KP -j DROP

# Rate limiting for ICMP
iptables -A INPUT -p icmp --icmp-type echo-request -m limit --limit 1/second -j ACCEPT
iptables -A INPUT -p icmp -j DROP

# Allow specific ICMP types
iptables -A INPUT -p icmp --icmp-type destination-unreachable -j ACCEPT
iptables -A INPUT -p icmp --icmp-type time-exceeded -j ACCEPT

# Log and drop everything else
iptables -A INPUT -j LOG --log-prefix "IPTABLES_DROPPED: " --log-level 4
iptables -A INPUT -j DROP

# Output rules (restrict outbound connections if needed)
echo "Setting up output rules..."

# Allow common outbound connections
iptables -A OUTPUT -p tcp --dport 80 -j ACCEPT   # HTTP
iptables -A OUTPUT -p tcp --dport 443 -j ACCEPT  # HTTPS
iptables -A OUTPUT -p tcp --dport 53 -j ACCEPT   # DNS TCP
iptables -A OUTPUT -p udp --dport 53 -j ACCEPT   # DNS UDP
iptables -A OUTPUT -p tcp --dport 25 -j ACCEPT   # SMTP
iptables -A OUTPUT -p tcp --dport 587 -j ACCEPT  # SMTP submission
iptables -A OUTPUT -p udp --dport 123 -j ACCEPT  # NTP

# Allow database replication if needed
# iptables -A OUTPUT -p tcp --dport 5432 -d BACKUP_SERVER_IP -j ACCEPT

# Block outbound to suspicious ports
iptables -A OUTPUT -p tcp --dport 6667 -j LOG --log-prefix "IRC_OUTBOUND: "
iptables -A OUTPUT -p tcp --dport 6667 -j DROP

# Log denied outbound connections
iptables -A OUTPUT -j LOG --log-prefix "OUTBOUND_DENIED: " --log-level 4

# Save rules
echo "Saving iptables rules..."
if command -v iptables-save >/dev/null 2>&1; then
    iptables-save > /etc/iptables/rules.v4
    echo "Rules saved to /etc/iptables/rules.v4"
else
    echo "Warning: iptables-save not found. Rules are temporary!"
fi

# Set up automatic rule restoration
echo "Setting up automatic rule restoration..."
cat > /etc/systemd/system/iptables-restore.service << 'EOF'
[Unit]
Description=Restore iptables rules
Before=network-pre.target
Wants=network-pre.target

[Service]
Type=oneshot
ExecStart=/sbin/iptables-restore /etc/iptables/rules.v4
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

systemctl enable iptables-restore.service

# Create rule monitoring script
cat > /usr/local/bin/monitor-firewall.sh << 'EOF'
#!/bin/bash
# Firewall monitoring script

LOG_FILE="/var/log/firewall-monitor.log"
ALERT_EMAIL="admin@yourdomain.com"

# Check for high rate of drops
DROPPED_COUNT=$(journalctl --since "5 minutes ago" | grep "IPTABLES_DROPPED" | wc -l)

if [ $DROPPED_COUNT -gt 100 ]; then
    echo "$(date): High number of dropped connections: $DROPPED_COUNT" >> $LOG_FILE
    echo "High firewall activity detected: $DROPPED_COUNT drops in last 5 minutes" | \
        mail -s "Firewall Alert" $ALERT_EMAIL
fi

# Check for brute force attempts
SSH_ATTACKS=$(journalctl --since "5 minutes ago" | grep "SSH_BRUTE_FORCE" | wc -l)

if [ $SSH_ATTACKS -gt 10 ]; then
    echo "$(date): SSH brute force attack detected: $SSH_ATTACKS attempts" >> $LOG_FILE
    echo "SSH brute force attack: $SSH_ATTACKS attempts in last 5 minutes" | \
        mail -s "SSH Attack Alert" $ALERT_EMAIL
fi
EOF

chmod +x /usr/local/bin/monitor-firewall.sh

# Add to crontab for monitoring
echo "*/5 * * * * /usr/local/bin/monitor-firewall.sh" | crontab -

echo "Advanced iptables configuration completed!"
echo "Current iptables rules:"
iptables -L -n -v

echo ""
echo "Configuration complete! Remember to:"
echo "1. Test all services are accessible"
echo "2. Adjust network ranges in the script as needed"
echo "3. Monitor logs in /var/log/kern.log for firewall activity"
echo "4. Set up log rotation for firewall logs"
echo "5. Configure email alerts for security events"