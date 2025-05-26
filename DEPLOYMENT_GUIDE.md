# Mem0AI VPS Deployment Guide

## 📝 Pre-Deployment Checklist

### Server Requirements
- [ ] Ubuntu 20.04 LTS or 22.04 LTS VPS
- [ ] Minimum 2GB RAM (4GB+ recommended)
- [ ] Minimum 10GB storage (20GB+ recommended)
- [ ] Root or sudo access
- [ ] Public IP address

### Domain & DNS Setup
- [ ] Domain name registered
- [ ] DNS A record pointing to VPS IP
- [ ] Subdomain for monitoring (optional): `monitoring.yourdomain.com`

### API Keys & Credentials
- [ ] OpenAI API key
- [ ] Email address for SSL certificates
- [ ] Admin email address

## 🚀 Step-by-Step Deployment

### Step 1: Initial Server Setup

1. **Connect to your VPS**
   ```bash
   ssh root@your-vps-ip
   ```

2. **Update the system**
   ```bash
   apt update && apt upgrade -y
   ```

3. **Install Git**
   ```bash
   apt install -y git
   ```

### Step 2: Download Deployment Files

1. **Clone the repository**
   ```bash
   cd /opt
   git clone <your-repo-url> mem0ai-deploy
   cd mem0ai-deploy
   ```

2. **Make scripts executable**
   ```bash
   chmod +x *.sh scripts/*.sh
   ```

### Step 3: Quick Deployment (Recommended)

**Option A: Full Automated Deployment**
```bash
sudo ./deploy.sh
```

**Option B: Custom Deployment**
```bash
# Skip security hardening
sudo ./deploy.sh --skip-security

# Skip SSL setup (configure later)
sudo ./deploy.sh --skip-ssl

# Skip both
sudo ./deploy.sh --skip-security --skip-ssl
```

### Step 4: Configuration

The deployment script will pause for configuration. Edit the `.env` file:

```bash
nano /opt/mem0ai/.env
```

**Required settings:**
```env
DOMAIN=yourdomain.com
ADMIN_EMAIL=admin@yourdomain.com
SSL_EMAIL=ssl@yourdomain.com
OPENAI_API_KEY=sk-your-openai-api-key-here
CORS_ORIGIN=https://yourdomain.com
```

### Step 5: Verify Deployment

1. **Check service status**
   ```bash
   systemctl status mem0ai.service
   ```

2. **Test HTTP endpoint**
   ```bash
   curl -f http://localhost/health
   ```

3. **Check all containers**
   ```bash
   cd /opt/mem0ai
   sudo -u mem0ai docker-compose ps
   ```

## 🔧 Manual Deployment (Advanced)

If you prefer step-by-step manual control:

### Step 1: Install Dependencies

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install additional tools
sudo apt install -y ufw fail2ban nginx-utils
```

### Step 2: Create System User

```bash
# Create mem0ai user
sudo useradd -r -s /bin/bash -d /opt/mem0ai -m mem0ai
sudo usermod -aG docker mem0ai

# Copy files
sudo cp -r /opt/mem0ai-deploy/* /opt/mem0ai/
sudo chown -R mem0ai:mem0ai /opt/mem0ai
sudo chmod +x /opt/mem0ai/scripts/*.sh
```

### Step 3: Generate Configuration

```bash
cd /opt/mem0ai
sudo -u mem0ai ./scripts/generate-secrets.sh
```

Edit the configuration:
```bash
sudo -u mem0ai nano .env
```

### Step 4: Setup Resource Optimization

Choose your VPS size and edit the override file:

```bash
# Copy optimization template
sudo -u mem0ai cp config/optimization/docker-compose.override.yml .

# Edit for your VPS size (small/medium/large)
sudo -u mem0ai nano docker-compose.override.yml
```

### Step 5: Start Services

```bash
cd /opt/mem0ai
sudo -u mem0ai docker-compose up -d
```

### Step 6: Setup SSL Certificates

```bash
cd /opt/mem0ai
sudo -u mem0ai ./scripts/ssl-setup.sh
```

### Step 7: Configure Systemd

```bash
# Copy service files
sudo cp config/systemd/*.service /etc/systemd/system/
sudo cp config/systemd/*.timer /etc/systemd/system/

# Update paths in service files
sudo sed -i "s|/opt/mem0ai|/opt/mem0ai|g" /etc/systemd/system/mem0ai*.service
sudo sed -i "s|/opt/mem0ai|/opt/mem0ai|g" /etc/systemd/system/mem0ai*.timer

# Enable services
sudo systemctl daemon-reload
sudo systemctl enable mem0ai.service
sudo systemctl enable mem0ai-backup.timer
sudo systemctl start mem0ai.service
sudo systemctl start mem0ai-backup.timer
```

### Step 8: Security Hardening

```bash
cd /opt/mem0ai
sudo ./scripts/security-hardening.sh
```

## 🔍 Post-Deployment Verification

### 1. Service Health Checks

```bash
# Check systemd service
sudo systemctl status mem0ai.service

# Check Docker containers
cd /opt/mem0ai
sudo -u mem0ai docker-compose ps

# Check application health
curl -f http://localhost/health
```

### 2. SSL Certificate Verification

```bash
# Check certificate installation
openssl s_client -connect yourdomain.com:443 -servername yourdomain.com

# Test SSL Labs rating
# Visit: https://www.ssllabs.com/ssltest/analyze.html?d=yourdomain.com
```

### 3. Security Verification

```bash
# Run security audit
sudo /usr/local/bin/security-audit.sh

# Check firewall status
sudo ufw status verbose

# Check fail2ban status
sudo fail2ban-client status
```

### 4. Monitoring Setup

```bash
# Access Grafana dashboard
# URL: https://monitoring.yourdomain.com
# Username: admin
# Password: (check .env file for GRAFANA_PASSWORD)

# Check Prometheus metrics
# URL: https://monitoring.yourdomain.com/prometheus/
```

### 5. Backup Verification

```bash
# Run manual backup test
cd /opt/mem0ai
sudo -u mem0ai ./scripts/backup.sh

# List backups
sudo -u mem0ai ./scripts/restore.sh --list

# Test SSL renewal
sudo -u mem0ai ./scripts/test-ssl-renewal.sh
```

## 🐛 Troubleshooting Common Issues

### Issue 1: Containers Won't Start

**Symptoms:**
- Docker containers in "Restarting" state
- Health check failures

**Solution:**
```bash
# Check container logs
cd /opt/mem0ai
sudo -u mem0ai docker-compose logs

# Check resource usage
sudo -u mem0ai docker stats

# Restart services
sudo -u mem0ai docker-compose down
sudo -u mem0ai docker-compose up -d
```

### Issue 2: SSL Certificate Problems

**Symptoms:**
- SSL certificate errors
- Let's Encrypt failures

**Solution:**
```bash
# Check domain configuration
dig yourdomain.com

# Test with staging certificates first
nano .env
# Set SSL_STAGING=true

# Re-run SSL setup
cd /opt/mem0ai
sudo -u mem0ai ./scripts/ssl-setup.sh
```

### Issue 3: Database Connection Errors

**Symptoms:**
- Application can't connect to databases
- Database health checks fail

**Solution:**
```bash
# Check database containers
cd /opt/mem0ai
sudo -u mem0ai docker-compose exec postgres pg_isready
sudo -u mem0ai docker-compose exec redis redis-cli ping
sudo -u mem0ai docker-compose exec qdrant curl http://localhost:6333/health

# Check environment variables
sudo -u mem0ai docker-compose config
```

### Issue 4: High Resource Usage

**Symptoms:**
- Server running out of memory
- High CPU usage

**Solution:**
```bash
# Check resource usage
htop
sudo -u mem0ai docker stats

# Optimize for smaller VPS
cd /opt/mem0ai
sudo -u mem0ai nano docker-compose.override.yml
# Uncomment small VPS configuration

# Restart with new limits
sudo -u mem0ai docker-compose up -d
```

### Issue 5: Backup Failures

**Symptoms:**
- Backup script errors
- Missing backup files

**Solution:**
```bash
# Check backup logs
sudo journalctl -u mem0ai-backup.service -n 50

# Test backup manually
cd /opt/mem0ai
sudo -u mem0ai ./scripts/backup.sh

# Check disk space
df -h

# Check permissions
ls -la /opt/mem0ai/backups/
```

## 🔄 Maintenance Procedures

### Daily Maintenance
```bash
# Check service status
sudo systemctl status mem0ai.service

# Check backup status
sudo journalctl -u mem0ai-backup.service --since "yesterday"

# Monitor disk usage
df -h
```

### Weekly Maintenance
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Update Docker images
cd /opt/mem0ai
sudo -u mem0ai docker-compose pull
sudo -u mem0ai docker-compose up -d

# Check SSL certificate expiry
sudo -u mem0ai ./scripts/test-ssl-renewal.sh

# Review security logs
sudo /usr/local/bin/security-audit.sh
```

### Monthly Maintenance
```bash
# Full security audit
sudo /usr/local/bin/security-audit.sh

# Test backup restoration
sudo -u mem0ai ./scripts/restore.sh --list

# Clean up old backups and Docker images
cd /opt/mem0ai
sudo -u mem0ai docker system prune -f

# Review and optimize resource usage
sudo -u mem0ai docker stats
```

## 📊 Monitoring and Alerts

### Key Metrics to Monitor

1. **Application Metrics**
   - Response times
   - Error rates
   - Memory usage

2. **System Metrics**
   - CPU usage
   - Memory usage
   - Disk space

3. **Security Metrics**
   - Failed login attempts
   - Firewall blocks
   - SSL certificate expiry

### Setting Up Alerts

1. **Grafana Alerts**
   - Access Grafana dashboard
   - Configure notification channels
   - Set up alert rules

2. **Email Notifications**
   - Configure SMTP settings
   - Set up fail2ban email alerts
   - Configure backup failure notifications

## 📋 Configuration Reference

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `DOMAIN` | Your domain name | `example.com` |
| `ADMIN_EMAIL` | Admin email | `admin@example.com` |
| `SSL_EMAIL` | SSL certificate email | `ssl@example.com` |
| `OPENAI_API_KEY` | OpenAI API key | `sk-...` |
| `POSTGRES_PASSWORD` | PostgreSQL password | Auto-generated |
| `REDIS_PASSWORD` | Redis password | Auto-generated |
| `JWT_SECRET` | JWT signing secret | Auto-generated |
| `GRAFANA_PASSWORD` | Grafana admin password | Auto-generated |

### Port Configuration

| Service | Internal Port | External Port | Purpose |
|---------|---------------|---------------|---------|
| Nginx | 80, 443 | 80, 443 | HTTP/HTTPS |
| Mem0AI | 8000 | - | Application |
| PostgreSQL | 5432 | - | Database |
| Redis | 6379 | - | Cache |
| Qdrant | 6333 | - | Vector DB |
| Prometheus | 9090 | - | Metrics |
| Grafana | 3000 | - | Dashboard |

### File Locations

| Component | Location |
|-----------|----------|
| Application | `/opt/mem0ai/` |
| Configuration | `/opt/mem0ai/config/` |
| Logs | `/opt/mem0ai/logs/` |
| Backups | `/opt/mem0ai/backups/` |
| SSL Certificates | `/opt/mem0ai/ssl/` |
| systemd Services | `/etc/systemd/system/mem0ai*` |

## 🆘 Emergency Procedures

### Complete Service Restart
```bash
sudo systemctl stop mem0ai.service
cd /opt/mem0ai
sudo -u mem0ai docker-compose down
sudo -u mem0ai docker-compose up -d
sudo systemctl start mem0ai.service
```

### Emergency Backup
```bash
cd /opt/mem0ai
sudo -u mem0ai ./scripts/backup.sh
```

### Disaster Recovery
```bash
# Stop services
sudo systemctl stop mem0ai.service

# Restore from backup
cd /opt/mem0ai
sudo -u mem0ai ./scripts/restore.sh --file backup_name

# Start services
sudo systemctl start mem0ai.service
```

### SSL Certificate Emergency Renewal
```bash
cd /opt/mem0ai
sudo -u mem0ai docker-compose run --rm certbot renew --force-renewal
sudo -u mem0ai docker-compose exec nginx nginx -s reload
```

---

**Need Help?** Check the [README.md](README.md) for more detailed information or create an issue in the repository.