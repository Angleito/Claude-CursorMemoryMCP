# Mem0AI Quick Reference Guide

## 🚀 Quick Commands

### Deployment
```bash
# Complete automated deployment
sudo ./deploy.sh

# Generate secrets only
sudo -u mem0ai ./scripts/generate-secrets.sh

# SSL setup only
sudo -u mem0ai ./scripts/ssl-setup.sh

# Security hardening only
sudo ./scripts/security-hardening.sh
```

### Service Management
```bash
# Start/Stop/Restart
sudo systemctl start mem0ai.service
sudo systemctl stop mem0ai.service
sudo systemctl restart mem0ai.service
sudo systemctl status mem0ai.service

# Enable/Disable auto-start
sudo systemctl enable mem0ai.service
sudo systemctl disable mem0ai.service

# View logs
sudo journalctl -u mem0ai.service -f
```

### Docker Management
```bash
# Navigate to project directory
cd /opt/mem0ai

# Container operations
sudo -u mem0ai docker-compose ps
sudo -u mem0ai docker-compose logs -f
sudo -u mem0ai docker-compose restart
sudo -u mem0ai docker-compose down
sudo -u mem0ai docker-compose up -d

# Individual container logs
sudo -u mem0ai docker-compose logs mem0ai
sudo -u mem0ai docker-compose logs postgres
sudo -u mem0ai docker-compose logs qdrant
sudo -u mem0ai docker-compose logs redis
```

### Health Checks
```bash
# Application health
curl -f http://localhost/health

# Service status
sudo systemctl is-active mem0ai.service

# Container health
cd /opt/mem0ai && sudo -u mem0ai docker-compose ps

# Database connectivity
sudo -u mem0ai docker-compose exec postgres pg_isready
sudo -u mem0ai docker-compose exec redis redis-cli ping
sudo -u mem0ai docker-compose exec qdrant curl http://localhost:6333/health
```

## 🔒 Security Commands

### Firewall Management
```bash
# UFW status and rules
sudo ufw status verbose
sudo ufw enable
sudo ufw disable

# Add/remove rules
sudo ufw allow 22/tcp
sudo ufw deny 22/tcp
sudo ufw delete allow 22/tcp
```

### Fail2ban Management
```bash
# Status and jails
sudo fail2ban-client status
sudo fail2ban-client status sshd

# Ban/unban IP
sudo fail2ban-client set sshd banip 192.168.1.100
sudo fail2ban-client set sshd unbanip 192.168.1.100
```

### Security Auditing
```bash
# Full security audit
sudo /usr/local/bin/security-audit.sh

# Manual security check
sudo /usr/local/bin/security-check.sh

# SSL certificate check
openssl s_client -connect yourdomain.com:443 -servername yourdomain.com
```

## 💾 Backup & Recovery

### Backup Operations
```bash
# Manual backup
cd /opt/mem0ai && sudo -u mem0ai ./scripts/backup.sh

# List available backups
sudo -u mem0ai ./scripts/restore.sh --list

# Test SSL renewal
sudo -u mem0ai ./scripts/test-ssl-renewal.sh
```

### Recovery Operations
```bash
# Restore from specific backup
sudo -u mem0ai ./scripts/restore.sh --file backup_name

# Interactive restore
sudo -u mem0ai ./scripts/restore.sh
```

## 📊 Monitoring

### Access URLs
- **Application**: `http://your-domain.com`
- **Health Check**: `http://your-domain.com/health`
- **Grafana**: `https://monitoring.your-domain.com`
- **Prometheus**: `https://monitoring.your-domain.com/prometheus/`

### Log Locations
```bash
# Application logs
tail -f /opt/mem0ai/logs/app.log

# Nginx logs
tail -f /opt/mem0ai/logs/nginx/access.log
tail -f /opt/mem0ai/logs/nginx/error.log

# System logs
sudo journalctl -u mem0ai.service -f
sudo tail -f /var/log/syslog
sudo tail -f /var/log/auth.log
```

### Performance Monitoring
```bash
# System resources
htop
free -h
df -h
iostat 1 5

# Docker resources
cd /opt/mem0ai && sudo -u mem0ai docker stats

# Network connections
sudo ss -tuln
sudo netstat -tuln
```

## 🔧 Configuration

### Important Files
```bash
# Main configuration
/opt/mem0ai/.env

# Docker configuration
/opt/mem0ai/docker-compose.yml
/opt/mem0ai/docker-compose.override.yml

# Nginx configuration
/opt/mem0ai/config/nginx/nginx.conf
/opt/mem0ai/config/nginx/sites-available/mem0ai.conf

# systemd services
/etc/systemd/system/mem0ai.service
/etc/systemd/system/mem0ai-backup.service
/etc/systemd/system/mem0ai-backup.timer
```

### Edit Configuration
```bash
# Environment variables
sudo -u mem0ai nano /opt/mem0ai/.env

# Docker Compose overrides
sudo -u mem0ai nano /opt/mem0ai/docker-compose.override.yml

# Nginx configuration
sudo -u mem0ai nano /opt/mem0ai/config/nginx/sites-available/mem0ai.conf

# Reload after changes
cd /opt/mem0ai && sudo -u mem0ai docker-compose up -d
sudo systemctl reload-or-restart mem0ai.service
```

## 🚨 Emergency Procedures

### Service Issues
```bash
# Complete restart
sudo systemctl stop mem0ai.service
cd /opt/mem0ai && sudo -u mem0ai docker-compose down
sudo -u mem0ai docker-compose up -d
sudo systemctl start mem0ai.service

# Force container rebuild
cd /opt/mem0ai && sudo -u mem0ai docker-compose down
sudo -u mem0ai docker-compose build --no-cache
sudo -u mem0ai docker-compose up -d
```

### SSL Issues
```bash
# Force certificate renewal
cd /opt/mem0ai
sudo -u mem0ai docker-compose run --rm certbot renew --force-renewal
sudo -u mem0ai docker-compose exec nginx nginx -s reload

# Debug SSL
openssl s_client -connect yourdomain.com:443 -servername yourdomain.com
```

### Database Issues
```bash
# PostgreSQL recovery
cd /opt/mem0ai
sudo -u mem0ai docker-compose exec postgres psql -U mem0ai -d mem0ai

# Redis recovery
sudo -u mem0ai docker-compose exec redis redis-cli
sudo -u mem0ai docker-compose restart redis

# Qdrant recovery
sudo -u mem0ai docker-compose exec qdrant curl http://localhost:6333/collections
sudo -u mem0ai docker-compose restart qdrant
```

## 📋 Environment Variables Quick Reference

### Required Variables
```bash
DOMAIN=your-domain.com
ADMIN_EMAIL=admin@your-domain.com
SSL_EMAIL=ssl@your-domain.com
OPENAI_API_KEY=sk-your-key-here
```

### Auto-Generated Variables
```bash
POSTGRES_PASSWORD=<generated>
REDIS_PASSWORD=<generated>
JWT_SECRET=<generated>
ENCRYPTION_KEY=<generated>
GRAFANA_PASSWORD=<generated>
```

### Optional Variables
```bash
# S3 Backup
S3_BUCKET_NAME=your-bucket
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
AWS_REGION=us-east-1

# Performance Tuning
MAX_MEMORY_SIZE=4096
VECTOR_DIMENSION=1536
WORKERS=4

# Security
CORS_ORIGIN=https://your-domain.com
RATE_LIMIT_PER_MINUTE=60
SESSION_TIMEOUT=3600
```

## 🔍 Troubleshooting Checklist

### Service Won't Start
- [ ] Check systemd service status
- [ ] Check Docker container status
- [ ] Check logs for errors
- [ ] Verify environment configuration
- [ ] Check disk space and memory

### SSL Certificate Issues
- [ ] Verify domain DNS configuration
- [ ] Check Let's Encrypt rate limits
- [ ] Verify nginx configuration
- [ ] Check firewall rules (ports 80/443)

### Database Connection Issues
- [ ] Check container health
- [ ] Verify environment variables
- [ ] Check network connectivity
- [ ] Review database logs

### Performance Issues
- [ ] Check resource usage (CPU/memory)
- [ ] Review Docker resource limits
- [ ] Check disk I/O
- [ ] Monitor network performance

### Backup Issues
- [ ] Check disk space
- [ ] Verify file permissions
- [ ] Check S3 credentials (if used)
- [ ] Review backup logs

## 📞 Getting Help

### Check These First
1. **Health endpoint**: `curl -f http://localhost/health`
2. **Service logs**: `sudo journalctl -u mem0ai.service -n 50`
3. **Container logs**: `cd /opt/mem0ai && sudo -u mem0ai docker-compose logs`
4. **System resources**: `htop` and `df -h`

### Useful Debug Commands
```bash
# Complete system status
sudo /usr/local/bin/security-audit.sh

# Docker environment
cd /opt/mem0ai && sudo -u mem0ai docker-compose config

# Network connectivity
sudo ss -tuln | grep -E "(80|443|8000|5432|6379|6333)"

# File permissions
ls -la /opt/mem0ai/
sudo -u mem0ai ls -la /opt/mem0ai/.env
```

---

**💡 Tip**: Bookmark this page for quick reference during operations and troubleshooting!