#!/bin/bash
set -euo pipefail

# SSL Certificate Setup Script for Mem0AI
# This script sets up Let's Encrypt SSL certificates using Certbot

# Load environment variables
if [[ -f .env ]]; then
    source .env
else
    echo "❌ .env file not found! Run generate-secrets.sh first."
    exit 1
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🔒 Mem0AI SSL Certificate Setup${NC}"
echo "=================================="

# Check required variables
if [[ -z "${DOMAIN:-}" ]]; then
    echo -e "${RED}❌ DOMAIN not set in .env file${NC}"
    exit 1
fi

if [[ -z "${SSL_EMAIL:-}" ]]; then
    echo -e "${RED}❌ SSL_EMAIL not set in .env file${NC}"
    exit 1
fi

DOMAIN=${DOMAIN}
EMAIL=${SSL_EMAIL}
STAGING=${SSL_STAGING:-false}

echo -e "${YELLOW}📋 Configuration:${NC}"
echo "   Domain: $DOMAIN"
echo "   Email: $EMAIL"
echo "   Staging: $STAGING"
echo ""

# Create necessary directories
mkdir -p ssl ssl-challenge
chmod 755 ssl ssl-challenge

# Create temporary nginx config for challenge
cat > config/nginx/sites-available/temp-ssl.conf << EOF
server {
    listen 80;
    listen [::]:80;
    server_name $DOMAIN www.$DOMAIN monitoring.$DOMAIN;
    
    location /.well-known/acme-challenge/ {
        root /var/www/certbot;
    }
    
    location / {
        return 200 'SSL setup in progress...';
        add_header Content-Type text/plain;
    }
}
EOF

# Enable temporary config
ln -sf ../sites-available/temp-ssl.conf config/nginx/sites-enabled/temp-ssl.conf

echo -e "${GREEN}🚀 Starting temporary nginx for SSL challenge...${NC}"

# Start nginx for challenge
docker-compose up -d nginx

# Wait for nginx to be ready
sleep 10

# Determine certbot arguments
CERTBOT_ARGS=""
if [[ "$STAGING" == "true" ]]; then
    CERTBOT_ARGS="--staging"
    echo -e "${YELLOW}⚠️  Using Let's Encrypt staging environment${NC}"
fi

echo -e "${GREEN}📜 Requesting SSL certificate...${NC}"

# Request certificate
docker-compose run --rm certbot certonly \
    --webroot \
    --webroot-path=/var/www/certbot \
    --email $EMAIL \
    --agree-tos \
    --no-eff-email \
    $CERTBOT_ARGS \
    -d $DOMAIN \
    -d www.$DOMAIN \
    -d monitoring.$DOMAIN

if [[ $? -eq 0 ]]; then
    echo -e "${GREEN}✅ SSL certificate obtained successfully!${NC}"
else
    echo -e "${RED}❌ Failed to obtain SSL certificate${NC}"
    exit 1
fi

# Update nginx configs with actual domain
echo -e "${GREEN}🔧 Updating nginx configuration...${NC}"

# Replace placeholder domain in nginx configs
find config/nginx/sites-available -name "*.conf" -type f -exec sed -i.bak "s/your-domain.com/$DOMAIN/g" {} \;
find config/nginx/sites-available -name "*.bak" -delete

# Enable actual sites
rm -f config/nginx/sites-enabled/temp-ssl.conf
ln -sf ../sites-available/mem0ai.conf config/nginx/sites-enabled/mem0ai.conf
ln -sf ../sites-available/monitoring.conf config/nginx/sites-enabled/monitoring.conf

# Test nginx configuration
echo -e "${GREEN}🧪 Testing nginx configuration...${NC}"
docker-compose exec nginx nginx -t

if [[ $? -eq 0 ]]; then
    echo -e "${GREEN}✅ Nginx configuration is valid${NC}"
    
    # Reload nginx
    echo -e "${GREEN}🔄 Reloading nginx...${NC}"
    docker-compose exec nginx nginx -s reload
    
    echo -e "${GREEN}🎉 SSL setup completed successfully!${NC}"
    echo ""
    echo -e "${YELLOW}📋 Next steps:${NC}"
    echo "1. Test your site: https://$DOMAIN"
    echo "2. Check SSL grade: https://www.ssllabs.com/ssltest/analyze.html?d=$DOMAIN"
    echo "3. Monitor certificate expiry (auto-renewal is configured)"
    echo ""
    echo -e "${GREEN}🔒 SSL certificate will auto-renew every 12 hours${NC}"
    
else
    echo -e "${RED}❌ Nginx configuration test failed${NC}"
    exit 1
fi

# Create certificate renewal test script
cat > scripts/test-ssl-renewal.sh << 'EOF'
#!/bin/bash
set -euo pipefail

echo "🔄 Testing SSL certificate renewal..."
docker-compose run --rm certbot renew --dry-run

if [[ $? -eq 0 ]]; then
    echo "✅ SSL renewal test passed"
else
    echo "❌ SSL renewal test failed"
    exit 1
fi
EOF

chmod +x scripts/test-ssl-renewal.sh

echo -e "${GREEN}📝 Created SSL renewal test script: scripts/test-ssl-renewal.sh${NC}"