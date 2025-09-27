#!/bin/bash

# AWS Deployment Script for Trader AI
# This script sets up the Trader AI application on AWS EC2

set -e

echo "🚀 Starting AWS deployment for Trader AI..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root. Please run as ec2-user."
   exit 1
fi

# Update system packages
print_status "Updating system packages..."
sudo yum update -y

# Install required packages
print_status "Installing required packages..."
sudo yum install -y docker git python3 python3-pip nodejs npm nginx

# Start and enable Docker
print_status "Starting Docker service..."
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -a -G docker ec2-user

# Install Docker Compose
print_status "Installing Docker Compose..."
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Create application directory
print_status "Setting up application directory..."
sudo mkdir -p /opt/trader_ai
sudo chown ec2-user:ec2-user /opt/trader_ai

# Copy application files
print_status "Copying application files..."
cp -r . /opt/trader_ai/
cd /opt/trader_ai

# Create log directory
print_status "Creating log directory..."
sudo mkdir -p /var/log/trader_ai
sudo chown ec2-user:ec2-user /var/log/trader_ai

# Set up environment variables
print_status "Setting up environment variables..."
if [ ! -f .env ]; then
    cp env.aws.production .env
    print_warning "Please update .env file with your actual AWS credentials and database endpoints"
fi

# Build and start services
print_status "Building Docker images..."
docker-compose -f docker-compose.aws.yml build

# Create systemd service for Trader AI
print_status "Creating systemd service..."
sudo tee /etc/systemd/system/trader-ai.service > /dev/null <<EOF
[Unit]
Description=Trader AI Application
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/trader_ai
ExecStart=/usr/local/bin/docker-compose -f docker-compose.aws.yml up -d
ExecStop=/usr/local/bin/docker-compose -f docker-compose.aws.yml down
TimeoutStartSec=0
User=ec2-user

[Install]
WantedBy=multi-user.target
EOF

# Configure Nginx
print_status "Configuring Nginx..."
sudo tee /etc/nginx/conf.d/trader-ai.conf > /dev/null <<EOF
upstream backend {
    server 127.0.0.1:8000;
}

upstream frontend {
    server 127.0.0.1:80;
}

server {
    listen 80;
    server_name _;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;

    # API routes
    location /api/ {
        proxy_pass http://backend;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # WebSocket routes
    location /ws {
        proxy_pass http://backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # Frontend routes
    location / {
        proxy_pass http://frontend;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    # Health check
    location /health {
        proxy_pass http://backend/health;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

# Start services
print_status "Starting services..."
sudo systemctl daemon-reload
sudo systemctl enable trader-ai
sudo systemctl start trader-ai

# Start Nginx
print_status "Starting Nginx..."
sudo systemctl start nginx
sudo systemctl enable nginx

# Configure firewall
print_status "Configuring firewall..."
sudo firewall-cmd --permanent --add-service=http
sudo firewall-cmd --permanent --add-service=https
sudo firewall-cmd --permanent --add-port=8000/tcp
sudo firewall-cmd --reload

# Wait for services to start
print_status "Waiting for services to start..."
sleep 30

# Check service status
print_status "Checking service status..."
if systemctl is-active --quiet trader-ai; then
    print_status "✅ Trader AI service is running"
else
    print_error "❌ Trader AI service failed to start"
    sudo systemctl status trader-ai
fi

if systemctl is-active --quiet nginx; then
    print_status "✅ Nginx is running"
else
    print_error "❌ Nginx failed to start"
    sudo systemctl status nginx
fi

# Display access information
print_status "🎉 Deployment completed!"
echo ""
echo "Access your application at:"
echo "  - Frontend: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)"
echo "  - API Documentation: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)/docs"
echo "  - Health Check: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)/health"
echo ""
echo "To view logs:"
echo "  - Application logs: sudo journalctl -u trader-ai -f"
echo "  - Docker logs: docker-compose -f docker-compose.aws.yml logs -f"
echo "  - Nginx logs: sudo tail -f /var/log/nginx/access.log"
echo ""
print_warning "Don't forget to:"
echo "  1. Update .env file with your actual AWS credentials"
echo "  2. Configure your domain name and SSL certificates"
echo "  3. Set up RDS and ElastiCache instances"
echo "  4. Update security groups to allow traffic on ports 80 and 443"
