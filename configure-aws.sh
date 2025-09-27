#!/bin/bash

# AWS Configuration Script for Trader AI
# This script helps configure environment variables for AWS deployment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_prompt() {
    echo -e "${BLUE}[PROMPT]${NC} $1"
}

echo "🔧 AWS Configuration Setup for Trader AI"
echo "========================================"
echo ""

# Get current public IP
PUBLIC_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo "unknown")
PRIVATE_IP=$(hostname -I | awk '{print $1}')

print_status "Detected server information:"
echo "  - Public IP: $PUBLIC_IP"
echo "  - Private IP: $PRIVATE_IP"
echo ""

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    print_status "Creating .env file from template..."
    cp env.aws.production .env
fi

# Function to update env variable
update_env() {
    local key=$1
    local value=$2
    local file=$3
    
    if grep -q "^$key=" "$file"; then
        sed -i "s|^$key=.*|$key=$value|" "$file"
    else
        echo "$key=$value" >> "$file"
    fi
}

# Get user input for configuration
echo "Please provide the following information:"
echo ""

# Database configuration
print_prompt "Enter your RDS PostgreSQL endpoint (e.g., trader-ai-db.abc123.us-east-1.rds.amazonaws.com):"
read -r DB_ENDPOINT

print_prompt "Enter your RDS database password:"
read -s DB_PASSWORD
echo ""

# Redis configuration
print_prompt "Enter your ElastiCache Redis endpoint (e.g., trader-ai-redis.abc123.cache.amazonaws.com):"
read -r REDIS_ENDPOINT

# AWS credentials
print_prompt "Enter your AWS region (e.g., us-east-1):"
read -r AWS_REGION

print_prompt "Enter your AWS Access Key ID:"
read -r AWS_ACCESS_KEY_ID

print_prompt "Enter your AWS Secret Access Key:"
read -s AWS_SECRET_ACCESS_KEY
echo ""

# Secret key for JWT
print_prompt "Enter a secret key for JWT tokens (or press Enter to generate one):"
read -r SECRET_KEY
if [ -z "$SECRET_KEY" ]; then
    SECRET_KEY=$(openssl rand -base64 32)
    print_status "Generated secret key: $SECRET_KEY"
fi

# Domain configuration
print_prompt "Enter your domain name (or press Enter to use IP address):"
read -r DOMAIN_NAME

if [ -z "$DOMAIN_NAME" ]; then
    DOMAIN_NAME="$PUBLIC_IP"
    print_warning "Using IP address: $DOMAIN_NAME"
fi

# Update environment variables
print_status "Updating environment variables..."

update_env "DB_ENDPOINT" "$DB_ENDPOINT" ".env"
update_env "DB_PASSWORD" "$DB_PASSWORD" ".env"
update_env "REDIS_ENDPOINT" "$REDIS_ENDPOINT" ".env"
update_env "AWS_REGION" "$AWS_REGION" ".env"
update_env "AWS_ACCESS_KEY_ID" "$AWS_ACCESS_KEY_ID" ".env"
update_env "AWS_SECRET_ACCESS_KEY" "$AWS_SECRET_ACCESS_KEY" ".env"
update_env "SECRET_KEY" "$SECRET_KEY" ".env"

# Update URLs based on domain
if [[ "$DOMAIN_NAME" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    # IP address
    update_env "ALLOWED_ORIGINS" "http://$DOMAIN_NAME,http://$PRIVATE_IP" ".env"
    update_env "REACT_APP_API_URL" "http://$DOMAIN_NAME:8000/api" ".env"
    update_env "REACT_APP_WS_URL" "ws://$DOMAIN_NAME:8000/ws" ".env"
else
    # Domain name
    update_env "ALLOWED_ORIGINS" "https://$DOMAIN_NAME,https://www.$DOMAIN_NAME,http://$PRIVATE_IP" ".env"
    update_env "REACT_APP_API_URL" "https://$DOMAIN_NAME/api" ".env"
    update_env "REACT_APP_WS_URL" "wss://$DOMAIN_NAME/ws" ".env"
fi

# Optional configurations
echo ""
print_prompt "Do you want to configure Angel One API? (y/n):"
read -r configure_angel_one

if [ "$configure_angel_one" = "y" ] || [ "$configure_angel_one" = "Y" ]; then
    print_prompt "Enter Angel One API Key:"
    read -r ANGEL_ONE_API_KEY
    print_prompt "Enter Angel One Client ID:"
    read -r ANGEL_ONE_CLIENT_ID
    print_prompt "Enter Angel One PIN:"
    read -r ANGEL_ONE_PIN
    print_prompt "Enter Angel One TOTP Secret:"
    read -r ANGEL_ONE_TOTP_SECRET
    
    update_env "ANGEL_ONE_API_KEY" "$ANGEL_ONE_API_KEY" ".env"
    update_env "ANGEL_ONE_CLIENT_ID" "$ANGEL_ONE_CLIENT_ID" ".env"
    update_env "ANGEL_ONE_PIN" "$ANGEL_ONE_PIN" ".env"
    update_env "ANGEL_ONE_TOTP_SECRET" "$ANGEL_ONE_TOTP_SECRET" ".env"
fi

print_prompt "Do you want to configure WhatsApp notifications? (y/n):"
read -r configure_whatsapp

if [ "$configure_whatsapp" = "y" ] || [ "$configure_whatsapp" = "Y" ]; then
    print_prompt "Enter WhatsApp API Key:"
    read -r WHATSAPP_API_KEY
    print_prompt "Enter WhatsApp Phone ID:"
    read -r WHATSAPP_PHONE_ID
    
    update_env "WHATSAPP_API_KEY" "$WHATSAPP_API_KEY" ".env"
    update_env "WHATSAPP_PHONE_ID" "$WHATSAPP_PHONE_ID" ".env"
fi

print_prompt "Do you want to configure Twilio SMS? (y/n):"
read -r configure_twilio

if [ "$configure_twilio" = "y" ] || [ "$configure_twilio" = "Y" ]; then
    print_prompt "Enter Twilio SID:"
    read -r TWILIO_SID
    print_prompt "Enter Twilio Token:"
    read -r TWILIO_TOKEN
    print_prompt "Enter Twilio Phone Number:"
    read -r TWILIO_PHONE
    
    update_env "TWILIO_SID" "$TWILIO_SID" ".env"
    update_env "TWILIO_TOKEN" "$TWILIO_TOKEN" ".env"
    update_env "TWILIO_PHONE" "$TWILIO_PHONE" ".env"
fi

# Set proper permissions
chmod 600 .env

print_status "✅ Configuration completed!"
echo ""
echo "Configuration summary:"
echo "  - Database: $DB_ENDPOINT"
echo "  - Redis: $REDIS_ENDPOINT"
echo "  - AWS Region: $AWS_REGION"
echo "  - Domain: $DOMAIN_NAME"
echo "  - API URL: $(grep REACT_APP_API_URL .env | cut -d'=' -f2)"
echo "  - WebSocket URL: $(grep REACT_APP_WS_URL .env | cut -d'=' -f2)"
echo ""
print_warning "Next steps:"
echo "  1. Make sure your RDS and ElastiCache instances are running"
echo "  2. Update security groups to allow traffic from this EC2 instance"
echo "  3. Run: ./deploy-aws.sh"
echo "  4. Configure SSL certificates if using a domain name"
