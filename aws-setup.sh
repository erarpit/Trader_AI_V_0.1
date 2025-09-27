#!/bin/bash

# AWS Setup Script for Trader AI
# This script helps you set up the AWS infrastructure and deploy the application

set -e

echo "🚀 AWS Setup for Trader AI"
echo "=========================="

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI is not installed. Please install it first:"
    echo "   curl 'https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip' -o 'awscliv2.zip'"
    echo "   unzip awscliv2.zip"
    echo "   sudo ./aws/install"
    exit 1
fi

# Check if Terraform is installed
if ! command -v terraform &> /dev/null; then
    echo "❌ Terraform is not installed. Please install it first:"
    echo "   wget https://releases.hashicorp.com/terraform/1.6.0/terraform_1.6.0_linux_amd64.zip"
    echo "   unzip terraform_1.6.0_linux_amd64.zip"
    echo "   sudo mv terraform /usr/local/bin/"
    exit 1
fi

# Check AWS credentials
echo "🔐 Checking AWS credentials..."
if ! aws sts get-caller-identity &> /dev/null; then
    echo "❌ AWS credentials not configured. Please run:"
    echo "   aws configure"
    exit 1
fi

echo "✅ AWS credentials configured"

# Get user input
echo ""
echo "📝 Please provide the following information:"
read -p "AWS Region (default: us-east-1): " aws_region
aws_region=${aws_region:-us-east-1}

read -p "Database Password: " db_password
if [ -z "$db_password" ]; then
    echo "❌ Database password is required"
    exit 1
fi

read -p "Secret Key for JWT: " secret_key
if [ -z "$secret_key" ]; then
    echo "❌ Secret key is required"
    exit 1
fi

read -p "Allowed Origins (comma-separated, default: http://localhost:3000): " allowed_origins
allowed_origins=${allowed_origins:-http://localhost:3000}

# Create terraform.tfvars
echo "📝 Creating terraform.tfvars..."
cat > terraform.tfvars << EOF
aws_region = "$aws_region"
db_password = "$db_password"
EOF

# Initialize and apply Terraform
echo "🏗️ Creating AWS infrastructure..."
terraform init
terraform plan
read -p "Do you want to apply these changes? (y/N): " confirm
if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
    terraform apply -auto-approve
else
    echo "❌ Infrastructure creation cancelled"
    exit 1
fi

# Get outputs
echo "📊 Getting infrastructure outputs..."
rds_endpoint=$(terraform output -raw rds_endpoint)
redis_endpoint=$(terraform output -raw redis_endpoint)
vpc_id=$(terraform output -raw vpc_id)

# Create environment file
echo "⚙️ Creating environment configuration..."
cat > .env.aws << EOF
# AWS Production Environment Configuration
DATABASE_URL=postgresql://trader_user:$db_password@$rds_endpoint:5432/trader_ai
REDIS_URL=redis://$redis_endpoint:6379

# NSE/BSE API Configuration
NSE_BASE_URL=https://www.nseindia.com/api
BSE_BASE_URL=https://api.bseindia.com/BseIndiaAPI/api
NSE_HEADERS_USER_AGENT=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36
NSE_HEADERS_ACCEPT=text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8
NSE_HEADERS_ACCEPT_LANGUAGE=en-US,en;q=0.5
NSE_HEADERS_ACCEPT_ENCODING=gzip, deflate
NSE_HEADERS_CONNECTION=keep-alive
NSE_HEADERS_UPGRADE_INSECURE_REQUESTS=1

# Angel One API (Optional - for live trading)
ANGEL_ONE_API_KEY=\${ANGEL_ONE_API_KEY}
ANGEL_ONE_CLIENT_ID=\${ANGEL_ONE_CLIENT_ID}
ANGEL_ONE_PIN=\${ANGEL_ONE_PIN}
ANGEL_ONE_TOTP_SECRET=\${ANGEL_ONE_TOTP_SECRET}

# WhatsApp Business API (Optional - for notifications)
WHATSAPP_API_KEY=\${WHATSAPP_API_KEY}
WHATSAPP_PHONE_ID=\${WHATSAPP_PHONE_ID}

# Twilio SMS API (Optional - for SMS notifications)
TWILIO_SID=\${TWILIO_SID}
TWILIO_TOKEN=\${TWILIO_TOKEN}
TWILIO_PHONE=\${TWILIO_PHONE}

# Application Configuration
SECRET_KEY=$secret_key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
DEBUG=False
HOST=0.0.0.0
PORT=8000

# WebSocket Configuration
WS_HOST=0.0.0.0
WS_PORT=8000

# Risk Management
MAX_POSITION_SIZE=100000
MAX_PORTFOLIO_RISK=0.02
DEFAULT_STOP_LOSS=0.05
DEFAULT_TAKE_PROFIT=0.10

# AWS Specific Configuration
AWS_REGION=$aws_region
AWS_ACCESS_KEY_ID=\${AWS_ACCESS_KEY_ID}
AWS_SECRET_ACCESS_KEY=\${AWS_SECRET_ACCESS_KEY}

# CORS Configuration for AWS
ALLOWED_ORIGINS=$allowed_origins

# Frontend Configuration
REACT_APP_API_URL=http://\$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8000
REACT_APP_WS_URL=ws://\$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8000
EOF

echo "✅ Environment configuration created"

# Deploy application
echo "🚀 Deploying application..."
chmod +x aws-deploy.sh
./aws-deploy.sh

echo ""
echo "🎉 Setup completed successfully!"
echo "================================"
echo "🌐 Your application is now running at:"
echo "   Frontend: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)"
echo "   Backend API: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8000"
echo "   API Docs: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8000/docs"
echo ""
echo "📊 Infrastructure details:"
echo "   RDS Endpoint: $rds_endpoint"
echo "   Redis Endpoint: $redis_endpoint"
echo "   VPC ID: $vpc_id"
echo ""
echo "🔧 To manage your infrastructure:"
echo "   terraform plan    # Review changes"
echo "   terraform apply   # Apply changes"
echo "   terraform destroy # Destroy infrastructure"
