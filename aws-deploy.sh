#!/bin/bash

# AWS Deployment Script for Trader AI
# This script sets up the application on AWS EC2

set -e

echo "🚀 Starting AWS deployment for Trader AI..."

# Update system packages
echo "📦 Updating system packages..."
sudo yum update -y

# Install Docker
echo "🐳 Installing Docker..."
sudo yum install -y docker
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -a -G docker ec2-user

# Install Docker Compose
echo "🔧 Installing Docker Compose..."
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install Git (if not already installed)
echo "📥 Installing Git..."
sudo yum install -y git

# Install curl (for health checks)
echo "🌐 Installing curl..."
sudo yum install -y curl

# Create application directory
echo "📁 Setting up application directory..."
cd /home/ec2-user
if [ -d "Trader_AI_V_0.1" ]; then
    echo "📂 Directory already exists, updating..."
    cd Trader_AI_V_0.1
    git pull origin main
else
    echo "📂 Cloning repository..."
    git clone <your-repo-url> Trader_AI_V_0.1
    cd Trader_AI_V_0.1
fi

# Create environment file
echo "⚙️ Creating environment configuration..."
if [ ! -f ".env.aws" ]; then
    echo "❌ .env.aws file not found. Please create it with your AWS-specific configuration."
    exit 1
fi

# Copy environment file
cp .env.aws .env

# Build and start services
echo "🏗️ Building and starting services..."
sudo docker-compose -f docker-compose.aws.yml down || true
sudo docker-compose -f docker-compose.aws.yml build --no-cache
sudo docker-compose -f docker-compose.aws.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# Health check
echo "🏥 Performing health check..."
if curl -f http://localhost:8000/health; then
    echo "✅ Backend is healthy!"
else
    echo "❌ Backend health check failed!"
    exit 1
fi

if curl -f http://localhost; then
    echo "✅ Frontend is healthy!"
else
    echo "❌ Frontend health check failed!"
    exit 1
fi

echo "🎉 Deployment completed successfully!"
echo "🌐 Application is running at:"
echo "   - Frontend: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)"
echo "   - Backend API: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8000"
echo "   - API Docs: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8000/docs"

# Show running containers
echo "📊 Running containers:"
sudo docker-compose -f docker-compose.aws.yml ps
