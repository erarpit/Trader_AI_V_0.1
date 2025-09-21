#!/bin/bash

# Trader AI Setup Script
# This script sets up the complete Trader AI platform

set -e

echo "🚀 Setting up Trader AI Platform..."
echo "=================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ and try again."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 16+ and try again."
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install npm and try again."
    exit 1
fi

echo "✅ Prerequisites check passed"

# Create virtual environment for backend
echo "🔄 Setting up Python virtual environment..."
cd backend
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "🔄 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create models directory
mkdir -p models

# Go back to root directory
cd ..

# Install Node.js dependencies
echo "🔄 Installing Node.js dependencies..."
cd frontend
npm install

# Go back to root directory
cd ..

# Create .env file if it doesn't exist
if [ ! -f backend/.env ]; then
    echo "🔄 Creating environment configuration..."
    cp env.example backend/.env
    echo "✅ Environment file created. Please edit backend/.env with your configuration."
fi

# Create necessary directories
echo "🔄 Creating necessary directories..."
mkdir -p logs
mkdir -p data
mkdir -p models

echo ""
echo "🎉 Setup completed successfully!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Edit backend/.env with your configuration"
echo "2. Start Redis server: redis-server"
echo "3. Start the platform: python start.py"
echo ""
echo "Or use Docker:"
echo "docker-compose up -d"
echo ""
echo "Happy Trading! 📈"
