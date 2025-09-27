#!/bin/bash

# Database Setup Script for Trader AI
# This script helps you choose between SQLite and PostgreSQL

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

print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

echo "🗄️  Database Setup for Trader AI"
echo "================================"
echo ""
echo "Choose your database configuration:"
echo "1. SQLite (Development/Testing) - Simple, file-based"
echo "2. PostgreSQL (Production) - Recommended for AWS"
echo "3. Use existing configuration"
echo ""

read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        print_header "Setting up SQLite Configuration"
        
        # Create SQLite environment file
        cat > .env << EOF
# SQLite Configuration for Development
DATABASE_URL=sqlite:///./trader_ai.db
REDIS_URL=redis://localhost:6379

# NSE/BSE API Configuration
NSE_BASE_URL=https://www.nseindia.com/api
BSE_BASE_URL=https://api.bseindia.com/BseIndiaAPI/api
NSE_HEADERS_USER_AGENT=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36
NSE_HEADERS_ACCEPT=text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8
NSE_HEADERS_ACCEPT_LANGUAGE=en-US,en;q=0.5
NSE_HEADERS_ACCEPT_ENCODING=gzip, deflate
NSE_HEADERS_CONNECTION=keep-alive
NSE_HEADERS_UPGRADE_INSECURE_REQUESTS=1

# Application Configuration
SECRET_KEY=$(openssl rand -base64 32)
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
DEBUG=True
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

# CORS Configuration
ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000

# Frontend Configuration
REACT_APP_API_URL=http://localhost:8000/api
REACT_APP_WS_URL=ws://localhost:8000/ws
EOF

        print_status "✅ SQLite configuration created"
        print_warning "Note: SQLite is suitable for development only. Use PostgreSQL for production."
        ;;
        
    2)
        print_header "Setting up PostgreSQL Configuration"
        
        echo "Please provide your PostgreSQL details:"
        echo ""
        
        read -p "Enter PostgreSQL host (e.g., localhost or RDS endpoint): " DB_HOST
        read -p "Enter PostgreSQL port (default: 5432): " DB_PORT
        DB_PORT=${DB_PORT:-5432}
        read -p "Enter database name (default: trader_ai): " DB_NAME
        DB_NAME=${DB_NAME:-trader_ai}
        read -p "Enter username (default: trader_user): " DB_USER
        DB_USER=${DB_USER:-trader_user}
        read -s -p "Enter password: " DB_PASSWORD
        echo ""
        
        # Create PostgreSQL environment file
        cat > .env << EOF
# PostgreSQL Configuration
DATABASE_URL=postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}
REDIS_URL=redis://localhost:6379

# NSE/BSE API Configuration
NSE_BASE_URL=https://www.nseindia.com/api
BSE_BASE_URL=https://api.bseindia.com/BseIndiaAPI/api
NSE_HEADERS_USER_AGENT=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36
NSE_HEADERS_ACCEPT=text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8
NSE_HEADERS_ACCEPT_LANGUAGE=en-US,en;q=0.5
NSE_HEADERS_ACCEPT_ENCODING=gzip, deflate
NSE_HEADERS_CONNECTION=keep-alive
NSE_HEADERS_UPGRADE_INSECURE_REQUESTS=1

# Application Configuration
SECRET_KEY=$(openssl rand -base64 32)
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
DEBUG=True
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

# CORS Configuration
ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000

# Frontend Configuration
REACT_APP_API_URL=http://localhost:8000/api
REACT_APP_WS_URL=ws://localhost:8000/ws
EOF

        print_status "✅ PostgreSQL configuration created"
        print_status "Database URL: postgresql://${DB_USER}:***@${DB_HOST}:${DB_PORT}/${DB_NAME}"
        ;;
        
    3)
        print_status "Using existing configuration"
        if [ -f ".env" ]; then
            print_status "Found existing .env file"
        else
            print_warning "No .env file found. Please create one manually."
        fi
        ;;
        
    *)
        print_error "Invalid choice. Please run the script again."
        exit 1
        ;;
esac

echo ""
print_header "Next Steps"
echo "1. Review your .env file: cat .env"
echo "2. Install dependencies: pip install -r requirements.txt"
echo "3. Start the application: python backend/main.py"
echo "4. For AWS deployment, use: ./configure-aws.sh"
echo ""

if [ "$choice" = "2" ]; then
    print_warning "Make sure your PostgreSQL server is running and accessible!"
fi
