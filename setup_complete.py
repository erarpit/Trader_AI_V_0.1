#!/usr/bin/env python3
"""
Complete Trader AI Setup Script
Sets up the entire platform with all features including:
- Advanced candlestick pattern recognition
- Real-time signal monitoring
- WhatsApp/SMS notifications
- Popup alerts
"""

import subprocess
import sys
import os
import time
import json
from pathlib import Path

def print_banner():
    """Print setup banner"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                    🚀 TRADER AI SETUP 🚀                     ║
    ║                                                              ║
    ║  Advanced Indian Stock Market Trading Platform              ║
    ║  with AI-Powered Signal Generation & Real-time Alerts       ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

def check_requirements():
    """Check if required tools are installed"""
    print("🔍 Checking system requirements...")
    
    required_tools = ['python3', 'node', 'npm', 'pip']
    missing_tools = []
    
    for tool in required_tools:
        if subprocess.run(['which', tool], capture_output=True).returncode != 0:
            missing_tools.append(tool)
    
    if missing_tools:
        print(f"❌ Missing required tools: {', '.join(missing_tools)}")
        print("Please install the missing tools and try again.")
        return False
    
    print("✅ All required tools are installed")
    return True

def setup_backend():
    """Setup backend with all dependencies"""
    print("\n🔧 Setting up backend...")
    
    try:
        # Create virtual environment
        print("Creating Python virtual environment...")
        subprocess.run(['python3', '-m', 'venv', 'backend/venv'], check=True)
        
        # Activate virtual environment and install dependencies
        if os.name == 'nt':  # Windows
            activate_script = 'backend/venv/Scripts/activate'
            pip_cmd = 'backend/venv/Scripts/pip'
        else:  # Unix/Linux/Mac
            activate_script = 'backend/venv/bin/activate'
            pip_cmd = 'backend/venv/bin/pip'
        
        print("Installing Python dependencies...")
        subprocess.run([pip_cmd, 'install', '--upgrade', 'pip'], check=True)
        subprocess.run([pip_cmd, 'install', '-r', 'backend/requirements.txt'], check=True)
        
        # Install additional packages for pattern recognition
        additional_packages = [
            'pyotp',  # For TOTP generation
            'aiohttp',  # For async HTTP requests
            'websockets'  # For WebSocket support
        ]
        
        for package in additional_packages:
            subprocess.run([pip_cmd, 'install', package], check=True)
        
        # Create necessary directories
        os.makedirs('backend/models', exist_ok=True)
        os.makedirs('backend/logs', exist_ok=True)
        os.makedirs('backend/data', exist_ok=True)
        
        print("✅ Backend setup completed")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Backend setup failed: {e}")
        return False

def setup_frontend():
    """Setup frontend with all dependencies"""
    print("\n🎨 Setting up frontend...")
    
    try:
        # Install Node.js dependencies
        print("Installing Node.js dependencies...")
        subprocess.run(['npm', 'install'], cwd='frontend', check=True)
        
        # Install additional packages for charts and notifications
        additional_packages = [
            'react-hot-toast',  # For notifications
            'lucide-react',  # For icons
            'recharts',  # For charts
            'tailwindcss',  # For styling
            'autoprefixer',  # For CSS
            'postcss'  # For CSS processing
        ]
        
        for package in additional_packages:
            subprocess.run(['npm', 'install', package], cwd='frontend', check=True)
        
        print("✅ Frontend setup completed")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Frontend setup failed: {e}")
        return False

def create_config_files():
    """Create configuration files"""
    print("\n📝 Creating configuration files...")
    
    try:
        # Create .env file if it doesn't exist
        if not os.path.exists('backend/.env'):
            print("Creating environment configuration...")
            with open('env.example', 'r') as f:
                env_content = f.read()
            
            with open('backend/.env', 'w') as f:
                f.write(env_content)
            
            print("✅ Environment file created at backend/.env")
            print("⚠️  Please edit backend/.env with your configuration")
        
        # Create frontend environment file
        frontend_env = """REACT_APP_API_URL=http://localhost:8000
REACT_APP_WS_URL=ws://localhost:8000
REACT_APP_APP_NAME=Trader AI
REACT_APP_VERSION=1.0.0
"""
        
        with open('frontend/.env', 'w') as f:
            f.write(frontend_env)
        
        print("✅ Frontend environment file created")
        
        # Create monitoring configuration
        monitoring_config = {
            "analysis_interval": 30,
            "alert_cooldown": 300,
            "min_confidence": 0.7,
            "max_symbols": 50,
            "notification_channels": ["websocket", "whatsapp", "sms"]
        }
        
        with open('backend/monitoring_config.json', 'w') as f:
            json.dump(monitoring_config, f, indent=2)
        
        print("✅ Monitoring configuration created")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration setup failed: {e}")
        return False

def create_startup_scripts():
    """Create startup scripts"""
    print("\n🚀 Creating startup scripts...")
    
    try:
        # Create comprehensive startup script
        startup_script = """#!/bin/bash
# Trader AI Complete Startup Script

echo "🚀 Starting Trader AI Platform..."

# Check if Redis is running
if ! pgrep -x "redis-server" > /dev/null; then
    echo "Starting Redis server..."
    redis-server --daemonize yes
    sleep 2
fi

# Start backend
echo "Starting backend API server..."
cd backend
source venv/bin/activate
python main.py &
BACKEND_PID=$!

# Wait for backend to start
sleep 5

# Start frontend
echo "Starting frontend development server..."
cd ../frontend
npm start &
FRONTEND_PID=$!

echo "✅ Trader AI Platform is running!"
echo "📊 Frontend: http://localhost:3000"
echo "🔧 Backend API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for user interrupt
trap "echo 'Stopping services...'; kill $BACKEND_PID $FRONTEND_PID; exit" INT
wait
"""
        
        with open('start_all.sh', 'w') as f:
            f.write(startup_script)
        
        os.chmod('start_all.sh', 0o755)
        
        # Create Windows batch file
        windows_script = """@echo off
echo 🚀 Starting Trader AI Platform...

REM Start Redis (if available)
redis-server --service-start 2>nul

REM Start backend
echo Starting backend API server...
cd backend
call venv\\Scripts\\activate
start "Backend" cmd /k "python main.py"

REM Wait for backend to start
timeout /t 5 /nobreak >nul

REM Start frontend
echo Starting frontend development server...
cd ..\\frontend
start "Frontend" cmd /k "npm start"

echo ✅ Trader AI Platform is starting!
echo 📊 Frontend: http://localhost:3000
echo 🔧 Backend API: http://localhost:8000
echo 📚 API Docs: http://localhost:8000/docs
pause
"""
        
        with open('start_all.bat', 'w') as f:
            f.write(windows_script)
        
        print("✅ Startup scripts created")
        return True
        
    except Exception as e:
        print(f"❌ Startup script creation failed: {e}")
        return False

def create_documentation():
    """Create comprehensive documentation"""
    print("\n📚 Creating documentation...")
    
    try:
        # Create feature documentation
        features_doc = """# 🎯 Trader AI Features

## 🚀 Core Features

### 1. Real-Time Data & Charts
- Live NSE/BSE price feeds
- Real-time candlestick charts
- WebSocket updates
- Historical data analysis

### 2. Advanced AI Analysis
- **Technical Analysis**: RSI, MACD, Bollinger Bands, Stochastic, Williams %R, ATR
- **Sentiment Analysis**: News and social media sentiment
- **Volume Analysis**: Volume patterns and order flow
- **Pattern Recognition**: 20+ candlestick patterns
- **AI Engine**: Machine learning-based signal generation

### 3. Candlestick Pattern Recognition
- **Reversal Patterns**: Hammer, Doji, Engulfing, Harami, Morning/Evening Star
- **Continuation Patterns**: Flag, Pennant, Rectangle
- **Indecision Patterns**: Spinning Top, High Wave
- **Volume Patterns**: Volume spikes, divergence analysis

### 4. Real-Time Signal Monitoring
- Continuous chart analysis
- Automatic signal generation
- Popup alerts and notifications
- WhatsApp/SMS integration
- WebSocket real-time updates

### 5. Risk Management
- Position sizing based on risk parameters
- Portfolio risk metrics (VaR, Sharpe ratio, Max Drawdown)
- Trade validation against risk criteria
- Risk recommendations and alerts

### 6. Trading Features
- Live order placement
- Portfolio management
- P&L tracking
- Order book management
- Angel One integration

## 🔧 API Endpoints

### Real-Time Data
- `GET /api/realtime/quote/{symbol}` - Live quotes
- `GET /api/realtime/historical/{symbol}` - Historical data
- `GET /api/realtime/market-status` - Market status

### AI Analysis
- `POST /api/ai/analyze/{symbol}` - Comprehensive analysis
- `POST /api/ai/patterns/{symbol}` - Candlestick patterns
- `POST /api/ai/generate-signal/{symbol}` - Generate signals
- `POST /api/ai/alert-signal/{symbol}` - Alert signals

### Monitoring
- `POST /api/monitoring/start-monitoring` - Start monitoring
- `GET /api/monitoring/signal-history` - Signal history
- `GET /api/monitoring/active-signals` - Active signals

### Notifications
- `POST /api/ai/test-notification` - Test notifications
- `GET /api/ai/notification-preferences` - Get preferences
- `POST /api/ai/notification-preferences` - Update preferences

## 📱 Notification Channels

### WhatsApp Integration
- Real-time trading signals
- Formatted messages with emojis
- Risk level indicators
- Price targets and stop-loss

### SMS Integration
- Twilio SMS support
- Concise signal alerts
- Urgency indicators

### WebSocket Alerts
- Real-time popup notifications
- Live chart updates
- Signal confirmations

## 🎨 Frontend Features

### Dashboard
- Portfolio overview
- Market performance
- Top gainers/losers
- Real-time updates

### Trading Terminal
- Live charts with candlestick patterns
- Order placement interface
- Real-time price feeds
- Technical indicators

### Analytics
- AI-powered insights
- Pattern analysis
- Risk metrics
- Performance tracking

### Settings
- Notification preferences
- Risk management settings
- Trading preferences
- User profile

## 🔒 Security & Risk

### Risk Management
- Position sizing algorithms
- Stop-loss automation
- Portfolio diversification
- Risk monitoring

### Data Security
- Encrypted communications
- Secure API endpoints
- User authentication
- Data privacy

## 🚀 Getting Started

1. **Setup**: Run `python setup_complete.py`
2. **Configure**: Edit `backend/.env` with your credentials
3. **Start**: Run `./start_all.sh` (Linux/Mac) or `start_all.bat` (Windows)
4. **Access**: Open http://localhost:3000

## 📞 Support

- Documentation: Check README.md
- API Docs: http://localhost:8000/docs
- Issues: Create GitHub issue
- Email: support@traderai.com

---

**Happy Trading! 📈**
"""
        
        with open('FEATURES.md', 'w') as f:
            f.write(features_doc)
        
        print("✅ Documentation created")
        return True
        
    except Exception as e:
        print(f"❌ Documentation creation failed: {e}")
        return False

def main():
    """Main setup function"""
    print_banner()
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Setup backend
    if not setup_backend():
        sys.exit(1)
    
    # Setup frontend
    if not setup_frontend():
        sys.exit(1)
    
    # Create configuration files
    if not create_config_files():
        sys.exit(1)
    
    # Create startup scripts
    if not create_startup_scripts():
        sys.exit(1)
    
    # Create documentation
    if not create_documentation():
        sys.exit(1)
    
    print("\n" + "="*60)
    print("🎉 TRADER AI SETUP COMPLETED SUCCESSFULLY! 🎉")
    print("="*60)
    print("\n📋 Next Steps:")
    print("1. Edit backend/.env with your API credentials")
    print("2. Configure notification settings (WhatsApp/SMS)")
    print("3. Start the platform:")
    print("   - Linux/Mac: ./start_all.sh")
    print("   - Windows: start_all.bat")
    print("   - Manual: python start.py")
    print("\n🌐 Access Points:")
    print("   - Frontend: http://localhost:3000")
    print("   - Backend API: http://localhost:8000")
    print("   - API Documentation: http://localhost:8000/docs")
    print("\n📚 Documentation:")
    print("   - README.md - Basic setup guide")
    print("   - FEATURES.md - Complete feature list")
    print("\n🚀 Features Included:")
    print("   ✅ Real-time NSE/BSE data")
    print("   ✅ Advanced candlestick pattern recognition")
    print("   ✅ AI-powered signal generation")
    print("   ✅ Popup alerts and notifications")
    print("   ✅ WhatsApp/SMS integration")
    print("   ✅ Risk management system")
    print("   ✅ Professional trading interface")
    print("\nHappy Trading! 📈")

if __name__ == "__main__":
    main()
