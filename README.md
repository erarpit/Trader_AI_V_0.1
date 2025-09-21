# 🚀 Trader AI - Indian Stock Market Platform

A comprehensive AI-powered trading platform for Indian stock markets (NSE/BSE) with real-time data, advanced analytics, and intelligent trading signals.

## ✨ Features

### 🎯 **Core Features**
- **Real-Time Data**: Live NSE/BSE price feeds with WebSocket updates
- **AI Trading Engine**: Multi-algorithm analysis with technical, fundamental, and sentiment analysis
- **Advanced Charts**: Real-time candlestick charts and technical indicators
- **Risk Management**: Comprehensive portfolio risk analysis and position sizing
- **Trading Terminal**: Professional trading interface with order placement
- **Portfolio Management**: Track P&L, positions, and performance metrics
- **Auto-Signals**: AI-powered buy/sell signals with popup notifications
- **Volume Analysis**: Advanced volume pattern recognition and order flow analysis

### 🔧 **Technical Stack**

**Backend:**
- FastAPI (Python) - High-performance API framework
- WebSocket - Real-time communication
- Redis - Caching and pub/sub
- PostgreSQL - Database
- NSE/BSE APIs - Market data
- Angel One SmartAPI - Live trading integration

**Frontend:**
- React 18 + TypeScript
- Tailwind CSS - Styling
- Recharts - Data visualization
- WebSocket Client - Real-time updates
- Responsive Design - Mobile support

**AI/ML:**
- Technical Analysis (RSI, MACD, Bollinger Bands, etc.)
- Sentiment Analysis (News and social media)
- Risk Models (VaR, Sharpe ratio, correlation)
- Ensemble Learning - Multi-algorithm approach

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- Redis
- PostgreSQL

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd Trader_AI
```

2. **Backend Setup**
```bash
cd backend
pip install -r requirements.txt
cp env.example .env
# Edit .env with your configuration
python main.py
```

3. **Frontend Setup**
```bash
cd frontend
npm install
npm start
```

4. **Database Setup**
```bash
# Create PostgreSQL database
createdb trader_ai

# Run migrations (if any)
# alembic upgrade head
```

### Configuration

1. **Environment Variables**
   - Copy `env.example` to `.env`
   - Configure database, Redis, and API credentials
   - Set up Angel One API credentials for live trading

2. **NSE/BSE API Setup**
   - No API key required for basic data
   - Configure headers in `env.example`

3. **Angel One Integration** (Optional)
   - Get API credentials from Angel One
   - Configure in environment variables

## 📊 API Endpoints

### Real-Time Data
- `GET /api/realtime/quote/{symbol}` - Live quotes
- `GET /api/realtime/historical/{symbol}` - Historical data
- `GET /api/realtime/market-status` - Market status
- `POST /api/realtime/start-feed` - Start real-time feed

### Trading
- `GET /api/trading/portfolio` - Portfolio data
- `GET /api/trading/orders` - Order book
- `POST /api/trading/place-order` - Place orders
- `DELETE /api/trading/cancel-order/{id}` - Cancel orders

### AI Analysis
- `POST /api/ai/analyze/{symbol}` - AI stock analysis
- `GET /api/ai/signals/{symbol}` - Trading signals
- `GET /api/ai/market-overview` - Market overview

### Risk Management
- `GET /api/risk/portfolio` - Portfolio risk metrics
- `POST /api/risk/position-size` - Calculate position size
- `POST /api/risk/validate-trade` - Validate trades

## 🎯 Usage

### 1. **Dashboard**
- View portfolio performance
- Monitor market trends
- Track P&L and returns

### 2. **Trading Terminal**
- Search and select stocks
- View real-time charts
- Place buy/sell orders
- Monitor order status

### 3. **Portfolio Management**
- Track all positions
- View detailed P&L
- Manage risk exposure

### 4. **AI Analytics**
- Get AI trading signals
- View risk metrics
- Analyze market sentiment
- Volume analysis

### 5. **Settings**
- Configure trading preferences
- Set risk limits
- Manage notifications

## 🔒 Security Features

- **Risk Management**: Position sizing, stop-loss, portfolio limits
- **Trade Validation**: Pre-trade risk checks
- **Secure APIs**: Authentication and authorization
- **Data Encryption**: Secure data transmission

## 📱 Mobile Support

- Responsive design for mobile devices
- Touch-friendly interface
- Real-time updates on mobile

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Trading in financial markets involves substantial risk of loss. Always do your own research and consider consulting with a financial advisor before making investment decisions.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the documentation
- Review the API endpoints

## 🔄 Updates

- **v1.0.0**: Initial release with core features
- Real-time data integration
- AI trading engine
- Risk management system
- Professional trading UI

---

**Happy Trading! 📈**
