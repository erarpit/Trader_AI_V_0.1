"""
Comprehensive Test Suite for Trader AI Platform
Tests all functionality including services, dashboard, and AI/ML models
"""

import pytest
import asyncio
import httpx
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TraderAITestSuite:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
        self.test_results = []
        self.test_symbols = ["RELIANCE", "TCS", "INFY", "HDFC", "ICICIBANK"]
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    def log_test_result(self, test_id: str, test_name: str, status: str, 
                       details: str = "", error: str = ""):
        """Log test result"""
        result = {
            "test_id": test_id,
            "test_name": test_name,
            "status": status,
            "details": details,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
        self.test_results.append(result)
        logger.info(f"Test {test_id}: {test_name} - {status}")

    async def test_health_check(self):
        """Test Case 1: Health Check"""
        test_id = "TC001"
        test_name = "Health Check Endpoint"
        
        try:
            response = await self.client.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    self.log_test_result(test_id, test_name, "PASS", 
                                       f"Health check successful: {data}")
                else:
                    self.log_test_result(test_id, test_name, "FAIL", 
                                       f"Unexpected status: {data}")
            else:
                self.log_test_result(test_id, test_name, "FAIL", 
                                   f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test_result(test_id, test_name, "ERROR", error=str(e))

    async def test_root_endpoint(self):
        """Test Case 2: Root Endpoint"""
        test_id = "TC002"
        test_name = "Root Endpoint"
        
        try:
            response = await self.client.get(f"{self.base_url}/")
            if response.status_code == 200:
                data = response.json()
                if "message" in data and "Trader AI" in data["message"]:
                    self.log_test_result(test_id, test_name, "PASS", 
                                       f"Root endpoint working: {data}")
                else:
                    self.log_test_result(test_id, test_name, "FAIL", 
                                       f"Unexpected response: {data}")
            else:
                self.log_test_result(test_id, test_name, "FAIL", 
                                   f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test_result(test_id, test_name, "ERROR", error=str(e))

    async def test_market_status(self):
        """Test Case 3: Market Status"""
        test_id = "TC003"
        test_name = "Market Status Endpoint"
        
        try:
            response = await self.client.get(f"{self.base_url}/api/realtime/market-status")
            if response.status_code == 200:
                data = response.json()
                if "nse" in data and "bse" in data:
                    self.log_test_result(test_id, test_name, "PASS", 
                                       f"Market status retrieved: NSE={data.get('nse')}, BSE={data.get('bse')}")
                else:
                    self.log_test_result(test_id, test_name, "FAIL", 
                                       f"Missing market data: {data}")
            else:
                self.log_test_result(test_id, test_name, "FAIL", 
                                   f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test_result(test_id, test_name, "ERROR", error=str(e))

    async def test_quote_endpoint(self):
        """Test Case 4: Quote Endpoint"""
        test_id = "TC004"
        test_name = "Quote Endpoint"
        
        for symbol in self.test_symbols[:2]:  # Test first 2 symbols
            try:
                response = await self.client.get(f"{self.base_url}/api/realtime/quote/{symbol}")
                if response.status_code == 200:
                    data = response.json()
                    if "symbol" in data or "price" in data:
                        self.log_test_result(test_id, test_name, "PASS", 
                                           f"Quote for {symbol}: {data}")
                    else:
                        self.log_test_result(test_id, test_name, "FAIL", 
                                           f"Invalid quote data for {symbol}: {data}")
                else:
                    self.log_test_result(test_id, test_name, "FAIL", 
                                       f"HTTP {response.status_code} for {symbol}")
            except Exception as e:
                self.log_test_result(test_id, test_name, "ERROR", 
                                   error=f"Error getting quote for {symbol}: {str(e)}")

    async def test_historical_data(self):
        """Test Case 5: Historical Data"""
        test_id = "TC005"
        test_name = "Historical Data Endpoint"
        
        try:
            symbol = self.test_symbols[0]
            response = await self.client.get(f"{self.base_url}/api/realtime/historical/{symbol}")
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    self.log_test_result(test_id, test_name, "PASS", 
                                       f"Historical data for {symbol}: {len(data)} records")
                else:
                    self.log_test_result(test_id, test_name, "FAIL", 
                                       f"Invalid historical data for {symbol}: {data}")
            else:
                self.log_test_result(test_id, test_name, "FAIL", 
                                   f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test_result(test_id, test_name, "ERROR", error=str(e))

    async def test_top_gainers(self):
        """Test Case 6: Top Gainers"""
        test_id = "TC006"
        test_name = "Top Gainers Endpoint"
        
        try:
            response = await self.client.get(f"{self.base_url}/api/realtime/top-gainers")
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    self.log_test_result(test_id, test_name, "PASS", 
                                       f"Top gainers retrieved: {len(data)} stocks")
                else:
                    self.log_test_result(test_id, test_name, "FAIL", 
                                       f"Invalid top gainers data: {data}")
            else:
                self.log_test_result(test_id, test_name, "FAIL", 
                                   f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test_result(test_id, test_name, "ERROR", error=str(e))

    async def test_top_losers(self):
        """Test Case 7: Top Losers"""
        test_id = "TC007"
        test_name = "Top Losers Endpoint"
        
        try:
            response = await self.client.get(f"{self.base_url}/api/realtime/top-losers")
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    self.log_test_result(test_id, test_name, "PASS", 
                                       f"Top losers retrieved: {len(data)} stocks")
                else:
                    self.log_test_result(test_id, test_name, "FAIL", 
                                       f"Invalid top losers data: {data}")
            else:
                self.log_test_result(test_id, test_name, "FAIL", 
                                   f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test_result(test_id, test_name, "ERROR", error=str(e))

    async def test_ai_analysis(self):
        """Test Case 8: AI Analysis"""
        test_id = "TC008"
        test_name = "AI Analysis Endpoint"
        
        try:
            symbol = self.test_symbols[0]
            response = await self.client.post(f"{self.base_url}/api/ai/analyze/{symbol}")
            if response.status_code == 200:
                data = response.json()
                if "analysis" in data or "signal" in data:
                    self.log_test_result(test_id, test_name, "PASS", 
                                       f"AI analysis for {symbol}: {data}")
                else:
                    self.log_test_result(test_id, test_name, "FAIL", 
                                       f"Invalid AI analysis data: {data}")
            else:
                self.log_test_result(test_id, test_name, "FAIL", 
                                   f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test_result(test_id, test_name, "ERROR", error=str(e))

    async def test_ai_signals(self):
        """Test Case 9: AI Signals"""
        test_id = "TC009"
        test_name = "AI Signals Endpoint"
        
        try:
            symbol = self.test_symbols[0]
            response = await self.client.get(f"{self.base_url}/api/ai/signals/{symbol}")
            if response.status_code == 200:
                data = response.json()
                if "signals" in data or "signal" in data:
                    self.log_test_result(test_id, test_name, "PASS", 
                                       f"AI signals for {symbol}: {data}")
                else:
                    self.log_test_result(test_id, test_name, "FAIL", 
                                       f"Invalid AI signals data: {data}")
            else:
                self.log_test_result(test_id, test_name, "FAIL", 
                                   f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test_result(test_id, test_name, "ERROR", error=str(e))

    async def test_market_overview(self):
        """Test Case 10: Market Overview"""
        test_id = "TC010"
        test_name = "Market Overview Endpoint"
        
        try:
            response = await self.client.get(f"{self.base_url}/api/ai/market-overview")
            if response.status_code == 200:
                data = response.json()
                if "overview" in data or "market" in data:
                    self.log_test_result(test_id, test_name, "PASS", 
                                       f"Market overview retrieved: {data}")
                else:
                    self.log_test_result(test_id, test_name, "FAIL", 
                                       f"Invalid market overview data: {data}")
            else:
                self.log_test_result(test_id, test_name, "FAIL", 
                                   f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test_result(test_id, test_name, "ERROR", error=str(e))

    async def test_portfolio_endpoint(self):
        """Test Case 11: Portfolio Endpoint"""
        test_id = "TC011"
        test_name = "Portfolio Endpoint"
        
        try:
            response = await self.client.get(f"{self.base_url}/api/trading/portfolio")
            if response.status_code == 200:
                data = response.json()
                self.log_test_result(test_id, test_name, "PASS", 
                                   f"Portfolio data retrieved: {data}")
            else:
                self.log_test_result(test_id, test_name, "FAIL", 
                                   f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test_result(test_id, test_name, "ERROR", error=str(e))

    async def test_orders_endpoint(self):
        """Test Case 12: Orders Endpoint"""
        test_id = "TC012"
        test_name = "Orders Endpoint"
        
        try:
            response = await self.client.get(f"{self.base_url}/api/trading/orders")
            if response.status_code == 200:
                data = response.json()
                self.log_test_result(test_id, test_name, "PASS", 
                                   f"Orders data retrieved: {data}")
            else:
                self.log_test_result(test_id, test_name, "FAIL", 
                                   f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test_result(test_id, test_name, "ERROR", error=str(e))

    async def test_risk_portfolio(self):
        """Test Case 13: Risk Portfolio"""
        test_id = "TC013"
        test_name = "Risk Portfolio Endpoint"
        
        try:
            response = await self.client.get(f"{self.base_url}/api/risk/portfolio")
            if response.status_code == 200:
                data = response.json()
                self.log_test_result(test_id, test_name, "PASS", 
                                   f"Risk portfolio data retrieved: {data}")
            else:
                self.log_test_result(test_id, test_name, "FAIL", 
                                   f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test_result(test_id, test_name, "ERROR", error=str(e))

    async def test_websocket_connection(self):
        """Test Case 14: WebSocket Connection"""
        test_id = "TC014"
        test_name = "WebSocket Connection"
        
        try:
            import websockets
            uri = f"ws://localhost:8000/ws"
            async with websockets.connect(uri) as websocket:
                # Send a test message
                await websocket.send(json.dumps({"type": "test", "message": "Hello"}))
                
                # Wait for response (with timeout)
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    self.log_test_result(test_id, test_name, "PASS", 
                                       f"WebSocket connected and responded: {response}")
                except asyncio.TimeoutError:
                    self.log_test_result(test_id, test_name, "PASS", 
                                       f"WebSocket connected (no response received)")
        except Exception as e:
            self.log_test_result(test_id, test_name, "ERROR", error=str(e))

    async def test_ai_engine_functionality(self):
        """Test Case 15: AI Engine Functionality"""
        test_id = "TC015"
        test_name = "AI Engine Functionality"
        
        try:
            # Test AI engine directly
            from backend.services.ai_engine import AIEngine
            
            ai_engine = AIEngine()
            
            # Create sample data
            sample_data = pd.DataFrame({
                'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
                'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
            })
            
            analysis_data = {
                'technical': {
                    'indicators': {
                        'rsi': pd.Series([50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]),
                        'macd': pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]),
                        'macd_signal': pd.Series([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05]),
                        'sma_20': pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]),
                        'sma_50': pd.Series([95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105]),
                        'bb_upper': pd.Series([110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120]),
                        'bb_lower': pd.Series([90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100])
                    },
                    'signals': {
                        'strength': 0.8
                    }
                },
                'sentiment': {
                    'sentiment_score': 0.7,
                    'confidence': 0.8
                }
            }
            
            # Generate signals
            signals = ai_engine.generate_signals(sample_data, analysis_data)
            
            if signals and 'signal' in signals:
                self.log_test_result(test_id, test_name, "PASS", 
                                   f"AI engine generated signals: {signals}")
            else:
                self.log_test_result(test_id, test_name, "FAIL", 
                                   f"AI engine failed to generate signals: {signals}")
                
        except Exception as e:
            self.log_test_result(test_id, test_name, "ERROR", error=str(e))

    async def test_technical_analysis(self):
        """Test Case 16: Technical Analysis"""
        test_id = "TC016"
        test_name = "Technical Analysis Service"
        
        try:
            from backend.services.technical_analysis import TechnicalAnalysis
            
            ta = TechnicalAnalysis()
            
            # Create sample data
            sample_data = pd.DataFrame({
                'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120],
                'high': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121],
                'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119],
                'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000]
            })
            
            # Calculate indicators
            indicators = ta.calculate_indicators(sample_data)
            
            if indicators and len(indicators) > 0:
                self.log_test_result(test_id, test_name, "PASS", 
                                   f"Technical analysis calculated {len(indicators)} indicators")
            else:
                self.log_test_result(test_id, test_name, "FAIL", 
                                   f"Technical analysis failed to calculate indicators")
                
        except Exception as e:
            self.log_test_result(test_id, test_name, "ERROR", error=str(e))

    async def test_sentiment_analysis(self):
        """Test Case 17: Sentiment Analysis"""
        test_id = "TC017"
        test_name = "Sentiment Analysis Service"
        
        try:
            from backend.services.sentiment_analysis import SentimentAnalyzer
            
            analyzer = SentimentAnalyzer()
            
            # Test sentiment analysis
            test_text = "This stock is performing very well and showing strong growth potential."
            sentiment = analyzer.analyze_sentiment(test_text)
            
            if sentiment and 'sentiment_score' in sentiment:
                self.log_test_result(test_id, test_name, "PASS", 
                                   f"Sentiment analysis result: {sentiment}")
            else:
                self.log_test_result(test_id, test_name, "FAIL", 
                                   f"Sentiment analysis failed: {sentiment}")
                
        except Exception as e:
            self.log_test_result(test_id, test_name, "ERROR", error=str(e))

    async def test_risk_management(self):
        """Test Case 18: Risk Management"""
        test_id = "TC018"
        test_name = "Risk Management Service"
        
        try:
            from backend.services.risk_manager import RiskManager
            
            risk_manager = RiskManager()
            
            # Test portfolio risk calculation
            portfolio_data = {
                'positions': [
                    {'symbol': 'RELIANCE', 'quantity': 100, 'price': 2500, 'value': 250000},
                    {'symbol': 'TCS', 'quantity': 50, 'price': 3500, 'value': 175000}
                ],
                'total_value': 425000
            }
            
            risk_metrics = risk_manager.calculate_portfolio_risk(portfolio_data)
            
            if risk_metrics and 'var' in risk_metrics:
                self.log_test_result(test_id, test_name, "PASS", 
                                   f"Risk management calculated metrics: {risk_metrics}")
            else:
                self.log_test_result(test_id, test_name, "FAIL", 
                                   f"Risk management failed: {risk_metrics}")
                
        except Exception as e:
            self.log_test_result(test_id, test_name, "ERROR", error=str(e))

    async def test_database_connection(self):
        """Test Case 19: Database Connection"""
        test_id = "TC019"
        test_name = "Database Connection"
        
        try:
            from backend.core.database import engine, Base
            from sqlalchemy import text
            
            # Test database connection
            with engine.connect() as connection:
                result = connection.execute(text("SELECT 1"))
                if result.fetchone()[0] == 1:
                    self.log_test_result(test_id, test_name, "PASS", 
                                       "Database connection successful")
                else:
                    self.log_test_result(test_id, test_name, "FAIL", 
                                       "Database connection failed")
                    
        except Exception as e:
            self.log_test_result(test_id, test_name, "ERROR", error=str(e))

    async def test_redis_connection(self):
        """Test Case 20: Redis Connection"""
        test_id = "TC020"
        test_name = "Redis Connection"
        
        try:
            from backend.core.redis_client import redis_client
            
            # Test Redis connection
            await redis_client.ping()
            self.log_test_result(test_id, test_name, "PASS", 
                               "Redis connection successful")
                
        except Exception as e:
            self.log_test_result(test_id, test_name, "ERROR", error=str(e))

    async def run_all_tests(self):
        """Run all test cases"""
        logger.info("Starting Trader AI Comprehensive Test Suite...")
        
        # API Tests
        await self.test_health_check()
        await self.test_root_endpoint()
        await self.test_market_status()
        await self.test_quote_endpoint()
        await self.test_historical_data()
        await self.test_top_gainers()
        await self.test_top_losers()
        await self.test_ai_analysis()
        await self.test_ai_signals()
        await self.test_market_overview()
        await self.test_portfolio_endpoint()
        await self.test_orders_endpoint()
        await self.test_risk_portfolio()
        await self.test_websocket_connection()
        
        # Service Tests
        await self.test_ai_engine_functionality()
        await self.test_technical_analysis()
        await self.test_sentiment_analysis()
        await self.test_risk_management()
        
        # Infrastructure Tests
        await self.test_database_connection()
        await self.test_redis_connection()
        
        logger.info("Test suite completed!")
        return self.test_results

    def generate_test_report(self):
        """Generate comprehensive test report"""
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r['status'] == 'PASS'])
        failed_tests = len([r for r in self.test_results if r['status'] == 'FAIL'])
        error_tests = len([r for r in self.test_results if r['status'] == 'ERROR'])
        
        report = {
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "errors": error_tests,
                "success_rate": f"{(passed_tests/total_tests)*100:.2f}%" if total_tests > 0 else "0%"
            },
            "test_results": self.test_results,
            "timestamp": datetime.now().isoformat()
        }
        
        return report

async def main():
    """Main test execution function"""
    async with TraderAITestSuite() as test_suite:
        await test_suite.run_all_tests()
        report = test_suite.generate_test_report()
        
        # Print summary
        print("\n" + "="*80)
        print("TRADER AI COMPREHENSIVE TEST REPORT")
        print("="*80)
        print(f"Total Tests: {report['summary']['total_tests']}")
        print(f"Passed: {report['summary']['passed']}")
        print(f"Failed: {report['summary']['failed']}")
        print(f"Errors: {report['summary']['errors']}")
        print(f"Success Rate: {report['summary']['success_rate']}")
        print("="*80)
        
        # Print detailed results
        print("\nDETAILED TEST RESULTS:")
        print("-" * 80)
        for result in test_suite.test_results:
            status_icon = "✅" if result['status'] == 'PASS' else "❌" if result['status'] == 'FAIL' else "⚠️"
            print(f"{status_icon} {result['test_id']}: {result['test_name']} - {result['status']}")
            if result['details']:
                print(f"   Details: {result['details']}")
            if result['error']:
                print(f"   Error: {result['error']}")
            print()
        
        # Save report to file
        with open("test_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\nDetailed report saved to: test_report.json")
        
        return report

if __name__ == "__main__":
    asyncio.run(main())
