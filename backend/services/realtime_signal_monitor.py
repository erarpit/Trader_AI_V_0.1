"""
Real-time Signal Monitoring Service
Continuously monitors charts and generates popup alerts
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set
import logging
from datetime import datetime, timedelta
import json

from services.signal_generator import SignalGenerator
from services.notification_service import NotificationService
from core.websocket_manager import WebSocketManager
from core.redis_client import redis_client

logger = logging.getLogger(__name__)

class RealtimeSignalMonitor:
    def __init__(self):
        self.signal_generator = SignalGenerator()
        self.notification_service = NotificationService()
        self.websocket_manager = WebSocketManager()
        self.monitored_symbols: Set[str] = set()
        self.signal_history: Dict[str, List[Dict]] = {}
        self.is_running = False
        
        # Signal generation intervals (in seconds)
        self.analysis_interval = 30  # Analyze every 30 seconds
        self.alert_cooldown = 300    # 5 minutes between alerts for same symbol
        
    async def start_monitoring(self, symbols: List[str]):
        """Start monitoring symbols for trading signals"""
        try:
            self.monitored_symbols.update(symbols)
            self.is_running = True
            
            logger.info(f"Started monitoring {len(symbols)} symbols: {symbols}")
            
            # Start monitoring loop
            asyncio.create_task(self._monitoring_loop())
            
            return {"status": "started", "symbols": list(self.monitored_symbols)}
            
        except Exception as e:
            logger.error(f"Error starting monitoring: {e}")
            return {"status": "error", "error": str(e)}
    
    async def stop_monitoring(self, symbols: List[str] = None):
        """Stop monitoring specific symbols or all symbols"""
        try:
            if symbols:
                self.monitored_symbols.difference_update(symbols)
                logger.info(f"Stopped monitoring symbols: {symbols}")
            else:
                self.monitored_symbols.clear()
                self.is_running = False
                logger.info("Stopped monitoring all symbols")
            
            return {"status": "stopped", "remaining_symbols": list(self.monitored_symbols)}
            
        except Exception as e:
            logger.error(f"Error stopping monitoring: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_running and self.monitored_symbols:
            try:
                # Analyze each symbol
                tasks = []
                for symbol in self.monitored_symbols:
                    task = asyncio.create_task(self._analyze_symbol(symbol))
                    tasks.append(task)
                
                # Wait for all analyses to complete
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                
                # Wait before next analysis cycle
                await asyncio.sleep(self.analysis_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def _analyze_symbol(self, symbol: str):
        """Analyze a single symbol for trading signals"""
        try:
            # Get recent data (last 50 candles)
            recent_data = await self._get_recent_data(symbol)
            if not recent_data or len(recent_data) < 20:
                return
            
            # Convert to DataFrame
            df = pd.DataFrame(recent_data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            
            # Generate signal
            signal_data = await self.signal_generator.generate_alert_signal(symbol, df, "NSE")
            
            if signal_data and self._should_generate_alert(symbol, signal_data):
                # Send WebSocket notification
                await self._send_websocket_alert(symbol, signal_data)
                
                # Store in history
                self._store_signal_history(symbol, signal_data)
                
                # Send external notifications
                await self._send_external_notifications(symbol, signal_data)
                
                logger.info(f"Generated alert for {symbol}: {signal_data['action']}")
            
        except Exception as e:
            logger.error(f"Error analyzing symbol {symbol}: {e}")
    
    async def _get_recent_data(self, symbol: str) -> List[Dict]:
        """Get recent market data for symbol"""
        try:
            # Try to get from cache first
            cached_data = await redis_client.get_cached_historical_data(symbol)
            if cached_data:
                return cached_data[-50:]  # Last 50 candles
            
            # If not in cache, generate mock data for demo
            return self._generate_mock_data(symbol)
            
        except Exception as e:
            logger.error(f"Error getting recent data for {symbol}: {e}")
            return []
    
    def _generate_mock_data(self, symbol: str) -> List[Dict]:
        """Generate mock data for demonstration"""
        data = []
        base_price = 1000 + hash(symbol) % 1000  # Different base price for each symbol
        current_price = base_price
        
        for i in range(50):
            date = datetime.now() - timedelta(days=50-i)
            
            # Generate realistic OHLC data
            open_price = current_price
            change = np.random.normal(0, 0.02)  # 2% daily volatility
            close_price = open_price * (1 + change)
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.01)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.01)))
            volume = int(np.random.uniform(100000, 1000000))
            
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': volume
            })
            
            current_price = close_price
        
        return data
    
    def _should_generate_alert(self, symbol: str, signal_data: Dict) -> bool:
        """Check if alert should be generated based on cooldown and criteria"""
        try:
            # Check cooldown period
            if symbol in self.signal_history:
                last_alert = self.signal_history[symbol][-1]
                last_time = datetime.fromisoformat(last_alert['timestamp'])
                if datetime.now() - last_time < timedelta(seconds=self.alert_cooldown):
                    return False
            
            # Check signal strength
            confidence = signal_data.get('confidence', 0)
            action = signal_data.get('action', 'HOLD')
            
            if confidence < 0.6 or action == 'HOLD':
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking alert criteria: {e}")
            return False
    
    async def _send_websocket_alert(self, symbol: str, signal_data: Dict):
        """Send alert via WebSocket"""
        try:
            alert_message = {
                "type": "trading_signal",
                "symbol": symbol,
                "signal": signal_data,
                "timestamp": datetime.now().isoformat()
            }
            
            # Send to all WebSocket subscribers
            await self.websocket_manager.broadcast(json.dumps(alert_message))
            
            # Send to symbol-specific subscribers
            await self.websocket_manager.send_to_subscribers(symbol, signal_data)
            
        except Exception as e:
            logger.error(f"Error sending WebSocket alert: {e}")
    
    def _store_signal_history(self, symbol: str, signal_data: Dict):
        """Store signal in history"""
        try:
            if symbol not in self.signal_history:
                self.signal_history[symbol] = []
            
            self.signal_history[symbol].append(signal_data)
            
            # Keep only last 10 signals per symbol
            if len(self.signal_history[symbol]) > 10:
                self.signal_history[symbol] = self.signal_history[symbol][-10:]
                
        except Exception as e:
            logger.error(f"Error storing signal history: {e}")
    
    async def _send_external_notifications(self, symbol: str, signal_data: Dict):
        """Send external notifications (WhatsApp, SMS)"""
        try:
            # Get user phone numbers from database or configuration
            # For demo, we'll use a default phone number
            user_phones = ["+919876543210"]  # Replace with actual user phones
            
            for phone in user_phones:
                await self.notification_service.send_trading_signal(signal_data, phone)
                
        except Exception as e:
            logger.error(f"Error sending external notifications: {e}")
    
    async def get_signal_history(self, symbol: str = None) -> Dict:
        """Get signal history for symbol or all symbols"""
        try:
            if symbol:
                return {
                    "symbol": symbol,
                    "signals": self.signal_history.get(symbol, []),
                    "count": len(self.signal_history.get(symbol, []))
                }
            else:
                return {
                    "all_signals": self.signal_history,
                    "total_symbols": len(self.signal_history),
                    "total_signals": sum(len(signals) for signals in self.signal_history.values())
                }
                
        except Exception as e:
            logger.error(f"Error getting signal history: {e}")
            return {"error": str(e)}
    
    async def get_monitoring_status(self) -> Dict:
        """Get current monitoring status"""
        try:
            return {
                "is_running": self.is_running,
                "monitored_symbols": list(self.monitored_symbols),
                "symbol_count": len(self.monitored_symbols),
                "analysis_interval": self.analysis_interval,
                "alert_cooldown": self.alert_cooldown,
                "total_signals_generated": sum(len(signals) for signals in self.signal_history.values())
            }
            
        except Exception as e:
            logger.error(f"Error getting monitoring status: {e}")
            return {"error": str(e)}
    
    async def update_monitoring_settings(self, settings: Dict):
        """Update monitoring settings"""
        try:
            if 'analysis_interval' in settings:
                self.analysis_interval = settings['analysis_interval']
            
            if 'alert_cooldown' in settings:
                self.alert_cooldown = settings['alert_cooldown']
            
            logger.info(f"Updated monitoring settings: {settings}")
            return {"status": "updated", "settings": settings}
            
        except Exception as e:
            logger.error(f"Error updating monitoring settings: {e}")
            return {"error": str(e)}

# Global instance
realtime_monitor = RealtimeSignalMonitor()
