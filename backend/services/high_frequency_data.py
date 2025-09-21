"""
High-Frequency Data Ingestion Service
Handles real-time order book data, tick data, and high-frequency market data
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import json
from collections import deque
import websockets
import aiohttp

logger = logging.getLogger(__name__)

class HighFrequencyDataIngestion:
    def __init__(self):
        self.order_book_data = {}
        self.tick_data = {}
        self.volume_profile = {}
        self.price_levels = {}
        
        # Data storage with time-based indexing
        self.data_buffer_size = 10000  # Keep last 10k ticks
        self.order_book_depth = 20  # Top 20 levels
        
        # Market microstructure metrics
        self.bid_ask_spread = {}
        self.order_flow_imbalance = {}
        self.volume_weighted_price = {}
        
    async def start_high_frequency_feed(self, symbols: List[str], exchange: str = "NSE"):
        """Start high-frequency data collection for symbols"""
        try:
            tasks = []
            for symbol in symbols:
                task = asyncio.create_task(self._collect_high_frequency_data(symbol, exchange))
                tasks.append(task)
            
            await asyncio.gather(*tasks)
            
        except Exception as e:
            logger.error(f"Error starting high-frequency feed: {e}")
    
    async def _collect_high_frequency_data(self, symbol: str, exchange: str):
        """Collect high-frequency data for a single symbol"""
        try:
            while True:
                # Get order book data
                order_book = await self._get_order_book_data(symbol, exchange)
                if order_book:
                    await self._process_order_book(symbol, order_book)
                
                # Get tick data
                tick_data = await self._get_tick_data(symbol, exchange)
                if tick_data:
                    await self._process_tick_data(symbol, tick_data)
                
                # Calculate microstructure metrics
                await self._calculate_microstructure_metrics(symbol)
                
                # Small delay for high-frequency updates
                await asyncio.sleep(0.1)  # 100ms updates
                
        except Exception as e:
            logger.error(f"Error collecting high-frequency data for {symbol}: {e}")
    
    async def _get_order_book_data(self, symbol: str, exchange: str) -> Optional[Dict]:
        """Get real-time order book data"""
        try:
            if exchange.upper() == "NSE":
                # Use NSE API for order book
                from core.nse_api import NSEAPI
                async with NSEAPI() as nse:
                    return await nse.get_order_book(symbol)
            elif exchange.upper() == "BSE":
                # Use BSE API for order book
                from core.bse_api import BSEAPI
                async with BSEAPI() as bse:
                    return await bse.get_order_book(symbol)
            else:
                # Generate mock order book for demo
                return self._generate_mock_order_book(symbol)
                
        except Exception as e:
            logger.error(f"Error getting order book data: {e}")
            return None
    
    async def _get_tick_data(self, symbol: str, exchange: str) -> Optional[Dict]:
        """Get real-time tick data"""
        try:
            if exchange.upper() == "NSE":
                from core.nse_api import NSEAPI
                async with NSEAPI() as nse:
                    quote = await nse.get_quote(symbol)
                    if quote:
                        return {
                            "symbol": symbol,
                            "price": quote.get("last_price", 0),
                            "volume": quote.get("volume", 0),
                            "timestamp": datetime.now(),
                            "bid": quote.get("bid", 0),
                            "ask": quote.get("ask", 0)
                        }
            else:
                # Generate mock tick data
                return self._generate_mock_tick_data(symbol)
                
        except Exception as e:
            logger.error(f"Error getting tick data: {e}")
            return None
    
    async def _process_order_book(self, symbol: str, order_book: Dict):
        """Process and store order book data"""
        try:
            if symbol not in self.order_book_data:
                self.order_book_data[symbol] = deque(maxlen=self.data_buffer_size)
            
            # Extract bid/ask levels
            bids = order_book.get("bids", [])[:self.order_book_depth]
            asks = order_book.get("asks", [])[:self.order_book_depth]
            
            processed_data = {
                "timestamp": datetime.now(),
                "bids": bids,
                "asks": asks,
                "bid_volume": sum([bid.get("volume", 0) for bid in bids]),
                "ask_volume": sum([ask.get("volume", 0) for ask in asks]),
                "best_bid": bids[0].get("price", 0) if bids else 0,
                "best_ask": asks[0].get("price", 0) if asks else 0
            }
            
            self.order_book_data[symbol].append(processed_data)
            
            # Update price levels
            self._update_price_levels(symbol, bids, asks)
            
        except Exception as e:
            logger.error(f"Error processing order book: {e}")
    
    async def _process_tick_data(self, symbol: str, tick_data: Dict):
        """Process and store tick data"""
        try:
            if symbol not in self.tick_data:
                self.tick_data[symbol] = deque(maxlen=self.data_buffer_size)
            
            self.tick_data[symbol].append(tick_data)
            
            # Update volume profile
            self._update_volume_profile(symbol, tick_data)
            
        except Exception as e:
            logger.error(f"Error processing tick data: {e}")
    
    def _update_price_levels(self, symbol: str, bids: List, asks: List):
        """Update price level data for volume analysis"""
        try:
            if symbol not in self.price_levels:
                self.price_levels[symbol] = {}
            
            # Aggregate volume by price level
            for bid in bids:
                price = bid.get("price", 0)
                volume = bid.get("volume", 0)
                if price in self.price_levels[symbol]:
                    self.price_levels[symbol][price] += volume
                else:
                    self.price_levels[symbol][price] = volume
            
            for ask in asks:
                price = ask.get("price", 0)
                volume = ask.get("volume", 0)
                if price in self.price_levels[symbol]:
                    self.price_levels[symbol][price] += volume
                else:
                    self.price_levels[symbol][price] = volume
                    
        except Exception as e:
            logger.error(f"Error updating price levels: {e}")
    
    def _update_volume_profile(self, symbol: str, tick_data: Dict):
        """Update volume profile for price-volume analysis"""
        try:
            if symbol not in self.volume_profile:
                self.volume_profile[symbol] = {}
            
            price = tick_data.get("price", 0)
            volume = tick_data.get("volume", 0)
            
            if price in self.volume_profile[symbol]:
                self.volume_profile[symbol][price] += volume
            else:
                self.volume_profile[symbol][price] = volume
                
        except Exception as e:
            logger.error(f"Error updating volume profile: {e}")
    
    async def _calculate_microstructure_metrics(self, symbol: str):
        """Calculate market microstructure metrics"""
        try:
            if symbol not in self.order_book_data or len(self.order_book_data[symbol]) < 2:
                return
            
            latest_data = self.order_book_data[symbol][-1]
            
            # Bid-ask spread
            best_bid = latest_data.get("best_bid", 0)
            best_ask = latest_data.get("best_ask", 0)
            if best_bid > 0 and best_ask > 0:
                spread = best_ask - best_bid
                spread_percent = (spread / best_bid) * 100
                self.bid_ask_spread[symbol] = {
                    "absolute": spread,
                    "percentage": spread_percent,
                    "timestamp": datetime.now()
                }
            
            # Order flow imbalance
            bid_volume = latest_data.get("bid_volume", 0)
            ask_volume = latest_data.get("ask_volume", 0)
            total_volume = bid_volume + ask_volume
            
            if total_volume > 0:
                imbalance = (bid_volume - ask_volume) / total_volume
                self.order_flow_imbalance[symbol] = {
                    "imbalance": imbalance,
                    "bid_volume": bid_volume,
                    "ask_volume": ask_volume,
                    "timestamp": datetime.now()
                }
            
            # Volume-weighted average price (VWAP)
            if symbol in self.tick_data and len(self.tick_data[symbol]) > 0:
                recent_ticks = list(self.tick_data[symbol])[-100:]  # Last 100 ticks
                total_value = sum(tick.get("price", 0) * tick.get("volume", 0) for tick in recent_ticks)
                total_volume = sum(tick.get("volume", 0) for tick in recent_ticks)
                
                if total_volume > 0:
                    vwap = total_value / total_volume
                    self.volume_weighted_price[symbol] = {
                        "vwap": vwap,
                        "total_volume": total_volume,
                        "timestamp": datetime.now()
                    }
                    
        except Exception as e:
            logger.error(f"Error calculating microstructure metrics: {e}")
    
    def get_order_book_snapshot(self, symbol: str) -> Optional[Dict]:
        """Get current order book snapshot"""
        try:
            if symbol in self.order_book_data and len(self.order_book_data[symbol]) > 0:
                return dict(self.order_book_data[symbol][-1])
            return None
        except Exception as e:
            logger.error(f"Error getting order book snapshot: {e}")
            return None
    
    def get_tick_history(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get recent tick history"""
        try:
            if symbol in self.tick_data:
                return list(self.tick_data[symbol])[-limit:]
            return []
        except Exception as e:
            logger.error(f"Error getting tick history: {e}")
            return []
    
    def get_microstructure_metrics(self, symbol: str) -> Dict:
        """Get current microstructure metrics"""
        try:
            return {
                "bid_ask_spread": self.bid_ask_spread.get(symbol, {}),
                "order_flow_imbalance": self.order_flow_imbalance.get(symbol, {}),
                "volume_weighted_price": self.volume_weighted_price.get(symbol, {}),
                "price_levels": self.price_levels.get(symbol, {}),
                "volume_profile": self.volume_profile.get(symbol, {})
            }
        except Exception as e:
            logger.error(f"Error getting microstructure metrics: {e}")
            return {}
    
    def _generate_mock_order_book(self, symbol: str) -> Dict:
        """Generate mock order book data for testing"""
        try:
            base_price = 100 + hash(symbol) % 1000  # Pseudo-random base price
            
            bids = []
            asks = []
            
            # Generate bid levels
            for i in range(20):
                price = base_price - (i * 0.5)
                volume = np.random.randint(100, 1000)
                bids.append({"price": price, "volume": volume})
            
            # Generate ask levels
            for i in range(20):
                price = base_price + ((i + 1) * 0.5)
                volume = np.random.randint(100, 1000)
                asks.append({"price": price, "volume": volume})
            
            return {
                "bids": bids,
                "asks": asks,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating mock order book: {e}")
            return {"bids": [], "asks": []}
    
    def _generate_mock_tick_data(self, symbol: str) -> Dict:
        """Generate mock tick data for testing"""
        try:
            base_price = 100 + hash(symbol) % 1000
            price_change = np.random.normal(0, 0.5)
            current_price = base_price + price_change
            
            return {
                "symbol": symbol,
                "price": max(current_price, 1),  # Ensure positive price
                "volume": np.random.randint(1, 1000),
                "timestamp": datetime.now(),
                "bid": current_price - 0.5,
                "ask": current_price + 0.5
            }
            
        except Exception as e:
            logger.error(f"Error generating mock tick data: {e}")
            return {}
