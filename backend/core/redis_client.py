"""
Redis client for caching and pub/sub
"""

import redis
import json
import asyncio
import logging
from typing import Optional, Dict, Any, List
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class RedisClient:
    def __init__(self):
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.client = None
        self.pubsub = None
        
    async def connect(self):
        """Connect to Redis"""
        try:
            self.client = redis.from_url(self.redis_url, decode_responses=True)
            self.pubsub = self.client.pubsub()
            logger.info("Connected to Redis")
        except Exception as e:
            logger.error(f"Error connecting to Redis: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from Redis"""
        try:
            if self.pubsub:
                await self.pubsub.close()
            if self.client:
                self.client.close()
            logger.info("Disconnected from Redis")
        except Exception as e:
            logger.error(f"Error disconnecting from Redis: {e}")
    
    async def set_cache(self, key: str, value: Any, expire: int = 300) -> bool:
        """Set cache with expiration"""
        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            return self.client.setex(key, expire, value)
        except Exception as e:
            logger.error(f"Error setting cache: {e}")
            return False
    
    async def get_cache(self, key: str) -> Optional[Any]:
        """Get cache value"""
        try:
            value = self.client.get(key)
            if value:
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            return None
        except Exception as e:
            logger.error(f"Error getting cache: {e}")
            return None
    
    async def delete_cache(self, key: str) -> bool:
        """Delete cache key"""
        try:
            return bool(self.client.delete(key))
        except Exception as e:
            logger.error(f"Error deleting cache: {e}")
            return False
    
    async def publish(self, channel: str, message: Dict[str, Any]) -> bool:
        """Publish message to channel"""
        try:
            message_str = json.dumps(message)
            result = self.client.publish(channel, message_str)
            return result > 0
        except Exception as e:
            logger.error(f"Error publishing message: {e}")
            return False
    
    async def subscribe(self, channels: List[str]):
        """Subscribe to channels"""
        try:
            for channel in channels:
                self.pubsub.subscribe(channel)
            logger.info(f"Subscribed to channels: {channels}")
        except Exception as e:
            logger.error(f"Error subscribing to channels: {e}")
    
    async def listen(self):
        """Listen for messages"""
        try:
            for message in self.pubsub.listen():
                if message['type'] == 'message':
                    try:
                        data = json.loads(message['data'])
                        yield {
                            'channel': message['channel'],
                            'data': data
                        }
                    except json.JSONDecodeError:
                        yield {
                            'channel': message['channel'],
                            'data': message['data']
                        }
        except Exception as e:
            logger.error(f"Error listening to messages: {e}")
    
    async def cache_quote(self, symbol: str, quote_data: Dict[str, Any], expire: int = 60) -> bool:
        """Cache quote data"""
        key = f"quote:{symbol}"
        return await self.set_cache(key, quote_data, expire)
    
    async def get_cached_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached quote data"""
        key = f"quote:{symbol}"
        return await self.get_cache(key)
    
    async def cache_historical_data(self, symbol: str, data: List[Dict[str, Any]], expire: int = 3600) -> bool:
        """Cache historical data"""
        key = f"historical:{symbol}"
        return await self.set_cache(key, data, expire)
    
    async def get_cached_historical_data(self, symbol: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached historical data"""
        key = f"historical:{symbol}"
        return await self.get_cache(key)
    
    async def cache_market_status(self, status_data: Dict[str, Any], expire: int = 300) -> bool:
        """Cache market status"""
        key = "market:status"
        return await self.set_cache(key, status_data, expire)
    
    async def get_cached_market_status(self) -> Optional[Dict[str, Any]]:
        """Get cached market status"""
        key = "market:status"
        return await self.get_cache(key)
    
    async def publish_price_update(self, symbol: str, price_data: Dict[str, Any]) -> bool:
        """Publish price update"""
        channel = f"price_updates:{symbol}"
        message = {
            "symbol": symbol,
            "data": price_data,
            "timestamp": price_data.get("timestamp")
        }
        return await self.publish(channel, message)
    
    async def publish_trading_signal(self, symbol: str, signal_data: Dict[str, Any]) -> bool:
        """Publish trading signal"""
        channel = "trading_signals"
        message = {
            "symbol": symbol,
            "signal": signal_data,
            "timestamp": signal_data.get("timestamp")
        }
        return await self.publish(channel, message)

# Global Redis client instance
redis_client = RedisClient()
