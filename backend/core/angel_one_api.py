"""
Angel One SmartAPI integration for live trading
"""

import asyncio
import aiohttp
import json
import hmac
import hashlib
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class AngelOneAPI:
    def __init__(self):
        self.api_key = os.getenv("ANGEL_ONE_API_KEY")
        self.client_id = os.getenv("ANGEL_ONE_CLIENT_ID")
        self.pin = os.getenv("ANGEL_ONE_PIN")
        self.totp_secret = os.getenv("ANGEL_ONE_TOTP_SECRET")
        
        self.base_url = "https://apiconnect.angelbroking.com"
        self.session = None
        self.access_token = None
        self.refresh_token = None
        self.feed_token = None
        self.jwt_token = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def login(self) -> bool:
        """Login to Angel One API"""
        try:
            if not all([self.api_key, self.client_id, self.pin]):
                logger.error("Angel One credentials not configured")
                return False
            
            # Generate TOTP
            totp = self._generate_totp()
            
            # Login request
            login_data = {
                "clientcode": self.client_id,
                "password": self.pin,
                "totp": totp
            }
            
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-UserType": "USER",
                "X-SourceID": "WEB",
                "X-ClientLocalIP": "127.0.0.1",
                "X-ClientPublicIP": "127.0.0.1",
                "X-MACAddress": "00:00:00:00:00:00",
                "X-PrivateKey": self.api_key
            }
            
            url = f"{self.base_url}/rest/auth/angelbroking/user/v1/loginByPassword"
            
            async with self.session.post(url, json=login_data, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("status"):
                        self.access_token = data["data"]["jwtToken"]
                        self.refresh_token = data["data"]["refreshToken"]
                        self.feed_token = data["data"]["feedToken"]
                        logger.info("Angel One login successful")
                        return True
                    else:
                        logger.error(f"Angel One login failed: {data.get('message')}")
                        return False
                else:
                    logger.error(f"Angel One login failed with status: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error logging into Angel One: {e}")
            return False
    
    async def get_profile(self) -> Optional[Dict]:
        """Get user profile"""
        try:
            if not self.access_token:
                await self.login()
            
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-UserType": "USER",
                "X-SourceID": "WEB",
                "X-ClientLocalIP": "127.0.0.1",
                "X-ClientPublicIP": "127.0.0.1",
                "X-MACAddress": "00:00:00:00:00:00",
                "X-PrivateKey": self.api_key
            }
            
            url = f"{self.base_url}/rest/secure/angelbroking/user/v1/getProfile"
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("data")
                else:
                    logger.error(f"Error getting profile: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting profile: {e}")
            return None
    
    async def get_holdings(self) -> Optional[List[Dict]]:
        """Get user holdings"""
        try:
            if not self.access_token:
                await self.login()
            
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-UserType": "USER",
                "X-SourceID": "WEB",
                "X-ClientLocalIP": "127.0.0.1",
                "X-ClientPublicIP": "127.0.0.1",
                "X-MACAddress": "00:00:00:00:00:00",
                "X-PrivateKey": self.api_key
            }
            
            url = f"{self.base_url}/rest/secure/angelbroking/portfolio/v1/getHolding"
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("data", [])
                else:
                    logger.error(f"Error getting holdings: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting holdings: {e}")
            return None
    
    async def place_order(self, order_data: Dict) -> Optional[Dict]:
        """Place an order"""
        try:
            if not self.access_token:
                await self.login()
            
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-UserType": "USER",
                "X-SourceID": "WEB",
                "X-ClientLocalIP": "127.0.0.1",
                "X-ClientPublicIP": "127.0.0.1",
                "X-MACAddress": "00:00:00:00:00:00",
                "X-PrivateKey": self.api_key
            }
            
            url = f"{self.base_url}/rest/secure/angelbroking/order/v1/placeOrder"
            
            async with self.session.post(url, json=order_data, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    logger.error(f"Error placing order: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None
    
    async def get_order_book(self) -> Optional[List[Dict]]:
        """Get order book"""
        try:
            if not self.access_token:
                await self.login()
            
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-UserType": "USER",
                "X-SourceID": "WEB",
                "X-ClientLocalIP": "127.0.0.1",
                "X-ClientPublicIP": "127.0.0.1",
                "X-MACAddress": "00:00:00:00:00:00",
                "X-PrivateKey": self.api_key
            }
            
            url = f"{self.base_url}/rest/secure/angelbroking/order/v1/getOrderBook"
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("data", [])
                else:
                    logger.error(f"Error getting order book: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting order book: {e}")
            return None
    
    async def cancel_order(self, order_id: str) -> Optional[Dict]:
        """Cancel an order"""
        try:
            if not self.access_token:
                await self.login()
            
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-UserType": "USER",
                "X-SourceID": "WEB",
                "X-ClientLocalIP": "127.0.0.1",
                "X-ClientPublicIP": "127.0.0.1",
                "X-MACAddress": "00:00:00:00:00:00",
                "X-PrivateKey": self.api_key
            }
            
            url = f"{self.base_url}/rest/secure/angelbroking/order/v1/cancelOrder"
            cancel_data = {"variety": "NORMAL", "orderid": order_id}
            
            async with self.session.post(url, json=cancel_data, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    logger.error(f"Error canceling order: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error canceling order: {e}")
            return None
    
    async def get_quote(self, symbol: str, exchange: str = "NSE") -> Optional[Dict]:
        """Get live quote"""
        try:
            if not self.access_token:
                await self.login()
            
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-UserType": "USER",
                "X-SourceID": "WEB",
                "X-ClientLocalIP": "127.0.0.1",
                "X-ClientPublicIP": "127.0.0.1",
                "X-MACAddress": "00:00:00:00:00:00",
                "X-PrivateKey": self.api_key
            }
            
            url = f"{self.base_url}/rest/secure/angelbroking/market/v1/quote"
            quote_data = {
                "mode": "FULL",
                "exchangeTokens": {
                    exchange: [symbol]
                }
            }
            
            async with self.session.post(url, json=quote_data, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("data", {}).get(f"{exchange}:{symbol}")
                else:
                    logger.error(f"Error getting quote: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting quote: {e}")
            return None
    
    def _generate_totp(self) -> str:
        """Generate TOTP for authentication"""
        try:
            import pyotp
            totp = pyotp.TOTP(self.totp_secret)
            return totp.now()
        except ImportError:
            logger.error("pyotp not installed. Install with: pip install pyotp")
            return "123456"  # Fallback for testing
        except Exception as e:
            logger.error(f"Error generating TOTP: {e}")
            return "123456"
    
    def _create_order_data(self, symbol: str, order_type: str, quantity: int, 
                          price: float, product: str = "CNC", 
                          variety: str = "NORMAL") -> Dict:
        """Create order data structure"""
        return {
            "variety": variety,
            "tradingsymbol": symbol,
            "symboltoken": "",  # Will be filled by API
            "transactiontype": order_type,
            "exchange": "NSE",
            "ordertype": "MARKET",
            "producttype": product,
            "duration": "DAY",
            "price": str(price),
            "squareoff": "0",
            "stoploss": "0",
            "quantity": str(quantity)
        }
