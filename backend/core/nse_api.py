"""
NSE API integration for Indian stock market data
"""

import httpx
import asyncio
import json
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class NSEAPI:
    def __init__(self):
        self.base_url = os.getenv("NSE_BASE_URL", "https://www.nseindia.com/api")
        self.session = None
        self.cookies = None
        
    async def __aenter__(self):
        self.session = httpx.AsyncClient(
            timeout=30.0,
            headers={
                "User-Agent": os.getenv("NSE_HEADERS_USER_AGENT", 
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"),
                "Accept": os.getenv("NSE_HEADERS_ACCEPT", 
                    "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"),
                "Accept-Language": os.getenv("NSE_HEADERS_ACCEPT_LANGUAGE", "en-US,en;q=0.5"),
                "Accept-Encoding": os.getenv("NSE_HEADERS_ACCEPT_ENCODING", "gzip, deflate"),
                "Connection": os.getenv("NSE_HEADERS_CONNECTION", "keep-alive"),
                "Upgrade-Insecure-Requests": "1"
            }
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.aclose()
            
    async def _get_session_cookies(self):
        """Get session cookies from NSE"""
        try:
            response = await self.session.get("https://www.nseindia.com/")
            self.cookies = response.cookies
            logger.info("NSE session cookies obtained")
        except Exception as e:
            logger.error(f"Error getting NSE session cookies: {e}")
            
    async def get_quote(self, symbol: str) -> Optional[Dict]:
        """Get live quote for a symbol"""
        try:
            if not self.cookies:
                await self._get_session_cookies()
                
            url = f"{self.base_url}/quote-equity"
            params = {"symbol": symbol}
            
            response = await self.session.get(url, params=params, cookies=self.cookies)
            response.raise_for_status()
            
            data = response.json()
            return self._parse_quote_data(data)
            
        except Exception as e:
            logger.error(f"Error getting quote for {symbol}: {e}")
            return None
            
    async def get_historical_data(self, symbol: str, series: str = "EQ", 
                                 from_date: str = None, to_date: str = None) -> Optional[List[Dict]]:
        """Get historical data for a symbol"""
        try:
            if not self.cookies:
                await self._get_session_cookies()
                
            if not from_date:
                from_date = (datetime.now() - timedelta(days=30)).strftime("%d-%m-%Y")
            if not to_date:
                to_date = datetime.now().strftime("%d-%m-%Y")
                
            url = f"{self.base_url}/historicalChart/equity/{symbol}"
            params = {
                "series": series,
                "from": from_date,
                "to": to_date
            }
            
            response = await self.session.get(url, params=params, cookies=self.cookies)
            response.raise_for_status()
            
            data = response.json()
            return self._parse_historical_data(data)
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return None
            
    async def get_market_status(self) -> Optional[Dict]:
        """Get current market status"""
        try:
            if not self.cookies:
                await self._get_session_cookies()
                
            url = f"{self.base_url}/marketStatus"
            response = await self.session.get(url, cookies=self.cookies)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error getting market status: {e}")
            return None
            
    async def get_top_gainers(self) -> Optional[List[Dict]]:
        """Get top gaining stocks"""
        try:
            if not self.cookies:
                await self._get_session_cookies()
                
            url = f"{self.base_url}/live-analysis-variations"
            params = {"index": "gainers"}
            
            response = await self.session.get(url, params=params, cookies=self.cookies)
            response.raise_for_status()
            
            data = response.json()
            return data.get("data", [])
            
        except Exception as e:
            logger.error(f"Error getting top gainers: {e}")
            return None
            
    async def get_top_losers(self) -> Optional[List[Dict]]:
        """Get top losing stocks"""
        try:
            if not self.cookies:
                await self._get_session_cookies()
                
            url = f"{self.base_url}/live-analysis-variations"
            params = {"index": "loosers"}
            
            response = await self.session.get(url, params=params, cookies=self.cookies)
            response.raise_for_status()
            
            data = response.json()
            return data.get("data", [])
            
        except Exception as e:
            logger.error(f"Error getting top losers: {e}")
            return None
            
    def _parse_quote_data(self, data: Dict) -> Dict:
        """Parse quote data from NSE response"""
        try:
            info = data.get("info", {})
            price_info = data.get("priceInfo", {})
            
            return {
                "symbol": info.get("symbol", ""),
                "company_name": info.get("companyName", ""),
                "last_price": price_info.get("lastPrice", 0),
                "change": price_info.get("change", 0),
                "change_percent": price_info.get("pChange", 0),
                "open": price_info.get("open", 0),
                "high": price_info.get("intraDayHighLow", {}).get("max", 0),
                "low": price_info.get("intraDayHighLow", {}).get("min", 0),
                "close": price_info.get("previousClose", 0),
                "volume": price_info.get("totalTradedVolume", 0),
                "value": price_info.get("totalTradedValue", 0),
                "market_cap": price_info.get("marketCap", 0),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error parsing quote data: {e}")
            return {}
            
    def _parse_historical_data(self, data: Dict) -> List[Dict]:
        """Parse historical data from NSE response"""
        try:
            historical_data = data.get("data", [])
            parsed_data = []
            
            for item in historical_data:
                parsed_data.append({
                    "date": item.get("mTIMESTAMP", ""),
                    "open": item.get("OPEN", 0),
                    "high": item.get("HIGH", 0),
                    "low": item.get("LOW", 0),
                    "close": item.get("CH_TIMESTAMP", 0),
                    "volume": item.get("CH_TIMESTAMP", 0),
                    "timestamp": datetime.now().isoformat()
                })
                
            return parsed_data
            
        except Exception as e:
            logger.error(f"Error parsing historical data: {e}")
            return []
