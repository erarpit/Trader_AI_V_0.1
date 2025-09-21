"""
BSE API integration for Indian stock market data
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

class BSEAPI:
    def __init__(self):
        self.base_url = os.getenv("BSE_BASE_URL", "https://api.bseindia.com/BseIndiaAPI/api")
        self.session = None
        
    async def __aenter__(self):
        self.session = httpx.AsyncClient(
            timeout=30.0,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://www.bseindia.com/",
                "Origin": "https://www.bseindia.com"
            }
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.aclose()
            
    async def get_quote(self, scrip_code: str) -> Optional[Dict]:
        """Get live quote for a BSE scrip code"""
        try:
            url = f"{self.base_url}/StockReachGraph/getScripFullDetails"
            params = {"scripcode": scrip_code}
            
            response = await self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            return self._parse_quote_data(data)
            
        except Exception as e:
            logger.error(f"Error getting BSE quote for {scrip_code}: {e}")
            return None
            
    async def get_historical_data(self, scrip_code: str, 
                                 from_date: str = None, to_date: str = None) -> Optional[List[Dict]]:
        """Get historical data for a BSE scrip"""
        try:
            if not from_date:
                from_date = (datetime.now() - timedelta(days=30)).strftime("%d-%m-%Y")
            if not to_date:
                to_date = datetime.now().strftime("%d-%m-%Y")
                
            url = f"{self.base_url}/StockReachGraph/getScripFullDetails"
            params = {
                "scripcode": scrip_code,
                "flag": "",
                "fromdate": from_date,
                "todate": to_date
            }
            
            response = await self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            return self._parse_historical_data(data)
            
        except Exception as e:
            logger.error(f"Error getting BSE historical data for {scrip_code}: {e}")
            return None
            
    async def get_market_status(self) -> Optional[Dict]:
        """Get BSE market status"""
        try:
            url = f"{self.base_url}/getMktStatus"
            response = await self.session.get(url)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error getting BSE market status: {e}")
            return None
            
    async def get_top_gainers(self) -> Optional[List[Dict]]:
        """Get top gaining stocks from BSE"""
        try:
            url = f"{self.base_url}/getTopGainersData"
            response = await self.session.get(url)
            response.raise_for_status()
            
            data = response.json()
            return data.get("Table", [])
            
        except Exception as e:
            logger.error(f"Error getting BSE top gainers: {e}")
            return None
            
    async def get_top_losers(self) -> Optional[List[Dict]]:
        """Get top losing stocks from BSE"""
        try:
            url = f"{self.base_url}/getTopLosersData"
            response = await self.session.get(url)
            response.raise_for_status()
            
            data = response.json()
            return data.get("Table", [])
            
        except Exception as e:
            logger.error(f"Error getting BSE top losers: {e}")
            return None
            
    def _parse_quote_data(self, data: Dict) -> Dict:
        """Parse quote data from BSE response"""
        try:
            # BSE API response structure may vary
            return {
                "symbol": data.get("ScripCode", ""),
                "company_name": data.get("CompanyName", ""),
                "last_price": data.get("LastPrice", 0),
                "change": data.get("Change", 0),
                "change_percent": data.get("PChange", 0),
                "open": data.get("Open", 0),
                "high": data.get("High", 0),
                "low": data.get("Low", 0),
                "close": data.get("PrevClose", 0),
                "volume": data.get("TotalTradedVolume", 0),
                "value": data.get("TotalTradedValue", 0),
                "market_cap": data.get("MarketCap", 0),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error parsing BSE quote data: {e}")
            return {}
            
    def _parse_historical_data(self, data: Dict) -> List[Dict]:
        """Parse historical data from BSE response"""
        try:
            historical_data = data.get("Table", [])
            parsed_data = []
            
            for item in historical_data:
                parsed_data.append({
                    "date": item.get("Date", ""),
                    "open": item.get("Open", 0),
                    "high": item.get("High", 0),
                    "low": item.get("Low", 0),
                    "close": item.get("Close", 0),
                    "volume": item.get("Volume", 0),
                    "timestamp": datetime.now().isoformat()
                })
                
            return parsed_data
            
        except Exception as e:
            logger.error(f"Error parsing BSE historical data: {e}")
            return []
