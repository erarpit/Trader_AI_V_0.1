"""
Sentiment Analysis Service
News and social media sentiment analysis for Indian stocks
"""

import requests
import asyncio
import aiohttp
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
import re
from textblob import TextBlob
import nltk
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        self.news_sources = [
            "https://economictimes.indiatimes.com",
            "https://www.moneycontrol.com",
            "https://www.business-standard.com",
            "https://www.livemint.com"
        ]
        
        # Download required NLTK data
        try:
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('punkt', quiet=True)
        except:
            pass
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Public method to analyze sentiment of text"""
        try:
            sentiment_score = self._analyze_text_sentiment(text)
            sentiment_label = self._get_sentiment_label(sentiment_score)
            
            return {
                "sentiment_score": sentiment_score,
                "sentiment_label": sentiment_label,
                "confidence": abs(sentiment_score),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {
                "sentiment_score": 0.0,
                "sentiment_label": "NEUTRAL",
                "confidence": 0.0,
                "timestamp": datetime.now().isoformat()
            }
    
    async def analyze_market_sentiment(self) -> Dict:
        """Analyze overall market sentiment"""
        try:
            # Get general market news
            market_articles = await self._fetch_market_news()
            
            # Analyze sentiment
            sentiment_scores = []
            for article in market_articles:
                sentiment = self._analyze_text_sentiment(article["content"])
                sentiment_scores.append(sentiment)
            
            # Calculate overall sentiment
            if sentiment_scores:
                avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                sentiment_label = self._get_sentiment_label(avg_sentiment)
                confidence = min(len(sentiment_scores) / 10, 1.0)  # More articles = higher confidence
            else:
                avg_sentiment = 0.0
                sentiment_label = "NEUTRAL"
                confidence = 0.0
            
            return {
                "market_sentiment_score": avg_sentiment,
                "market_sentiment_label": sentiment_label,
                "article_count": len(sentiment_scores),
                "confidence": confidence,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market sentiment: {e}")
            return {
                "market_sentiment_score": 0.0,
                "market_sentiment_label": "NEUTRAL",
                "article_count": 0,
                "confidence": 0.0,
                "timestamp": datetime.now().isoformat()
            }
    
    async def analyze_stock_sentiment(self, symbol: str) -> Dict:
        """Analyze sentiment for a specific stock"""
        try:
            # Get news articles
            news_articles = await self._fetch_news_articles(symbol)
            
            # Analyze sentiment
            sentiment_scores = []
            for article in news_articles:
                sentiment = self._analyze_text_sentiment(article["content"])
                sentiment_scores.append(sentiment)
            
            # Calculate overall sentiment
            if sentiment_scores:
                avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                sentiment_label = self._get_sentiment_label(avg_sentiment)
            else:
                avg_sentiment = 0
                sentiment_label = "NEUTRAL"
            
            return {
                "symbol": symbol,
                "sentiment_score": avg_sentiment,
                "sentiment_label": sentiment_label,
                "article_count": len(news_articles),
                "confidence": min(len(news_articles) / 10, 1.0),  # Confidence based on article count
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment for {symbol}: {e}")
            return {
                "symbol": symbol,
                "sentiment_score": 0,
                "sentiment_label": "NEUTRAL",
                "article_count": 0,
                "confidence": 0,
                "timestamp": datetime.now().isoformat()
            }
    
    async def analyze_market_sentiment(self) -> Dict:
        """Analyze overall market sentiment"""
        try:
            # Get market news
            market_articles = await self._fetch_market_news()
            
            # Analyze sentiment
            sentiment_scores = []
            for article in market_articles:
                sentiment = self._analyze_text_sentiment(article["content"])
                sentiment_scores.append(sentiment)
            
            # Calculate overall market sentiment
            if sentiment_scores:
                avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                sentiment_label = self._get_sentiment_label(avg_sentiment)
            else:
                avg_sentiment = 0
                sentiment_label = "NEUTRAL"
            
            return {
                "market_sentiment_score": avg_sentiment,
                "market_sentiment_label": sentiment_label,
                "article_count": len(market_articles),
                "confidence": min(len(market_articles) / 20, 1.0),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market sentiment: {e}")
            return {
                "market_sentiment_score": 0,
                "market_sentiment_label": "NEUTRAL",
                "article_count": 0,
                "confidence": 0,
                "timestamp": datetime.now().isoformat()
            }
    
    async def _fetch_news_articles(self, symbol: str) -> List[Dict]:
        """Fetch news articles related to a stock symbol"""
        articles = []
        
        try:
            # Search for news articles
            search_queries = [
                f"{symbol} stock news",
                f"{symbol} share price",
                f"{symbol} company news",
                f"{symbol} financial results"
            ]
            
            for query in search_queries:
                # Simulate news fetching (in real implementation, use news APIs)
                mock_articles = self._get_mock_news_articles(symbol, query)
                articles.extend(mock_articles)
            
            return articles[:10]  # Limit to 10 articles
            
        except Exception as e:
            logger.error(f"Error fetching news articles: {e}")
            return []
    
    async def _fetch_market_news(self) -> List[Dict]:
        """Fetch general market news"""
        try:
            # Simulate market news fetching
            mock_articles = [
                {
                    "title": "Indian Stock Market Update",
                    "content": "The Indian stock market showed mixed signals today with Nifty 50 gaining 0.5% while Sensex remained flat. Banking stocks led the gains while IT stocks faced selling pressure.",
                    "source": "Economic Times",
                    "published_at": datetime.now().isoformat()
                },
                {
                    "title": "Market Analysis: Bullish Trends Continue",
                    "content": "Analysts remain optimistic about the Indian equity markets with strong fundamentals and positive global cues supporting the rally.",
                    "source": "Money Control",
                    "published_at": datetime.now().isoformat()
                }
            ]
            
            return mock_articles
            
        except Exception as e:
            logger.error(f"Error fetching market news: {e}")
            return []
    
    async def _fetch_market_news(self) -> List[Dict]:
        """Fetch general market news articles"""
        try:
            # For now, return mock market news
            return self._get_mock_market_news()
            
        except Exception as e:
            logger.error(f"Error fetching market news: {e}")
            return []
    
    def _get_mock_news_articles(self, symbol: str, query: str) -> List[Dict]:
        """Generate mock news articles for testing"""
        mock_articles = [
            {
                "title": f"{symbol} Stock Analysis: Strong Fundamentals",
                "content": f"The {symbol} stock has shown strong performance with positive quarterly results and good market sentiment. Analysts are bullish on the stock with target price upgrades.",
                "source": "Economic Times",
                "published_at": datetime.now().isoformat()
            },
            {
                "title": f"{symbol} Share Price Movement",
                "content": f"{symbol} shares gained 2.5% in today's trading session on the back of positive news flow and strong institutional buying interest.",
                "source": "Money Control",
                "published_at": datetime.now().isoformat()
            },
            {
                "title": f"Market Update: {symbol} Performance",
                "content": f"The {symbol} stock continues to trade above its key resistance levels with strong volume support. Technical indicators suggest further upside potential.",
                "source": "Business Standard",
                "published_at": datetime.now().isoformat()
            }
        ]
        
        return mock_articles
    
    def _get_mock_market_news(self) -> List[Dict]:
        """Generate mock market news articles for testing"""
        mock_articles = [
            {
                "title": "Indian Stock Market Shows Strong Performance",
                "content": "The Indian stock market continues to show strong performance with Nifty 50 gaining 1.5% in today's session. Positive global cues and strong domestic fundamentals are driving the market higher.",
                "source": "Economic Times",
                "published_at": datetime.now().isoformat()
            },
            {
                "title": "Market Sentiment Remains Positive",
                "content": "Market sentiment remains positive with strong institutional buying and positive earnings outlook. Analysts expect the market to continue its upward trajectory in the coming weeks.",
                "source": "Money Control",
                "published_at": datetime.now().isoformat()
            },
            {
                "title": "FIIs Continue to Invest in Indian Markets",
                "content": "Foreign Institutional Investors (FIIs) continue to show strong interest in Indian markets with net inflows of over $500 million this week. This is a positive sign for market stability.",
                "source": "Business Standard",
                "published_at": datetime.now().isoformat()
            }
        ]
        
        return mock_articles
    
    def _analyze_text_sentiment(self, text: str) -> float:
        """Analyze sentiment of text using TextBlob"""
        try:
            # Clean text
            text = re.sub(r'[^\w\s]', '', text.lower())
            
            # Analyze sentiment
            blob = TextBlob(text)
            sentiment = blob.sentiment.polarity  # Range: -1 to 1
            
            return sentiment
            
        except Exception as e:
            logger.error(f"Error analyzing text sentiment: {e}")
            return 0.0
    
    def _get_sentiment_label(self, sentiment_score: float) -> str:
        """Convert sentiment score to label"""
        if sentiment_score > 0.1:
            return "POSITIVE"
        elif sentiment_score < -0.1:
            return "NEGATIVE"
        else:
            return "NEUTRAL"
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text"""
        try:
            # Simple keyword extraction
            words = re.findall(r'\b\w+\b', text.lower())
            
            # Filter out common words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
            
            keywords = [word for word in words if word not in stop_words and len(word) > 3]
            
            return keywords[:10]  # Return top 10 keywords
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
    
    async def get_sentiment_trends(self, symbol: str, days: int = 7) -> Dict:
        """Get sentiment trends over time"""
        try:
            # This would typically fetch historical sentiment data
            # For now, return mock data
            trends = []
            
            for i in range(days):
                date = datetime.now() - timedelta(days=i)
                sentiment_score = 0.1 + (i * 0.05)  # Mock trend
                
                trends.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "sentiment_score": sentiment_score,
                    "sentiment_label": self._get_sentiment_label(sentiment_score)
                })
            
            return {
                "symbol": symbol,
                "trends": trends,
                "period_days": days,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting sentiment trends: {e}")
            return {"symbol": symbol, "trends": [], "period_days": days, "timestamp": datetime.now().isoformat()}
