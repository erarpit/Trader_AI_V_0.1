"""
Technical Analysis Service
Advanced technical indicators and pattern recognition
"""

import pandas as pd
import numpy as np
import ta
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class TechnicalAnalyzer:
    def __init__(self):
        self.indicators = {}
        
    def analyze(self, df: pd.DataFrame) -> Dict:
        """Perform comprehensive technical analysis"""
        try:
            if len(df) < 20:
                return {"error": "Insufficient data for technical analysis"}
                
            # Calculate all technical indicators
            indicators = self._calculate_indicators(df)
            
            # Generate signals
            signals = self._generate_signals(df, indicators)
            
            # Calculate support and resistance
            support_resistance = self._calculate_support_resistance(df)
            
            # Pattern recognition
            patterns = self._detect_patterns(df)
            
            return {
                "indicators": indicators,
                "signals": signals,
                "support_resistance": support_resistance,
                "patterns": patterns,
                "summary": self._generate_summary(signals, indicators)
            }
            
        except Exception as e:
            logger.error(f"Error in technical analysis: {e}")
            return {"error": str(e)}
    
    def calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """Public method to calculate technical indicators"""
        return self._calculate_indicators(df)
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate all technical indicators"""
        try:
            indicators = {}
            data_length = len(df)
            
            # Price-based indicators (with data length checks)
            if data_length >= 20:
                indicators["sma_20"] = ta.trend.sma_indicator(df["close"], window=20)
            if data_length >= 50:
                indicators["sma_50"] = ta.trend.sma_indicator(df["close"], window=50)
            if data_length >= 200:
                indicators["sma_200"] = ta.trend.sma_indicator(df["close"], window=200)
            if data_length >= 12:
                indicators["ema_12"] = ta.trend.ema_indicator(df["close"], window=12)
            if data_length >= 26:
                indicators["ema_26"] = ta.trend.ema_indicator(df["close"], window=26)
            
            # Momentum indicators
            if data_length >= 14:
                indicators["rsi"] = ta.momentum.rsi(df["close"], window=14)
            if data_length >= 14:
                indicators["stoch"] = ta.momentum.stoch(df["high"], df["low"], df["close"])
            if data_length >= 14:
                indicators["williams_r"] = ta.momentum.williams_r(df["high"], df["low"], df["close"])
            if data_length >= 10:
                indicators["roc"] = ta.momentum.roc(df["close"], window=10)
            
            # Trend indicators (guard each calc)
            if data_length >= 12:
                try:
                    indicators["macd"] = ta.trend.macd(df["close"])
                    indicators["macd_signal"] = ta.trend.macd_signal(df["close"])
                    indicators["macd_histogram"] = ta.trend.macd_diff(df["close"])
                except Exception:
                    pass
            if data_length >= 14:
                try:
                    indicators["adx"] = ta.trend.adx(df["high"], df["low"], df["close"])
                except Exception:
                    pass
                try:
                    indicators["cci"] = ta.trend.cci(df["high"], df["low"], df["close"])
                except Exception:
                    pass
            
            # Volatility indicators
            if data_length >= 20:
                try:
                    indicators["bb_upper"] = ta.volatility.bollinger_hband(df["close"])
                    indicators["bb_middle"] = ta.volatility.bollinger_mavg(df["close"])
                    indicators["bb_lower"] = ta.volatility.bollinger_lband(df["close"])
                except Exception:
                    pass
            if data_length >= 14:
                try:
                    indicators["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"])
                except Exception:
                    pass
                try:
                    indicators["keltner_upper"] = ta.volatility.keltner_channel_hband(df["high"], df["low"], df["close"])
                    indicators["keltner_lower"] = ta.volatility.keltner_channel_lband(df["high"], df["low"], df["close"])
                except Exception:
                    pass
            
            # Volume indicators
            if "volume" in df.columns and data_length >= 14:
                try:
                    indicators["obv"] = ta.volume.on_balance_volume(df["close"], df["volume"])
                except Exception:
                    pass
                try:
                    indicators["vwap"] = ta.volume.volume_weighted_average_price(df["high"], df["low"], df["close"], df["volume"])
                except Exception:
                    pass
                try:
                    indicators["mfi"] = ta.volume.money_flow_index(df["high"], df["low"], df["close"], df["volume"])
                except Exception:
                    pass
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return {}
    
    def _generate_signals(self, df: pd.DataFrame, indicators: Dict) -> Dict:
        """Generate trading signals based on indicators"""
        try:
            signals = {
                "overall_signal": "HOLD",
                "strength": 0,
                "signals": []
            }
            
            current_price = df["close"].iloc[-1]
            signal_count = 0
            buy_signals = 0
            sell_signals = 0
            
            # RSI signals
            if "rsi" in indicators and not pd.isna(indicators["rsi"].iloc[-1]):
                rsi = indicators["rsi"].iloc[-1]
                if rsi < 30:
                    signals["signals"].append({"indicator": "RSI", "signal": "BUY", "value": rsi, "strength": 0.8})
                    buy_signals += 1
                    signal_count += 1
                elif rsi > 70:
                    signals["signals"].append({"indicator": "RSI", "signal": "SELL", "value": rsi, "strength": 0.8})
                    sell_signals += 1
                    signal_count += 1
            
            # MACD signals
            if "macd" in indicators and "macd_signal" in indicators:
                macd = indicators["macd"].iloc[-1]
                macd_signal = indicators["macd_signal"].iloc[-1]
                if not pd.isna(macd) and not pd.isna(macd_signal):
                    if macd > macd_signal:
                        signals["signals"].append({"indicator": "MACD", "signal": "BUY", "value": macd - macd_signal, "strength": 0.7})
                        buy_signals += 1
                        signal_count += 1
                    else:
                        signals["signals"].append({"indicator": "MACD", "signal": "SELL", "value": macd - macd_signal, "strength": 0.7})
                        sell_signals += 1
                        signal_count += 1
            
            # Moving Average signals
            if "sma_20" in indicators and "sma_50" in indicators:
                sma_20 = indicators["sma_20"].iloc[-1]
                sma_50 = indicators["sma_50"].iloc[-1]
                if not pd.isna(sma_20) and not pd.isna(sma_50):
                    if sma_20 > sma_50 and current_price > sma_20:
                        signals["signals"].append({"indicator": "MA", "signal": "BUY", "value": (sma_20 - sma_50) / sma_50, "strength": 0.6})
                        buy_signals += 1
                        signal_count += 1
                    elif sma_20 < sma_50 and current_price < sma_20:
                        signals["signals"].append({"indicator": "MA", "signal": "SELL", "value": (sma_20 - sma_50) / sma_50, "strength": 0.6})
                        sell_signals += 1
                        signal_count += 1
            
            # Bollinger Bands signals
            if "bb_upper" in indicators and "bb_lower" in indicators:
                bb_upper = indicators["bb_upper"].iloc[-1]
                bb_lower = indicators["bb_lower"].iloc[-1]
                if not pd.isna(bb_upper) and not pd.isna(bb_lower):
                    if current_price <= bb_lower:
                        signals["signals"].append({"indicator": "BB", "signal": "BUY", "value": (bb_lower - current_price) / current_price, "strength": 0.7})
                        buy_signals += 1
                        signal_count += 1
                    elif current_price >= bb_upper:
                        signals["signals"].append({"indicator": "BB", "signal": "SELL", "value": (current_price - bb_upper) / current_price, "strength": 0.7})
                        sell_signals += 1
                        signal_count += 1
            
            # Determine overall signal
            if signal_count > 0:
                if buy_signals > sell_signals:
                    signals["overall_signal"] = "BUY"
                    signals["strength"] = buy_signals / signal_count
                elif sell_signals > buy_signals:
                    signals["overall_signal"] = "SELL"
                    signals["strength"] = sell_signals / signal_count
                else:
                    signals["overall_signal"] = "HOLD"
                    signals["strength"] = 0.5
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return {"overall_signal": "HOLD", "strength": 0, "signals": []}
    
    def _calculate_support_resistance(self, df: pd.DataFrame) -> Dict:
        """Calculate support and resistance levels"""
        try:
            # Simple pivot points
            high = df["high"].rolling(window=20).max()
            low = df["low"].rolling(window=20).min()
            
            current_price = df["close"].iloc[-1]
            
            # Find recent highs and lows
            recent_highs = df["high"].tail(50).nlargest(3).values
            recent_lows = df["low"].tail(50).nsmallest(3).values
            
            # Calculate pivot levels
            pivot = (df["high"].iloc[-1] + df["low"].iloc[-1] + df["close"].iloc[-1]) / 3
            resistance1 = 2 * pivot - df["low"].iloc[-1]
            support1 = 2 * pivot - df["high"].iloc[-1]
            resistance2 = pivot + (df["high"].iloc[-1] - df["low"].iloc[-1])
            support2 = pivot - (df["high"].iloc[-1] - df["low"].iloc[-1])
            
            return {
                "pivot": pivot,
                "resistance_levels": [resistance1, resistance2] + recent_highs.tolist(),
                "support_levels": [support1, support2] + recent_lows.tolist(),
                "current_price": current_price
            }
            
        except Exception as e:
            logger.error(f"Error calculating support/resistance: {e}")
            return {"pivot": 0, "resistance_levels": [], "support_levels": [], "current_price": 0}
    
    def _detect_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect chart patterns"""
        try:
            patterns = []
            
            # Simple pattern detection
            if len(df) >= 20:
                # Head and Shoulders (simplified)
                recent_data = df.tail(20)
                highs = recent_data["high"].values
                
                if len(highs) >= 5:
                    # Look for three peaks
                    peak_indices = []
                    for i in range(2, len(highs) - 2):
                        if highs[i] > highs[i-1] and highs[i] > highs[i+1] and \
                           highs[i] > highs[i-2] and highs[i] > highs[i+2]:
                            peak_indices.append(i)
                    
                    if len(peak_indices) >= 3:
                        patterns.append({
                            "name": "Potential Head and Shoulders",
                            "type": "Reversal",
                            "confidence": 0.6
                        })
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
            return []
    
    def _generate_summary(self, signals: Dict, indicators: Dict) -> Dict:
        """Generate analysis summary"""
        try:
            summary = {
                "trend": "NEUTRAL",
                "momentum": "NEUTRAL",
                "volatility": "NORMAL",
                "overall_sentiment": "NEUTRAL"
            }
            
            # Determine trend
            if "sma_20" in indicators and "sma_50" in indicators:
                sma_20 = indicators["sma_20"].iloc[-1]
                sma_50 = indicators["sma_50"].iloc[-1]
                if not pd.isna(sma_20) and not pd.isna(sma_50):
                    if sma_20 > sma_50:
                        summary["trend"] = "BULLISH"
                    elif sma_20 < sma_50:
                        summary["trend"] = "BEARISH"
            
            # Determine momentum
            if "rsi" in indicators:
                rsi = indicators["rsi"].iloc[-1]
                if not pd.isna(rsi):
                    if rsi > 50:
                        summary["momentum"] = "BULLISH"
                    elif rsi < 50:
                        summary["momentum"] = "BEARISH"
            
            # Determine volatility
            if "atr" in indicators:
                atr = indicators["atr"].iloc[-1]
                if not pd.isna(atr):
                    # Simple volatility assessment
                    if atr > df["close"].iloc[-1] * 0.03:  # 3% ATR
                        summary["volatility"] = "HIGH"
                    elif atr < df["close"].iloc[-1] * 0.01:  # 1% ATR
                        summary["volatility"] = "LOW"
            
            # Overall sentiment
            if signals["overall_signal"] == "BUY":
                summary["overall_sentiment"] = "BULLISH"
            elif signals["overall_signal"] == "SELL":
                summary["overall_sentiment"] = "BEARISH"
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return {"trend": "NEUTRAL", "momentum": "NEUTRAL", "volatility": "NORMAL", "overall_sentiment": "NEUTRAL"}
