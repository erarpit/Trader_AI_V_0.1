"""
Advanced Candlestick Pattern Recognition Service
Detects all major candlestick patterns for trading signals
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class CandlestickPatternAnalyzer:
    def __init__(self):
        self.patterns = {}
        
    def analyze_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze all candlestick patterns in the data"""
        try:
            if len(df) < 5:
                return {"error": "Insufficient data for pattern analysis"}
            
            # Ensure we have required columns
            required_columns = ['open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required_columns):
                return {"error": "Missing required OHLC columns"}
            
            # Calculate pattern signals
            patterns = {
                "reversal_patterns": self._detect_reversal_patterns(df),
                "continuation_patterns": self._detect_continuation_patterns(df),
                "indecision_patterns": self._detect_indecision_patterns(df),
                "volume_patterns": self._detect_volume_patterns(df),
                "current_signals": self._get_current_signals(df),
                "pattern_strength": self._calculate_pattern_strength(df)
            }
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing candlestick patterns: {e}")
            return {"error": str(e)}
    
    def _detect_reversal_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect reversal patterns"""
        patterns = []
        
        # Hammer and Hanging Man
        hammer_patterns = self._detect_hammer_patterns(df)
        patterns.extend(hammer_patterns)
        
        # Doji patterns
        doji_patterns = self._detect_doji_patterns(df)
        patterns.extend(doji_patterns)
        
        # Engulfing patterns
        engulfing_patterns = self._detect_engulfing_patterns(df)
        patterns.extend(engulfing_patterns)
        
        # Harami patterns
        harami_patterns = self._detect_harami_patterns(df)
        patterns.extend(harami_patterns)
        
        # Morning/Evening Star
        star_patterns = self._detect_star_patterns(df)
        patterns.extend(star_patterns)
        
        # Three White Soldiers / Three Black Crows
        three_patterns = self._detect_three_patterns(df)
        patterns.extend(three_patterns)
        
        return patterns
    
    def _detect_continuation_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect continuation patterns"""
        patterns = []
        
        # Flag patterns
        flag_patterns = self._detect_flag_patterns(df)
        patterns.extend(flag_patterns)
        
        # Pennant patterns
        pennant_patterns = self._detect_pennant_patterns(df)
        patterns.extend(pennant_patterns)
        
        # Rectangle patterns
        rectangle_patterns = self._detect_rectangle_patterns(df)
        patterns.extend(rectangle_patterns)
        
        return patterns
    
    def _detect_indecision_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect indecision patterns"""
        patterns = []
        
        # Spinning tops
        spinning_top_patterns = self._detect_spinning_tops(df)
        patterns.extend(spinning_top_patterns)
        
        # High wave patterns
        high_wave_patterns = self._detect_high_wave_patterns(df)
        patterns.extend(high_wave_patterns)
        
        return patterns
    
    def _detect_volume_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect volume-based patterns"""
        patterns = []
        
        if 'volume' not in df.columns:
            return patterns
        
        # Volume spikes
        volume_spike_patterns = self._detect_volume_spikes(df)
        patterns.extend(volume_spike_patterns)
        
        # Volume divergence
        volume_divergence_patterns = self._detect_volume_divergence(df)
        patterns.extend(volume_divergence_patterns)
        
        return patterns
    
    def _detect_hammer_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect Hammer and Hanging Man patterns"""
        patterns = []
        
        for i in range(1, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            
            # Calculate body and shadow sizes
            body_size = abs(current['close'] - current['open'])
            total_range = current['high'] - current['low']
            upper_shadow = current['high'] - max(current['open'], current['close'])
            lower_shadow = min(current['open'], current['close']) - current['low']
            
            if total_range == 0:
                continue
                
            # Hammer criteria
            is_hammer = (
                lower_shadow >= 2 * body_size and  # Long lower shadow
                upper_shadow <= body_size and      # Small upper shadow
                body_size <= total_range * 0.3     # Small body
            )
            
            if is_hammer:
                pattern_type = "Hammer" if current['close'] > current['open'] else "Hanging Man"
                signal = "BULLISH" if current['close'] > current['open'] else "BEARISH"
                
                patterns.append({
                    "pattern": pattern_type,
                    "signal": signal,
                    "index": i,
                    "date": df.index[i],
                    "confidence": self._calculate_hammer_confidence(current, prev),
                    "price": current['close'],
                    "description": f"{pattern_type} pattern detected - {signal} reversal signal"
                })
        
        return patterns
    
    def _detect_doji_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect Doji patterns"""
        patterns = []
        
        for i in range(len(df)):
            current = df.iloc[i]
            
            # Calculate body size
            body_size = abs(current['close'] - current['open'])
            total_range = current['high'] - current['low']
            
            if total_range == 0:
                continue
            
            # Doji criteria (body is very small)
            is_doji = body_size <= total_range * 0.1
            
            if is_doji:
                # Determine doji type
                upper_shadow = current['high'] - max(current['open'], current['close'])
                lower_shadow = min(current['open'], current['close']) - current['low']
                
                if upper_shadow > lower_shadow * 2:
                    doji_type = "Gravestone Doji"
                    signal = "BEARISH"
                elif lower_shadow > upper_shadow * 2:
                    doji_type = "Dragonfly Doji"
                    signal = "BULLISH"
                else:
                    doji_type = "Standard Doji"
                    signal = "NEUTRAL"
                
                patterns.append({
                    "pattern": doji_type,
                    "signal": signal,
                    "index": i,
                    "date": df.index[i],
                    "confidence": 0.7,
                    "price": current['close'],
                    "description": f"{doji_type} pattern detected - {signal} signal"
                })
        
        return patterns
    
    def _detect_engulfing_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect Engulfing patterns"""
        patterns = []
        
        for i in range(1, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            
            # Bullish Engulfing
            bullish_engulfing = (
                prev['close'] < prev['open'] and  # Previous candle is bearish
                current['close'] > current['open'] and  # Current candle is bullish
                current['open'] < prev['close'] and  # Current opens below previous close
                current['close'] > prev['open']  # Current closes above previous open
            )
            
            # Bearish Engulfing
            bearish_engulfing = (
                prev['close'] > prev['open'] and  # Previous candle is bullish
                current['close'] < current['open'] and  # Current candle is bearish
                current['open'] > prev['close'] and  # Current opens above previous close
                current['close'] < prev['open']  # Current closes below previous open
            )
            
            if bullish_engulfing:
                patterns.append({
                    "pattern": "Bullish Engulfing",
                    "signal": "BULLISH",
                    "index": i,
                    "date": df.index[i],
                    "confidence": self._calculate_engulfing_confidence(current, prev),
                    "price": current['close'],
                    "description": "Bullish Engulfing pattern - Strong bullish reversal signal"
                })
            
            if bearish_engulfing:
                patterns.append({
                    "pattern": "Bearish Engulfing",
                    "signal": "BEARISH",
                    "index": i,
                    "date": df.index[i],
                    "confidence": self._calculate_engulfing_confidence(current, prev),
                    "price": current['close'],
                    "description": "Bearish Engulfing pattern - Strong bearish reversal signal"
                })
        
        return patterns
    
    def _detect_harami_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect Harami patterns"""
        patterns = []
        
        for i in range(1, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            
            # Calculate body sizes
            prev_body = abs(prev['close'] - prev['open'])
            current_body = abs(current['close'] - current['open'])
            
            if prev_body == 0 or current_body == 0:
                continue
            
            # Harami criteria
            is_harami = (
                prev_body > current_body * 2 and  # Previous body is much larger
                current['high'] < prev['high'] and  # Current high is lower
                current['low'] > prev['low']  # Current low is higher
            )
            
            if is_harami:
                # Determine signal based on previous candle
                if prev['close'] < prev['open']:  # Previous was bearish
                    pattern_type = "Bullish Harami"
                    signal = "BULLISH"
                else:  # Previous was bullish
                    pattern_type = "Bearish Harami"
                    signal = "BEARISH"
                
                patterns.append({
                    "pattern": pattern_type,
                    "signal": signal,
                    "index": i,
                    "date": df.index[i],
                    "confidence": 0.6,
                    "price": current['close'],
                    "description": f"{pattern_type} pattern - {signal} reversal signal"
                })
        
        return patterns
    
    def _detect_star_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect Morning/Evening Star patterns"""
        patterns = []
        
        for i in range(2, len(df)):
            first = df.iloc[i-2]
            second = df.iloc[i-1]
            third = df.iloc[i]
            
            # Morning Star (Bullish)
            morning_star = (
                first['close'] < first['open'] and  # First candle is bearish
                abs(second['close'] - second['open']) < (first['high'] - first['low']) * 0.3 and  # Second is small
                third['close'] > third['open'] and  # Third is bullish
                third['close'] > (first['open'] + first['close']) / 2  # Third closes above first midpoint
            )
            
            # Evening Star (Bearish)
            evening_star = (
                first['close'] > first['open'] and  # First candle is bullish
                abs(second['close'] - second['open']) < (first['high'] - first['low']) * 0.3 and  # Second is small
                third['close'] < third['open'] and  # Third is bearish
                third['close'] < (first['open'] + first['close']) / 2  # Third closes below first midpoint
            )
            
            if morning_star:
                patterns.append({
                    "pattern": "Morning Star",
                    "signal": "BULLISH",
                    "index": i,
                    "date": df.index[i],
                    "confidence": 0.8,
                    "price": third['close'],
                    "description": "Morning Star pattern - Strong bullish reversal signal"
                })
            
            if evening_star:
                patterns.append({
                    "pattern": "Evening Star",
                    "signal": "BEARISH",
                    "index": i,
                    "date": df.index[i],
                    "confidence": 0.8,
                    "price": third['close'],
                    "description": "Evening Star pattern - Strong bearish reversal signal"
                })
        
        return patterns
    
    def _detect_three_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect Three White Soldiers / Three Black Crows"""
        patterns = []
        
        for i in range(2, len(df)):
            first = df.iloc[i-2]
            second = df.iloc[i-1]
            third = df.iloc[i]
            
            # Three White Soldiers (Bullish)
            three_white_soldiers = (
                first['close'] > first['open'] and
                second['close'] > second['open'] and
                third['close'] > third['open'] and
                second['open'] > first['close'] and
                third['open'] > second['close'] and
                first['close'] > first['open'] and
                second['close'] > second['open'] and
                third['close'] > third['open']
            )
            
            # Three Black Crows (Bearish)
            three_black_crows = (
                first['close'] < first['open'] and
                second['close'] < second['open'] and
                third['close'] < third['open'] and
                second['open'] < first['close'] and
                third['open'] < second['close']
            )
            
            if three_white_soldiers:
                patterns.append({
                    "pattern": "Three White Soldiers",
                    "signal": "BULLISH",
                    "index": i,
                    "date": df.index[i],
                    "confidence": 0.7,
                    "price": third['close'],
                    "description": "Three White Soldiers - Strong bullish continuation"
                })
            
            if three_black_crows:
                patterns.append({
                    "pattern": "Three Black Crows",
                    "signal": "BEARISH",
                    "index": i,
                    "date": df.index[i],
                    "confidence": 0.7,
                    "price": third['close'],
                    "description": "Three Black Crows - Strong bearish continuation"
                })
        
        return patterns
    
    def _detect_flag_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect Flag patterns"""
        patterns = []
        
        # Look for flag patterns in recent data
        for i in range(10, len(df)):
            recent_data = df.iloc[i-10:i+1]
            
            # Check for upward flag
            if self._is_upward_flag(recent_data):
                patterns.append({
                    "pattern": "Bull Flag",
                    "signal": "BULLISH",
                    "index": i,
                    "date": df.index[i],
                    "confidence": 0.6,
                    "price": df.iloc[i]['close'],
                    "description": "Bull Flag pattern - Bullish continuation"
                })
            
            # Check for downward flag
            if self._is_downward_flag(recent_data):
                patterns.append({
                    "pattern": "Bear Flag",
                    "signal": "BEARISH",
                    "index": i,
                    "date": df.index[i],
                    "confidence": 0.6,
                    "price": df.iloc[i]['close'],
                    "description": "Bear Flag pattern - Bearish continuation"
                })
        
        return patterns
    
    def _detect_pennant_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect Pennant patterns"""
        patterns = []
        
        for i in range(10, len(df)):
            recent_data = df.iloc[i-10:i+1]
            
            if self._is_pennant(recent_data):
                patterns.append({
                    "pattern": "Pennant",
                    "signal": "NEUTRAL",
                    "index": i,
                    "date": df.index[i],
                    "confidence": 0.5,
                    "price": df.iloc[i]['close'],
                    "description": "Pennant pattern - Consolidation before breakout"
                })
        
        return patterns
    
    def _detect_rectangle_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect Rectangle patterns"""
        patterns = []
        
        for i in range(10, len(df)):
            recent_data = df.iloc[i-10:i+1]
            
            if self._is_rectangle(recent_data):
                patterns.append({
                    "pattern": "Rectangle",
                    "signal": "NEUTRAL",
                    "index": i,
                    "date": df.index[i],
                    "confidence": 0.5,
                    "price": df.iloc[i]['close'],
                    "description": "Rectangle pattern - Sideways consolidation"
                })
        
        return patterns
    
    def _detect_spinning_tops(self, df: pd.DataFrame) -> List[Dict]:
        """Detect Spinning Top patterns"""
        patterns = []
        
        for i in range(len(df)):
            current = df.iloc[i]
            
            body_size = abs(current['close'] - current['open'])
            total_range = current['high'] - current['low']
            upper_shadow = current['high'] - max(current['open'], current['close'])
            lower_shadow = min(current['open'], current['close']) - current['low']
            
            if total_range == 0:
                continue
            
            # Spinning top criteria
            is_spinning_top = (
                body_size <= total_range * 0.3 and  # Small body
                upper_shadow >= body_size and  # Upper shadow at least as long as body
                lower_shadow >= body_size  # Lower shadow at least as long as body
            )
            
            if is_spinning_top:
                patterns.append({
                    "pattern": "Spinning Top",
                    "signal": "NEUTRAL",
                    "index": i,
                    "date": df.index[i],
                    "confidence": 0.5,
                    "price": current['close'],
                    "description": "Spinning Top pattern - Indecision in market"
                })
        
        return patterns
    
    def _detect_high_wave_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect High Wave patterns"""
        patterns = []
        
        for i in range(len(df)):
            current = df.iloc[i]
            
            body_size = abs(current['close'] - current['open'])
            total_range = current['high'] - current['low']
            upper_shadow = current['high'] - max(current['open'], current['close'])
            lower_shadow = min(current['open'], current['close']) - current['low']
            
            if total_range == 0:
                continue
            
            # High wave criteria
            is_high_wave = (
                body_size <= total_range * 0.1 and  # Very small body
                upper_shadow >= total_range * 0.4 and  # Long upper shadow
                lower_shadow >= total_range * 0.4  # Long lower shadow
            )
            
            if is_high_wave:
                patterns.append({
                    "pattern": "High Wave",
                    "signal": "NEUTRAL",
                    "index": i,
                    "date": df.index[i],
                    "confidence": 0.6,
                    "price": current['close'],
                    "description": "High Wave pattern - High volatility, indecision"
                })
        
        return patterns
    
    def _detect_volume_spikes(self, df: pd.DataFrame) -> List[Dict]:
        """Detect volume spike patterns"""
        patterns = []
        
        if 'volume' not in df.columns:
            return patterns
        
        volume_ma = df['volume'].rolling(window=20).mean()
        
        for i in range(20, len(df)):
            current_volume = df.iloc[i]['volume']
            avg_volume = volume_ma.iloc[i]
            
            if avg_volume == 0:
                continue
            
            volume_ratio = current_volume / avg_volume
            
            if volume_ratio >= 2.0:  # 2x average volume
                patterns.append({
                    "pattern": "Volume Spike",
                    "signal": "NEUTRAL",
                    "index": i,
                    "date": df.index[i],
                    "confidence": min(volume_ratio / 3, 1.0),
                    "price": df.iloc[i]['close'],
                    "description": f"Volume spike - {volume_ratio:.1f}x average volume"
                })
        
        return patterns
    
    def _detect_volume_divergence(self, df: pd.DataFrame) -> List[Dict]:
        """Detect volume divergence patterns"""
        patterns = []
        
        if 'volume' not in df.columns or len(df) < 20:
            return patterns
        
        # Look for price vs volume divergence
        price_trend = df['close'].tail(10).pct_change().mean()
        volume_trend = df['volume'].tail(10).pct_change().mean()
        
        if abs(price_trend) > 0.02 and abs(volume_trend) > 0.02:
            if (price_trend > 0 and volume_trend < 0) or (price_trend < 0 and volume_trend > 0):
                signal = "BEARISH" if price_trend > 0 else "BULLISH"
                patterns.append({
                    "pattern": "Volume Divergence",
                    "signal": signal,
                    "index": len(df) - 1,
                    "date": df.index[-1],
                    "confidence": 0.6,
                    "price": df.iloc[-1]['close'],
                    "description": f"Volume divergence - {signal} signal"
                })
        
        return patterns
    
    def _get_current_signals(self, df: pd.DataFrame) -> Dict:
        """Get current trading signals based on recent patterns"""
        recent_patterns = []
        
        # Get patterns from last 5 candles
        for pattern_type in ['reversal_patterns', 'continuation_patterns', 'indecision_patterns']:
            patterns = self.analyze_patterns(df) if 'patterns' not in locals() else patterns
            if pattern_type in patterns:
                recent_patterns.extend([p for p in patterns[pattern_type] if p['index'] >= len(df) - 5])
        
        # Calculate overall signal
        bullish_count = len([p for p in recent_patterns if p['signal'] == 'BULLISH'])
        bearish_count = len([p for p in recent_patterns if p['signal'] == 'BEARISH'])
        
        if bullish_count > bearish_count:
            overall_signal = 'BUY'
            confidence = min(bullish_count / 3, 1.0)
        elif bearish_count > bullish_count:
            overall_signal = 'SELL'
            confidence = min(bearish_count / 3, 1.0)
        else:
            overall_signal = 'HOLD'
            confidence = 0.5
        
        return {
            "signal": overall_signal,
            "confidence": confidence,
            "pattern_count": len(recent_patterns),
            "recent_patterns": recent_patterns[-3:] if recent_patterns else []
        }
    
    def _calculate_pattern_strength(self, df: pd.DataFrame) -> float:
        """Calculate overall pattern strength"""
        all_patterns = []
        
        for pattern_type in ['reversal_patterns', 'continuation_patterns', 'indecision_patterns']:
            patterns = self.analyze_patterns(df) if 'patterns' not in locals() else patterns
            if pattern_type in patterns:
                all_patterns.extend(patterns[pattern_type])
        
        if not all_patterns:
            return 0.0
        
        # Calculate weighted average confidence
        total_confidence = sum(p['confidence'] for p in all_patterns)
        return min(total_confidence / len(all_patterns), 1.0)
    
    def _calculate_hammer_confidence(self, current: pd.Series, prev: pd.Series) -> float:
        """Calculate confidence for hammer patterns"""
        body_size = abs(current['close'] - current['open'])
        total_range = current['high'] - current['low']
        lower_shadow = min(current['open'], current['close']) - current['low']
        
        if total_range == 0:
            return 0.0
        
        # Higher confidence for longer lower shadow
        shadow_ratio = lower_shadow / total_range
        return min(shadow_ratio * 2, 1.0)
    
    def _calculate_engulfing_confidence(self, current: pd.Series, prev: pd.Series) -> float:
        """Calculate confidence for engulfing patterns"""
        current_body = abs(current['close'] - current['open'])
        prev_body = abs(prev['close'] - prev['open'])
        
        if prev_body == 0:
            return 0.0
        
        # Higher confidence for larger engulfing
        engulfing_ratio = current_body / prev_body
        return min(engulfing_ratio / 2, 1.0)
    
    def _is_upward_flag(self, data: pd.DataFrame) -> bool:
        """Check if data forms an upward flag pattern"""
        if len(data) < 5:
            return False
        
        # Check for initial upward move
        first_half = data.iloc[:len(data)//2]
        second_half = data.iloc[len(data)//2:]
        
        first_trend = first_half['close'].iloc[-1] - first_half['close'].iloc[0]
        second_trend = second_half['close'].iloc[-1] - second_half['close'].iloc[0]
        
        return first_trend > 0 and abs(second_trend) < abs(first_trend) * 0.5
    
    def _is_downward_flag(self, data: pd.DataFrame) -> bool:
        """Check if data forms a downward flag pattern"""
        if len(data) < 5:
            return False
        
        first_half = data.iloc[:len(data)//2]
        second_half = data.iloc[len(data)//2:]
        
        first_trend = first_half['close'].iloc[-1] - first_half['close'].iloc[0]
        second_trend = second_half['close'].iloc[-1] - second_half['close'].iloc[0]
        
        return first_trend < 0 and abs(second_trend) < abs(first_trend) * 0.5
    
    def _is_pennant(self, data: pd.DataFrame) -> bool:
        """Check if data forms a pennant pattern"""
        if len(data) < 5:
            return False
        
        # Check for converging trend lines
        highs = data['high'].values
        lows = data['low'].values
        
        # Simple check for converging pattern
        high_slope = (highs[-1] - highs[0]) / len(highs)
        low_slope = (lows[-1] - lows[0]) / len(lows)
        
        return abs(high_slope) < 0.1 and abs(low_slope) < 0.1
    
    def _is_rectangle(self, data: pd.DataFrame) -> bool:
        """Check if data forms a rectangle pattern"""
        if len(data) < 5:
            return False
        
        # Check for sideways movement
        price_range = data['high'].max() - data['low'].min()
        avg_price = data['close'].mean()
        
        return price_range < avg_price * 0.05  # Less than 5% range
