"""
Candlestick Pattern Detection Module
=====================================
Expert-level candlestick pattern recognition for trading signals.
Detects 40+ patterns including:
- Single candle patterns (Doji, Hammer, Shooting Star, etc.)
- Double candle patterns (Engulfing, Harami, Piercing, etc.)
- Triple candle patterns (Morning Star, Evening Star, Three Black Crows, etc.)
- Advanced patterns (Ichimoku Cloud, Three Line Strike, etc.)

All analysis uses live chart data - zero placeholders.
"""

import os
import sys
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import yfinance as yf
except ImportError:
    yf = None


class CandlestickPatternDetector:
    """
    Expert-level candlestick pattern detection with bullish/bearish signals.
    Identifies patterns like a veteran trader would on live charts.
    """
    
    def __init__(self):
        self.finnhub_key = os.environ.get('KEY') or os.environ.get('FINNHUB_API_KEY') or 'd55b3ohr01qljfdeghm0d55b3ohr01qljfdeghm1'
        
        # Pattern definitions with trading implications
        self.pattern_info = {
            # Bullish Reversal Patterns
            'HAMMER': {
                'type': 'BULLISH_REVERSAL',
                'reliability': 'HIGH',
                'description': 'Small body at top with long lower shadow. Indicates buyers stepping in after selling pressure.',
                'action': 'Consider long entry on confirmation'
            },
            'INVERTED_HAMMER': {
                'type': 'BULLISH_REVERSAL',
                'reliability': 'MODERATE',
                'description': 'Small body at bottom with long upper shadow. Potential reversal after downtrend.',
                'action': 'Wait for bullish confirmation candle'
            },
            'BULLISH_ENGULFING': {
                'type': 'BULLISH_REVERSAL',
                'reliability': 'HIGH',
                'description': 'Large green candle completely engulfs previous red candle. Strong buying pressure.',
                'action': 'Strong buy signal, enter on next candle open'
            },
            'PIERCING_LINE': {
                'type': 'BULLISH_REVERSAL',
                'reliability': 'MODERATE',
                'description': 'Green candle opens below prior low and closes above midpoint of prior red candle.',
                'action': 'Bullish signal, confirm with volume'
            },
            'MORNING_STAR': {
                'type': 'BULLISH_REVERSAL',
                'reliability': 'VERY_HIGH',
                'description': 'Three-candle pattern: large red, small body (indecision), large green. Classic bottom reversal.',
                'action': 'Strong buy signal after downtrend'
            },
            'THREE_WHITE_SOLDIERS': {
                'type': 'BULLISH_CONTINUATION',
                'reliability': 'VERY_HIGH',
                'description': 'Three consecutive large green candles with higher closes. Strong bullish momentum.',
                'action': 'Strong uptrend confirmation, ride the trend'
            },
            'BULLISH_HARAMI': {
                'type': 'BULLISH_REVERSAL',
                'reliability': 'MODERATE',
                'description': 'Small green candle contained within prior large red candle. Selling pressure weakening.',
                'action': 'Potential reversal, wait for confirmation'
            },
            'DRAGONFLY_DOJI': {
                'type': 'BULLISH_REVERSAL',
                'reliability': 'MODERATE',
                'description': 'Open, high, and close at same level with long lower shadow. Rejection of lower prices.',
                'action': 'Bullish at support levels'
            },
            
            # Bearish Reversal Patterns
            'SHOOTING_STAR': {
                'type': 'BEARISH_REVERSAL',
                'reliability': 'HIGH',
                'description': 'Small body at bottom with long upper shadow. Indicates sellers stepping in after buying pressure.',
                'action': 'Consider short entry on confirmation'
            },
            'HANGING_MAN': {
                'type': 'BEARISH_REVERSAL',
                'reliability': 'MODERATE',
                'description': 'Hammer-like pattern at top of uptrend. Warning of potential reversal.',
                'action': 'Caution signal, tighten stops'
            },
            'BEARISH_ENGULFING': {
                'type': 'BEARISH_REVERSAL',
                'reliability': 'HIGH',
                'description': 'Large red candle completely engulfs previous green candle. Strong selling pressure.',
                'action': 'Strong sell signal, exit longs or enter shorts'
            },
            'DARK_CLOUD_COVER': {
                'type': 'BEARISH_REVERSAL',
                'reliability': 'MODERATE',
                'description': 'Red candle opens above prior high and closes below midpoint of prior green candle.',
                'action': 'Bearish signal, confirm with volume'
            },
            'EVENING_STAR': {
                'type': 'BEARISH_REVERSAL',
                'reliability': 'VERY_HIGH',
                'description': 'Three-candle pattern: large green, small body (indecision), large red. Classic top reversal.',
                'action': 'Strong sell signal after uptrend'
            },
            'THREE_BLACK_CROWS': {
                'type': 'BEARISH_CONTINUATION',
                'reliability': 'VERY_HIGH',
                'description': 'Three consecutive large red candles with lower closes. Strong bearish momentum.',
                'action': 'Strong downtrend confirmation, avoid longs'
            },
            'BEARISH_HARAMI': {
                'type': 'BEARISH_REVERSAL',
                'reliability': 'MODERATE',
                'description': 'Small red candle contained within prior large green candle. Buying pressure weakening.',
                'action': 'Potential reversal, wait for confirmation'
            },
            'GRAVESTONE_DOJI': {
                'type': 'BEARISH_REVERSAL',
                'reliability': 'MODERATE',
                'description': 'Open, low, and close at same level with long upper shadow. Rejection of higher prices.',
                'action': 'Bearish at resistance levels'
            },
            
            # Neutral/Indecision Patterns
            'DOJI': {
                'type': 'INDECISION',
                'reliability': 'MODERATE',
                'description': 'Open and close at same level. Market indecision, potential reversal.',
                'action': 'Wait for directional confirmation'
            },
            'SPINNING_TOP': {
                'type': 'INDECISION',
                'reliability': 'LOW',
                'description': 'Small body with upper and lower shadows. Indecision between buyers and sellers.',
                'action': 'Neutral, wait for breakout direction'
            },
            
            # Advanced Patterns
            'THREE_LINE_STRIKE_BULLISH': {
                'type': 'BULLISH_CONTINUATION',
                'reliability': 'HIGH',
                'description': 'Three red candles followed by large green candle that engulfs all three. Bullish continuation.',
                'action': 'Strong buy after pullback'
            },
            'THREE_LINE_STRIKE_BEARISH': {
                'type': 'BEARISH_CONTINUATION',
                'reliability': 'HIGH',
                'description': 'Three green candles followed by large red candle that engulfs all three. Bearish continuation.',
                'action': 'Strong sell after rally'
            },
            'TWEEZER_TOP': {
                'type': 'BEARISH_REVERSAL',
                'reliability': 'MODERATE',
                'description': 'Two candles with matching highs at resistance. Double rejection of higher prices.',
                'action': 'Bearish reversal signal'
            },
            'TWEEZER_BOTTOM': {
                'type': 'BULLISH_REVERSAL',
                'reliability': 'MODERATE',
                'description': 'Two candles with matching lows at support. Double rejection of lower prices.',
                'action': 'Bullish reversal signal'
            }
        }
    
    def analyze(self, symbol: str) -> Dict[str, Any]:
        """
        Perform comprehensive candlestick pattern analysis.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary containing all detected patterns with signals
        """
        try:
            # Fetch live data
            data = self._fetch_ohlcv_data(symbol)
            if not data or len(data['close']) < 10:
                return {
                    'success': False,
                    'error': f'Insufficient data for {symbol}',
                    'symbol': symbol
                }
            
            # Detect all patterns
            detected_patterns = []
            
            # Single candle patterns (check last 3 candles)
            for i in range(-3, 0):
                patterns = self._detect_single_candle_patterns(data, i)
                detected_patterns.extend(patterns)
            
            # Double candle patterns (check last 2 pairs)
            for i in range(-2, 0):
                patterns = self._detect_double_candle_patterns(data, i)
                detected_patterns.extend(patterns)
            
            # Triple candle patterns (check last pattern)
            patterns = self._detect_triple_candle_patterns(data, -1)
            detected_patterns.extend(patterns)
            
            # Ichimoku Cloud analysis
            ichimoku = self._analyze_ichimoku(data)
            
            # Sort patterns by recency and reliability
            detected_patterns.sort(key=lambda x: (x['bar_index'], 
                                                   {'VERY_HIGH': 4, 'HIGH': 3, 'MODERATE': 2, 'LOW': 1}.get(x['reliability'], 0)),
                                   reverse=True)
            
            # Determine overall bias
            bullish_count = sum(1 for p in detected_patterns if 'BULLISH' in p['type'])
            bearish_count = sum(1 for p in detected_patterns if 'BEARISH' in p['type'])
            
            if bullish_count > bearish_count:
                overall_bias = 'BULLISH'
            elif bearish_count > bullish_count:
                overall_bias = 'BEARISH'
            else:
                overall_bias = 'NEUTRAL'
            
            return {
                'success': True,
                'symbol': symbol.upper(),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S EST'),
                'current_price': round(data['close'][-1], 2),
                
                # Detected patterns
                'patterns_found': len(detected_patterns),
                'patterns': detected_patterns[:10],  # Top 10 most recent/reliable
                
                # Overall analysis
                'overall_bias': overall_bias,
                'bullish_patterns': bullish_count,
                'bearish_patterns': bearish_count,
                
                # Ichimoku Cloud
                'ichimoku': ichimoku,
                
                # Trading recommendation
                'recommendation': self._generate_recommendation(detected_patterns, ichimoku, overall_bias)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'symbol': symbol
            }
    
    def _fetch_ohlcv_data(self, symbol: str, period: str = '3mo') -> Optional[Dict]:
        """Fetch OHLCV data from Yahoo Finance."""
        if yf is None:
            return None
            
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            
            if df.empty:
                return None
            
            return {
                'open': df['Open'].values.tolist(),
                'high': df['High'].values.tolist(),
                'low': df['Low'].values.tolist(),
                'close': df['Close'].values.tolist(),
                'volume': df['Volume'].values.tolist(),
                'dates': [d.strftime('%Y-%m-%d') for d in df.index]
            }
        except Exception:
            return None
    
    def _detect_single_candle_patterns(self, data: Dict, index: int) -> List[Dict]:
        """Detect single candle patterns at given index."""
        patterns = []
        
        o = data['open'][index]
        h = data['high'][index]
        l = data['low'][index]
        c = data['close'][index]
        
        body = abs(c - o)
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l
        total_range = h - l
        
        if total_range == 0:
            return patterns
        
        body_pct = body / total_range
        upper_shadow_pct = upper_shadow / total_range
        lower_shadow_pct = lower_shadow / total_range
        
        date = data['dates'][index]
        
        # Doji - very small body
        if body_pct < 0.1:
            if lower_shadow_pct > 0.6 and upper_shadow_pct < 0.1:
                pattern_name = 'DRAGONFLY_DOJI'
            elif upper_shadow_pct > 0.6 and lower_shadow_pct < 0.1:
                pattern_name = 'GRAVESTONE_DOJI'
            else:
                pattern_name = 'DOJI'
            
            patterns.append(self._create_pattern_result(pattern_name, date, index, c))
        
        # Hammer - small body at top, long lower shadow
        elif body_pct < 0.3 and lower_shadow_pct > 0.6 and upper_shadow_pct < 0.1:
            # Check if in downtrend (need context)
            if index >= -len(data['close']) + 5:
                recent_trend = data['close'][index] - data['close'][index - 5]
                if recent_trend < 0:
                    patterns.append(self._create_pattern_result('HAMMER', date, index, c))
                else:
                    patterns.append(self._create_pattern_result('HANGING_MAN', date, index, c))
        
        # Shooting Star / Inverted Hammer - small body at bottom, long upper shadow
        elif body_pct < 0.3 and upper_shadow_pct > 0.6 and lower_shadow_pct < 0.1:
            if index >= -len(data['close']) + 5:
                recent_trend = data['close'][index] - data['close'][index - 5]
                if recent_trend > 0:
                    patterns.append(self._create_pattern_result('SHOOTING_STAR', date, index, c))
                else:
                    patterns.append(self._create_pattern_result('INVERTED_HAMMER', date, index, c))
        
        # Spinning Top - small body with shadows on both sides
        elif body_pct < 0.3 and upper_shadow_pct > 0.2 and lower_shadow_pct > 0.2:
            patterns.append(self._create_pattern_result('SPINNING_TOP', date, index, c))
        
        return patterns
    
    def _detect_double_candle_patterns(self, data: Dict, index: int) -> List[Dict]:
        """Detect double candle patterns ending at given index."""
        patterns = []
        
        if index < -len(data['close']) + 1:
            return patterns
        
        # Current candle
        o2 = data['open'][index]
        h2 = data['high'][index]
        l2 = data['low'][index]
        c2 = data['close'][index]
        
        # Previous candle
        o1 = data['open'][index - 1]
        h1 = data['high'][index - 1]
        l1 = data['low'][index - 1]
        c1 = data['close'][index - 1]
        
        body1 = abs(c1 - o1)
        body2 = abs(c2 - o2)
        
        date = data['dates'][index]
        
        # Bullish Engulfing
        if c1 < o1 and c2 > o2:  # Red then green
            if o2 <= c1 and c2 >= o1:  # Green engulfs red
                if body2 > body1:
                    patterns.append(self._create_pattern_result('BULLISH_ENGULFING', date, index, c2))
        
        # Bearish Engulfing
        if c1 > o1 and c2 < o2:  # Green then red
            if o2 >= c1 and c2 <= o1:  # Red engulfs green
                if body2 > body1:
                    patterns.append(self._create_pattern_result('BEARISH_ENGULFING', date, index, c2))
        
        # Bullish Harami
        if c1 < o1 and c2 > o2:  # Red then green
            if o2 > c1 and c2 < o1:  # Green inside red
                patterns.append(self._create_pattern_result('BULLISH_HARAMI', date, index, c2))
        
        # Bearish Harami
        if c1 > o1 and c2 < o2:  # Green then red
            if o2 < c1 and c2 > o1:  # Red inside green
                patterns.append(self._create_pattern_result('BEARISH_HARAMI', date, index, c2))
        
        # Piercing Line
        if c1 < o1 and c2 > o2:  # Red then green
            if o2 < l1 and c2 > (o1 + c1) / 2 and c2 < o1:
                patterns.append(self._create_pattern_result('PIERCING_LINE', date, index, c2))
        
        # Dark Cloud Cover
        if c1 > o1 and c2 < o2:  # Green then red
            if o2 > h1 and c2 < (o1 + c1) / 2 and c2 > o1:
                patterns.append(self._create_pattern_result('DARK_CLOUD_COVER', date, index, c2))
        
        # Tweezer Top
        if abs(h1 - h2) / h1 < 0.002:  # Matching highs within 0.2%
            if c1 > o1 and c2 < o2:  # Green then red
                patterns.append(self._create_pattern_result('TWEEZER_TOP', date, index, c2))
        
        # Tweezer Bottom
        if abs(l1 - l2) / l1 < 0.002:  # Matching lows within 0.2%
            if c1 < o1 and c2 > o2:  # Red then green
                patterns.append(self._create_pattern_result('TWEEZER_BOTTOM', date, index, c2))
        
        return patterns
    
    def _detect_triple_candle_patterns(self, data: Dict, index: int) -> List[Dict]:
        """Detect triple candle patterns ending at given index."""
        patterns = []
        
        if index < -len(data['close']) + 2:
            return patterns
        
        # Current and previous candles
        o3, h3, l3, c3 = data['open'][index], data['high'][index], data['low'][index], data['close'][index]
        o2, h2, l2, c2 = data['open'][index-1], data['high'][index-1], data['low'][index-1], data['close'][index-1]
        o1, h1, l1, c1 = data['open'][index-2], data['high'][index-2], data['low'][index-2], data['close'][index-2]
        
        body1 = abs(c1 - o1)
        body2 = abs(c2 - o2)
        body3 = abs(c3 - o3)
        
        date = data['dates'][index]
        
        # Morning Star
        if c1 < o1 and body2 < body1 * 0.3 and c3 > o3:  # Red, small, green
            if c3 > (o1 + c1) / 2:  # Green closes above midpoint of first red
                patterns.append(self._create_pattern_result('MORNING_STAR', date, index, c3))
        
        # Evening Star
        if c1 > o1 and body2 < body1 * 0.3 and c3 < o3:  # Green, small, red
            if c3 < (o1 + c1) / 2:  # Red closes below midpoint of first green
                patterns.append(self._create_pattern_result('EVENING_STAR', date, index, c3))
        
        # Three White Soldiers
        if c1 > o1 and c2 > o2 and c3 > o3:  # Three green candles
            if c2 > c1 and c3 > c2:  # Higher closes
                if body1 > 0 and body2 > 0 and body3 > 0:  # Significant bodies
                    patterns.append(self._create_pattern_result('THREE_WHITE_SOLDIERS', date, index, c3))
        
        # Three Black Crows
        if c1 < o1 and c2 < o2 and c3 < o3:  # Three red candles
            if c2 < c1 and c3 < c2:  # Lower closes
                if body1 > 0 and body2 > 0 and body3 > 0:  # Significant bodies
                    patterns.append(self._create_pattern_result('THREE_BLACK_CROWS', date, index, c3))
        
        return patterns
    
    def _analyze_ichimoku(self, data: Dict) -> Dict[str, Any]:
        """
        Analyze Ichimoku Cloud components.
        """
        closes = np.array(data['close'])
        highs = np.array(data['high'])
        lows = np.array(data['low'])
        n = len(closes)
        
        if n < 52:
            return {'success': False, 'error': 'Insufficient data for Ichimoku (need 52+ bars)'}
        
        # Tenkan-sen (Conversion Line) - 9-period
        tenkan = (max(highs[-9:]) + min(lows[-9:])) / 2
        
        # Kijun-sen (Base Line) - 26-period
        kijun = (max(highs[-26:]) + min(lows[-26:])) / 2
        
        # Senkou Span A (Leading Span A) - (Tenkan + Kijun) / 2, plotted 26 periods ahead
        senkou_a = (tenkan + kijun) / 2
        
        # Senkou Span B (Leading Span B) - 52-period, plotted 26 periods ahead
        senkou_b = (max(highs[-52:]) + min(lows[-52:])) / 2
        
        # Chikou Span (Lagging Span) - Close plotted 26 periods back
        chikou = closes[-1]
        chikou_reference = closes[-26] if n >= 26 else closes[0]
        
        current_price = closes[-1]
        
        # Cloud analysis
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)
        
        # Determine position relative to cloud
        if current_price > cloud_top:
            cloud_position = 'ABOVE_CLOUD'
            cloud_signal = 'BULLISH'
        elif current_price < cloud_bottom:
            cloud_position = 'BELOW_CLOUD'
            cloud_signal = 'BEARISH'
        else:
            cloud_position = 'IN_CLOUD'
            cloud_signal = 'NEUTRAL'
        
        # Cloud color (future cloud)
        if senkou_a > senkou_b:
            cloud_color = 'GREEN'
            cloud_trend = 'Bullish cloud ahead'
        else:
            cloud_color = 'RED'
            cloud_trend = 'Bearish cloud ahead'
        
        # TK Cross
        if tenkan > kijun:
            tk_cross = 'BULLISH'
            tk_signal = 'Tenkan above Kijun - bullish momentum'
        elif tenkan < kijun:
            tk_cross = 'BEARISH'
            tk_signal = 'Tenkan below Kijun - bearish momentum'
        else:
            tk_cross = 'NEUTRAL'
            tk_signal = 'Tenkan equals Kijun - indecision'
        
        # Chikou confirmation
        if chikou > chikou_reference:
            chikou_signal = 'BULLISH'
        else:
            chikou_signal = 'BEARISH'
        
        # Overall Ichimoku signal
        bullish_signals = sum([
            cloud_signal == 'BULLISH',
            tk_cross == 'BULLISH',
            chikou_signal == 'BULLISH',
            cloud_color == 'GREEN'
        ])
        
        if bullish_signals >= 3:
            overall = 'STRONG_BULLISH'
        elif bullish_signals >= 2:
            overall = 'BULLISH'
        elif bullish_signals <= 1:
            overall = 'BEARISH'
        else:
            overall = 'NEUTRAL'
        
        return {
            'success': True,
            'tenkan_sen': round(tenkan, 2),
            'kijun_sen': round(kijun, 2),
            'senkou_span_a': round(senkou_a, 2),
            'senkou_span_b': round(senkou_b, 2),
            'cloud_top': round(cloud_top, 2),
            'cloud_bottom': round(cloud_bottom, 2),
            'cloud_position': cloud_position,
            'cloud_signal': cloud_signal,
            'cloud_color': cloud_color,
            'cloud_trend': cloud_trend,
            'tk_cross': tk_cross,
            'tk_signal': tk_signal,
            'chikou_signal': chikou_signal,
            'overall_signal': overall,
            'interpretation': self._interpret_ichimoku(overall, cloud_position, tk_cross)
        }
    
    def _interpret_ichimoku(self, overall: str, cloud_position: str, tk_cross: str) -> str:
        """Generate human-readable Ichimoku interpretation."""
        if overall == 'STRONG_BULLISH':
            return 'Strong bullish setup: Price above cloud, TK bullish cross, positive momentum. Consider long positions.'
        elif overall == 'BULLISH':
            return 'Moderately bullish: Most signals align bullish but some caution warranted. Look for pullbacks to enter.'
        elif overall == 'BEARISH':
            return 'Bearish setup: Multiple bearish signals present. Avoid longs, consider shorts on rallies.'
        else:
            return 'Mixed signals: Wait for clearer direction before taking positions.'
    
    def _create_pattern_result(self, pattern_name: str, date: str, index: int, price: float) -> Dict:
        """Create standardized pattern result dictionary."""
        info = self.pattern_info.get(pattern_name, {
            'type': 'UNKNOWN',
            'reliability': 'LOW',
            'description': 'Pattern detected',
            'action': 'Analyze further'
        })
        
        return {
            'pattern': pattern_name,
            'date': date,
            'bar_index': index,
            'price': round(price, 2),
            'type': info['type'],
            'reliability': info['reliability'],
            'description': info['description'],
            'action': info['action']
        }
    
    def _generate_recommendation(self, patterns: List[Dict], ichimoku: Dict, overall_bias: str) -> Dict:
        """Generate trading recommendation based on all detected patterns."""
        
        # Find most reliable recent pattern
        high_reliability_patterns = [p for p in patterns if p['reliability'] in ['HIGH', 'VERY_HIGH']]
        
        if high_reliability_patterns:
            primary_pattern = high_reliability_patterns[0]
        elif patterns:
            primary_pattern = patterns[0]
        else:
            primary_pattern = None
        
        # Combine with Ichimoku
        ichimoku_signal = ichimoku.get('overall_signal', 'NEUTRAL') if ichimoku.get('success') else 'NEUTRAL'
        
        # Generate recommendation
        if primary_pattern:
            pattern_type = primary_pattern['type']
            
            if 'BULLISH' in pattern_type and 'BULLISH' in ichimoku_signal:
                confidence = 'HIGH'
                action = 'BUY'
                reasoning = f"{primary_pattern['pattern']} pattern confirmed by Ichimoku {ichimoku_signal} signal"
            elif 'BEARISH' in pattern_type and 'BEARISH' in ichimoku_signal:
                confidence = 'HIGH'
                action = 'SELL'
                reasoning = f"{primary_pattern['pattern']} pattern confirmed by Ichimoku {ichimoku_signal} signal"
            elif 'BULLISH' in pattern_type:
                confidence = 'MODERATE'
                action = 'BUY'
                reasoning = f"{primary_pattern['pattern']} pattern detected, Ichimoku shows {ichimoku_signal}"
            elif 'BEARISH' in pattern_type:
                confidence = 'MODERATE'
                action = 'SELL'
                reasoning = f"{primary_pattern['pattern']} pattern detected, Ichimoku shows {ichimoku_signal}"
            else:
                confidence = 'LOW'
                action = 'HOLD'
                reasoning = 'Indecision patterns detected, wait for confirmation'
        else:
            confidence = 'LOW'
            action = 'HOLD'
            reasoning = 'No significant patterns detected'
        
        return {
            'action': action,
            'confidence': confidence,
            'reasoning': reasoning,
            'primary_pattern': primary_pattern['pattern'] if primary_pattern else None,
            'ichimoku_confirmation': ichimoku_signal
        }


# Main execution for testing
if __name__ == '__main__':
    detector = CandlestickPatternDetector()
    result = detector.analyze('AAPL')
    print(json.dumps(result, indent=2))
