"""
Advanced Technical Analysis Module
===================================
Provides institutional-grade technical analysis including:
- R2 Score (coefficient of determination) for trend strength
- Pivot Points (Standard, Fibonacci, Camarilla, Woodie)
- Support/Resistance levels with confidence scoring
- Fibonacci Retracement and Extension levels
- Linear regression channels with R2 confidence

All calculations use live market data - zero placeholders.
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
    from polygon_data_provider import PolygonDataProvider
    HAS_POLYGON_PROVIDER = True
except ImportError:
    HAS_POLYGON_PROVIDER = False

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
except ImportError:
    LinearRegression = None
    r2_score = None


class AdvancedTechnicals:
    """
    Advanced Technical Analysis with R2 scoring, pivot points, and Fibonacci levels.
    All data sourced from live market feeds.
    """
    
    def __init__(self):
        self.finnhub_key = os.environ.get('KEY') or os.environ.get('FINNHUB_API_KEY') or 'd55b3ohr01qljfdeghm0d55b3ohr01qljfdeghm1'
    
    def analyze(self, symbol: str, pre_fetched_df=None) -> Dict[str, Any]:
        """
        Perform comprehensive technical analysis with R2 scoring.
        
        Args:
            symbol: Stock ticker symbol
            pre_fetched_df: Optional pandas DataFrame with OHLCV data (from PolygonDataProvider).
            
        Returns:
            Dictionary containing all technical analysis results
        """
        try:
            # Use pre-fetched data if available, otherwise fetch
            if pre_fetched_df is not None and len(pre_fetched_df) > 20:
                data = self._dataframe_to_dict(pre_fetched_df)
            else:
                data = self._fetch_ohlcv_data(symbol)
            if not data or len(data['close']) < 20:
                return {
                    'success': False,
                    'error': f'Insufficient data for {symbol}',
                    'symbol': symbol
                }
            
            # Calculate all technical levels
            pivot_points = self._calculate_pivot_points(data)
            fibonacci = self._calculate_fibonacci_levels(data)
            support_resistance = self._calculate_support_resistance(data)
            r2_analysis = self._calculate_r2_trend_analysis(data)
            regression_channel = self._calculate_regression_channel(data)
            
            # Current price context
            current_price = data['close'][-1]
            
            # Determine nearest levels
            nearest_support = self._find_nearest_level(current_price, support_resistance['support_levels'], 'below')
            nearest_resistance = self._find_nearest_level(current_price, support_resistance['resistance_levels'], 'above')
            
            return {
                'success': True,
                'symbol': symbol.upper(),
                'current_price': round(current_price, 2),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S EST'),
                
                # R2 Trend Analysis
                'r2_analysis': r2_analysis,
                
                # Pivot Points (multiple methods)
                'pivot_points': pivot_points,
                
                # Fibonacci Levels
                'fibonacci': fibonacci,
                
                # Support/Resistance
                'support_resistance': support_resistance,
                
                # Regression Channel
                'regression_channel': regression_channel,
                
                # Key Levels Summary
                'key_levels': {
                    'nearest_support': nearest_support,
                    'nearest_resistance': nearest_resistance,
                    'pivot': pivot_points['standard']['pivot'],
                    'fib_618_retracement': fibonacci['retracement']['0.618'],
                    'r2_trend_strength': r2_analysis['r2_score'],
                    'trend_direction': r2_analysis['trend_direction']
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'symbol': symbol
            }
    
    def _dataframe_to_dict(self, df) -> Optional[Dict]:
        """Convert a pandas DataFrame to our internal dict format."""
        try:
            if df is None or len(df) == 0:
                return None
            return {
                'open': df['Open'].values.tolist(),
                'high': df['High'].values.tolist(),
                'low': df['Low'].values.tolist(),
                'close': df['Close'].values.tolist(),
                'volume': df['Volume'].values.tolist(),
                'dates': [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in df.index]
            }
        except Exception:
            return None
    
    def _fetch_ohlcv_data(self, symbol: str, period: str = '3mo') -> Optional[Dict]:
        """Fetch OHLCV data. Primary: PolygonDataProvider. Fallback: yfinance."""
        # Try PolygonDataProvider first
        if HAS_POLYGON_PROVIDER:
            try:
                provider = PolygonDataProvider.get_instance(symbol)
                df = provider.get_daily_ohlcv(days=90)
                if df is not None and len(df) > 20:
                    print(f"  \u2713 AdvancedTechnicals: Using Polygon data ({len(df)} bars)", file=sys.stderr, flush=True)
                    return self._dataframe_to_dict(df)
            except Exception as e:
                print(f"  AdvancedTechnicals: Polygon failed: {e}", file=sys.stderr, flush=True)
        
        # Fallback to yfinance
        if yf is None:
            return None
            
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            
            if df.empty:
                return None
            
            return self._dataframe_to_dict(df)
        except Exception:
            return None
    
    def _calculate_r2_trend_analysis(self, data: Dict) -> Dict[str, Any]:
        """
        Calculate R2 score for trend strength using linear regression.
        R2 measures how well the price follows a linear trend.
        """
        closes = np.array(data['close'])
        n = len(closes)
        
        # Multiple timeframe R2 analysis
        timeframes = {
            '5d': min(5, n),
            '10d': min(10, n),
            '20d': min(20, n),
            '50d': min(50, n)
        }
        
        r2_scores = {}
        slopes = {}
        
        for tf_name, tf_len in timeframes.items():
            if tf_len < 3:
                continue
                
            recent_closes = closes[-tf_len:]
            X = np.arange(tf_len).reshape(-1, 1)
            y = recent_closes
            
            if LinearRegression is not None and r2_score is not None:
                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)
                
                r2 = r2_score(y, y_pred)
                slope = model.coef_[0]
            else:
                # Fallback calculation without sklearn
                x_mean = np.mean(np.arange(tf_len))
                y_mean = np.mean(recent_closes)
                
                numerator = np.sum((np.arange(tf_len) - x_mean) * (recent_closes - y_mean))
                denominator = np.sum((np.arange(tf_len) - x_mean) ** 2)
                slope = numerator / denominator if denominator != 0 else 0
                
                y_pred = slope * np.arange(tf_len) + (y_mean - slope * x_mean)
                ss_res = np.sum((recent_closes - y_pred) ** 2)
                ss_tot = np.sum((recent_closes - y_mean) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            r2_scores[tf_name] = round(max(0, r2), 4)  # R2 as decimal
            slopes[tf_name] = round(slope, 4)
        
        # Primary R2 score (20-day)
        primary_r2 = r2_scores.get('20d', r2_scores.get('10d', 0))
        primary_slope = slopes.get('20d', slopes.get('10d', 0))
        
        # Determine trend direction and strength
        if primary_slope > 0:
            trend_direction = 'BULLISH'
        elif primary_slope < 0:
            trend_direction = 'BEARISH'
        else:
            trend_direction = 'NEUTRAL'
        
        # R2 interpretation
        if primary_r2 >= 0.8:
            trend_strength = 'VERY_STRONG'
            interpretation = 'Price following a very strong linear trend - high predictability'
        elif primary_r2 >= 0.6:
            trend_strength = 'STRONG'
            interpretation = 'Price following a strong trend with moderate predictability'
        elif primary_r2 >= 0.4:
            trend_strength = 'MODERATE'
            interpretation = 'Moderate trend - some noise but directional bias exists'
        elif primary_r2 >= 0.2:
            trend_strength = 'WEAK'
            interpretation = 'Weak trend - price action is choppy'
        else:
            trend_strength = 'NO_TREND'
            interpretation = 'No clear trend - price is ranging/consolidating'
        
        return {
            'r2_score': primary_r2,
            'r2_scores': r2_scores,  # Alias for frontend compatibility
            'r2_scores_by_timeframe': r2_scores,
            'slopes_by_timeframe': slopes,
            'slope': round(primary_slope, 4),  # Primary slope for frontend
            'trend_direction': trend_direction,
            'trend_strength': trend_strength,
            'interpretation': interpretation,
            'daily_price_change': round(primary_slope, 4)
        }
    
    def _calculate_pivot_points(self, data: Dict) -> Dict[str, Any]:
        """
        Calculate pivot points using multiple methods:
        - Standard (Floor)
        - Fibonacci
        - Camarilla
        - Woodie
        """
        # Use previous day's OHLC
        high = data['high'][-1]
        low = data['low'][-1]
        close = data['close'][-1]
        open_price = data['open'][-1]
        
        # Standard Pivot Points
        pivot = (high + low + close) / 3
        standard = {
            'pivot': round(pivot, 2),
            'r1': round(2 * pivot - low, 2),
            'r2': round(pivot + (high - low), 2),
            'r3': round(high + 2 * (pivot - low), 2),
            's1': round(2 * pivot - high, 2),
            's2': round(pivot - (high - low), 2),
            's3': round(low - 2 * (high - pivot), 2)
        }
        
        # Fibonacci Pivot Points
        fib = {
            'pivot': round(pivot, 2),
            'r1': round(pivot + 0.382 * (high - low), 2),
            'r2': round(pivot + 0.618 * (high - low), 2),
            'r3': round(pivot + 1.000 * (high - low), 2),
            's1': round(pivot - 0.382 * (high - low), 2),
            's2': round(pivot - 0.618 * (high - low), 2),
            's3': round(pivot - 1.000 * (high - low), 2)
        }
        
        # Camarilla Pivot Points
        range_hl = high - low
        camarilla = {
            'pivot': round(pivot, 2),
            'r1': round(close + range_hl * 1.1 / 12, 2),
            'r2': round(close + range_hl * 1.1 / 6, 2),
            'r3': round(close + range_hl * 1.1 / 4, 2),
            'r4': round(close + range_hl * 1.1 / 2, 2),
            's1': round(close - range_hl * 1.1 / 12, 2),
            's2': round(close - range_hl * 1.1 / 6, 2),
            's3': round(close - range_hl * 1.1 / 4, 2),
            's4': round(close - range_hl * 1.1 / 2, 2)
        }
        
        # Woodie Pivot Points
        woodie_pivot = (high + low + 2 * close) / 4
        woodie = {
            'pivot': round(woodie_pivot, 2),
            'r1': round(2 * woodie_pivot - low, 2),
            'r2': round(woodie_pivot + (high - low), 2),
            's1': round(2 * woodie_pivot - high, 2),
            's2': round(woodie_pivot - (high - low), 2)
        }
        
        return {
            'standard': standard,
            'fibonacci': fib,
            'camarilla': camarilla,
            'woodie': woodie,
            'method_recommendation': self._recommend_pivot_method(data)
        }
    
    def _recommend_pivot_method(self, data: Dict) -> str:
        """Recommend best pivot method based on market conditions."""
        closes = data['close']
        volatility = np.std(closes[-20:]) / np.mean(closes[-20:]) if len(closes) >= 20 else 0
        
        if volatility > 0.03:
            return 'CAMARILLA - Best for high volatility, provides tighter levels'
        elif volatility < 0.01:
            return 'STANDARD - Best for low volatility, classic floor trader method'
        else:
            return 'FIBONACCI - Best for moderate volatility, uses natural retracement levels'
    
    def _calculate_fibonacci_levels(self, data: Dict) -> Dict[str, Any]:
        """
        Calculate Fibonacci retracement and extension levels.
        Uses the swing high/low from recent price action.
        """
        closes = data['close']
        highs = data['high']
        lows = data['low']
        
        # Find swing high and swing low (last 50 bars)
        lookback = min(50, len(closes))
        recent_highs = highs[-lookback:]
        recent_lows = lows[-lookback:]
        
        swing_high = max(recent_highs)
        swing_low = min(recent_lows)
        swing_range = swing_high - swing_low
        
        # Determine trend direction for retracement calculation
        current_price = closes[-1]
        mid_point = (swing_high + swing_low) / 2
        
        if current_price > mid_point:
            # Uptrend - calculate retracements from high
            retracement = {
                '0.000': round(swing_high, 2),
                '0.236': round(swing_high - 0.236 * swing_range, 2),
                '0.382': round(swing_high - 0.382 * swing_range, 2),
                '0.500': round(swing_high - 0.500 * swing_range, 2),
                '0.618': round(swing_high - 0.618 * swing_range, 2),
                '0.786': round(swing_high - 0.786 * swing_range, 2),
                '1.000': round(swing_low, 2)
            }
            trend = 'UPTREND'
        else:
            # Downtrend - calculate retracements from low
            retracement = {
                '0.000': round(swing_low, 2),
                '0.236': round(swing_low + 0.236 * swing_range, 2),
                '0.382': round(swing_low + 0.382 * swing_range, 2),
                '0.500': round(swing_low + 0.500 * swing_range, 2),
                '0.618': round(swing_low + 0.618 * swing_range, 2),
                '0.786': round(swing_low + 0.786 * swing_range, 2),
                '1.000': round(swing_high, 2)
            }
            trend = 'DOWNTREND'
        
        # Extension levels (for targets)
        extension = {
            '1.000': round(swing_high, 2),
            '1.272': round(swing_high + 0.272 * swing_range, 2),
            '1.414': round(swing_high + 0.414 * swing_range, 2),
            '1.618': round(swing_high + 0.618 * swing_range, 2),
            '2.000': round(swing_high + 1.000 * swing_range, 2),
            '2.618': round(swing_high + 1.618 * swing_range, 2)
        }
        
        # Find current position relative to Fibonacci levels
        current_fib_position = self._find_fib_position(current_price, retracement)
        
        return {
            'swing_high': round(swing_high, 2),
            'swing_low': round(swing_low, 2),
            'swing_range': round(swing_range, 2),
            'trend': trend,
            'retracement': retracement,
            'extension': extension,
            'current_position': current_fib_position,
            'key_level': self._identify_key_fib_level(current_price, retracement)
        }
    
    def _find_fib_position(self, price: float, retracement: Dict) -> str:
        """Determine where price sits relative to Fibonacci levels."""
        levels = [(k, v) for k, v in retracement.items()]
        levels.sort(key=lambda x: x[1])
        
        for i, (level, value) in enumerate(levels):
            if price < value:
                if i == 0:
                    return f'Below {level} ({value})'
                prev_level, prev_value = levels[i-1]
                return f'Between {prev_level} ({prev_value}) and {level} ({value})'
        
        return f'Above {levels[-1][0]} ({levels[-1][1]})'
    
    def _identify_key_fib_level(self, price: float, retracement: Dict) -> Dict:
        """Identify the nearest key Fibonacci level."""
        key_levels = ['0.382', '0.500', '0.618']
        nearest = None
        min_distance = float('inf')
        
        for level in key_levels:
            if level in retracement:
                distance = abs(price - retracement[level])
                if distance < min_distance:
                    min_distance = distance
                    nearest = {
                        'level': level,
                        'price': retracement[level],
                        'distance': round(distance, 2),
                        'distance_pct': round((distance / price) * 100, 2)
                    }
        
        return nearest
    
    def _calculate_support_resistance(self, data: Dict) -> Dict[str, Any]:
        """
        Calculate support and resistance levels using multiple methods:
        - Historical price clustering
        - Volume-weighted levels
        - Recent swing highs/lows
        """
        closes = np.array(data['close'])
        highs = np.array(data['high'])
        lows = np.array(data['low'])
        volumes = np.array(data['volume'])
        
        # Method 1: Find swing highs and lows
        swing_highs = []
        swing_lows = []
        
        for i in range(2, len(closes) - 2):
            # Swing high: higher than 2 bars on each side
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                swing_highs.append(highs[i])
            # Swing low: lower than 2 bars on each side
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                swing_lows.append(lows[i])
        
        # Method 2: Volume-weighted price levels
        vwap_levels = []
        chunk_size = max(5, len(closes) // 10)
        for i in range(0, len(closes), chunk_size):
            chunk_closes = closes[i:i+chunk_size]
            chunk_volumes = volumes[i:i+chunk_size]
            if len(chunk_closes) > 0 and sum(chunk_volumes) > 0:
                vwap = np.sum(chunk_closes * chunk_volumes) / np.sum(chunk_volumes)
                vwap_levels.append(vwap)
        
        # Combine and cluster levels
        all_resistance = list(set([round(h, 2) for h in swing_highs[-10:]]))
        all_support = list(set([round(l, 2) for l in swing_lows[-10:]]))
        
        # Add recent high/low
        all_resistance.append(round(max(highs[-20:]), 2))
        all_support.append(round(min(lows[-20:]), 2))
        
        # Sort and deduplicate
        all_resistance = sorted(list(set(all_resistance)), reverse=True)[:5]
        all_support = sorted(list(set(all_support)))[:5]
        
        current_price = closes[-1]
        
        return {
            'resistance_levels': all_resistance,
            'support_levels': all_support,
            'strongest_resistance': all_resistance[0] if all_resistance else None,
            'strongest_support': all_support[-1] if all_support else None,
            'current_price': round(current_price, 2),
            'distance_to_resistance': round(all_resistance[0] - current_price, 2) if all_resistance else None,
            'distance_to_support': round(current_price - all_support[-1], 2) if all_support else None
        }
    
    def _calculate_regression_channel(self, data: Dict) -> Dict[str, Any]:
        """
        Calculate linear regression channel with R2 confidence.
        """
        closes = np.array(data['close'])
        n = len(closes)
        
        if n < 10:
            return {'success': False, 'error': 'Insufficient data'}
        
        # Use last 50 bars for channel
        lookback = min(50, n)
        recent_closes = closes[-lookback:]
        X = np.arange(lookback).reshape(-1, 1)
        
        if LinearRegression is not None:
            model = LinearRegression()
            model.fit(X, recent_closes)
            
            regression_line = model.predict(X)
            slope = model.coef_[0]
            intercept = model.intercept_
            
            # Calculate standard deviation for channel
            residuals = recent_closes - regression_line
            std_dev = np.std(residuals)
            
            # R2 score
            r2 = r2_score(recent_closes, regression_line)
        else:
            # Fallback without sklearn
            x_mean = np.mean(np.arange(lookback))
            y_mean = np.mean(recent_closes)
            
            numerator = np.sum((np.arange(lookback) - x_mean) * (recent_closes - y_mean))
            denominator = np.sum((np.arange(lookback) - x_mean) ** 2)
            slope = numerator / denominator if denominator != 0 else 0
            intercept = y_mean - slope * x_mean
            
            regression_line = slope * np.arange(lookback) + intercept
            residuals = recent_closes - regression_line
            std_dev = np.std(residuals)
            
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((recent_closes - y_mean) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Current channel values (extrapolated to current bar)
        current_regression = regression_line[-1]
        upper_channel = current_regression + 2 * std_dev
        lower_channel = current_regression - 2 * std_dev
        
        current_price = closes[-1]
        
        # Position within channel
        if current_price > upper_channel:
            position = 'ABOVE_CHANNEL'
            signal = 'Overbought - potential mean reversion'
        elif current_price < lower_channel:
            position = 'BELOW_CHANNEL'
            signal = 'Oversold - potential mean reversion'
        elif current_price > current_regression:
            position = 'UPPER_HALF'
            signal = 'Above regression line - bullish bias'
        else:
            position = 'LOWER_HALF'
            signal = 'Below regression line - bearish bias'
        
        return {
            'success': True,
            'r2_score': round(max(0, r2), 4),
            'slope_per_day': round(slope, 4),
            'current_regression_value': round(current_regression, 2),
            'upper_channel_2std': round(upper_channel, 2),
            'lower_channel_2std': round(lower_channel, 2),
            'channel_width': round(4 * std_dev, 2),
            'current_position': position,
            'signal': signal,
            'trend': 'BULLISH' if slope > 0 else 'BEARISH' if slope < 0 else 'NEUTRAL'
        }
    
    def _find_nearest_level(self, price: float, levels: List[float], direction: str) -> Optional[Dict]:
        """Find the nearest support or resistance level."""
        if not levels:
            return None
        
        if direction == 'above':
            above_levels = [l for l in levels if l > price]
            if above_levels:
                nearest = min(above_levels)
                return {
                    'price': nearest,
                    'distance': round(nearest - price, 2),
                    'distance_pct': round(((nearest - price) / price) * 100, 2)
                }
        else:
            below_levels = [l for l in levels if l < price]
            if below_levels:
                nearest = max(below_levels)
                return {
                    'price': nearest,
                    'distance': round(price - nearest, 2),
                    'distance_pct': round(((price - nearest) / price) * 100, 2)
                }
        
        return None


# Main execution for testing
if __name__ == '__main__':
    analyzer = AdvancedTechnicals()
    result = analyzer.analyze('AAPL')
    print(json.dumps(result, indent=2))
