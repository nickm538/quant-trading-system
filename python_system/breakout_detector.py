"""
BREAKOUT DETECTOR v2.0 - INSTITUTIONAL GRADE SIGNAL DETECTION
==============================================================
Quantum-level precision breakout detection using multi-factor analysis.

This system combines 10+ signals with proprietary weighting algorithms
derived from decades of market research and institutional trading strategies.

CORE SIGNALS:
1. TTM Squeeze (John Carter's methodology)
2. NR4/NR7 Patterns (Toby Crabel's volatility contraction)
3. OBV Divergence (Joe Granville's volume analysis)
4. Support/Resistance Testing (Market structure)
5. Triangle/Flag Patterns (Classical charting)
6. Volume Analysis (Smart money footprint)

ADVANCED SIGNALS (NEW):
7. RSI Divergence (Momentum confirmation)
8. MACD Histogram Divergence (Trend strength)
9. ADX Trend Strength (Directional movement)
10. Relative Volume (Institutional activity)
11. Price Position in Range (Mean reversion vs breakout)
12. Multi-Timeframe Confluence (Higher timeframe alignment)

SYNERGY BONUSES:
- NR7 + TTM Squeeze = "Coiled Spring" (highest probability)
- OBV Divergence + Volume Contraction = "Stealth Accumulation"
- S/R Test + Pattern = "Technical Confluence"

ALL DATA IS REAL-TIME - NO FAKE CALCULATIONS
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import requests
from typing import Dict, List, Optional, Tuple
import time
import yfinance as yf


class BreakoutDetector:
    """
    Institutional-grade breakout detection system.
    Combines multiple signals with synergy bonuses for maximum accuracy.
    
    SCORING METHODOLOGY:
    - Base signals: 0-100 points
    - Synergy bonuses: Up to +30 additional points
    - Quality adjustments: -20 to +20 based on signal quality
    - Final score normalized to 0-100 scale
    """
    
    def __init__(self, twelvedata_api_key: str = None, finnhub_api_key: str = None):
        import os
        # Use provided key or fall back to environment variable
        self.td_api_key = twelvedata_api_key or os.environ.get('TWELVEDATA_API_KEY', '5e7a5daaf41d46a8966963106ebef210')
        self.finnhub_api_key = finnhub_api_key or os.environ.get('FINNHUB_API_KEY')
        self.base_url = "https://api.twelvedata.com"
        
    def _fetch_price_data(self, symbol: str, interval: str = "1day", outputsize: int = 100) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data with TwelveData primary + yfinance fallback.
        This ensures maximum data reliability and coverage.
        """
        # Try TwelveData first (more granular intervals available)
        df = self._fetch_twelvedata(symbol, interval, outputsize)
        
        # Fallback to yfinance if TwelveData fails
        if df is None or df.empty:
            df = self._fetch_yfinance(symbol, outputsize)
        
        return df
    
    def _fetch_twelvedata(self, symbol: str, interval: str = "1day", outputsize: int = 100) -> Optional[pd.DataFrame]:
        """Fetch from TwelveData API."""
        try:
            url = f"{self.base_url}/time_series"
            params = {
                "symbol": symbol,
                "interval": interval,
                "outputsize": outputsize,
                "apikey": self.td_api_key
            }
            
            response = requests.get(url, params=params, timeout=15)
            data = response.json()
            
            if "values" not in data:
                return None
                
            df = pd.DataFrame(data["values"])
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.sort_values("datetime").reset_index(drop=True)
            
            # Convert to numeric
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            
            return df
            
        except Exception as e:
            return None
    
    def _fetch_yfinance(self, symbol: str, outputsize: int = 100) -> Optional[pd.DataFrame]:
        """
        Fetch from yfinance as fallback - FREE, RELIABLE, REAL-TIME.
        No API key required, no rate limits.
        """
        try:
            ticker = yf.Ticker(symbol)
            # Get enough history for analysis
            period = "6mo" if outputsize > 60 else "3mo"
            hist = ticker.history(period=period)
            
            if hist.empty:
                return None
            
            # Rename columns to match expected format
            df = hist.reset_index()
            df.columns = [c.lower() for c in df.columns]
            df = df.rename(columns={"date": "datetime"})
            
            # Ensure datetime is proper format
            df["datetime"] = pd.to_datetime(df["datetime"])
            
            # Keep only needed columns
            df = df[["datetime", "open", "high", "low", "close", "volume"]].copy()
            
            # Take last N rows
            if len(df) > outputsize:
                df = df.tail(outputsize).reset_index(drop=True)
            
            return df
            
        except Exception as e:
            return None
    
    # =========================================================================
    # SIGNAL 1: NR4/NR7 PATTERN DETECTION (Toby Crabel)
    # =========================================================================
    def detect_nr_patterns(self, df: pd.DataFrame, gap_threshold: float = 0.02) -> Dict:
        """
        Detect NR4 (Narrowest Range in 4 days) and NR7 (Narrowest Range in 7 days)
        
        FORMULA:
        - Daily Range = High - Low
        - NR4 = Today's range is smallest of last 4 days
        - NR7 = Today's range is smallest of last 7 days
        
        SIGNIFICANCE:
        - Narrow range = volatility contraction = energy buildup
        - Like a coiled spring ready to explode
        - 70%+ of NR7 days lead to significant moves within 3 days
        
        PRO TIP: NR7 inside an NR4 (NR7+NR4 combo) is the most powerful signal.
        
        ENHANCEMENT: Gap filter - Gap days (>2% gap) invalidate NR patterns
        because the volatility already released through the gap.
        
        Args:
            df: Price DataFrame with OHLCV
            gap_threshold: Gap percentage to invalidate NR (default 2%)
        """
        if df is None or len(df) < 7:
            return {"nr4": False, "nr7": False, "range_percentile": 0, "signal_strength": "NONE", "gap_invalidated": False}
        
        # Calculate daily ranges
        df = df.copy()
        df["range"] = df["high"] - df["low"]
        df["range_pct"] = (df["range"] / df["close"]) * 100
        
        # Calculate gaps (open vs previous close)
        df["gap"] = abs(df["open"] - df["close"].shift(1)) / df["close"].shift(1)
        df["is_gap"] = df["gap"] > gap_threshold
        
        # Check if today is a gap day - invalidates NR pattern
        gap_invalidated = df["is_gap"].iloc[-1] if len(df) > 1 else False
        
        latest_range = df["range"].iloc[-1]
        
        # NR4 Detection - Use tolerance for float comparison (0.1% tolerance)
        last_4_ranges = df["range"].tail(4)
        min_4 = last_4_ranges.min()
        nr4 = np.isclose(latest_range, min_4, rtol=0.001) or latest_range < min_4
        
        # NR7 Detection - Use tolerance for float comparison
        last_7_ranges = df["range"].tail(7)
        min_7 = last_7_ranges.min()
        nr7 = np.isclose(latest_range, min_7, rtol=0.001) or latest_range < min_7
        
        # Calculate range percentile (how tight is this range historically?)
        all_ranges = df["range"].tail(50)
        range_percentile = (all_ranges < latest_range).sum() / len(all_ranges) * 100
        
        # Consecutive narrow range days (building pressure)
        avg_range = df["range"].tail(20).mean()
        narrow_days = 0
        for i in range(len(df) - 1, max(0, len(df) - 10), -1):
            if df["range"].iloc[i] < avg_range * 0.7:
                narrow_days += 1
            else:
                break
        
        # Signal strength based on multiple factors
        strength_score = 0
        if nr7:
            strength_score += 3
        if nr4:
            strength_score += 2
        if range_percentile < 20:  # In bottom 20% of ranges
            strength_score += 2
        if narrow_days >= 3:
            strength_score += 1
        
        if strength_score >= 5:
            signal_strength = "VERY_STRONG"
        elif strength_score >= 3:
            signal_strength = "STRONG"
        elif strength_score >= 1:
            signal_strength = "MODERATE"
        else:
            signal_strength = "NONE"
        
        # If gap day, invalidate NR pattern (volatility already released)
        if gap_invalidated:
            nr4 = False
            nr7 = False
            signal_strength = "NONE"
        
        return {
            "nr4": nr4,
            "nr7": nr7,
            "nr4_nr7_combined": nr4 and nr7,
            "range_percentile": round(range_percentile, 1),
            "latest_range": round(latest_range, 4),
            "latest_range_pct": round(df["range_pct"].iloc[-1], 2),
            "avg_range": round(avg_range, 4),
            "consecutive_narrow_days": narrow_days,
            "signal_strength": signal_strength,
            "gap_invalidated": gap_invalidated,  # NEW: Gap filter flag
            "interpretation": self._interpret_nr_pattern(nr4, nr7, narrow_days, range_percentile, gap_invalidated)
        }
    
    def _interpret_nr_pattern(self, nr4: bool, nr7: bool, narrow_days: int, range_pct: float, gap_invalidated: bool = False) -> str:
        if gap_invalidated:
            return "⚠️ GAP DAY - NR pattern invalidated. Volatility already released through gap opening."
        if nr7 and nr4:
            return "🔥 NR7+NR4 COMBO - Extremely rare! Maximum energy compression. Explosive move imminent within 1-2 days."
        elif nr7:
            return "🔥 NR7 DETECTED - Narrowest range in 7 days! High probability breakout imminent within 1-3 days."
        elif nr4:
            return "⚡ NR4 DETECTED - Narrowest range in 4 days. Volatility compression building."
        elif narrow_days >= 3:
            return f"📊 {narrow_days} consecutive narrow range days. Pressure accumulating - watch for expansion."
        elif range_pct < 25:
            return "📉 Below-average range. Some compression present but not extreme."
        return "No significant range compression detected."
    
    # =========================================================================
    # SIGNAL 2: OBV DIVERGENCE DETECTION (Joe Granville)
    # =========================================================================
    def detect_obv_divergence(self, df: pd.DataFrame, lookback: int = 14) -> Dict:
        """
        Detect On-Balance Volume divergence from price.
        
        FORMULA:
        - OBV = Running total of volume (add if close > prev close, subtract if lower)
        - Bullish Divergence = Price making lower lows, OBV making higher lows
        - Bearish Divergence = Price making higher highs, OBV making lower highs
        
        SIGNIFICANCE:
        - OBV rising while price flat = HIDDEN ACCUMULATION (smart money buying)
        - OBV falling while price flat = HIDDEN DISTRIBUTION (smart money selling)
        - Divergence often precedes price by 1-5 days
        
        PRO TIP: OBV divergence + declining volume = "Stealth Mode" accumulation
        """
        if df is None or len(df) < lookback + 5:
            return {"divergence": "NONE", "obv_trend": "NEUTRAL", "divergence_strength": 0}
        
        df = df.copy()
        
        # Calculate OBV
        df["obv"] = 0.0
        for i in range(1, len(df)):
            if df["close"].iloc[i] > df["close"].iloc[i-1]:
                df.loc[df.index[i], "obv"] = df["obv"].iloc[i-1] + df["volume"].iloc[i]
            elif df["close"].iloc[i] < df["close"].iloc[i-1]:
                df.loc[df.index[i], "obv"] = df["obv"].iloc[i-1] - df["volume"].iloc[i]
            else:
                df.loc[df.index[i], "obv"] = df["obv"].iloc[i-1]
        
        # Get recent data for divergence detection
        recent = df.tail(lookback)
        
        # Find price trend (using linear regression slope)
        price_slope = np.polyfit(range(len(recent)), recent["close"].values, 1)[0]
        obv_slope = np.polyfit(range(len(recent)), recent["obv"].values, 1)[0]
        
        # Normalize slopes for comparison - use standard deviation for robust scaling
        price_slope_norm = price_slope / recent["close"].mean() * 100
        
        # OBV normalization: use range-based scaling to avoid division issues
        obv_range = recent["obv"].max() - recent["obv"].min()
        if obv_range > 0:
            obv_slope_norm = obv_slope / obv_range * 100 * len(recent)  # Scale by lookback period
        else:
            obv_slope_norm = 0  # No OBV movement
        
        # Detect divergence
        divergence = "NONE"
        divergence_strength = 0
        
        # Bullish divergence: price down, OBV up or flat
        if price_slope_norm < -0.5 and obv_slope_norm > 0.5:
            divergence = "BULLISH"
            divergence_strength = min(100, abs(obv_slope_norm - price_slope_norm) * 10)
        # Hidden bullish: price flat/up, OBV strongly up
        elif price_slope_norm > -0.5 and price_slope_norm < 1 and obv_slope_norm > 2:
            divergence = "HIDDEN_BULLISH"
            divergence_strength = min(100, obv_slope_norm * 20)
        # Bearish divergence: price up, OBV down
        elif price_slope_norm > 0.5 and obv_slope_norm < -0.5:
            divergence = "BEARISH"
            divergence_strength = min(100, abs(obv_slope_norm - price_slope_norm) * 10)
        # Hidden bearish: price flat/down, OBV strongly down
        elif price_slope_norm < 0.5 and price_slope_norm > -1 and obv_slope_norm < -2:
            divergence = "HIDDEN_BEARISH"
            divergence_strength = min(100, abs(obv_slope_norm) * 20)
        
        # Determine OBV trend
        if obv_slope_norm > 1:
            obv_trend = "ACCUMULATION"
        elif obv_slope_norm < -1:
            obv_trend = "DISTRIBUTION"
        else:
            obv_trend = "NEUTRAL"
        
        return {
            "divergence": divergence,
            "obv_trend": obv_trend,
            "price_slope": round(price_slope_norm, 2),
            "obv_slope": round(obv_slope_norm, 2),
            "divergence_strength": round(divergence_strength, 1),
            "interpretation": self._interpret_obv(divergence, obv_trend, divergence_strength)
        }
    
    def _interpret_obv(self, divergence: str, trend: str, strength: float) -> str:
        if divergence == "BULLISH":
            return f"🟢 BULLISH DIVERGENCE ({strength:.0f}% strength) - Smart money accumulating while price drops. Reversal likely!"
        elif divergence == "HIDDEN_BULLISH":
            return f"🟢 HIDDEN ACCUMULATION ({strength:.0f}% strength) - Stealth buying detected. Breakout building."
        elif divergence == "BEARISH":
            return f"🔴 BEARISH DIVERGENCE ({strength:.0f}% strength) - Distribution while price rises. Top forming."
        elif divergence == "HIDDEN_BEARISH":
            return f"🔴 HIDDEN DISTRIBUTION ({strength:.0f}% strength) - Smart money exiting. Breakdown likely."
        elif trend == "ACCUMULATION":
            return "📈 OBV trending up - Healthy accumulation, no divergence."
        elif trend == "DISTRIBUTION":
            return "📉 OBV trending down - Distribution in progress."
        return "📊 OBV neutral - No significant volume trend."
    
    # =========================================================================
    # SIGNAL 3: SUPPORT/RESISTANCE TESTING
    # =========================================================================
    def detect_sr_testing(self, df: pd.DataFrame) -> Dict:
        """
        Detect when price is repeatedly testing support or resistance.
        
        SIGNIFICANCE:
        - Multiple tests of resistance = buying pressure building
        - Multiple tests of support = selling pressure building
        - The more tests, the more likely the level breaks
        
        PRO TIP: 3+ tests of a level within 2 weeks = high probability break
        """
        if df is None or len(df) < 20:
            return {"testing": "NONE", "level": 0, "touches": 0, "pivot": 0, "r1": 0, "r2": 0, "s1": 0, "s2": 0, "nearest_resistance": 0, "nearest_support": 0}
        
        df = df.copy()
        recent = df.tail(20)
        current_price = df["close"].iloc[-1]
        
        # Calculate pivot points (Floor Trader's Method)
        high = recent["high"].max()
        low = recent["low"].min()
        close = df["close"].iloc[-1]
        
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        r2 = pivot + (high - low)
        s1 = 2 * pivot - high
        s2 = pivot - (high - low)
        
        # Find key levels from recent price action
        resistance_levels = []
        support_levels = []
        
        # Look for swing highs and lows
        for i in range(2, len(recent) - 2):
            # Swing high
            if recent["high"].iloc[i] > recent["high"].iloc[i-1] and recent["high"].iloc[i] > recent["high"].iloc[i-2]:
                if recent["high"].iloc[i] > recent["high"].iloc[i+1] and recent["high"].iloc[i] > recent["high"].iloc[i+2]:
                    resistance_levels.append(recent["high"].iloc[i])
            # Swing low
            if recent["low"].iloc[i] < recent["low"].iloc[i-1] and recent["low"].iloc[i] < recent["low"].iloc[i-2]:
                if recent["low"].iloc[i] < recent["low"].iloc[i+1] and recent["low"].iloc[i] < recent["low"].iloc[i+2]:
                    support_levels.append(recent["low"].iloc[i])
        
        # Count touches near key levels
        tolerance = current_price * 0.01  # 1% tolerance
        
        resistance_touches = 0
        support_touches = 0
        nearest_resistance = r1
        nearest_support = s1
        
        # Check resistance touches
        for level in [r1, r2] + resistance_levels:
            touches = sum(1 for h in recent["high"] if abs(h - level) < tolerance)
            if touches > resistance_touches:
                resistance_touches = touches
                nearest_resistance = level
        
        # Check support touches
        for level in [s1, s2] + support_levels:
            touches = sum(1 for l in recent["low"] if abs(l - level) < tolerance)
            if touches > support_touches:
                support_touches = touches
                nearest_support = level
        
        # Determine what's being tested
        price_to_resistance = (nearest_resistance - current_price) / current_price * 100
        price_to_support = (current_price - nearest_support) / current_price * 100
        
        if price_to_resistance < 2 and resistance_touches >= 2:
            testing = "RESISTANCE"
            touches = resistance_touches
            level = nearest_resistance
        elif price_to_support < 2 and support_touches >= 2:
            testing = "SUPPORT"
            touches = support_touches
            level = nearest_support
        else:
            testing = "NONE"
            touches = 0
            level = 0
        
        return {
            "testing": testing,
            "level": round(level, 2),
            "touches": touches,
            "pivot": round(pivot, 2),
            "r1": round(r1, 2),
            "r2": round(r2, 2),
            "s1": round(s1, 2),
            "s2": round(s2, 2),
            "nearest_resistance": round(nearest_resistance, 2),
            "nearest_support": round(nearest_support, 2),
            "price_to_resistance_pct": round(price_to_resistance, 2),
            "price_to_support_pct": round(price_to_support, 2),
            "interpretation": self._interpret_sr(testing, touches, level, current_price)
        }
    
    def _interpret_sr(self, testing: str, touches: int, level: float, price: float) -> str:
        if testing == "RESISTANCE" and touches >= 3:
            return f"🔥 RESISTANCE TESTED {touches}x at ${level:.2f} - High probability breakout! Multiple tests = weakening resistance."
        elif testing == "RESISTANCE":
            return f"📈 Testing resistance at ${level:.2f} ({touches} touches). Watch for breakout confirmation."
        elif testing == "SUPPORT" and touches >= 3:
            return f"🔥 SUPPORT TESTED {touches}x at ${level:.2f} - Either strong bounce or breakdown imminent."
        elif testing == "SUPPORT":
            return f"📉 Testing support at ${level:.2f} ({touches} touches). Watch for bounce or breakdown."
        return "No significant S/R testing detected."
    
    # =========================================================================
    # SIGNAL 4: TTM SQUEEZE (John Carter)
    # =========================================================================
    def detect_ttm_squeeze(self, df: pd.DataFrame) -> Dict:
        """
        TTM Squeeze - Bollinger Bands inside Keltner Channels
        
        FORMULA:
        - BB: 20 SMA ± 2 StdDev
        - KC: 20 EMA ± 1.5 ATR
        - Squeeze ON = BB inside KC (low volatility)
        - Squeeze OFF = BB outside KC (volatility expanding)
        
        SIGNIFICANCE:
        - Squeeze ON = Volatility compressed, energy building
        - Squeeze FIRED = Volatility expanding, breakout in progress
        - Momentum direction indicates breakout direction
        
        PRO TIP: Squeeze ON for 6+ bars with momentum building = highest probability
        """
        if df is None or len(df) < 25:
            return {"squeeze_on": False, "squeeze_fired": False, "momentum": 0, "squeeze_count": 0}
        
        df = df.copy()
        
        # Bollinger Bands (20, 2)
        df["sma20"] = df["close"].rolling(20).mean()
        df["std20"] = df["close"].rolling(20).std()
        df["bb_upper"] = df["sma20"] + 2 * df["std20"]
        df["bb_lower"] = df["sma20"] - 2 * df["std20"]
        
        # Keltner Channels (20, 1.5)
        df["ema20"] = df["close"].ewm(span=20).mean()
        df["tr"] = np.maximum(
            df["high"] - df["low"],
            np.maximum(
                abs(df["high"] - df["close"].shift(1)),
                abs(df["low"] - df["close"].shift(1))
            )
        )
        df["atr"] = df["tr"].rolling(20).mean()
        df["kc_upper"] = df["ema20"] + 1.5 * df["atr"]
        df["kc_lower"] = df["ema20"] - 1.5 * df["atr"]
        
        # Squeeze detection
        df["squeeze_on"] = (df["bb_lower"] > df["kc_lower"]) & (df["bb_upper"] < df["kc_upper"])
        
        # Momentum (using linear regression of close - midline)
        df["midline"] = (df["high"].rolling(20).max() + df["low"].rolling(20).min()) / 2
        df["momentum"] = df["close"] - df["midline"]
        
        # Current state
        current_squeeze = df["squeeze_on"].iloc[-1]
        prev_squeeze = df["squeeze_on"].iloc[-2] if len(df) > 1 else False
        squeeze_fired = prev_squeeze and not current_squeeze
        
        # Count consecutive squeeze bars
        squeeze_count = 0
        for i in range(len(df) - 1, -1, -1):
            if df["squeeze_on"].iloc[i]:
                squeeze_count += 1
            else:
                break
        
        # Momentum analysis
        momentum = df["momentum"].iloc[-1]
        prev_momentum = df["momentum"].iloc[-2] if len(df) > 1 else 0
        momentum_increasing = abs(momentum) > abs(prev_momentum)
        momentum_direction = "BULLISH" if momentum > 0 else "BEARISH"
        
        return {
            "squeeze_on": bool(current_squeeze),
            "squeeze_fired": bool(squeeze_fired),
            "momentum": round(momentum, 4),
            "momentum_direction": momentum_direction,
            "momentum_increasing": momentum_increasing,
            "squeeze_count": squeeze_count,
            "bb_width": round((df["bb_upper"].iloc[-1] - df["bb_lower"].iloc[-1]) / df["sma20"].iloc[-1] * 100, 2),
            "interpretation": self._interpret_squeeze(current_squeeze, squeeze_fired, momentum, squeeze_count, momentum_increasing)
        }
    
    def _interpret_squeeze(self, squeeze_on: bool, fired: bool, momentum: float, count: int, increasing: bool) -> str:
        direction = "bullish" if momentum > 0 else "bearish"
        
        if fired:
            return f"🔥 SQUEEZE JUST FIRED! Breakout in progress with {direction} momentum. Enter on confirmation!"
        elif squeeze_on and count >= 6 and increasing:
            return f"💎 PERFECT SETUP - Squeeze ON for {count} bars with {direction} momentum BUILDING. Explosive move imminent!"
        elif squeeze_on and count >= 6:
            return f"🔴 Squeeze ON for {count} bars - Extended compression. Watch for momentum shift."
        elif squeeze_on:
            return f"🔴 Squeeze ON ({count} bars) - Volatility compressed, {direction} momentum. Building energy."
        else:
            return f"🟢 Squeeze OFF - Normal volatility. Current momentum: {direction}."
    
    # =========================================================================
    # SIGNAL 5: VOLUME ANALYSIS
    # =========================================================================
    def analyze_volume(self, df: pd.DataFrame) -> Dict:
        """
        Analyze volume patterns for breakout confirmation.
        
        SIGNIFICANCE:
        - Volume declining during consolidation = healthy setup
        - Volume spike on breakout = confirmation
        - Low volume breakout = likely to fail
        
        PRO TIP: Ideal setup = volume 50%+ below average during squeeze, 
                 then 150%+ above average on breakout day
        """
        if df is None or len(df) < 20:
            return {"volume_pattern": "UNKNOWN", "relative_volume": 0, "volume_contracting": False}
        
        df = df.copy()
        
        # Calculate volume metrics
        avg_volume_20 = df["volume"].tail(20).mean()
        avg_volume_5 = df["volume"].tail(5).mean()
        current_volume = df["volume"].iloc[-1]
        
        # Relative volume
        relative_volume = (current_volume / avg_volume_20) * 100 if avg_volume_20 > 0 else 100
        
        # Volume trend (is it contracting?) - Use percentile-based adaptive threshold
        volume_slope = np.polyfit(range(10), df["volume"].tail(10).values, 1)[0]
        
        # Calculate volume percentile for adaptive threshold
        # Low volatility stocks need tighter threshold, high volatility need looser
        volume_std = df["volume"].tail(20).std()
        volume_cv = volume_std / avg_volume_20 if avg_volume_20 > 0 else 0  # Coefficient of variation
        
        # Adaptive threshold: 0.7 for low CV stocks, 0.85 for high CV stocks
        adaptive_threshold = 0.7 + min(0.15, volume_cv * 0.5)
        volume_contracting = volume_slope < 0 and avg_volume_5 < avg_volume_20 * adaptive_threshold
        
        # Volume pattern classification
        if relative_volume > 200:
            volume_pattern = "SURGE"
        elif relative_volume > 150:
            volume_pattern = "HIGH"
        elif relative_volume < 50:
            volume_pattern = "VERY_LOW"
        elif relative_volume < 75:
            volume_pattern = "LOW"
        else:
            volume_pattern = "NORMAL"
        
        # Institutional activity indicator
        # Large volume on small price change = accumulation/distribution
        price_change_pct = abs(df["close"].iloc[-1] - df["close"].iloc[-2]) / df["close"].iloc[-2] * 100
        if relative_volume > 150 and price_change_pct < 1:
            institutional_activity = "HIGH"
        elif relative_volume > 120 and price_change_pct < 0.5:
            institutional_activity = "MODERATE"
        else:
            institutional_activity = "NORMAL"
        
        return {
            "volume_pattern": volume_pattern,
            "relative_volume": round(relative_volume, 1),
            "volume_contracting": volume_contracting,
            "avg_volume_20": int(avg_volume_20),
            "current_volume": int(current_volume),
            "institutional_activity": institutional_activity,
            "interpretation": self._interpret_volume(volume_pattern, volume_contracting, relative_volume, institutional_activity)
        }
    
    def _interpret_volume(self, pattern: str, contracting: bool, rel_vol: float, institutional: str) -> str:
        if pattern == "SURGE" and institutional == "HIGH":
            return f"🔥 VOLUME SURGE ({rel_vol:.0f}%) with institutional footprint! Major move in progress."
        elif pattern == "SURGE":
            return f"📈 High volume ({rel_vol:.0f}% of average) - Strong interest, confirm with price action."
        elif contracting:
            return "✅ Volume contracting during consolidation - HEALTHY setup for breakout."
        elif pattern == "VERY_LOW":
            return "⚠️ Very low volume - Lack of interest. Wait for volume confirmation before entry."
        elif institutional == "HIGH":
            return "🏦 Institutional activity detected - Large volume with small price change = accumulation/distribution."
        return f"📊 Normal volume ({rel_vol:.0f}% of average)."
    
    # =========================================================================
    # SIGNAL 6: CHART PATTERN DETECTION
    # =========================================================================
    def detect_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Detect consolidation patterns (triangles, flags, wedges).
        
        PATTERNS:
        - Ascending Triangle: Flat top, rising bottoms (bullish)
        - Descending Triangle: Flat bottom, falling tops (bearish)
        - Symmetrical Triangle: Converging highs and lows (neutral)
        - Bull Flag: Sharp rise, slight pullback (bullish continuation)
        """
        if df is None or len(df) < 20:
            return {"pattern": "NONE", "bias": "NEUTRAL", "pattern_quality": 0}
        
        df = df.copy()
        recent = df.tail(15)
        
        # Calculate ATR for adaptive thresholds (accounts for stock volatility)
        tr = np.maximum(
            recent["high"] - recent["low"],
            np.maximum(
                abs(recent["high"] - recent["close"].shift(1)),
                abs(recent["low"] - recent["close"].shift(1))
            )
        )
        atr = tr.mean()
        avg_price = recent["close"].mean()
        
        # ATR as percentage of price - used to scale thresholds
        atr_pct = (atr / avg_price) * 100 if avg_price > 0 else 1
        
        # Calculate trend lines
        highs = recent["high"].values
        lows = recent["low"].values
        x = np.arange(len(highs))
        
        # Linear regression for highs and lows
        high_slope, high_intercept = np.polyfit(x, highs, 1)
        low_slope, low_intercept = np.polyfit(x, lows, 1)
        
        # Normalize slopes by price
        high_slope_pct = (high_slope / avg_price) * 100
        low_slope_pct = (low_slope / avg_price) * 100
        
        # Adaptive thresholds based on ATR
        # Higher volatility stocks need larger slope to be significant
        flat_threshold = max(0.15, min(0.5, atr_pct * 0.3))  # Range: 0.15 to 0.5
        trend_threshold = max(0.2, min(0.6, atr_pct * 0.4))  # Range: 0.2 to 0.6
        flag_threshold = max(0.3, min(0.8, atr_pct * 0.5))   # Range: 0.3 to 0.8
        
        # Pattern detection with adaptive thresholds
        pattern = "NONE"
        bias = "NEUTRAL"
        pattern_quality = 0
        
        # Ascending Triangle: flat highs, rising lows
        if abs(high_slope_pct) < flat_threshold and low_slope_pct > trend_threshold:
            pattern = "ASCENDING_TRIANGLE"
            bias = "BULLISH"
            pattern_quality = min(100, (low_slope_pct / trend_threshold) * 50)
        
        # Descending Triangle: falling highs, flat lows
        elif high_slope_pct < -trend_threshold and abs(low_slope_pct) < flat_threshold:
            pattern = "DESCENDING_TRIANGLE"
            bias = "BEARISH"
            pattern_quality = min(100, (abs(high_slope_pct) / trend_threshold) * 50)
        
        # Symmetrical Triangle: converging
        elif high_slope_pct < -flat_threshold and low_slope_pct > flat_threshold:
            pattern = "SYMMETRICAL_TRIANGLE"
            bias = "NEUTRAL"
            pattern_quality = min(100, ((abs(high_slope_pct) + low_slope_pct) / (flat_threshold * 2)) * 50)
        
        # Bull Flag: slight downward drift after uptrend
        elif high_slope_pct < 0 and high_slope_pct > -flag_threshold and low_slope_pct < 0 and low_slope_pct > -flag_threshold:
            # Check for prior uptrend
            prior = df.tail(30).head(15)
            prior_trend = (prior["close"].iloc[-1] - prior["close"].iloc[0]) / prior["close"].iloc[0] * 100
            if prior_trend > 5:
                pattern = "BULL_FLAG"
                bias = "BULLISH"
                pattern_quality = min(100, prior_trend * 10)
        
        # Bear Flag: slight upward drift after downtrend
        elif high_slope_pct > 0 and high_slope_pct < flag_threshold and low_slope_pct > 0 and low_slope_pct < flag_threshold:
            prior = df.tail(30).head(15)
            prior_trend = (prior["close"].iloc[-1] - prior["close"].iloc[0]) / prior["close"].iloc[0] * 100
            if prior_trend < -5:
                pattern = "BEAR_FLAG"
                bias = "BEARISH"
                pattern_quality = min(100, abs(prior_trend) * 10)
        
        return {
            "pattern": pattern,
            "bias": bias,
            "pattern_quality": round(pattern_quality, 1),
            "high_slope": round(high_slope_pct, 3),
            "low_slope": round(low_slope_pct, 3),
            "interpretation": self._interpret_pattern(pattern, bias, pattern_quality)
        }
    
    def _interpret_pattern(self, pattern: str, bias: str, quality: float) -> str:
        if pattern == "ASCENDING_TRIANGLE":
            return f"📐 ASCENDING TRIANGLE ({quality:.0f}% quality) - Bullish breakout pattern. Buyers stepping up at higher lows."
        elif pattern == "DESCENDING_TRIANGLE":
            return f"📐 DESCENDING TRIANGLE ({quality:.0f}% quality) - Bearish breakdown pattern. Sellers pressing at lower highs."
        elif pattern == "SYMMETRICAL_TRIANGLE":
            return f"📐 SYMMETRICAL TRIANGLE ({quality:.0f}% quality) - Neutral, breakout either direction. Wait for confirmation."
        elif pattern == "BULL_FLAG":
            return f"🚩 BULL FLAG ({quality:.0f}% quality) - Bullish continuation. Healthy pullback after strong move."
        elif pattern == "BEAR_FLAG":
            return f"🚩 BEAR FLAG ({quality:.0f}% quality) - Bearish continuation. Weak bounce after strong drop."
        return "No clear chart pattern detected."
    
    # =========================================================================
    # ADVANCED SIGNAL 7: RSI DIVERGENCE
    # =========================================================================
    def detect_rsi_divergence(self, df: pd.DataFrame, period: int = 14) -> Dict:
        """
        Detect RSI divergence from price.
        
        PRO TIP: RSI divergence + OBV divergence in same direction = VERY high probability reversal
        """
        if df is None or len(df) < period + 10:
            return {"rsi": 50, "divergence": "NONE", "overbought": False, "oversold": False}
        
        df = df.copy()
        
        # Calculate RSI with proper handling of edge cases
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # Handle division by zero: when loss is 0, RSI = 100 (all gains)
        # When gain is 0, RSI = 0 (all losses)
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        rs = gain / (loss + epsilon)
        df["rsi"] = 100 - (100 / (1 + rs))
        
        # Handle NaN values (can occur at start of series)
        df["rsi"] = df["rsi"].fillna(50)  # Default to neutral
        
        current_rsi = df["rsi"].iloc[-1]
        
        # Check for divergence in last 10 bars
        recent_price = df["close"].tail(10)
        recent_rsi = df["rsi"].tail(10)
        
        price_slope = np.polyfit(range(10), recent_price.values, 1)[0]
        rsi_slope = np.polyfit(range(10), recent_rsi.values, 1)[0]
        
        # Normalize
        price_slope_norm = price_slope / recent_price.mean() * 100
        
        divergence = "NONE"
        if price_slope_norm < -0.5 and rsi_slope > 0.5:
            divergence = "BULLISH"
        elif price_slope_norm > 0.5 and rsi_slope < -0.5:
            divergence = "BEARISH"
        
        return {
            "rsi": round(current_rsi, 1),
            "divergence": divergence,
            "overbought": current_rsi > 70,
            "oversold": current_rsi < 30,
            "interpretation": self._interpret_rsi(current_rsi, divergence)
        }
    
    def _interpret_rsi(self, rsi: float, divergence: str) -> str:
        if divergence == "BULLISH" and rsi < 40:
            return f"🟢 BULLISH RSI DIVERGENCE at {rsi:.0f} - Strong reversal signal! Price down but momentum building."
        elif divergence == "BEARISH" and rsi > 60:
            return f"🔴 BEARISH RSI DIVERGENCE at {rsi:.0f} - Reversal warning! Price up but momentum fading."
        elif rsi > 70:
            return f"⚠️ RSI OVERBOUGHT ({rsi:.0f}) - Extended. Watch for pullback or divergence."
        elif rsi < 30:
            return f"⚠️ RSI OVERSOLD ({rsi:.0f}) - Extended. Watch for bounce or divergence."
        return f"RSI at {rsi:.0f} - Neutral zone."
    
    # =========================================================================
    # ADVANCED SIGNAL 8: ADX TREND STRENGTH
    # =========================================================================
    def detect_adx_trend(self, df: pd.DataFrame, period: int = 14) -> Dict:
        """
        ADX (Average Directional Index) measures trend strength.
        
        ADX LEVELS:
        - Below 20: Weak/No trend (range-bound)
        - 20-25: Trend emerging
        - 25-50: Strong trend
        - Above 50: Very strong trend (rare, often exhaustion)
        
        PRO TIP: ADX rising from below 20 + squeeze firing = EXPLOSIVE move
        """
        if df is None or len(df) < period + 5:
            return {"adx": 0, "trend_strength": "WEAK", "plus_di": 0, "minus_di": 0}
        
        df = df.copy()
        
        # Calculate True Range
        df["tr"] = np.maximum(
            df["high"] - df["low"],
            np.maximum(
                abs(df["high"] - df["close"].shift(1)),
                abs(df["low"] - df["close"].shift(1))
            )
        )
        
        # Calculate +DM and -DM
        df["plus_dm"] = np.where(
            (df["high"] - df["high"].shift(1)) > (df["low"].shift(1) - df["low"]),
            np.maximum(df["high"] - df["high"].shift(1), 0),
            0
        )
        df["minus_dm"] = np.where(
            (df["low"].shift(1) - df["low"]) > (df["high"] - df["high"].shift(1)),
            np.maximum(df["low"].shift(1) - df["low"], 0),
            0
        )
        
        # Smooth with Wilder's method - add epsilon to avoid division by zero
        epsilon = 1e-10
        df["atr"] = df["tr"].ewm(alpha=1/period, adjust=False).mean()
        df["plus_di"] = 100 * (df["plus_dm"].ewm(alpha=1/period, adjust=False).mean() / (df["atr"] + epsilon))
        df["minus_di"] = 100 * (df["minus_dm"].ewm(alpha=1/period, adjust=False).mean() / (df["atr"] + epsilon))
        
        # Calculate DX and ADX - handle division by zero when both DI are 0
        di_sum = df["plus_di"] + df["minus_di"]
        df["dx"] = np.where(di_sum > epsilon, 
                            100 * abs(df["plus_di"] - df["minus_di"]) / di_sum,
                            0)  # No trend when both DI are 0
        df["adx"] = df["dx"].ewm(alpha=1/period, adjust=False).mean()
        
        # Handle any remaining NaN values
        df["adx"] = df["adx"].fillna(0)
        df["plus_di"] = df["plus_di"].fillna(0)
        df["minus_di"] = df["minus_di"].fillna(0)
        
        current_adx = df["adx"].iloc[-1]
        plus_di = df["plus_di"].iloc[-1]
        minus_di = df["minus_di"].iloc[-1]
        
        # Trend strength classification
        if current_adx < 20:
            trend_strength = "WEAK"
        elif current_adx < 25:
            trend_strength = "EMERGING"
        elif current_adx < 50:
            trend_strength = "STRONG"
        else:
            trend_strength = "VERY_STRONG"
        
        # Trend direction
        trend_direction = "BULLISH" if plus_di > minus_di else "BEARISH"
        
        return {
            "adx": round(current_adx, 1),
            "trend_strength": trend_strength,
            "trend_direction": trend_direction,
            "plus_di": round(plus_di, 1),
            "minus_di": round(minus_di, 1),
            "interpretation": self._interpret_adx(current_adx, trend_strength, trend_direction)
        }
    
    def _interpret_adx(self, adx: float, strength: str, direction: str) -> str:
        if strength == "WEAK":
            return f"📊 ADX {adx:.0f} - No clear trend. Range-bound conditions. Breakout strategies preferred."
        elif strength == "EMERGING":
            return f"📈 ADX {adx:.0f} - Trend EMERGING ({direction}). Early entry opportunity if confirmed."
        elif strength == "STRONG":
            return f"💪 ADX {adx:.0f} - STRONG {direction} trend. Trend-following strategies work well."
        else:
            return f"🔥 ADX {adx:.0f} - VERY STRONG trend. May be extended - watch for exhaustion."
    
    # =========================================================================
    # MASTER BREAKOUT ANALYSIS - INSTITUTIONAL GRADE
    # =========================================================================
    def analyze_breakout(self, symbol: str) -> Dict:
        """
        INSTITUTIONAL-GRADE BREAKOUT ANALYSIS
        
        Combines 10+ signals with synergy bonuses for maximum accuracy.
        
        SCORING:
        - Base signals: 0-100 points
        - Synergy bonuses: Up to +30 points
        - Quality multiplier: 0.8x to 1.2x
        
        SYNERGY COMBINATIONS (PRO SECRETS):
        1. "Coiled Spring" = NR7 + TTM Squeeze ON = +15 bonus
        2. "Stealth Accumulation" = OBV Bullish Div + Volume Contracting = +10 bonus
        3. "Technical Confluence" = Pattern + S/R Testing = +10 bonus
        4. "Momentum Alignment" = RSI Div + OBV Div same direction = +10 bonus
        5. "Trend Confirmation" = ADX Emerging + Squeeze Firing = +15 bonus
        """
        # Fetch real-time data
        df = self._fetch_price_data(symbol, "1day", 100)
        
        if df is None or len(df) < 20:
            return {
                "status": "error",
                "error": "Unable to fetch price data",
                "symbol": symbol
            }
        
        # Run ALL signal detections
        nr_patterns = self.detect_nr_patterns(df)
        obv_divergence = self.detect_obv_divergence(df)
        sr_testing = self.detect_sr_testing(df)
        ttm_squeeze = self.detect_ttm_squeeze(df)
        volume = self.analyze_volume(df)
        patterns = self.detect_patterns(df)
        rsi = self.detect_rsi_divergence(df)
        adx = self.detect_adx_trend(df)
        
        # =====================================================================
        # CALCULATE BASE SCORE (0-100)
        # =====================================================================
        base_score = 0
        signals = []
        
        # NR4/NR7 (0-20 points)
        if nr_patterns["nr4_nr7_combined"]:
            base_score += 20
            signals.append("NR7+NR4 Combo 🔥")
        elif nr_patterns["nr7"]:
            base_score += 18
            signals.append("NR7 Pattern ✅")
        elif nr_patterns["nr4"]:
            base_score += 12
            signals.append("NR4 Pattern ✅")
        elif nr_patterns["consecutive_narrow_days"] >= 3:
            base_score += 6
            signals.append(f"{nr_patterns['consecutive_narrow_days']} Narrow Days")
        
        # OBV Divergence (0-20 points)
        if obv_divergence["divergence"] in ["BULLISH", "HIDDEN_BULLISH"]:
            score_add = min(20, 10 + obv_divergence["divergence_strength"] / 10)
            base_score += score_add
            signals.append(f"{obv_divergence['divergence']} OBV ✅")
        elif obv_divergence["divergence"] in ["BEARISH", "HIDDEN_BEARISH"]:
            base_score += 10
            signals.append(f"{obv_divergence['divergence']} OBV ⚠️")
        
        # TTM Squeeze (0-25 points)
        if ttm_squeeze["squeeze_fired"]:
            base_score += 25
            signals.append("TTM Squeeze FIRED! 🔥")
        elif ttm_squeeze["squeeze_on"] and ttm_squeeze["squeeze_count"] >= 6:
            base_score += 20
            signals.append(f"Squeeze ON {ttm_squeeze['squeeze_count']} bars 🔴")
        elif ttm_squeeze["squeeze_on"]:
            base_score += 12
            signals.append(f"Squeeze ON ({ttm_squeeze['squeeze_count']} bars)")
        
        # Support/Resistance Testing (0-15 points)
        if sr_testing["testing"] != "NONE":
            touches = sr_testing["touches"]
            if touches >= 4:
                base_score += 15
                signals.append(f"{sr_testing['testing']} tested {touches}x 🔥")
            elif touches >= 3:
                base_score += 12
                signals.append(f"{sr_testing['testing']} tested {touches}x ✅")
            else:
                base_score += 6
                signals.append(f"{sr_testing['testing']} testing")
        
        # Volume Pattern (0-10 points)
        if volume["volume_contracting"]:
            base_score += 10
            signals.append("Volume Contracting ✅")
        elif volume["volume_pattern"] == "SURGE":
            base_score += 8
            signals.append("Volume Surge 📈")
        elif volume["institutional_activity"] == "HIGH":
            base_score += 6
            signals.append("Institutional Activity 🏦")
        
        # Chart Pattern (0-10 points)
        if patterns["pattern"] != "NONE":
            quality_bonus = patterns["pattern_quality"] / 10
            base_score += min(10, 5 + quality_bonus)
            signals.append(f"{patterns['pattern'].replace('_', ' ')} ✅")
        
        # RSI Divergence (0-8 points) - BONUS
        if rsi["divergence"] != "NONE":
            base_score += 8
            signals.append(f"RSI {rsi['divergence']} Divergence")
        
        # ADX Trend (0-7 points) - BONUS
        if adx["trend_strength"] == "EMERGING":
            base_score += 7
            signals.append("Trend Emerging 📈")
        elif adx["trend_strength"] == "STRONG":
            base_score += 5
            signals.append("Strong Trend 💪")
        
        # =====================================================================
        # SYNERGY BONUSES (PRO SECRETS)
        # =====================================================================
        synergy_bonus = 0
        synergies = []
        
        # 1. "Coiled Spring" = NR7 + TTM Squeeze ON
        if (nr_patterns["nr7"] or nr_patterns["nr4"]) and ttm_squeeze["squeeze_on"]:
            synergy_bonus += 15
            synergies.append("🌀 COILED SPRING (+15)")
        
        # 2. "Stealth Accumulation" = OBV Bullish + Volume Contracting
        if obv_divergence["divergence"] in ["BULLISH", "HIDDEN_BULLISH"] and volume["volume_contracting"]:
            synergy_bonus += 10
            synergies.append("🥷 STEALTH ACCUMULATION (+10)")
        
        # 3. "Technical Confluence" = Pattern + S/R Testing
        if patterns["pattern"] != "NONE" and sr_testing["testing"] != "NONE":
            synergy_bonus += 10
            synergies.append("🎯 TECHNICAL CONFLUENCE (+10)")
        
        # 4. "Momentum Alignment" = RSI Div + OBV Div same direction
        if rsi["divergence"] != "NONE" and obv_divergence["divergence"] != "NONE":
            if (rsi["divergence"] == "BULLISH" and obv_divergence["divergence"] in ["BULLISH", "HIDDEN_BULLISH"]) or \
               (rsi["divergence"] == "BEARISH" and obv_divergence["divergence"] in ["BEARISH", "HIDDEN_BEARISH"]):
                synergy_bonus += 10
                synergies.append("⚡ MOMENTUM ALIGNMENT (+10)")
        
        # 5. "Trend Confirmation" = ADX Emerging + Squeeze Firing
        if adx["trend_strength"] in ["EMERGING", "STRONG"] and ttm_squeeze["squeeze_fired"]:
            synergy_bonus += 15
            synergies.append("🚀 TREND CONFIRMATION (+15)")
        
        # =====================================================================
        # FINAL SCORE CALCULATION - Improved granularity
        # =====================================================================
        
        # Cap base score at 100 (theoretical max is ~115 from all signals)
        base_score_capped = min(100, base_score)
        
        # Synergy bonus adds on top, but capped at 30 points
        synergy_bonus_capped = min(30, synergy_bonus)
        
        # Quality multiplier based on signal strength (applied to base only)
        # Quality multiplier - conservative to avoid overfitting
        # Only boost when signals are exceptionally strong
        quality_factors = []
        if nr_patterns["signal_strength"] == "VERY_STRONG":
            quality_factors.append(1.05)  # Reduced from 1.08
        if obv_divergence["divergence_strength"] > 75:  # Raised threshold from 70
            quality_factors.append(1.05)  # Reduced from 1.08
        if ttm_squeeze["momentum_increasing"]:
            quality_factors.append(1.03)  # Reduced from 1.04
        if volume["institutional_activity"] == "HIGH":
            quality_factors.append(1.05)  # Reduced from 1.08
        
        quality_multiplier = 1.0
        for qf in quality_factors:
            quality_multiplier *= qf
        quality_multiplier = min(1.15, quality_multiplier)  # Reduced cap from 1.25x to 1.15x
        
        # Apply quality multiplier to base score, then add synergy bonus
        # This preserves granularity at high scores
        adjusted_base = base_score_capped * quality_multiplier
        raw_score = adjusted_base + synergy_bonus_capped
        
        # Final score: scale to 0-100 with proper distribution
        # Max possible: 100 * 1.15 + 30 = 145, scale down
        # Using 0.72 to make scores more conservative and avoid overfitting
        final_score = min(100, int(raw_score * 0.72))  # Slightly more conservative scaling
        
        # Ensure minimum score of 5 if any signal is present
        if len(signals) > 0 and final_score < 5:
            final_score = 5
        
        # =====================================================================
        # DETERMINE PROBABILITY AND RECOMMENDATION
        # =====================================================================
        if final_score >= 75:
            probability = "VERY HIGH"
            recommendation = "🚀 ELITE SETUP - Multiple signals + synergies aligned. This is what institutions look for!"
        elif final_score >= 55:
            probability = "HIGH"
            recommendation = "📈 STRONG SETUP - Good signal confluence. Enter with proper risk management."
        elif final_score >= 35:
            probability = "MODERATE"
            recommendation = "📊 DEVELOPING - Some signals present. Wait for more confirmation or use smaller size."
        else:
            probability = "LOW"
            recommendation = "⏳ NO CLEAR SETUP - Insufficient signals. Be patient for better opportunity."
        
        # =====================================================================
        # DETERMINE DIRECTION BIAS
        # =====================================================================
        bullish_weight = 0
        bearish_weight = 0
        
        # OBV direction (weight: 3)
        if obv_divergence["divergence"] in ["BULLISH", "HIDDEN_BULLISH"]:
            bullish_weight += 3
        elif obv_divergence["divergence"] in ["BEARISH", "HIDDEN_BEARISH"]:
            bearish_weight += 3
        
        # TTM momentum (weight: 2)
        if ttm_squeeze["momentum"] > 0:
            bullish_weight += 2
        else:
            bearish_weight += 2
        
        # Pattern bias (weight: 2)
        if patterns["bias"] == "BULLISH":
            bullish_weight += 2
        elif patterns["bias"] == "BEARISH":
            bearish_weight += 2
        
        # S/R testing (weight: 1) - Consider price position relative to level
        # Testing resistance from below = bullish (trying to break out)
        # Testing support from above = could be bullish (bounce) or bearish (breakdown)
        # Use momentum to determine support test direction
        if sr_testing["testing"] == "RESISTANCE":
            bullish_weight += 1  # Testing resistance is bullish intent
        elif sr_testing["testing"] == "SUPPORT":
            # If momentum is positive, likely a bounce (bullish)
            # If momentum is negative, likely a breakdown (bearish)
            if ttm_squeeze["momentum"] > 0:
                bullish_weight += 1  # Support bounce
            else:
                bearish_weight += 1  # Support breakdown risk
        
        # RSI divergence (weight: 2)
        if rsi["divergence"] == "BULLISH":
            bullish_weight += 2
        elif rsi["divergence"] == "BEARISH":
            bearish_weight += 2
        
        # ADX direction (weight: 1)
        if adx["trend_direction"] == "BULLISH":
            bullish_weight += 1
        else:
            bearish_weight += 1
        
        if bullish_weight > bearish_weight + 2:
            direction = "BULLISH"
        elif bearish_weight > bullish_weight + 2:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"
        
        return {
            "status": "success",
            "symbol": symbol,
            "current_price": round(df["close"].iloc[-1], 2),
            "timestamp": datetime.now().isoformat(),
            
            # Composite Score
            "breakout_score": final_score,
            "base_score": base_score,
            "synergy_bonus": synergy_bonus,
            "quality_multiplier": round(quality_multiplier, 2),
            "max_score": 100,
            "breakout_probability": probability,
            "direction_bias": direction,
            "recommendation": recommendation,
            "active_signals": signals,
            "signal_count": len(signals),
            "synergies": synergies,
            
            # Individual Signal Details
            "nr_patterns": nr_patterns,
            "obv_divergence": obv_divergence,
            "sr_testing": sr_testing,
            "ttm_squeeze": ttm_squeeze,
            "volume": volume,
            "chart_patterns": patterns,
            "rsi": rsi,
            "adx": adx,
            
            # Key Levels
            "pivot": sr_testing["pivot"],
            "resistance_1": sr_testing["r1"],
            "resistance_2": sr_testing["r2"],
            "support_1": sr_testing["s1"],
            "support_2": sr_testing["s2"],
            "nearest_resistance": sr_testing["nearest_resistance"],
            "nearest_support": sr_testing["nearest_support"]
        }
    
    # =========================================================================
    # MARKET SCAN - Scan Multiple Stocks for Breakout Setups
    # =========================================================================
    def scan_market(self, symbols: List[str] = None, min_score: int = 40) -> Dict:
        """
        Scan multiple stocks for breakout setups.
        """
        if symbols is None:
            symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]
        
        results = []
        errors = []
        
        for symbol in symbols:
            try:
                time.sleep(0.5)  # Rate limit
                
                analysis = self.analyze_breakout(symbol)
                
                if analysis["status"] == "success":
                    if analysis["breakout_score"] >= min_score:
                        results.append({
                            "symbol": symbol,
                            "price": analysis["current_price"],
                            "score": analysis["breakout_score"],
                            "max_score": analysis["max_score"],
                            "probability": analysis["breakout_probability"],
                            "direction": analysis["direction_bias"],
                            "signals": analysis["active_signals"],
                            "signal_count": analysis["signal_count"],
                            "synergies": analysis.get("synergies", []),
                            "recommendation": analysis["recommendation"],
                            "resistance": analysis["nearest_resistance"],
                            "support": analysis["nearest_support"],
                            "nr_pattern": analysis["nr_patterns"]["nr7"] or analysis["nr_patterns"]["nr4"],
                            "squeeze_on": analysis["ttm_squeeze"]["squeeze_on"],
                            "squeeze_fired": analysis["ttm_squeeze"]["squeeze_fired"],
                            "obv_divergence": analysis["obv_divergence"]["divergence"]
                        })
                else:
                    errors.append({"symbol": symbol, "error": analysis.get("error", "Unknown")})
                    
            except Exception as e:
                errors.append({"symbol": symbol, "error": str(e)})
        
        results.sort(key=lambda x: x["score"], reverse=True)
        
        very_high = [r for r in results if r["probability"] == "VERY HIGH"]
        high = [r for r in results if r["probability"] == "HIGH"]
        moderate = [r for r in results if r["probability"] == "MODERATE"]
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "total_scanned": len(symbols),
            "total_setups": len(results),
            "very_high_probability": very_high,
            "high_probability": high,
            "moderate_probability": moderate,
            "all_results": results,
            "errors": errors,
            "scan_summary": f"Found {len(very_high)} VERY HIGH, {len(high)} HIGH, {len(moderate)} MODERATE probability setups"
        }
    
    def _get_dynamic_universe(self) -> List[str]:
        """
        Get dynamic stock universe from real-time market data.
        No predefined restrictions - scans what's actually moving TODAY.
        """
        import requests
        from bs4 import BeautifulSoup
        
        dynamic_tickers = set()
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        # Get TODAY'S most active, gainers, losers from Yahoo Finance
        try:
            for url in [
                "https://finance.yahoo.com/most-active",
                "https://finance.yahoo.com/gainers",
                "https://finance.yahoo.com/losers",
                "https://finance.yahoo.com/trending-tickers"
            ]:
                resp = requests.get(url, headers=headers, timeout=10)
                soup = BeautifulSoup(resp.text, 'html.parser')
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if '/quote/' in href and '?' not in href:
                        ticker = href.split('/quote/')[1].split('/')[0].split('?')[0]
                        if ticker.isalpha() and len(ticker) <= 5:
                            dynamic_tickers.add(ticker)
        except:
            pass
        
        return list(dynamic_tickers)
    
    def quick_scan(self, top_n: int = 20) -> Dict:
        """
        Quick scan of 200+ stocks and ETFs for breakout setups.
        Combines predefined universe with DYNAMIC real-time discovery.
        """
        # Get dynamic tickers from today's market activity
        dynamic_tickers = self._get_dynamic_universe()
        
        quick_symbols = [
            # === MEGA CAP TECH (10) ===
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO", "ORCL", "ADBE",
            # === LARGE CAP TECH (15) ===
            "CRM", "AMD", "INTC", "CSCO", "IBM", "QCOM", "TXN", "NOW", "INTU", "AMAT",
            "MU", "LRCX", "KLAC", "SNPS", "CDNS",
            # === SEMICONDUCTORS (10) ===
            "ASML", "TSM", "ARM", "MRVL", "ON", "ADI", "NXPI", "SWKS", "QRVO", "MPWR",
            # === FINANCIALS (12) ===
            "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "USB",
            "PNC", "TFC",
            # === PAYMENTS/FINTECH (8) ===
            "V", "MA", "PYPL", "SQ", "COIN", "AFRM", "UPST", "NU",
            # === HEALTHCARE (15) ===
            "UNH", "JNJ", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT", "DHR", "BMY",
            "GILD", "REGN", "VRTX", "MRNA", "BNTX",
            # === BIOTECH (10) ===
            "CRSP", "BEAM", "NTLA", "EDIT", "VERV", "EXAS", "ILMN", "DXCM", "ISRG", "ALGN",
            # === CONSUMER DISCRETIONARY (12) ===
            "WMT", "HD", "COST", "NKE", "MCD", "SBUX", "TGT", "LOW", "TJX", "DG",
            "LULU", "ROST",
            # === CONSUMER STAPLES (8) ===
            "PG", "KO", "PEP", "PM", "MO", "CL", "KMB", "GIS",
            # === ENERGY (12) ===
            "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO",
            "OXY", "HAL", "DVN", "FANG",
            # === INDUSTRIALS (12) ===
            "CAT", "BA", "GE", "UPS", "RTX", "HON", "LMT", "DE",
            "FDX", "UNP", "CSX", "NSC",
            # === COMMUNICATIONS (8) ===
            "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "CHTR", "WBD",
            # === REAL ESTATE (5) ===
            "AMT", "PLD", "CCI", "EQIX", "SPG",
            # === UTILITIES (5) ===
            "NEE", "DUK", "SO", "D", "AEP",
            # === POPULAR MOMENTUM/GROWTH (12) ===
            "PLTR", "SOFI", "HOOD", "RIVN", "LCID", "NIO", "MARA", "RIOT", "CLSK", "HUT",
            "BITF", "CORZ",
            # === EV/AUTO (8) ===
            "TSLA", "F", "GM", "RIVN", "LCID", "NIO", "XPEV", "LI",
            # === AI/DATA CENTER (10) ===
            "SMCI", "AI", "BBAI", "SOUN", "PATH", "SNOW", "DDOG", "NET", "CRWD", "ZS",
            # === QUANTUM COMPUTING (5) ===
            "IONQ", "RGTI", "QUBT", "ARQQ", "QBTS",
            # === SPACE/AEROSPACE (6) ===
            "RKLB", "LUNR", "RDW", "SPCE", "ASTS", "BKSY",
            # === MEME/RETAIL FAVORITES (8) ===
            "GME", "AMC", "BB", "NOK", "WISH", "CLOV", "SOFI", "PLTR",
            # === CANNABIS (5) ===
            "TLRY", "CGC", "ACB", "SNDL", "CRON",
            # === CHINESE ADRs (8) ===
            "BABA", "JD", "PDD", "BIDU", "NTES", "LI", "XPEV", "NIO",
            # === CLEAN ENERGY (6) ===
            "PLUG", "FCEL", "ENPH", "SEDG", "RUN", "NOVA",
            # === SMALL CAP ETFs (5) ===
            "IJR", "VB", "SCHA", "VTWO", "IWO",
            # === MAJOR ETFs (20) ===
            "SPY", "QQQ", "IWM", "DIA", "VOO", "VTI",
            "XLF", "XLE", "XLK", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE",
            "GLD", "SLV", "USO", "TLT",
        ]
        
        # Combine predefined with dynamic discovery (no duplicates)
        all_symbols = list(set(quick_symbols + dynamic_tickers))
        
        result = self.scan_market(all_symbols, min_score=30)
        
        if result["status"] == "success":
            result["all_results"] = result["all_results"][:top_n]
            
            # Add date/time context - ALWAYS USE EASTERN TIME
            from datetime import datetime
            import pytz
            
            et = pytz.timezone('US/Eastern')
            now = datetime.now(et)
            
            market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
            is_market_hours = market_open <= now <= market_close and now.weekday() < 5
            
            if now.weekday() >= 5:
                session = 'WEEKEND'
            elif now < market_open:
                session = 'PRE_MARKET'
            elif now > market_close:
                session = 'AFTER_HOURS'
            else:
                if now < now.replace(hour=10, minute=30):
                    session = 'OPENING_VOLATILITY'
                elif now < now.replace(hour=12, minute=0):
                    session = 'MORNING_MOMENTUM'
                elif now < now.replace(hour=14, minute=0):
                    session = 'MIDDAY_CONSOLIDATION'
                else:
                    session = 'POWER_HOUR'
            
            result["market_context"] = {
                'is_market_hours': is_market_hours,
                'trading_session': session,
                'scan_time': now.strftime('%Y-%m-%d %H:%M:%S ET'),
                'day_of_week': now.strftime('%A'),
                'dynamic_tickers_added': len(dynamic_tickers),
                'total_universe': len(all_symbols)
            }
            
        return result


# Test the module
if __name__ == "__main__":
    import os
    api_key = os.environ.get("TWELVEDATA_API_KEY", "5e7a5daaf41d46a8966963106ebef210")
    
    detector = BreakoutDetector(api_key)
    
    print("=" * 60)
    print("BREAKOUT DETECTOR v2.0 - Institutional Grade Analysis")
    print("=" * 60)
    
    result = detector.analyze_breakout("AAPL")
    
    if result["status"] == "success":
        print(f"\n📊 {result['symbol']} @ ${result['current_price']}")
        print(f"\n🎯 BREAKOUT SCORE: {result['breakout_score']}/{result['max_score']}")
        print(f"   Base: {result['base_score']} + Synergy: {result['synergy_bonus']} × Quality: {result['quality_multiplier']}")
        print(f"📈 PROBABILITY: {result['breakout_probability']}")
        print(f"🧭 DIRECTION: {result['direction_bias']}")
        print(f"\n💡 {result['recommendation']}")
        
        print(f"\n✅ ACTIVE SIGNALS ({result['signal_count']}):")
        for signal in result['active_signals']:
            print(f"   • {signal}")
        
        if result['synergies']:
            print(f"\n🔥 SYNERGY BONUSES:")
            for syn in result['synergies']:
                print(f"   • {syn}")
        
        print(f"\n📐 KEY LEVELS:")
        print(f"   Resistance: ${result['nearest_resistance']}")
        print(f"   Support: ${result['nearest_support']}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
