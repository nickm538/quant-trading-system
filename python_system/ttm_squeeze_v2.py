"""
TTM SQUEEZE INDICATOR - Production Grade Implementation
========================================================

The TTM Squeeze is a volatility and momentum indicator developed by John Carter.
It identifies periods of low volatility (squeeze) that often precede significant price moves.

MATHEMATICAL FORMULAS:
======================

1. BOLLINGER BANDS (BB):
   - Middle Band = 20-period SMA
   - Upper Band = Middle + (2 × 20-period StdDev)
   - Lower Band = Middle - (2 × 20-period StdDev)

2. KELTNER CHANNELS (KC):
   - Middle Line = 20-period EMA
   - Upper Channel = Middle + (1.5 × ATR)
   - Lower Channel = Middle - (1.5 × ATR)
   - ATR = 20-period Average True Range

3. SQUEEZE DETECTION:
   - SQUEEZE ON (Red dots): BB Lower > KC Lower AND BB Upper < KC Upper
     (Bollinger Bands are INSIDE Keltner Channels = low volatility)
   - SQUEEZE OFF (Green dots): BB Lower < KC Lower OR BB Upper > KC Upper
     (Bollinger Bands are OUTSIDE Keltner Channels = volatility expanding)

4. MOMENTUM HISTOGRAM:
   - Value = Linear Regression (12 periods) of (Close - Average(Highest High, Lowest Low, SMA))
   - Midline = (Highest High (20) + Lowest Low (20)) / 2
   - Momentum = Close - ((Midline + SMA(20)) / 2)
   - Histogram = Linear Regression of Momentum over 12 periods

   Color coding:
   - Dark Green: Momentum positive and increasing
   - Light Green: Momentum positive but decreasing
   - Dark Red: Momentum negative and decreasing
   - Light Red: Momentum negative but increasing

DATA SOURCE: TwelveData API (Real-time)
NO FALLBACKS, NO DEMO DATA, NO ASSUMPTIONS
"""

import requests
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime
import time


class TTMSqueeze:
    """
    Production-grade TTM Squeeze indicator with strict mathematical accuracy.
    
    Uses ONLY real-time data from TwelveData API.
    NO fallbacks, NO demo data, NO assumptions.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize TTM Squeeze calculator.
        
        Args:
            api_key: TwelveData API key for real-time data
        """
        self.api_key = api_key
        self.base_url = "https://api.twelvedata.com"
        
        # Standard TTM Squeeze parameters (John Carter's original settings)
        self.bb_period = 20      # Bollinger Bands period
        self.bb_std = 2.0        # Bollinger Bands standard deviation multiplier
        self.kc_period = 20      # Keltner Channel period
        self.kc_atr_mult = 1.5   # Keltner Channel ATR multiplier (standard squeeze)
        self.kc_atr_mult_tight = 1.0  # Tight squeeze detection (more compression)
        self.momentum_period = 12  # Linear regression period for momentum
        
        # Squeeze intensity thresholds (BB width / KC width ratio)
        self.squeeze_intensity_extreme = 0.5   # Very tight compression
        self.squeeze_intensity_high = 0.7      # High compression
        self.squeeze_intensity_moderate = 0.85 # Moderate compression
        
    def _fetch_price_data(self, symbol: str, interval: str = '1day', outputsize: int = 100) -> Optional[List[Dict]]:
        """
        Fetch OHLCV price data from TwelveData.
        
        Returns raw price bars - NO FALLBACKS.
        """
        url = f"{self.base_url}/time_series"
        params = {
            'symbol': symbol,
            'interval': interval,
            'outputsize': outputsize,
            'apikey': self.api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') == 'error':
                print(f"❌ TwelveData API Error: {data.get('message')}")
                return None
            
            values = data.get('values', [])
            if not values:
                print(f"❌ No price data returned for {symbol}")
                return None
            
            return values
            
        except Exception as e:
            print(f"❌ Error fetching price data: {e}")
            return None
    
    def _calculate_sma(self, prices: List[float], period: int) -> List[float]:
        """Calculate Simple Moving Average"""
        sma = []
        for i in range(len(prices)):
            if i < period - 1:
                sma.append(None)
            else:
                sma.append(np.mean(prices[i - period + 1:i + 1]))
        return sma
    
    def _calculate_ema(self, prices: List[float], period: int) -> List[float]:
        """Calculate Exponential Moving Average"""
        ema = []
        multiplier = 2 / (period + 1)
        
        for i in range(len(prices)):
            if i < period - 1:
                ema.append(None)
            elif i == period - 1:
                # First EMA is SMA
                ema.append(np.mean(prices[:period]))
            else:
                ema.append((prices[i] - ema[-1]) * multiplier + ema[-1])
        
        return ema
    
    def _calculate_std(self, prices: List[float], period: int) -> List[float]:
        """Calculate Standard Deviation"""
        std = []
        for i in range(len(prices)):
            if i < period - 1:
                std.append(None)
            else:
                std.append(np.std(prices[i - period + 1:i + 1], ddof=0))
        return std
    
    def _calculate_true_range(self, highs: List[float], lows: List[float], closes: List[float]) -> List[float]:
        """
        Calculate True Range.
        TR = max(High - Low, |High - Previous Close|, |Low - Previous Close|)
        """
        tr = []
        for i in range(len(highs)):
            if i == 0:
                tr.append(highs[i] - lows[i])
            else:
                tr.append(max(
                    highs[i] - lows[i],
                    abs(highs[i] - closes[i - 1]),
                    abs(lows[i] - closes[i - 1])
                ))
        return tr
    
    def _calculate_atr(self, highs: List[float], lows: List[float], closes: List[float], period: int) -> List[float]:
        """Calculate Average True Range using Wilder's smoothing method"""
        tr = self._calculate_true_range(highs, lows, closes)
        atr = []
        
        for i in range(len(tr)):
            if i < period - 1:
                atr.append(None)
            elif i == period - 1:
                # First ATR is simple average of TR
                atr.append(np.mean(tr[:period]))
            else:
                # Wilder's smoothing: ATR = ((Prior ATR × (period-1)) + Current TR) / period
                atr.append((atr[-1] * (period - 1) + tr[i]) / period)
        
        return atr
    
    def _calculate_linear_regression(self, values: List[float], period: int) -> List[float]:
        """
        Calculate Linear Regression value (endpoint of regression line).
        This is used for the momentum histogram.
        """
        lr = []
        for i in range(len(values)):
            if i < period - 1 or values[i - period + 1] is None:
                lr.append(None)
            else:
                y = np.array(values[i - period + 1:i + 1])
                x = np.arange(period)
                
                # Linear regression: y = mx + b
                # We want the value at the end (x = period - 1)
                if None in y:
                    lr.append(None)
                else:
                    slope, intercept = np.polyfit(x, y, 1)
                    lr.append(slope * (period - 1) + intercept)
        
        return lr
    
    def _calculate_highest_high(self, highs: List[float], period: int) -> List[float]:
        """Calculate Highest High over period"""
        hh = []
        for i in range(len(highs)):
            if i < period - 1:
                hh.append(None)
            else:
                hh.append(max(highs[i - period + 1:i + 1]))
        return hh
    
    def _calculate_lowest_low(self, lows: List[float], period: int) -> List[float]:
        """Calculate Lowest Low over period"""
        ll = []
        for i in range(len(lows)):
            if i < period - 1:
                ll.append(None)
            else:
                ll.append(min(lows[i - period + 1:i + 1]))
        return ll
    
    def calculate_squeeze(self, symbol: str, interval: str = '1day') -> Dict:
        """
        Calculate TTM Squeeze indicator with full mathematical precision.
        
        Returns:
            Dict with squeeze status, momentum, and all component values.
            Returns error status if data cannot be fetched (NO FALLBACKS).
        """
        print(f"🔍 Calculating TTM Squeeze for {symbol} ({interval})...")
        
        # Fetch real-time price data
        price_data = self._fetch_price_data(symbol, interval, outputsize=100)
        
        if not price_data:
            return {
                'status': 'error',
                'error': 'Failed to fetch real-time price data',
                'symbol': symbol,
                'timestamp': datetime.now().isoformat()
            }
        
        # Extract OHLCV data (reverse to chronological order - oldest first)
        price_data = list(reversed(price_data))
        
        closes = [float(bar['close']) for bar in price_data]
        highs = [float(bar['high']) for bar in price_data]
        lows = [float(bar['low']) for bar in price_data]
        
        if len(closes) < self.bb_period + self.momentum_period:
            return {
                'status': 'error',
                'error': f'Insufficient data: need {self.bb_period + self.momentum_period} bars, got {len(closes)}',
                'symbol': symbol,
                'timestamp': datetime.now().isoformat()
            }
        
        # ========== BOLLINGER BANDS ==========
        bb_sma = self._calculate_sma(closes, self.bb_period)
        bb_std = self._calculate_std(closes, self.bb_period)
        
        bb_upper = []
        bb_lower = []
        for i in range(len(closes)):
            if bb_sma[i] is None or bb_std[i] is None:
                bb_upper.append(None)
                bb_lower.append(None)
            else:
                bb_upper.append(bb_sma[i] + (self.bb_std * bb_std[i]))
                bb_lower.append(bb_sma[i] - (self.bb_std * bb_std[i]))
        
        # ========== KELTNER CHANNELS ==========
        kc_ema = self._calculate_ema(closes, self.kc_period)
        atr = self._calculate_atr(highs, lows, closes, self.kc_period)
        
        kc_upper = []
        kc_lower = []
        for i in range(len(closes)):
            if kc_ema[i] is None or atr[i] is None:
                kc_upper.append(None)
                kc_lower.append(None)
            else:
                kc_upper.append(kc_ema[i] + (self.kc_atr_mult * atr[i]))
                kc_lower.append(kc_ema[i] - (self.kc_atr_mult * atr[i]))
        
        # ========== SQUEEZE DETECTION ==========
        # Squeeze ON = BB inside KC (low volatility)
        # Squeeze OFF = BB outside KC (volatility expanding)
        squeeze_on = []
        for i in range(len(closes)):
            if bb_lower[i] is None or kc_lower[i] is None:
                squeeze_on.append(None)
            else:
                # Squeeze is ON when BB is INSIDE KC
                is_squeeze = bb_lower[i] > kc_lower[i] and bb_upper[i] < kc_upper[i]
                squeeze_on.append(is_squeeze)
        
        # ========== MOMENTUM HISTOGRAM ==========
        # Momentum = Close - Average(Highest High, Lowest Low, SMA)
        highest_high = self._calculate_highest_high(highs, self.bb_period)
        lowest_low = self._calculate_lowest_low(lows, self.bb_period)
        
        momentum_raw = []
        for i in range(len(closes)):
            if highest_high[i] is None or lowest_low[i] is None or bb_sma[i] is None:
                momentum_raw.append(None)
            else:
                midline = (highest_high[i] + lowest_low[i]) / 2
                avg = (midline + bb_sma[i]) / 2
                momentum_raw.append(closes[i] - avg)
        
        # Apply linear regression to smooth momentum
        momentum = self._calculate_linear_regression(momentum_raw, self.momentum_period)
        
        # ========== MOMENTUM COLOR ==========
        # Determine momentum direction and color
        def get_momentum_color(current: float, previous: float) -> str:
            if current is None or previous is None:
                return 'gray'
            if current >= 0:
                if current > previous:
                    return 'dark_green'  # Positive and increasing
                else:
                    return 'light_green'  # Positive but decreasing
            else:
                if current < previous:
                    return 'dark_red'  # Negative and decreasing
                else:
                    return 'light_red'  # Negative but increasing
        
        # Get current values (most recent)
        idx = -1  # Latest bar
        
        current_squeeze = squeeze_on[idx] if squeeze_on[idx] is not None else None
        current_momentum = momentum[idx] if momentum[idx] is not None else None
        previous_momentum = momentum[idx - 1] if len(momentum) > 1 and momentum[idx - 1] is not None else None
        
        momentum_color = get_momentum_color(current_momentum, previous_momentum)
        
        # Count consecutive squeeze bars
        squeeze_count = 0
        for i in range(len(squeeze_on) - 1, -1, -1):
            if squeeze_on[i] == current_squeeze:
                squeeze_count += 1
            else:
                break
        
        # ========== SIGNAL INTERPRETATION ==========
        signal = 'NEUTRAL'
        signal_strength = 'WEAK'
        
        if current_squeeze is True:
            # Squeeze is ON - waiting for breakout
            signal = 'SQUEEZE_ON'
            if current_momentum is not None:
                if current_momentum > 0:
                    signal_strength = 'BULLISH_BUILDING'
                else:
                    signal_strength = 'BEARISH_BUILDING'
        elif current_squeeze is False:
            # Squeeze fired - momentum determines direction
            if current_momentum is not None:
                if current_momentum > 0:
                    signal = 'BULLISH_BREAKOUT'
                    if momentum_color == 'dark_green':
                        signal_strength = 'STRONG'
                    else:
                        signal_strength = 'WEAKENING'
                else:
                    signal = 'BEARISH_BREAKOUT'
                    if momentum_color == 'dark_red':
                        signal_strength = 'STRONG'
                    else:
                        signal_strength = 'WEAKENING'
        
        # Get latest price data
        latest_bar = price_data[-1]
        current_price = float(latest_bar['close'])
        
        # Calculate squeeze intensity (how tight is the compression)
        bb_width_val = (bb_upper[idx] - bb_lower[idx]) if bb_upper[idx] and bb_lower[idx] else 0
        kc_width_val = (kc_upper[idx] - kc_lower[idx]) if kc_upper[idx] and kc_lower[idx] else 0
        
        squeeze_intensity = 'NONE'
        squeeze_intensity_ratio = 0
        if kc_width_val > 0 and current_squeeze:
            squeeze_intensity_ratio = bb_width_val / kc_width_val
            if squeeze_intensity_ratio < self.squeeze_intensity_extreme:
                squeeze_intensity = 'EXTREME'  # Very tight - explosive move likely
            elif squeeze_intensity_ratio < self.squeeze_intensity_high:
                squeeze_intensity = 'HIGH'  # High compression
            elif squeeze_intensity_ratio < self.squeeze_intensity_moderate:
                squeeze_intensity = 'MODERATE'  # Moderate compression
            else:
                squeeze_intensity = 'LOW'  # Mild compression
        
        # Calculate tight squeeze (using 1.0x ATR Keltner)
        kc_upper_tight = kc_ema[idx] + (self.kc_atr_mult_tight * atr[idx]) if kc_ema[idx] and atr[idx] else None
        kc_lower_tight = kc_ema[idx] - (self.kc_atr_mult_tight * atr[idx]) if kc_ema[idx] and atr[idx] else None
        tight_squeeze = False
        if kc_upper_tight and kc_lower_tight and bb_upper[idx] and bb_lower[idx]:
            tight_squeeze = bb_lower[idx] > kc_lower_tight and bb_upper[idx] < kc_upper_tight
        
        result = {
            'status': 'success',
            'symbol': symbol,
            'interval': interval,
            'timestamp': datetime.now().isoformat(),
            'data_source': 'TwelveData API (Real-time)',
            
            # Current values
            'current_price': current_price,
            'squeeze_on': current_squeeze,
            'squeeze_count': squeeze_count,
            'momentum': round(current_momentum, 4) if current_momentum else None,
            'momentum_color': momentum_color,
            
            # NEW: Squeeze intensity metrics
            'squeeze_intensity': squeeze_intensity,
            'squeeze_intensity_ratio': round(squeeze_intensity_ratio, 3) if squeeze_intensity_ratio else None,
            'tight_squeeze': tight_squeeze,  # Using 1.0x ATR Keltner
            
            # Signal interpretation
            'signal': signal,
            'signal_strength': signal_strength,
            
            # Bollinger Bands
            'bb_upper': round(bb_upper[idx], 4) if bb_upper[idx] else None,
            'bb_middle': round(bb_sma[idx], 4) if bb_sma[idx] else None,
            'bb_lower': round(bb_lower[idx], 4) if bb_lower[idx] else None,
            
            # Keltner Channels
            'kc_upper': round(kc_upper[idx], 4) if kc_upper[idx] else None,
            'kc_middle': round(kc_ema[idx], 4) if kc_ema[idx] else None,
            'kc_lower': round(kc_lower[idx], 4) if kc_lower[idx] else None,
            
            # Volatility
            'atr': round(atr[idx], 4) if atr[idx] else None,
            'bb_width': round((bb_upper[idx] - bb_lower[idx]) / bb_sma[idx] * 100, 2) if bb_upper[idx] and bb_lower[idx] and bb_sma[idx] else None,
            
            # Historical squeeze data (last 10 bars)
            'squeeze_history': squeeze_on[-10:],
            'momentum_history': [round(m, 4) if m else None for m in momentum[-10:]]
        }
        
        print(f"✅ TTM Squeeze calculated: {'🔴 SQUEEZE ON' if current_squeeze else '🟢 SQUEEZE OFF'}")
        print(f"   Momentum: {current_momentum:.4f} ({momentum_color})" if current_momentum else "   Momentum: N/A")
        print(f"   Signal: {signal} ({signal_strength})")
        
        return result


# Test function
if __name__ == "__main__":
    # Test with TwelveData API key
    API_KEY = "5e7a5daaf41d46a8966963106ebef210"
    
    squeeze = TTMSqueeze(API_KEY)
    result = squeeze.calculate_squeeze("AAPL", "1day")
    
    print("\n" + "="*60)
    print("TTM SQUEEZE RESULT")
    print("="*60)
    
    if result['status'] == 'success':
        print(f"Symbol: {result['symbol']}")
        print(f"Price: ${result['current_price']:.2f}")
        print(f"Squeeze: {'🔴 ON' if result['squeeze_on'] else '🟢 OFF'} ({result['squeeze_count']} bars)")
        print(f"Momentum: {result['momentum']} ({result['momentum_color']})")
        print(f"Signal: {result['signal']} ({result['signal_strength']})")
        print(f"\nBollinger Bands: {result['bb_lower']:.2f} | {result['bb_middle']:.2f} | {result['bb_upper']:.2f}")
        print(f"Keltner Channels: {result['kc_lower']:.2f} | {result['kc_middle']:.2f} | {result['kc_upper']:.2f}")
    else:
        print(f"Error: {result['error']}")
