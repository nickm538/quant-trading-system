"""
TTM Squeeze Indicator - Volatility Compression Detection
Identifies when Bollinger Bands squeeze inside Keltner Channels
Signals potential breakout opportunities
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from .technical_indicators import TechnicalIndicators


class TTMSqueeze:
    """
    TTM Squeeze Indicator
    - Detects when volatility is compressed (Bollinger Bands inside Keltner Channels)
    - Calculates momentum direction
    - Provides breakout signals
    """
    
    @staticmethod
    def calculate_squeeze(high: pd.Series, low: pd.Series, close: pd.Series,
                         bb_period: int = 20, bb_std: float = 2.0,
                         kc_period: int = 20, kc_mult: float = 1.5) -> Dict:
        """
        Calculate TTM Squeeze indicator
        
        Returns:
            Dict with:
                - squeeze_on: bool (True if squeeze is active)
                - squeeze_bars: int (number of consecutive squeeze bars)
                - momentum: float (current momentum value)
                - momentum_direction: str ('bullish' or 'bearish')
                - signal: str ('long', 'short', 'active', 'none')
                - expected_move_pct: float (estimated move percentage)
        """
        # Calculate Bollinger Bands
        bb_upper, bb_mid, bb_lower = TechnicalIndicators.calculate_bollinger_bands(
            close, period=bb_period, std_dev=bb_std
        )
        
        # Calculate Keltner Channels
        kc_upper, kc_mid, kc_lower = TechnicalIndicators.calculate_keltner_channels(
            high, low, close, period=kc_period, multiplier=kc_mult
        )
        
        # Squeeze is ON when Bollinger Bands are inside Keltner Channels
        squeeze_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)
        
        # Calculate momentum (using linear regression of close prices)
        momentum = TTMSqueeze._calculate_momentum(high, low, close)
        
        # Get current values (last bar)
        current_squeeze = squeeze_on.iloc[-1] if len(squeeze_on) > 0 else False
        current_momentum = momentum.iloc[-1] if len(momentum) > 0 else 0.0
        
        # Count consecutive squeeze bars
        squeeze_bars = TTMSqueeze._count_squeeze_bars(squeeze_on)
        
        # Determine momentum direction
        momentum_direction = 'bullish' if current_momentum > 0 else 'bearish'
        
        # Generate signal
        signal = TTMSqueeze._generate_signal(squeeze_on, momentum)
        
        # Estimate expected move (based on ATR and squeeze duration)
        atr = TechnicalIndicators.calculate_atr(high, low, close, period=14)
        current_atr = atr.iloc[-1] if len(atr) > 0 else 0.0
        current_price = close.iloc[-1] if len(close) > 0 else 1.0
        
        # Expected move increases with squeeze duration (up to 10 bars)
        squeeze_multiplier = min(squeeze_bars / 10.0, 1.0) + 1.0
        expected_move_pct = (current_atr * squeeze_multiplier / current_price) * 100
        
        return {
            'squeeze_on': bool(current_squeeze),
            'squeeze_bars': int(squeeze_bars),
            'momentum': float(current_momentum),
            'momentum_direction': momentum_direction,
            'signal': signal,
            'expected_move_pct': float(expected_move_pct)
        }
    
    @staticmethod
    def _calculate_momentum(high: pd.Series, low: pd.Series, close: pd.Series,
                           period: int = 20) -> pd.Series:
        """
        Calculate momentum using linear regression
        Similar to StockCharts.com implementation
        """
        # Calculate highest high and lowest low
        highest = high.rolling(window=period).max()
        lowest = low.rolling(window=period).min()
        
        # Calculate midpoint
        mid = (highest + lowest) / 2
        
        # Linear regression of (close - mid)
        momentum = close - mid
        
        # Smooth with SMA
        momentum = momentum.rolling(window=3).mean()
        
        return momentum
    
    @staticmethod
    def _count_squeeze_bars(squeeze_on: pd.Series) -> int:
        """Count consecutive squeeze bars from the end"""
        if len(squeeze_on) == 0:
            return 0
        
        count = 0
        for i in range(len(squeeze_on) - 1, -1, -1):
            if squeeze_on.iloc[i]:
                count += 1
            else:
                break
        
        return count
    
    @staticmethod
    def _generate_signal(squeeze_on: pd.Series, momentum: pd.Series) -> str:
        """
        Generate trading signal based on squeeze and momentum
        
        Returns:
            'long' - Squeeze released, bullish momentum
            'short' - Squeeze released, bearish momentum
            'active' - Squeeze is active (wait)
            'none' - No squeeze, no signal
        """
        if len(squeeze_on) < 2 or len(momentum) < 2:
            return 'none'
        
        current_squeeze = squeeze_on.iloc[-1]
        previous_squeeze = squeeze_on.iloc[-2]
        current_momentum = momentum.iloc[-1]
        
        # Squeeze just released (was on, now off)
        if previous_squeeze and not current_squeeze:
            if current_momentum > 0:
                return 'long'
            else:
                return 'short'
        
        # Squeeze is active
        if current_squeeze:
            return 'active'
        
        # No squeeze
        return 'none'
