"""
Intraday Minute-Level Analysis
===============================

Real-time intraday analysis down to the minute for dynamic trading decisions.
Implements momentum detection, volume analysis, and intraday pattern recognition.

Author: Institutional Trading System
Date: 2025-11-20
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class IntradaySignal:
    """Intraday trading signal"""
    timestamp: datetime
    signal_type: str  # 'buy', 'sell', 'hold'
    strength: float  # 0-100
    price: float
    volume: int
    indicators: Dict[str, float]
    reason: str
    confidence: float


class IntradayAnalyzer:
    """
    Minute-level intraday analysis for dynamic trading.
    Detects momentum shifts, volume spikes, and pattern breaks in real-time.
    """
    
    def __init__(self):
        """Initialize intraday analyzer"""
        self.logger = logging.getLogger(__name__)
    
    def analyze_intraday_momentum(
        self,
        intraday_data: pd.DataFrame,
        lookback_minutes: int = 60
    ) -> Dict[str, float]:
        """
        Analyze intraday momentum using minute-level data.
        
        Args:
            intraday_data: DataFrame with 1-min OHLCV data
            lookback_minutes: Minutes to look back for momentum
            
        Returns:
            Dict of momentum indicators
        """
        if intraday_data is None or len(intraday_data) < lookback_minutes:
            return {}
        
        # Use last N minutes
        recent = intraday_data.tail(lookback_minutes).copy()
        
        # Calculate returns
        recent['returns'] = recent['close'].pct_change()
        
        # Momentum indicators
        momentum = {}
        
        # 1. Price momentum (% change)
        momentum['price_change_pct'] = (
            (recent['close'].iloc[-1] - recent['close'].iloc[0]) / 
            recent['close'].iloc[0] * 100
        )
        
        # 2. Volume-weighted momentum
        recent['volume_weighted_return'] = recent['returns'] * recent['volume']
        momentum['vwap_momentum'] = recent['volume_weighted_return'].sum() / recent['volume'].sum()
        
        # 3. Acceleration (change in momentum)
        mid_point = len(recent) // 2
        first_half_momentum = (
            (recent['close'].iloc[mid_point] - recent['close'].iloc[0]) / 
            recent['close'].iloc[0]
        )
        second_half_momentum = (
            (recent['close'].iloc[-1] - recent['close'].iloc[mid_point]) / 
            recent['close'].iloc[mid_point]
        )
        momentum['acceleration'] = second_half_momentum - first_half_momentum
        
        # 4. Volume trend
        first_half_volume = recent['volume'].iloc[:mid_point].mean()
        second_half_volume = recent['volume'].iloc[mid_point:].mean()
        momentum['volume_trend'] = (
            (second_half_volume - first_half_volume) / first_half_volume * 100
            if first_half_volume > 0 else 0
        )
        
        # 5. Volatility (intraday)
        momentum['intraday_volatility'] = recent['returns'].std() * np.sqrt(390)  # Annualized
        
        # 6. Trend strength (R-squared of linear regression)
        x = np.arange(len(recent))
        y = recent['close'].values
        if len(x) > 1:
            correlation = np.corrcoef(x, y)[0, 1]
            momentum['trend_strength'] = correlation ** 2  # R-squared
        else:
            momentum['trend_strength'] = 0
        
        # 7. High-Low range
        momentum['hl_range_pct'] = (
            (recent['high'].max() - recent['low'].min()) / recent['close'].iloc[0] * 100
        )
        
        return momentum
    
    def detect_volume_spikes(
        self,
        intraday_data: pd.DataFrame,
        threshold_std: float = 2.0
    ) -> List[Dict]:
        """
        Detect unusual volume spikes (potential whale activity).
        
        Args:
            intraday_data: DataFrame with minute-level data
            threshold_std: Number of std devs above mean to flag
            
        Returns:
            List of volume spike events
        """
        if intraday_data is None or len(intraday_data) < 30:
            return []
        
        # Calculate rolling volume statistics
        volume_mean = intraday_data['volume'].rolling(window=30, min_periods=10).mean()
        volume_std = intraday_data['volume'].rolling(window=30, min_periods=10).std()
        
        # Detect spikes
        spikes = []
        for idx in range(len(intraday_data)):
            if pd.isna(volume_mean.iloc[idx]) or pd.isna(volume_std.iloc[idx]):
                continue
            
            current_volume = intraday_data['volume'].iloc[idx]
            mean_vol = volume_mean.iloc[idx]
            std_vol = volume_std.iloc[idx]
            
            if std_vol > 0:
                z_score = (current_volume - mean_vol) / std_vol
                
                if z_score > threshold_std:
                    spikes.append({
                        'timestamp': intraday_data.index[idx],
                        'volume': current_volume,
                        'mean_volume': mean_vol,
                        'z_score': z_score,
                        'price': intraday_data['close'].iloc[idx],
                        'price_change': intraday_data['close'].pct_change().iloc[idx] * 100
                    })
        
        return spikes
    
    def calculate_vwap(
        self,
        intraday_data: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate Volume-Weighted Average Price (VWAP).
        Critical benchmark for institutional trading.
        
        Args:
            intraday_data: DataFrame with OHLCV data
            
        Returns:
            Series with VWAP values
        """
        if intraday_data is None or len(intraday_data) == 0:
            return pd.Series()
        
        # Typical price
        typical_price = (intraday_data['high'] + intraday_data['low'] + intraday_data['close']) / 3
        
        # Cumulative volume * typical price
        cumulative_tp_volume = (typical_price * intraday_data['volume']).cumsum()
        cumulative_volume = intraday_data['volume'].cumsum()
        
        # VWAP
        vwap = cumulative_tp_volume / cumulative_volume
        
        return vwap
    
    def detect_breakouts(
        self,
        intraday_data: pd.DataFrame,
        lookback_minutes: int = 60
    ) -> Optional[Dict]:
        """
        Detect price breakouts from intraday ranges.
        
        Args:
            intraday_data: DataFrame with minute data
            lookback_minutes: Period to establish range
            
        Returns:
            Breakout info or None
        """
        if intraday_data is None or len(intraday_data) < lookback_minutes + 1:
            return None
        
        # Establish range from lookback period
        range_data = intraday_data.iloc[-(lookback_minutes+1):-1]
        current = intraday_data.iloc[-1]
        
        high_range = range_data['high'].max()
        low_range = range_data['low'].min()
        range_size = high_range - low_range
        
        # Check for breakout
        breakout = None
        
        if current['close'] > high_range:
            breakout = {
                'type': 'upside_breakout',
                'price': current['close'],
                'resistance': high_range,
                'breakout_pct': ((current['close'] - high_range) / high_range) * 100,
                'volume': current['volume'],
                'timestamp': current.name
            }
        elif current['close'] < low_range:
            breakout = {
                'type': 'downside_breakout',
                'price': current['close'],
                'support': low_range,
                'breakout_pct': ((low_range - current['close']) / low_range) * 100,
                'volume': current['volume'],
                'timestamp': current.name
            }
        
        return breakout
    
    def generate_intraday_signal(
        self,
        intraday_data: pd.DataFrame,
        current_price: float
    ) -> IntradaySignal:
        """
        Generate comprehensive intraday trading signal.
        
        Args:
            intraday_data: Minute-level OHLCV data
            current_price: Current market price
            
        Returns:
            IntradaySignal with recommendation
        """
        if intraday_data is None or len(intraday_data) < 60:
            return IntradaySignal(
                timestamp=datetime.now(),
                signal_type='hold',
                strength=0,
                price=current_price,
                volume=0,
                indicators={},
                reason="Insufficient intraday data",
                confidence=0
            )
        
        # Calculate all indicators
        momentum = self.analyze_intraday_momentum(intraday_data)
        volume_spikes = self.detect_volume_spikes(intraday_data)
        vwap = self.calculate_vwap(intraday_data)
        breakout = self.detect_breakouts(intraday_data)
        
        # Current VWAP
        current_vwap = vwap.iloc[-1] if len(vwap) > 0 else current_price
        vwap_deviation = ((current_price - current_vwap) / current_vwap) * 100
        
        # Signal logic
        bullish_score = 0
        bearish_score = 0
        reasons = []
        
        # 1. Momentum
        if momentum.get('price_change_pct', 0) > 0.5:
            bullish_score += 20
            reasons.append(f"Strong upward momentum (+{momentum['price_change_pct']:.2f}%)")
        elif momentum.get('price_change_pct', 0) < -0.5:
            bearish_score += 20
            reasons.append(f"Strong downward momentum ({momentum['price_change_pct']:.2f}%)")
        
        # 2. Acceleration
        if momentum.get('acceleration', 0) > 0.001:
            bullish_score += 15
            reasons.append("Positive acceleration")
        elif momentum.get('acceleration', 0) < -0.001:
            bearish_score += 15
            reasons.append("Negative acceleration")
        
        # 3. Volume trend
        if momentum.get('volume_trend', 0) > 20:
            bullish_score += 15
            reasons.append(f"Volume increasing (+{momentum['volume_trend']:.1f}%)")
        
        # 4. VWAP position
        if vwap_deviation > 0.3:
            bearish_score += 10
            reasons.append(f"Price above VWAP (+{vwap_deviation:.2f}%) - potential pullback")
        elif vwap_deviation < -0.3:
            bullish_score += 10
            reasons.append(f"Price below VWAP ({vwap_deviation:.2f}%) - potential bounce")
        
        # 5. Volume spikes
        if len(volume_spikes) > 0:
            recent_spike = volume_spikes[-1]
            if recent_spike['price_change'] > 0:
                bullish_score += 15
                reasons.append(f"Whale buying detected (vol spike: {recent_spike['z_score']:.1f}σ)")
            else:
                bearish_score += 15
                reasons.append(f"Whale selling detected (vol spike: {recent_spike['z_score']:.1f}σ)")
        
        # 6. Breakouts
        if breakout:
            if breakout['type'] == 'upside_breakout':
                bullish_score += 25
                reasons.append(f"Upside breakout (+{breakout['breakout_pct']:.2f}%)")
            else:
                bearish_score += 25
                reasons.append(f"Downside breakdown (-{breakout['breakout_pct']:.2f}%)")
        
        # Determine signal
        net_score = bullish_score - bearish_score
        
        if net_score > 30:
            signal_type = 'buy'
            strength = min(100, bullish_score)
            confidence = min(100, strength * 0.8)  # Conservative confidence
        elif net_score < -30:
            signal_type = 'sell'
            strength = min(100, bearish_score)
            confidence = min(100, strength * 0.8)
        else:
            signal_type = 'hold'
            strength = abs(net_score)
            confidence = 50
        
        return IntradaySignal(
            timestamp=datetime.now(),
            signal_type=signal_type,
            strength=strength,
            price=current_price,
            volume=int(intraday_data['volume'].iloc[-1]),
            indicators={
                'momentum': momentum,
                'vwap': current_vwap,
                'vwap_deviation_pct': vwap_deviation,
                'volume_spikes_count': len(volume_spikes),
                'breakout': breakout
            },
            reason="; ".join(reasons) if reasons else "Neutral intraday conditions",
            confidence=confidence
        )
    
    def get_intraday_summary(
        self,
        intraday_data: pd.DataFrame
    ) -> Dict:
        """
        Get comprehensive intraday summary.
        
        Args:
            intraday_data: Minute-level data
            
        Returns:
            Summary dict with all intraday metrics
        """
        if intraday_data is None or len(intraday_data) == 0:
            return {}
        
        current_price = intraday_data['close'].iloc[-1]
        
        summary = {
            'current_price': current_price,
            'day_open': intraday_data['open'].iloc[0],
            'day_high': intraday_data['high'].max(),
            'day_low': intraday_data['low'].min(),
            'day_volume': int(intraday_data['volume'].sum()),
            'minutes_traded': len(intraday_data),
            'avg_minute_volume': int(intraday_data['volume'].mean()),
            'momentum': self.analyze_intraday_momentum(intraday_data),
            'volume_spikes': self.detect_volume_spikes(intraday_data),
            'breakout': self.detect_breakouts(intraday_data),
            'signal': self.generate_intraday_signal(intraday_data, current_price)
        }
        
        # Calculate VWAP
        vwap = self.calculate_vwap(intraday_data)
        if len(vwap) > 0:
            summary['vwap'] = vwap.iloc[-1]
            summary['vwap_deviation_pct'] = (
                (current_price - summary['vwap']) / summary['vwap'] * 100
            )
        
        return summary


# Global instance
intraday_analyzer = IntradayAnalyzer()
