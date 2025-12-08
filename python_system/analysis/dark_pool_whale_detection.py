"""
Dark Pool Detection & Whale Alert Monitoring
============================================

Detects institutional activity through:
- Dark pool transaction analysis
- Unusual volume spikes
- Large block trades
- Options flow (smart money indicators)
- Insider trading patterns

Author: Institutional Trading System
Date: 2025-11-20
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class WhaleAlert:
    """Alert for whale/institutional activity"""
    timestamp: datetime
    alert_type: str  # 'dark_pool', 'block_trade', 'unusual_volume', 'options_flow', 'insider'
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    metrics: Dict[str, float]
    signal: str  # 'bullish', 'bearish', 'neutral'
    confidence: float  # 0-100


class DarkPoolDetector:
    """
    Detects dark pool activity and institutional positioning.
    
    Dark pools are private exchanges where institutions trade large blocks
    without moving the public market. Detecting this activity gives us an
    edge on institutional intent.
    """
    
    def __init__(self):
        """Initialize dark pool detector"""
        self.logger = logging.getLogger(__name__)
    
    def detect_unusual_volume(
        self,
        data: pd.DataFrame,
        volume_column: str = 'volume',
        lookback_days: int = 20,
        threshold_sigma: float = 3.0
    ) -> Optional[WhaleAlert]:
        """
        Detect unusual volume spikes (potential whale activity).
        
        Args:
            data: OHLCV DataFrame
            volume_column: Volume column name
            lookback_days: Days for baseline calculation
            threshold_sigma: Standard deviations for alert
            
        Returns:
            WhaleAlert if unusual volume detected
        """
        if len(data) < lookback_days + 1:
            return None
        
        # Calculate baseline statistics
        baseline_volume = data[volume_column].iloc[-(lookback_days+1):-1]
        mean_volume = baseline_volume.mean()
        std_volume = baseline_volume.std()
        
        if std_volume == 0:
            return None
        
        # Current volume
        current_volume = data[volume_column].iloc[-1]
        
        # Z-score
        z_score = (current_volume - mean_volume) / std_volume
        
        if z_score > threshold_sigma:
            # Determine if bullish or bearish
            current_close = data['close'].iloc[-1]
            current_open = data['open'].iloc[-1]
            
            if current_close > current_open:
                signal = 'bullish'
                desc = "Unusual buying volume detected"
            elif current_close < current_open:
                signal = 'bearish'
                desc = "Unusual selling volume detected"
            else:
                signal = 'neutral'
                desc = "Unusual volume with no clear direction"
            
            severity = 'critical' if z_score > 5 else ('high' if z_score > 4 else 'medium')
            
            return WhaleAlert(
                timestamp=datetime.now(),
                alert_type='unusual_volume',
                severity=severity,
                description=f"{desc} ({z_score:.2f}σ above average)",
                metrics={
                    'current_volume': current_volume,
                    'average_volume': mean_volume,
                    'z_score': z_score,
                    'volume_ratio': current_volume / mean_volume
                },
                signal=signal,
                confidence=min(95, 60 + z_score * 5)
            )
        
        return None
    
    def detect_block_trades(
        self,
        data: pd.DataFrame,
        min_block_size: float = 10000  # Minimum shares for block trade
    ) -> List[WhaleAlert]:
        """
        Detect large block trades (institutional activity).
        
        Block trades are large single transactions, often indicating
        institutional positioning.
        
        Args:
            data: Intraday OHLCV data
            min_block_size: Minimum size to consider a block trade
            
        Returns:
            List of WhaleAlerts for block trades
        """
        alerts = []
        
        if 'volume' not in data.columns or len(data) < 2:
            return alerts
        
        # Calculate average trade size
        avg_volume = data['volume'].mean()
        
        # Look for volume spikes within the day
        for idx in range(1, len(data)):
            current_vol = data['volume'].iloc[idx]
            prev_vol = data['volume'].iloc[idx-1]
            
            # Spike detection
            if current_vol > min_block_size and current_vol > avg_volume * 5:
                current_price = data['close'].iloc[idx]
                prev_price = data['close'].iloc[idx-1]
                
                # Determine direction
                if current_price > prev_price:
                    signal = 'bullish'
                    desc = "Large block buy detected"
                elif current_price < prev_price:
                    signal = 'bearish'
                    desc = "Large block sell detected"
                else:
                    signal = 'neutral'
                    desc = "Large block trade (neutral)"
                
                alerts.append(WhaleAlert(
                    timestamp=data.index[idx] if hasattr(data.index[idx], 'to_pydatetime') else datetime.now(),
                    alert_type='block_trade',
                    severity='high',
                    description=desc,
                    metrics={
                        'block_size': current_vol,
                        'average_volume': avg_volume,
                        'price': current_price
                    },
                    signal=signal,
                    confidence=75
                ))
        
        return alerts
    
    def analyze_price_volume_divergence(
        self,
        data: pd.DataFrame,
        window: int = 20
    ) -> Optional[WhaleAlert]:
        """
        Detect price-volume divergence (accumulation/distribution).
        
        Divergence between price and volume can indicate smart money
        accumulation (bullish) or distribution (bearish).
        
        Args:
            data: OHLCV DataFrame
            window: Analysis window
            
        Returns:
            WhaleAlert if significant divergence detected
        """
        if len(data) < window:
            return None
        
        recent_data = data.iloc[-window:]
        
        # Calculate price trend
        price_change = (recent_data['close'].iloc[-1] / recent_data['close'].iloc[0] - 1) * 100
        
        # Calculate volume trend
        first_half_vol = recent_data['volume'].iloc[:window//2].mean()
        second_half_vol = recent_data['volume'].iloc[window//2:].mean()
        volume_change = (second_half_vol / first_half_vol - 1) * 100 if first_half_vol > 0 else 0
        
        # Detect divergence
        # Bullish: Price down but volume up (accumulation)
        # Bearish: Price up but volume down (distribution)
        
        if price_change < -2 and volume_change > 20:
            return WhaleAlert(
                timestamp=datetime.now(),
                alert_type='dark_pool',
                severity='high',
                description="Accumulation pattern: Price declining with increasing volume (smart money buying)",
                metrics={
                    'price_change_pct': price_change,
                    'volume_change_pct': volume_change
                },
                signal='bullish',
                confidence=80
            )
        elif price_change > 2 and volume_change < -20:
            return WhaleAlert(
                timestamp=datetime.now(),
                alert_type='dark_pool',
                severity='high',
                description="Distribution pattern: Price rising with decreasing volume (smart money selling)",
                metrics={
                    'price_change_pct': price_change,
                    'volume_change_pct': volume_change
                },
                signal='bearish',
                confidence=80
            )
        
        return None


class OptionsFlowDetector:
    """
    Analyzes options flow for smart money indicators.
    
    Large options trades often precede major price moves.
    Institutions use options for leverage and hedging.
    """
    
    def __init__(self):
        """Initialize options flow detector"""
        self.logger = logging.getLogger(__name__)
    
    def analyze_put_call_ratio(
        self,
        put_volume: float,
        call_volume: float,
        put_oi: float,
        call_oi: float
    ) -> Dict[str, any]:
        """
        Analyze put/call ratios for sentiment.
        
        Args:
            put_volume: Put option volume
            call_volume: Call option volume
            put_oi: Put open interest
            call_oi: Call open interest
            
        Returns:
            Dict with analysis results
        """
        # Volume-based P/C ratio
        pcr_volume = put_volume / call_volume if call_volume > 0 else 0
        
        # OI-based P/C ratio
        pcr_oi = put_oi / call_oi if call_oi > 0 else 0
        
        # Interpretation
        # PCR < 0.7: Bullish (more calls)
        # PCR > 1.0: Bearish (more puts)
        # PCR 0.7-1.0: Neutral
        
        if pcr_volume < 0.7:
            sentiment = 'bullish'
            confidence = min(90, 50 + (0.7 - pcr_volume) * 100)
        elif pcr_volume > 1.0:
            sentiment = 'bearish'
            confidence = min(90, 50 + (pcr_volume - 1.0) * 50)
        else:
            sentiment = 'neutral'
            confidence = 50
        
        return {
            'pcr_volume': pcr_volume,
            'pcr_oi': pcr_oi,
            'sentiment': sentiment,
            'confidence': confidence,
            'interpretation': self._interpret_pcr(pcr_volume)
        }
    
    def _interpret_pcr(self, pcr: float) -> str:
        """Interpret P/C ratio value"""
        if pcr < 0.5:
            return "Extremely bullish - heavy call buying"
        elif pcr < 0.7:
            return "Bullish - more calls than puts"
        elif pcr < 0.9:
            return "Slightly bullish"
        elif pcr < 1.1:
            return "Neutral - balanced put/call activity"
        elif pcr < 1.3:
            return "Slightly bearish"
        elif pcr < 1.5:
            return "Bearish - more puts than calls"
        else:
            return "Extremely bearish - heavy put buying"
    
    def detect_unusual_options_activity(
        self,
        options_data: pd.DataFrame,
        volume_threshold: float = 1000,
        oi_volume_ratio: float = 0.5
    ) -> List[WhaleAlert]:
        """
        Detect unusual options activity (smart money).
        
        Args:
            options_data: Options chain data
            volume_threshold: Minimum volume for alert
            oi_volume_ratio: Volume/OI ratio threshold
            
        Returns:
            List of WhaleAlerts
        """
        alerts = []
        
        if options_data.empty:
            return alerts
        
        for _, option in options_data.iterrows():
            volume = option.get('volume', 0)
            oi = option.get('openInterest', 1)
            
            # High volume relative to OI indicates new positioning
            if volume > volume_threshold and volume / oi > oi_volume_ratio:
                option_type = 'call' if option.get('type') == 'call' else 'put'
                strike = option.get('strike', 0)
                
                signal = 'bullish' if option_type == 'call' else 'bearish'
                
                alerts.append(WhaleAlert(
                    timestamp=datetime.now(),
                    alert_type='options_flow',
                    severity='high',
                    description=f"Unusual {option_type} activity at ${strike} strike",
                    metrics={
                        'volume': volume,
                        'open_interest': oi,
                        'volume_oi_ratio': volume / oi,
                        'strike': strike
                    },
                    signal=signal,
                    confidence=70
                ))
        
        return alerts


class InsiderActivityDetector:
    """
    Monitors insider trading activity.
    
    Insider buys are often bullish signals.
    Insider sells can be neutral (diversification) or bearish.
    """
    
    def __init__(self):
        """Initialize insider activity detector"""
        self.logger = logging.getLogger(__name__)
    
    def analyze_insider_transactions(
        self,
        insider_data: List[Dict]
    ) -> List[WhaleAlert]:
        """
        Analyze insider trading transactions.
        
        Args:
            insider_data: List of insider transactions from Finnhub
            
        Returns:
            List of WhaleAlerts
        """
        alerts = []
        
        if not insider_data:
            return alerts
        
        # Aggregate recent transactions
        recent_buys = 0
        recent_sells = 0
        total_buy_value = 0
        total_sell_value = 0
        
        for transaction in insider_data:
            shares = transaction.get('share', 0)
            price = transaction.get('price', 0)
            trans_type = transaction.get('transactionCode', '')
            
            value = shares * price
            
            # P = Purchase, S = Sale
            if trans_type in ['P', 'M']:  # Purchase or option exercise
                recent_buys += 1
                total_buy_value += value
            elif trans_type == 'S':
                recent_sells += 1
                total_sell_value += value
        
        # Analyze pattern
        if recent_buys > recent_sells and total_buy_value > 100000:
            alerts.append(WhaleAlert(
                timestamp=datetime.now(),
                alert_type='insider',
                severity='high',
                description=f"Insider buying detected: {recent_buys} buys vs {recent_sells} sells",
                metrics={
                    'buy_count': recent_buys,
                    'sell_count': recent_sells,
                    'total_buy_value': total_buy_value,
                    'total_sell_value': total_sell_value
                },
                signal='bullish',
                confidence=75
            ))
        elif recent_sells > recent_buys * 3 and total_sell_value > 1000000:
            alerts.append(WhaleAlert(
                timestamp=datetime.now(),
                alert_type='insider',
                severity='medium',
                description=f"Heavy insider selling: {recent_sells} sells vs {recent_buys} buys",
                metrics={
                    'buy_count': recent_buys,
                    'sell_count': recent_sells,
                    'total_buy_value': total_buy_value,
                    'total_sell_value': total_sell_value
                },
                signal='bearish',
                confidence=60  # Lower confidence - could be diversification
            ))
        
        return alerts


# Global instances
dark_pool_detector = DarkPoolDetector()
options_flow_detector = OptionsFlowDetector()
insider_detector = InsiderActivityDetector()
