"""
TIM BOHEN 5:1 RISK/REWARD SCANNER
=================================

Implements Tim Bohen's (StocksToTrade) 5:1 Risk/Reward methodology.

Core Principles:
1. MINIMUM 5:1 Reward/Risk ratio on every trade
2. Clear support level for stop loss placement
3. Clear resistance/target for profit taking
4. Volume confirmation (above average)
5. Catalyst-driven moves preferred

Tim Bohen's Philosophy:
- "If you can't find 5:1, don't take the trade"
- "Risk small, win big"
- "The stop loss is non-negotiable"
- "Let winners run, cut losers fast"

This module calculates precise entry, stop, and target levels
to achieve minimum 5:1 R/R on every recommended trade.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class Bohen5to1Scanner:
    """
    Tim Bohen's 5:1 Risk/Reward Scanner
    
    Identifies setups with minimum 5:1 reward/risk ratio using:
    - Support/Resistance analysis
    - Volume confirmation
    - ATR-based stop placement
    - Multiple target levels
    """
    
    def __init__(self, min_rr_ratio: float = 5.0):
        """
        Initialize the scanner.
        
        Args:
            min_rr_ratio: Minimum reward/risk ratio (default 5.0)
        """
        self.min_rr_ratio = min_rr_ratio
        
    def analyze(self, symbol: str, price_data: pd.DataFrame = None) -> Dict:
        """
        Analyze a stock for 5:1 setups.
        
        Args:
            symbol: Stock ticker symbol
            price_data: Optional DataFrame with OHLCV data. If None, fetches from yfinance.
            
        Returns:
            Dict with setup analysis, entry/stop/target levels, and R/R ratio
        """
        try:
            # Fetch data if not provided
            if price_data is None:
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                price_data = ticker.history(period='3mo')
                
                if price_data.empty:
                    return {'error': f'No data available for {symbol}'}
            
            # Standardize column names
            price_data.columns = [c.lower() for c in price_data.columns]
            
            # Get current price
            current_price = price_data['close'].iloc[-1]
            
            # Calculate key levels
            support_levels = self._find_support_levels(price_data)
            resistance_levels = self._find_resistance_levels(price_data)
            
            # Calculate ATR for stop placement
            atr = self._calculate_atr(price_data, period=14)
            current_atr = atr.iloc[-1]
            
            # Calculate volume metrics
            avg_volume = price_data['volume'].rolling(20).mean().iloc[-1]
            current_volume = price_data['volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Find best 5:1 setup
            long_setup = self._find_long_setup(
                current_price, support_levels, resistance_levels, current_atr
            )
            short_setup = self._find_short_setup(
                current_price, support_levels, resistance_levels, current_atr
            )
            
            # Determine best setup
            best_setup = None
            setup_type = None
            
            if long_setup and long_setup['rr_ratio'] >= self.min_rr_ratio:
                if short_setup and short_setup['rr_ratio'] > long_setup['rr_ratio']:
                    best_setup = short_setup
                    setup_type = 'SHORT'
                else:
                    best_setup = long_setup
                    setup_type = 'LONG'
            elif short_setup and short_setup['rr_ratio'] >= self.min_rr_ratio:
                best_setup = short_setup
                setup_type = 'SHORT'
            
            # Calculate trend strength
            trend = self._calculate_trend(price_data)
            
            # Build result
            result = {
                'symbol': symbol,
                'current_price': round(current_price, 2),
                'atr': round(current_atr, 2),
                'atr_pct': round((current_atr / current_price) * 100, 2),
                'volume_ratio': round(volume_ratio, 2),
                'volume_signal': 'HIGH' if volume_ratio > 1.5 else 'NORMAL' if volume_ratio > 0.8 else 'LOW',
                'trend': trend,
                'support_levels': [round(s, 2) for s in support_levels[:3]],
                'resistance_levels': [round(r, 2) for r in resistance_levels[:3]],
                'long_setup': long_setup,
                'short_setup': short_setup,
                'best_setup': best_setup,
                'setup_type': setup_type,
                'meets_5to1': best_setup is not None and best_setup['rr_ratio'] >= self.min_rr_ratio,
                'bohen_verdict': self._get_bohen_verdict(best_setup, volume_ratio, trend)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return {'error': str(e), 'symbol': symbol}
    
    def _find_support_levels(self, data: pd.DataFrame, lookback: int = 60) -> List[float]:
        """
        Find support levels using pivot point analysis.
        """
        recent_data = data.tail(lookback)
        lows = recent_data['low'].values
        
        # Find local minima (pivot lows)
        support_levels = []
        for i in range(2, len(lows) - 2):
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
               lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                support_levels.append(lows[i])
        
        # Also add recent swing lows
        recent_low = recent_data['low'].min()
        if recent_low not in support_levels:
            support_levels.append(recent_low)
        
        # Sort by proximity to current price (closest first)
        current_price = data['close'].iloc[-1]
        support_levels = [s for s in support_levels if s < current_price]
        support_levels.sort(key=lambda x: current_price - x)
        
        return support_levels
    
    def _find_resistance_levels(self, data: pd.DataFrame, lookback: int = 60) -> List[float]:
        """
        Find resistance levels using pivot point analysis.
        """
        recent_data = data.tail(lookback)
        highs = recent_data['high'].values
        
        # Find local maxima (pivot highs)
        resistance_levels = []
        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
               highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                resistance_levels.append(highs[i])
        
        # Also add recent swing highs
        recent_high = recent_data['high'].max()
        if recent_high not in resistance_levels:
            resistance_levels.append(recent_high)
        
        # Sort by proximity to current price (closest first)
        current_price = data['close'].iloc[-1]
        resistance_levels = [r for r in resistance_levels if r > current_price]
        resistance_levels.sort(key=lambda x: x - current_price)
        
        return resistance_levels
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range.
        """
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def _find_long_setup(
        self, 
        current_price: float, 
        support_levels: List[float],
        resistance_levels: List[float],
        atr: float
    ) -> Optional[Dict]:
        """
        Find best long setup with 5:1 R/R.
        
        Entry: Current price or slight pullback
        Stop: Below nearest support (1 ATR buffer)
        Target: Resistance levels that give 5:1+
        """
        if not support_levels or not resistance_levels:
            return None
        
        # Entry at current price
        entry = current_price
        
        # Stop below nearest support with ATR buffer
        nearest_support = support_levels[0]
        stop_loss = nearest_support - (atr * 0.5)  # Half ATR below support
        
        # Risk per share
        risk = entry - stop_loss
        if risk <= 0:
            return None
        
        # Find target that gives 5:1
        required_reward = risk * self.min_rr_ratio
        target_price = entry + required_reward
        
        # Check if there's a resistance level near our target
        actual_target = None
        for resistance in resistance_levels:
            if resistance >= target_price:
                actual_target = resistance
                break
        
        if actual_target is None and resistance_levels:
            actual_target = resistance_levels[-1]  # Use highest resistance
        
        if actual_target is None:
            return None
        
        # Calculate actual R/R
        reward = actual_target - entry
        rr_ratio = reward / risk if risk > 0 else 0
        
        # Calculate percentages
        risk_pct = (risk / entry) * 100
        reward_pct = (reward / entry) * 100
        
        return {
            'direction': 'LONG',
            'entry': round(entry, 2),
            'stop_loss': round(stop_loss, 2),
            'target': round(actual_target, 2),
            'risk': round(risk, 2),
            'reward': round(reward, 2),
            'risk_pct': round(risk_pct, 2),
            'reward_pct': round(reward_pct, 2),
            'rr_ratio': round(rr_ratio, 2),
            'meets_5to1': rr_ratio >= self.min_rr_ratio,
            'support_used': round(nearest_support, 2),
            'resistance_used': round(actual_target, 2)
        }
    
    def _find_short_setup(
        self, 
        current_price: float, 
        support_levels: List[float],
        resistance_levels: List[float],
        atr: float
    ) -> Optional[Dict]:
        """
        Find best short setup with 5:1 R/R.
        
        Entry: Current price
        Stop: Above nearest resistance (1 ATR buffer)
        Target: Support levels that give 5:1+
        """
        if not support_levels or not resistance_levels:
            return None
        
        # Entry at current price
        entry = current_price
        
        # Stop above nearest resistance with ATR buffer
        nearest_resistance = resistance_levels[0]
        stop_loss = nearest_resistance + (atr * 0.5)  # Half ATR above resistance
        
        # Risk per share
        risk = stop_loss - entry
        if risk <= 0:
            return None
        
        # Find target that gives 5:1
        required_reward = risk * self.min_rr_ratio
        target_price = entry - required_reward
        
        # Check if there's a support level near our target
        actual_target = None
        for support in support_levels:
            if support <= target_price:
                actual_target = support
                break
        
        if actual_target is None and support_levels:
            actual_target = support_levels[-1]  # Use lowest support
        
        if actual_target is None:
            return None
        
        # Calculate actual R/R
        reward = entry - actual_target
        rr_ratio = reward / risk if risk > 0 else 0
        
        # Calculate percentages
        risk_pct = (risk / entry) * 100
        reward_pct = (reward / entry) * 100
        
        return {
            'direction': 'SHORT',
            'entry': round(entry, 2),
            'stop_loss': round(stop_loss, 2),
            'target': round(actual_target, 2),
            'risk': round(risk, 2),
            'reward': round(reward, 2),
            'risk_pct': round(risk_pct, 2),
            'reward_pct': round(reward_pct, 2),
            'rr_ratio': round(rr_ratio, 2),
            'meets_5to1': rr_ratio >= self.min_rr_ratio,
            'resistance_used': round(nearest_resistance, 2),
            'support_used': round(actual_target, 2)
        }
    
    def _calculate_trend(self, data: pd.DataFrame) -> Dict:
        """
        Calculate trend direction and strength.
        """
        close = data['close']
        
        # Calculate EMAs
        ema_10 = close.ewm(span=10).mean()
        ema_20 = close.ewm(span=20).mean()
        ema_50 = close.ewm(span=50).mean()
        
        current_price = close.iloc[-1]
        
        # Determine trend
        above_10 = current_price > ema_10.iloc[-1]
        above_20 = current_price > ema_20.iloc[-1]
        above_50 = current_price > ema_50.iloc[-1]
        
        ema_10_above_20 = ema_10.iloc[-1] > ema_20.iloc[-1]
        ema_20_above_50 = ema_20.iloc[-1] > ema_50.iloc[-1]
        
        # Score trend (0-100)
        trend_score = 50
        if above_10: trend_score += 10
        if above_20: trend_score += 10
        if above_50: trend_score += 10
        if ema_10_above_20: trend_score += 10
        if ema_20_above_50: trend_score += 10
        
        # Determine direction
        if trend_score >= 80:
            direction = 'STRONG_UPTREND'
        elif trend_score >= 60:
            direction = 'UPTREND'
        elif trend_score >= 40:
            direction = 'NEUTRAL'
        elif trend_score >= 20:
            direction = 'DOWNTREND'
        else:
            direction = 'STRONG_DOWNTREND'
        
        return {
            'direction': direction,
            'score': trend_score,
            'price_vs_ema10': 'above' if above_10 else 'below',
            'price_vs_ema20': 'above' if above_20 else 'below',
            'price_vs_ema50': 'above' if above_50 else 'below',
            'ema_alignment': 'bullish' if ema_10_above_20 and ema_20_above_50 else 'bearish' if not ema_10_above_20 and not ema_20_above_50 else 'mixed'
        }
    
    def _get_bohen_verdict(
        self, 
        setup: Optional[Dict], 
        volume_ratio: float, 
        trend: Dict
    ) -> str:
        """
        Get Tim Bohen's verdict on this setup.
        """
        if setup is None:
            return "❌ NO TRADE - Cannot find 5:1 setup. Tim says: 'If you can't find 5:1, don't take the trade.'"
        
        rr = setup['rr_ratio']
        direction = setup['direction']
        
        # Check volume
        volume_ok = volume_ratio >= 1.0
        
        # Check trend alignment
        trend_aligned = (
            (direction == 'LONG' and trend['score'] >= 50) or
            (direction == 'SHORT' and trend['score'] <= 50)
        )
        
        if rr >= 5.0 and volume_ok and trend_aligned:
            return f"✅ TAKE THE TRADE - {rr:.1f}:1 R/R with volume confirmation and trend alignment. Tim says: 'This is exactly what we look for - risk small, win big!'"
        elif rr >= 5.0 and volume_ok:
            return f"⚠️ PROCEED WITH CAUTION - {rr:.1f}:1 R/R with volume, but trend not aligned. Tim says: 'The setup is there, but be ready to cut quickly if it goes against you.'"
        elif rr >= 5.0:
            return f"⚠️ WAIT FOR VOLUME - {rr:.1f}:1 R/R but volume is low ({volume_ratio:.1f}x avg). Tim says: 'Volume confirms conviction. Wait for the move to prove itself.'"
        else:
            return f"❌ PASS - Only {rr:.1f}:1 R/R. Tim says: 'We need minimum 5:1. This doesn't meet our criteria.'"
    
    def scan_watchlist(self, symbols: List[str]) -> List[Dict]:
        """
        Scan a watchlist for 5:1 setups.
        
        Args:
            symbols: List of ticker symbols
            
        Returns:
            List of setups sorted by R/R ratio (best first)
        """
        results = []
        
        for symbol in symbols:
            try:
                analysis = self.analyze(symbol)
                if analysis.get('meets_5to1'):
                    results.append(analysis)
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                continue
        
        # Sort by R/R ratio (highest first)
        results.sort(key=lambda x: x.get('best_setup', {}).get('rr_ratio', 0), reverse=True)
        
        return results


# Standalone test
if __name__ == '__main__':
    scanner = Bohen5to1Scanner()
    
    # Test with a few symbols
    test_symbols = ['AAPL', 'NVDA', 'TSLA', 'AMD', 'SPY']
    
    print("=" * 60)
    print("TIM BOHEN 5:1 SCANNER TEST")
    print("=" * 60)
    
    for symbol in test_symbols:
        print(f"\n--- {symbol} ---")
        result = scanner.analyze(symbol)
        
        if 'error' in result:
            print(f"Error: {result['error']}")
            continue
        
        print(f"Current Price: ${result['current_price']}")
        print(f"ATR: ${result['atr']} ({result['atr_pct']}%)")
        print(f"Volume: {result['volume_signal']} ({result['volume_ratio']}x avg)")
        print(f"Trend: {result['trend']['direction']} (score: {result['trend']['score']})")
        
        if result['best_setup']:
            setup = result['best_setup']
            print(f"\nBest Setup: {setup['direction']}")
            print(f"  Entry: ${setup['entry']}")
            print(f"  Stop: ${setup['stop_loss']} (Risk: {setup['risk_pct']}%)")
            print(f"  Target: ${setup['target']} (Reward: {setup['reward_pct']}%)")
            print(f"  R/R Ratio: {setup['rr_ratio']}:1")
        
        print(f"\n{result['bohen_verdict']}")
    
    print("\n" + "=" * 60)
    print("SCAN COMPLETE")
    print("=" * 60)
