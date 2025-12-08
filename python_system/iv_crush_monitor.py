#!/usr/bin/env python3
"""
IV Crush Monitor - Production Grade
Uses yfinance to fetch REAL implied volatility data and detect IV crush around earnings
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import sys

class IVCrushMonitor:
    def __init__(self):
        pass
    
    def get_current_iv(self, symbol):
        """
        Get current implied volatility from options chain
        Returns average IV across ATM options
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Get current stock price
            info = ticker.info
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            
            if not current_price:
                return None
            
            # Get options expirations
            expirations = ticker.options
            
            if not expirations or len(expirations) == 0:
                return None
            
            # Use nearest expiration (typically weekly)
            nearest_exp = expirations[0]
            
            # Get options chain
            opt_chain = ticker.option_chain(nearest_exp)
            calls = opt_chain.calls
            puts = opt_chain.puts
            
            # Find ATM options (within 5% of current price)
            atm_range = current_price * 0.05
            
            atm_calls = calls[
                (calls['strike'] >= current_price - atm_range) &
                (calls['strike'] <= current_price + atm_range)
            ]
            
            atm_puts = puts[
                (puts['strike'] >= current_price - atm_range) &
                (puts['strike'] <= current_price + atm_range)
            ]
            
            # Calculate average IV from ATM options
            call_ivs = atm_calls['impliedVolatility'].dropna()
            put_ivs = atm_puts['impliedVolatility'].dropna()
            
            all_ivs = pd.concat([call_ivs, put_ivs])
            
            if len(all_ivs) == 0:
                return None
            
            avg_iv = all_ivs.mean()
            
            # Convert to percentage (yfinance returns decimal)
            avg_iv_pct = avg_iv * 100
            
            return {
                'symbol': symbol,
                'current_price': float(current_price),
                'expiration': nearest_exp,
                'avg_iv': float(avg_iv_pct),
                'call_iv': float(call_ivs.mean() * 100) if len(call_ivs) > 0 else None,
                'put_iv': float(put_ivs.mean() * 100) if len(put_ivs) > 0 else None,
                'iv_skew': float((put_ivs.mean() - call_ivs.mean()) * 100) if len(call_ivs) > 0 and len(put_ivs) > 0 else None,
                'atm_call_count': len(atm_calls),
                'atm_put_count': len(atm_puts),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            print(f"Error getting current IV: {e}")
            return None
    
    def calculate_iv_rank(self, symbol, current_iv):
        """
        Calculate IV Rank (IV percentile over past year)
        IV Rank = (Current IV - Min IV) / (Max IV - Min IV) * 100
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Get historical data for past year
            hist = ticker.history(period='1y', interval='1d')
            
            if hist.empty:
                return None
            
            # Calculate historical volatility as proxy for IV history
            # (Real IV history requires options chain historical data which is premium)
            returns = hist['Close'].pct_change().dropna()
            
            # Calculate rolling 30-day volatility
            rolling_vol = returns.rolling(window=30).std() * np.sqrt(252) * 100
            
            if len(rolling_vol.dropna()) == 0:
                return None
            
            min_vol = rolling_vol.min()
            max_vol = rolling_vol.max()
            
            if max_vol == min_vol:
                return 50.0  # If no range, return middle
            
            # Calculate IV Rank
            iv_rank = ((current_iv - min_vol) / (max_vol - min_vol)) * 100
            
            return {
                'iv_rank': float(iv_rank),
                'min_iv_1y': float(min_vol),
                'max_iv_1y': float(max_vol),
                'current_iv': float(current_iv),
                'interpretation': self._interpret_iv_rank(iv_rank)
            }
            
        except Exception as e:
            print(f"Error calculating IV rank: {e}")
            return None
    
    def _interpret_iv_rank(self, iv_rank):
        """Interpret IV rank for trading decisions"""
        if iv_rank < 20:
            return "Very Low - Consider buying options (cheap premium)"
        elif iv_rank < 40:
            return "Low - Neutral to bullish on options buying"
        elif iv_rank < 60:
            return "Moderate - No strong directional bias"
        elif iv_rank < 80:
            return "High - Consider selling options (expensive premium)"
        else:
            return "Very High - Strong opportunity for option selling"
    
    def calculate_historical_iv_crush(self, symbol):
        """
        Calculate historical IV crush magnitude from past earnings events
        Uses 1-year historical volatility spikes as proxy for earnings IV patterns
        """
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='1y', interval='1d')
            
            if hist.empty:
                return None
            
            # Calculate rolling 30-day volatility
            returns = hist['Close'].pct_change().dropna()
            rolling_vol = returns.rolling(window=30).std() * np.sqrt(252) * 100
            
            # Find volatility spikes (likely earnings events)
            # Spike = vol > 1.5x median vol
            median_vol = rolling_vol.median()
            vol_spikes = rolling_vol[rolling_vol > median_vol * 1.5]
            
            if len(vol_spikes) < 2:
                # Not enough data, use conservative 40% crush estimate
                return {
                    'avg_crush_pct': 40.0,
                    'sample_size': 0,
                    'confidence': 'low'
                }
            
            # Calculate average crush magnitude
            # Crush = (peak_vol - post_peak_vol) / peak_vol
            crush_magnitudes = []
            
            for idx in vol_spikes.index:
                # Find position in rolling_vol
                pos = rolling_vol.index.get_loc(idx)
                
                # Look 5-10 days after spike for post-earnings vol
                if pos + 10 < len(rolling_vol):
                    peak_vol = rolling_vol.iloc[pos]
                    post_vol = rolling_vol.iloc[pos + 5:pos + 10].mean()
                    
                    if not np.isnan(post_vol) and peak_vol > 0:
                        crush_pct = ((peak_vol - post_vol) / peak_vol) * 100
                        if 0 < crush_pct < 80:  # Sanity check
                            crush_magnitudes.append(crush_pct)
            
            if len(crush_magnitudes) == 0:
                return {
                    'avg_crush_pct': 40.0,
                    'sample_size': 0,
                    'confidence': 'low'
                }
            
            avg_crush = np.mean(crush_magnitudes)
            std_crush = np.std(crush_magnitudes)
            
            # Confidence based on sample size
            if len(crush_magnitudes) >= 4:
                confidence = 'high'
            elif len(crush_magnitudes) >= 2:
                confidence = 'medium'
            else:
                confidence = 'low'
            
            return {
                'avg_crush_pct': float(avg_crush),
                'std_crush_pct': float(std_crush),
                'min_crush_pct': float(min(crush_magnitudes)),
                'max_crush_pct': float(max(crush_magnitudes)),
                'sample_size': len(crush_magnitudes),
                'confidence': confidence
            }
            
        except Exception as e:
            print(f"Error calculating historical IV crush: {e}")
            return {
                'avg_crush_pct': 40.0,
                'sample_size': 0,
                'confidence': 'low'
            }
    
    def detect_earnings_iv_crush(self, symbol):
        """
        Detect potential IV crush around earnings with historical crush magnitude
        """
        try:
            iv_data = self.get_current_iv(symbol)
            
            if not iv_data:
                return None
            
            current_iv = iv_data['avg_iv']
            
            # Calculate IV rank
            iv_rank_data = self.calculate_iv_rank(symbol, current_iv)
            
            if not iv_rank_data:
                return None
            
            iv_rank = iv_rank_data['iv_rank']
            
            # High IV rank (>70) suggests potential earnings event
            potential_earnings = iv_rank > 70
            
            # Calculate historical IV crush magnitude
            historical_crush = self.calculate_historical_iv_crush(symbol)
            crush_pct = historical_crush['avg_crush_pct']
            
            # Estimate post-earnings IV using historical crush
            estimated_post_earnings_iv = current_iv * (1 - crush_pct / 100)
            
            # Calculate expected option value loss
            # Options lose value proportional to IV drop (vega risk)
            estimated_value_loss_pct = crush_pct
            
            return {
                'symbol': symbol,
                'current_iv': float(current_iv),
                'iv_rank': float(iv_rank),
                'potential_earnings_event': potential_earnings,
                'estimated_post_earnings_iv': float(estimated_post_earnings_iv),
                'estimated_iv_crush_pct': float(estimated_value_loss_pct),
                'estimated_option_value_loss_pct': float(estimated_value_loss_pct),
                'historical_crush_data': historical_crush,
                'recommendation': self._get_iv_crush_recommendation(iv_rank, potential_earnings, crush_pct),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            print(f"Error detecting IV crush: {e}")
            return None
    
    def _get_iv_crush_recommendation(self, iv_rank, potential_earnings, expected_crush_pct):
        """Get trading recommendation based on IV analysis with expected crush magnitude"""
        if potential_earnings:
            if iv_rank > 80 and expected_crush_pct > 40:
                return f"AVOID BUYING OPTIONS - Extreme IV (rank {iv_rank:.0f}), expected {expected_crush_pct:.0f}% IV crush. Options will lose ~{expected_crush_pct:.0f}% value post-earnings even if stock doesn't move. Consider selling premium if experienced."
            elif iv_rank > 70 and expected_crush_pct > 30:
                return f"CAUTION - Elevated IV suggests earnings soon. Expected {expected_crush_pct:.0f}% IV crush will hurt long options. Stock must move >{expected_crush_pct:.0f}% to break even. Only buy if expecting massive move."
            elif expected_crush_pct > 25:
                return f"MODERATE RISK - Expected {expected_crush_pct:.0f}% IV crush. Factor this into position sizing and profit targets."
            else:
                return "NEUTRAL - Monitor for IV changes"
        else:
            if iv_rank < 30:
                return "OPPORTUNITY - Low IV, consider buying options for directional plays"
            elif iv_rank > 70:
                return "EXPENSIVE - High IV, consider selling premium strategies"
            else:
                return "NEUTRAL - Normal IV levels"
    
    def get_full_iv_analysis(self, symbol):
        """
        Get comprehensive IV analysis including current IV, IV rank, and crush detection
        """
        try:
            # Get current IV
            current_iv_data = self.get_current_iv(symbol)
            
            if not current_iv_data:
                return {
                    'error': 'Unable to fetch IV data',
                    'symbol': symbol
                }
            
            # Get IV rank
            iv_rank_data = self.calculate_iv_rank(symbol, current_iv_data['avg_iv'])
            
            # Get IV crush analysis
            iv_crush_data = self.detect_earnings_iv_crush(symbol)
            
            return {
                'symbol': symbol,
                'timestamp': datetime.utcnow().isoformat(),
                'current_iv': current_iv_data,
                'iv_rank': iv_rank_data,
                'iv_crush_analysis': iv_crush_data
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'symbol': symbol
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 iv_crush_monitor.py <SYMBOL>")
        sys.exit(1)
    
    symbol = sys.argv[1].upper()
    
    monitor = IVCrushMonitor()
    result = monitor.get_full_iv_analysis(symbol)
    
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
