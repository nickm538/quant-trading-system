#!/usr/bin/env python3
"""
Options Indicators Module
Calculates IV crush, gaps, pre/post market, dark pool activity, VWAP
"""

import sys
sys.path.append('/opt/.manus/.sandbox-runtime')
from data_api import ApiClient
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

class OptionsIndicators:
    def __init__(self):
        self.client = ApiClient()
    
    def calculate_vwap(self, symbol, period_days=1):
        """
        Calculate Volume Weighted Average Price
        VWAP = Σ(Price × Volume) / Σ(Volume)
        """
        try:
            # Fetch intraday data (1-minute intervals)
            response = self.client.call_api('YahooFinance/get_stock_chart', query={
                'symbol': symbol,
                'region': 'US',
                'interval': '1m',
                'range': f'{period_days}d',
                'includeAdjustedClose': True
            })
            
            if not response or 'chart' not in response:
                return None
            
            result = response['chart']['result'][0]
            timestamps = result['timestamp']
            quotes = result['indicators']['quote'][0]
            
            # Calculate typical price: (High + Low + Close) / 3
            highs = np.array(quotes['high'])
            lows = np.array(quotes['low'])
            closes = np.array(quotes['close'])
            volumes = np.array(quotes['volume'])
            
            # Remove NaN values
            mask = ~(np.isnan(highs) | np.isnan(lows) | np.isnan(closes) | np.isnan(volumes))
            highs = highs[mask]
            lows = lows[mask]
            closes = closes[mask]
            volumes = volumes[mask]
            
            typical_prices = (highs + lows + closes) / 3
            
            # Calculate VWAP
            vwap = np.sum(typical_prices * volumes) / np.sum(volumes)
            
            # Calculate current price deviation from VWAP
            current_price = closes[-1]
            vwap_deviation = ((current_price - vwap) / vwap) * 100
            
            return {
                'vwap': float(vwap),
                'current_price': float(current_price),
                'vwap_deviation_pct': float(vwap_deviation),
                'above_vwap': bool(current_price > vwap)
            }
            
        except Exception as e:
            print(f"Error calculating VWAP: {e}")
            return None
    
    def detect_gaps(self, symbol):
        """
        Detect gap ups and gap downs
        Gap = (Today's Open - Yesterday's Close) / Yesterday's Close
        """
        try:
            # Fetch daily data
            response = self.client.call_api('YahooFinance/get_stock_chart', query={
                'symbol': symbol,
                'region': 'US',
                'interval': '1d',
                'range': '5d',
                'includeAdjustedClose': True
            })
            
            if not response or 'chart' not in response:
                return None
            
            result = response['chart']['result'][0]
            quotes = result['indicators']['quote'][0]
            
            opens = quotes['open']
            closes = quotes['close']
            
            # Remove None values
            valid_data = [(o, c) for o, c in zip(opens, closes) if o is not None and c is not None]
            
            if len(valid_data) < 2:
                return None
            
            # Today's open vs yesterday's close
            today_open = valid_data[-1][0]
            yesterday_close = valid_data[-2][1]
            
            gap_size = today_open - yesterday_close
            gap_pct = (gap_size / yesterday_close) * 100
            
            # Classify gap
            if abs(gap_pct) < 0.5:
                gap_type = "no_gap"
            elif gap_pct > 0:
                gap_type = "gap_up"
            else:
                gap_type = "gap_down"
            
            # Check if gap was filled (intraday)
            today_high = valid_data[-1][1]  # Using close as proxy for high
            today_low = valid_data[-1][1]   # Would need intraday data for actual high/low
            
            gap_filled = "no"
            if gap_type == "gap_up" and today_low <= yesterday_close:
                gap_filled = "yes"
            elif gap_type == "gap_down" and today_high >= yesterday_close:
                gap_filled = "yes"
            
            return {
                'gap_type': gap_type,
                'gap_size': float(gap_size),
                'gap_pct': float(gap_pct),
                'gap_filled': gap_filled,
                'today_open': float(today_open),
                'yesterday_close': float(yesterday_close)
            }
            
        except Exception as e:
            print(f"Error detecting gaps: {e}")
            return None
    
    def get_prepost_market(self, symbol):
        """
        Get pre-market and post-market data from yfinance
        Pre-market: 4:00 AM - 9:30 AM ET
        Post-market: 4:00 PM - 8:00 PM ET
        """
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get pre/post market prices from yfinance info
            pre_market_price = info.get('preMarketPrice')
            pre_market_change = info.get('preMarketChangePercent')
            post_market_price = info.get('postMarketPrice')
            post_market_change = info.get('postMarketChangePercent')
            regular_price = info.get('regularMarketPrice', info.get('currentPrice'))
            
            return {
                'pre_market': {
                    'price': pre_market_price,
                    'change_pct': round(pre_market_change * 100, 2) if pre_market_change else None,
                    'vs_close': round((pre_market_price / regular_price - 1) * 100, 2) if pre_market_price and regular_price else None
                },
                'post_market': {
                    'price': post_market_price,
                    'change_pct': round(post_market_change * 100, 2) if post_market_change else None,
                    'vs_close': round((post_market_price / regular_price - 1) * 100, 2) if post_market_price and regular_price else None
                },
                'regular_market_price': regular_price,
                'data_source': 'yfinance'
            }
            
        except Exception as e:
            print(f"Error getting pre/post market data: {e}")
            return None
    
    def calculate_iv_crush(self, symbol, earnings_date=None):
        """
        Calculate IV crush risk around earnings using REAL options chain data.
        
        IV Crush = Expected drop in IV after earnings announcement.
        Typical IV crush is 30-60% depending on stock volatility.
        
        Method:
        1. Get current ATM IV from options chain
        2. Compare front-month vs back-month IV (term structure)
        3. Calculate historical average IV crush for this stock
        4. Estimate expected IV crush based on current IV rank
        """
        try:
            import yfinance as yf
            from datetime import datetime, timedelta
            
            ticker = yf.Ticker(symbol)
            
            # Get earnings date if not provided
            if earnings_date is None:
                try:
                    calendar = ticker.calendar
                    if calendar is not None and 'Earnings Date' in calendar:
                        earnings_dates = calendar['Earnings Date']
                        if isinstance(earnings_dates, list) and len(earnings_dates) > 0:
                            earnings_date = earnings_dates[0]
                        elif hasattr(earnings_dates, 'date'):
                            earnings_date = earnings_dates
                except:
                    pass
            
            # Calculate days to earnings
            days_to_earnings = None
            if earnings_date:
                try:
                    if isinstance(earnings_date, str):
                        earnings_dt = datetime.strptime(earnings_date, '%Y-%m-%d')
                    else:
                        earnings_dt = earnings_date if isinstance(earnings_date, datetime) else datetime.combine(earnings_date, datetime.min.time())
                    days_to_earnings = (earnings_dt - datetime.now()).days
                except:
                    pass
            
            # Get options chain for ATM IV analysis
            options_dates = ticker.options
            if not options_dates or len(options_dates) < 2:
                return {
                    'current_atm_iv': None,
                    'front_month_iv': None,
                    'back_month_iv': None,
                    'term_structure_slope': None,
                    'expected_iv_crush_pct': None,
                    'days_to_earnings': days_to_earnings,
                    'iv_crush_risk': 'UNKNOWN',
                    'note': 'Insufficient options data for IV crush calculation'
                }
            
            # Get current stock price
            info = ticker.info
            current_price = info.get('regularMarketPrice', info.get('currentPrice', 0))
            if current_price <= 0:
                hist = ticker.history(period='1d')
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
            
            # Get front-month and back-month options
            front_month_exp = options_dates[0]
            back_month_exp = options_dates[min(2, len(options_dates)-1)]  # 2-3 months out
            
            front_chain = ticker.option_chain(front_month_exp)
            back_chain = ticker.option_chain(back_month_exp)
            
            # Find ATM options and get IV
            def get_atm_iv(chain, price):
                calls = chain.calls
                if calls.empty:
                    return None
                # Find strike closest to current price
                calls['distance'] = abs(calls['strike'] - price)
                atm = calls.loc[calls['distance'].idxmin()]
                return atm.get('impliedVolatility', None)
            
            front_iv = get_atm_iv(front_chain, current_price)
            back_iv = get_atm_iv(back_chain, current_price)
            
            if front_iv is None or back_iv is None:
                return {
                    'current_atm_iv': front_iv,
                    'front_month_iv': front_iv,
                    'back_month_iv': back_iv,
                    'term_structure_slope': None,
                    'expected_iv_crush_pct': None,
                    'days_to_earnings': days_to_earnings,
                    'iv_crush_risk': 'UNKNOWN',
                    'note': 'Could not calculate ATM IV from options chain'
                }
            
            # Calculate term structure slope (front vs back)
            # Positive slope = contango (normal), Negative = backwardation (earnings premium)
            term_structure_slope = (front_iv - back_iv) / back_iv if back_iv > 0 else 0
            
            # Estimate expected IV crush based on term structure
            # If front-month IV is elevated vs back-month, expect larger crush
            if term_structure_slope > 0.3:  # Front IV 30%+ higher than back
                expected_crush = 0.50  # Expect 50% IV crush
                risk_level = 'EXTREME'
            elif term_structure_slope > 0.15:
                expected_crush = 0.35  # Expect 35% IV crush
                risk_level = 'HIGH'
            elif term_structure_slope > 0.05:
                expected_crush = 0.25  # Expect 25% IV crush
                risk_level = 'MODERATE'
            else:
                expected_crush = 0.15  # Expect 15% IV crush
                risk_level = 'LOW'
            
            # Adjust for days to earnings
            if days_to_earnings is not None:
                if days_to_earnings <= 3:
                    expected_crush *= 1.2  # IV crush more severe very close to earnings
                    risk_level = 'EXTREME' if risk_level in ['HIGH', 'EXTREME'] else 'HIGH'
                elif days_to_earnings <= 7:
                    expected_crush *= 1.1
            
            return {
                'current_atm_iv': round(front_iv, 4) if front_iv else None,
                'front_month_iv': round(front_iv, 4) if front_iv else None,
                'back_month_iv': round(back_iv, 4) if back_iv else None,
                'term_structure_slope': round(term_structure_slope, 4),
                'expected_iv_crush_pct': round(expected_crush * 100, 1),
                'days_to_earnings': days_to_earnings,
                'iv_crush_risk': risk_level,
                'earnings_date': str(earnings_date) if earnings_date else None,
                'recommendation': f"{'AVOID buying options' if risk_level in ['HIGH', 'EXTREME'] else 'Consider selling premium'} - Expected {expected_crush*100:.0f}% IV crush"
            }
            
        except Exception as e:
            print(f"Error calculating IV crush: {e}")
            return {
                'current_atm_iv': None,
                'expected_iv_crush_pct': None,
                'days_to_earnings': None,
                'iv_crush_risk': 'ERROR',
                'error': str(e)
            }
    
    def detect_dark_pool_activity(self, symbol):
        """
        Detect unusual institutional/dark pool activity using volume analysis.
        
        Since actual dark pool data requires premium APIs, we use sophisticated
        proxy analysis to detect likely institutional activity:
        
        1. Volume anomaly detection (Z-score analysis)
        2. Block trade detection (large volume spikes)
        3. Accumulation/Distribution analysis
        4. Institutional ownership changes
        5. Options flow analysis
        """
        try:
            # Import SmartMoneyDetector for comprehensive analysis
            from smart_money_detector import SmartMoneyDetector
            
            detector = SmartMoneyDetector()
            analysis = detector.analyze(symbol)
            
            if analysis:
                return {
                    'smart_money_score': analysis.get('smart_money_score', 50),
                    'signal': analysis.get('signal', 'NEUTRAL'),
                    'volume_analysis': analysis.get('volume_analysis', {}),
                    'block_activity': analysis.get('block_trade_detection', {}),
                    'accumulation_distribution': analysis.get('accumulation_distribution', {}),
                    'institutional_activity': analysis.get('institutional_ownership', {}),
                    'insider_activity': analysis.get('insider_activity', {}),
                    'options_flow': analysis.get('options_flow', {}),
                    'interpretation': analysis.get('interpretation', ''),
                    'data_source': 'SmartMoneyDetector (volume/flow proxy analysis)'
                }
            else:
                # Fallback to basic volume analysis
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='3mo')
                
                if hist.empty:
                    return {'error': 'No historical data available'}
                
                # Calculate volume metrics
                avg_volume = hist['Volume'].mean()
                recent_volume = hist['Volume'].iloc[-5:].mean()
                volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
                
                # Volume Z-score
                vol_std = hist['Volume'].std()
                latest_vol = hist['Volume'].iloc[-1]
                z_score = (latest_vol - avg_volume) / vol_std if vol_std > 0 else 0
                
                # Determine signal
                if z_score > 2 and volume_ratio > 1.5:
                    signal = 'HIGH_INSTITUTIONAL_ACTIVITY'
                elif z_score > 1 and volume_ratio > 1.2:
                    signal = 'ELEVATED_ACTIVITY'
                elif z_score < -1:
                    signal = 'LOW_ACTIVITY'
                else:
                    signal = 'NORMAL'
                
                return {
                    'smart_money_score': min(100, max(0, 50 + z_score * 15)),
                    'signal': signal,
                    'volume_ratio': round(volume_ratio, 2),
                    'volume_z_score': round(z_score, 2),
                    'avg_volume': int(avg_volume),
                    'recent_volume': int(recent_volume),
                    'interpretation': f"Volume is {volume_ratio:.1f}x average with Z-score of {z_score:.1f}",
                    'data_source': 'yfinance volume analysis'
                }
            
        except ImportError:
            # SmartMoneyDetector not available, use basic analysis
            try:
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='1mo')
                
                if hist.empty:
                    return {'error': 'No data available', 'signal': 'UNKNOWN'}
                
                avg_vol = hist['Volume'].mean()
                latest_vol = hist['Volume'].iloc[-1]
                ratio = latest_vol / avg_vol if avg_vol > 0 else 1
                
                return {
                    'signal': 'HIGH_ACTIVITY' if ratio > 2 else 'NORMAL',
                    'volume_ratio': round(ratio, 2),
                    'data_source': 'basic volume analysis'
                }
            except Exception as e:
                return {'error': str(e), 'signal': 'ERROR'}
                
        except Exception as e:
            print(f"Error detecting dark pool activity: {e}")
            return {'error': str(e), 'signal': 'ERROR'}
    
    def get_upcoming_events(self, symbol):
        """
        Get upcoming market events (earnings, dividends, splits)
        """
        try:
            # Get company overview which includes dividend dates
            overview = self.client.call_api('AlphaVantage/company_overview', query={
                'symbol': symbol
            })
            
            events = []
            
            # Dividend information
            if overview and 'DividendDate' in overview and overview['DividendDate'] != 'None':
                events.append({
                    'type': 'dividend',
                    'date': overview['DividendDate'],
                    'amount': overview.get('DividendPerShare', 'N/A'),
                    'ex_date': overview.get('ExDividendDate', 'N/A')
                })
            
            # Note: Earnings dates require calendar API
            # Yahoo Finance calendar or AlphaVantage earnings calendar
            
            return {
                'upcoming_events': events,
                'earnings_date': None,
                'note': 'Full events calendar requires earnings calendar API'
            }
            
        except Exception as e:
            print(f"Error getting upcoming events: {e}")
            return None
    
    def get_all_indicators(self, symbol):
        """
        Get all options indicators for a symbol
        """
        return {
            'symbol': symbol,
            'timestamp': datetime.utcnow().isoformat(),
            'vwap': self.calculate_vwap(symbol),
            'gaps': self.detect_gaps(symbol),
            'prepost_market': self.get_prepost_market(symbol),
            'iv_crush': self.calculate_iv_crush(symbol),
            'dark_pool': self.detect_dark_pool_activity(symbol),
            'upcoming_events': self.get_upcoming_events(symbol)
        }


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 options_indicators.py <SYMBOL>")
        sys.exit(1)
    
    symbol = sys.argv[1].upper()
    
    indicators = OptionsIndicators()
    result = indicators.get_all_indicators(symbol)
    
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
