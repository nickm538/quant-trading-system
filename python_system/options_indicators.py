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
        Calculate IV crush around earnings
        IV Crush = (Pre-Earnings IV - Post-Earnings IV) / Pre-Earnings IV
        """
        try:
            # Note: IV data requires options chain data
            # Yahoo Finance doesn't provide historical IV directly
            # This would require options chain historical data
            
            return {
                'pre_earnings_iv': None,
                'post_earnings_iv': None,
                'iv_crush_pct': None,
                'days_to_earnings': None,
                'note': 'IV crush calculation requires historical options chain data'
            }
            
        except Exception as e:
            print(f"Error calculating IV crush: {e}")
            return None
    
    def detect_dark_pool_activity(self, symbol):
        """
        Detect unusual dark pool activity
        Note: Dark pool data is proprietary and not available via free APIs
        """
        try:
            return {
                'large_blocks': [],
                'total_dark_pool_volume': None,
                'dark_pool_pct_of_total': None,
                'sentiment': 'neutral',
                'note': 'Dark pool data requires premium data source (Quiver Quantitative, Unusual Whales)'
            }
            
        except Exception as e:
            print(f"Error detecting dark pool activity: {e}")
            return None
    
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
