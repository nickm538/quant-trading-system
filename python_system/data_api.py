"""
Data API Client for Manus API Hub
Provides access to financial data through the Manus API Hub with multi-source fallbacks.

Data Source Priority:
1. Manus API Hub (primary)
2. yfinance (fallback #1)
3. Twelve Data (fallback #2)
"""

import os
import sys
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

# Import Twelve Data client for additional fallback
try:
    from twelvedata_client import TwelveDataClient
    TWELVEDATA_AVAILABLE = True
except ImportError:
    TWELVEDATA_AVAILABLE = False


class ApiClient:
    """
    Client for accessing Manus API Hub endpoints with multi-source fallbacks.
    
    Fallback chain: Manus API Hub -> yfinance -> Twelve Data
    """
    
    def __init__(self):
        # Get API base URL from environment or use default
        self.base_url = os.environ.get('MANUS_API_HUB_URL', 'https://api-hub.manus.app')
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Quant-Trading-System/1.0'
        })
        # Enable fallbacks when Manus API Hub is unavailable
        self.use_yfinance_fallback = True
        
        # Initialize Twelve Data client as additional fallback
        self.twelvedata_client = None
        if TWELVEDATA_AVAILABLE:
            try:
                self.twelvedata_client = TwelveDataClient()
                print("✓ Twelve Data fallback initialized", file=sys.stderr)
            except Exception as e:
                print(f"Twelve Data fallback unavailable: {e}", file=sys.stderr)
    
    def call_api(self, endpoint: str, query: Optional[Dict[str, Any]] = None, method: str = 'GET') -> Dict[str, Any]:
        """
        Call a Manus API Hub endpoint with yfinance fallback

        Args:
            endpoint: API endpoint path (e.g., 'YahooFinance/get_stock_chart')
            query: Query parameters as dictionary
            method: HTTP method (GET or POST)

        Returns:
            API response as dictionary
        """
        # PRIORITY: Try Manus API Hub FIRST, then yfinance as fallback
        url = f"{self.base_url}/{endpoint}"

        try:
            if method.upper() == 'GET':
                response = self.session.get(url, params=query, timeout=30)
            else:
                response = self.session.post(url, json=query, timeout=30)

            response.raise_for_status()
            data = response.json()

            # Validate response based on endpoint type
            if 'get_stock_chart' in endpoint:
                # Chart endpoint needs chart.result data
                if data and 'chart' in data and data['chart'].get('result'):
                    return data
                else:
                    raise ValueError("Chart API returned empty or invalid data")
            else:
                # Other endpoints - just check for non-empty response
                if data:
                    return data
                else:
                    raise ValueError("API returned empty data")

        except Exception as e:
            print(f"Manus API Hub failed: {e}", file=sys.stderr)
            # FALLBACK to yfinance for Yahoo Finance endpoints
            if 'YahooFinance' in endpoint and self.use_yfinance_fallback:
                print(f"Trying yfinance fallback...", file=sys.stderr)
                if 'get_stock_chart' in endpoint:
                    result = self._get_stock_chart_yfinance(query)
                    if result:
                        return result
                elif 'get_stock_insights' in endpoint:
                    result = self._get_stock_insights_yfinance(query)
                    if result:
                        return result
                
                # If yfinance also failed, try Twelve Data
                if self.twelvedata_client:
                    print(f"Trying Twelve Data fallback...", file=sys.stderr)
                    if 'get_stock_chart' in endpoint:
                        return self._get_stock_chart_twelvedata(query)
                    elif 'get_stock_insights' in endpoint:
                        return self._get_stock_insights_twelvedata(query)
            return {}
    
    def get_stock_chart(self, symbol: str, interval: str = '1d', range_period: str = '1y') -> Dict[str, Any]:
        """
        Get stock chart data from Yahoo Finance via Manus API Hub
        
        Args:
            symbol: Stock ticker symbol
            interval: Data interval (1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo)
            range_period: Time range (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        
        Returns:
            Chart data with OHLCV
        """
        return self.call_api('YahooFinance/get_stock_chart', {
            'symbol': symbol,
            'region': 'US',
            'interval': interval,
            'range': range_period,
            'includeAdjustedClose': True
        })
    
    def get_stock_insights(self, symbol: str) -> Dict[str, Any]:
        """
        Get stock insights and fundamentals from Yahoo Finance via Manus API Hub
        
        Args:
            symbol: Stock ticker symbol
        
        Returns:
            Stock insights including fundamentals, analyst ratings, etc.
        """
        return self.call_api('YahooFinance/get_stock_insights', {
            'symbol': symbol,
            'region': 'US'
        })
    
    def _get_stock_chart_yfinance(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get stock chart data using yfinance as fallback
        """
        try:
            symbol = query.get('symbol')
            interval = query.get('interval', '1d')
            range_period = query.get('range', '1y')
            
            # Convert interval format
            interval_map = {
                '1m': '1m', '2m': '2m', '5m': '5m', '15m': '15m', '30m': '30m',
                '60m': '60m', '90m': '90m', '1h': '1h', '1d': '1d', '5d': '5d',
                '1wk': '1wk', '1mo': '1mo', '3mo': '3mo'
            }
            yf_interval = interval_map.get(interval, '1d')
            
            # Convert range to period
            period_map = {
                '1d': '1d', '5d': '5d', '1mo': '1mo', '3mo': '3mo',
                '6mo': '6mo', '1y': '1y', '2y': '2y', '5y': '5y',
                '10y': '10y', 'ytd': 'ytd', 'max': 'max'
            }
            period = period_map.get(range_period, '1y')
            
            # Fetch data from yfinance
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval=yf_interval)
            
            if hist.empty:
                return {}
            
            # Convert to Yahoo Finance API format
            timestamps = [int(ts.timestamp()) for ts in hist.index]
            
            result = {
                'chart': {
                    'result': [{
                        'meta': {
                            'currency': 'USD',
                            'symbol': symbol,
                            'regularMarketPrice': float(hist['Close'].iloc[-1]),
                            'previousClose': float(hist['Close'].iloc[-2]) if len(hist) > 1 else float(hist['Close'].iloc[-1])
                        },
                        'timestamp': timestamps,
                        'indicators': {
                            'quote': [{
                                'open': hist['Open'].tolist(),
                                'high': hist['High'].tolist(),
                                'low': hist['Low'].tolist(),
                                'close': hist['Close'].tolist(),
                                'volume': hist['Volume'].tolist()
                            }]
                        }
                    }]
                }
            }
            
            return result
            
        except Exception as e:
            print(f"yfinance chart fallback failed: {e}", file=sys.stderr)
            return {}
    
    def _get_stock_insights_yfinance(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get stock insights using yfinance as fallback
        """
        try:
            symbol = query.get('symbol')
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Convert to insights format
            result = {
                'symbol': symbol,
                'companyName': info.get('longName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'marketCap': info.get('marketCap', 0),
                'peRatio': info.get('trailingPE', 0),
                'dividendYield': info.get('dividendYield', 0),
                'beta': info.get('beta', 1.0),
                'fiftyTwoWeekHigh': info.get('fiftyTwoWeekHigh', 0),
                'fiftyTwoWeekLow': info.get('fiftyTwoWeekLow', 0)
            }
            
            return result
            
        except Exception as e:
            print(f"yfinance insights fallback failed: {e}", file=sys.stderr)
            return {}
    
    def _get_stock_chart_twelvedata(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get stock chart data using Twelve Data as fallback
        """
        try:
            symbol = query.get('symbol')
            interval = query.get('interval', '1d')
            range_period = query.get('range', '1y')
            
            # Convert range to outputsize (approximate trading days)
            range_to_days = {
                '1d': 1, '5d': 5, '1mo': 22, '3mo': 66,
                '6mo': 126, '1y': 252, '2y': 504, '5y': 1260
            }
            outputsize = range_to_days.get(range_period, 252)
            
            # Convert interval format
            interval_map = {
                '1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min',
                '1h': '1h', '1d': '1day', '1wk': '1week', '1mo': '1month'
            }
            td_interval = interval_map.get(interval, '1day')
            
            # Fetch from Twelve Data
            df = self.twelvedata_client.get_time_series(
                symbol, interval=td_interval, outputsize=outputsize
            )
            
            if df.empty:
                return {}
            
            # Get current quote for meta info
            quote = self.twelvedata_client.get_quote(symbol)
            
            # Convert to Yahoo Finance API format
            timestamps = [int(ts.timestamp()) for ts in df.index]
            
            result = {
                'chart': {
                    'result': [{
                        'meta': {
                            'currency': quote.get('currency', 'USD'),
                            'symbol': symbol,
                            'regularMarketPrice': float(df['close'].iloc[-1]),
                            'previousClose': float(df['close'].iloc[-2]) if len(df) > 1 else float(df['close'].iloc[-1])
                        },
                        'timestamp': timestamps,
                        'indicators': {
                            'quote': [{
                                'open': df['open'].tolist(),
                                'high': df['high'].tolist(),
                                'low': df['low'].tolist(),
                                'close': df['close'].tolist(),
                                'volume': df['volume'].tolist() if 'volume' in df.columns else [0] * len(df)
                            }]
                        }
                    }]
                }
            }
            
            print(f"✓ Twelve Data chart fallback successful for {symbol}", file=sys.stderr)
            return result
            
        except Exception as e:
            print(f"Twelve Data chart fallback failed: {e}", file=sys.stderr)
            return {}
    
    def _get_stock_insights_twelvedata(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get stock insights using Twelve Data as fallback
        """
        try:
            symbol = query.get('symbol')
            quote = self.twelvedata_client.get_quote(symbol)
            
            if 'error' in quote:
                return {}
            
            # Convert to insights format
            result = {
                'symbol': symbol,
                'companyName': quote.get('name', symbol),
                'sector': 'Unknown',  # Twelve Data basic plan doesn't include sector
                'industry': 'Unknown',
                'marketCap': 0,  # Would need fundamentals endpoint
                'peRatio': 0,
                'dividendYield': 0,
                'beta': 1.0,
                'fiftyTwoWeekHigh': quote.get('fifty_two_week', {}).get('high', 0),
                'fiftyTwoWeekLow': quote.get('fifty_two_week', {}).get('low', 0)
            }
            
            print(f"✓ Twelve Data insights fallback successful for {symbol}", file=sys.stderr)
            return result
            
        except Exception as e:
            print(f"Twelve Data insights fallback failed: {e}", file=sys.stderr)
            return {}
