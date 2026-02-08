"""
Data API Client - Multi-Source Financial Data with Robust Fallbacks
===================================================================

Data Source Priority (ordered by reliability in production):
1. Polygon.io (primary - paid API, most reliable)
2. Finnhub (fallback #1 - paid API)
3. yfinance (fallback #2 - free, works most environments)
4. Twelve Data (fallback #3 - free tier)
5. Manus API Hub (fallback #4 - may be unavailable)

Each source converts data to a unified Yahoo Finance-compatible format
so the rest of the pipeline doesn't need to change.
"""

import os
import sys
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

# Import yfinance
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("yfinance not available", file=sys.stderr)

# Import Polygon
try:
    from polygon import RESTClient as PolygonClient
    POLYGON_AVAILABLE = True
except ImportError:
    POLYGON_AVAILABLE = False
    print("polygon-api-client not available", file=sys.stderr)

# Import Twelve Data client
try:
    from twelvedata_client import TwelveDataClient
    TWELVEDATA_AVAILABLE = True
except ImportError:
    TWELVEDATA_AVAILABLE = False


class ApiClient:
    """
    Client for accessing financial market data with multi-source fallbacks.
    
    Fallback chain: Polygon -> Finnhub -> yfinance -> Twelve Data -> Manus API Hub
    """
    
    # Hardcoded API keys per user preference
    POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', '')
    FINNHUB_API_KEY = os.environ.get('FINNHUB_API_KEY', 'd55b3ohr01qljfdeghm0d55b3ohr01qljfdeghmg')
    FMP_API_KEY = 'LTecnRjOFtd8bFOTCRLpcncjxrqaZlqq'
    
    def __init__(self):
        self.base_url = os.environ.get('MANUS_API_HUB_URL', 'https://api-hub.manus.app')
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Quant-Trading-System/2.0'
        })
        
        # Initialize Polygon client
        self.polygon_client = None
        if POLYGON_AVAILABLE and self.POLYGON_API_KEY:
            try:
                self.polygon_client = PolygonClient(api_key=self.POLYGON_API_KEY)
                print("✓ Polygon.io initialized (primary data source)", file=sys.stderr)
            except Exception as e:
                print(f"Polygon.io init failed: {e}", file=sys.stderr)
        
        # Initialize Twelve Data client
        self.twelvedata_client = None
        if TWELVEDATA_AVAILABLE:
            try:
                self.twelvedata_client = TwelveDataClient()
                print("✓ Twelve Data fallback initialized", file=sys.stderr)
            except Exception as e:
                print(f"Twelve Data fallback unavailable: {e}", file=sys.stderr)
        
        # Track which source succeeded for logging
        self.last_source = None
    
    def call_api(self, endpoint: str, query: Optional[Dict[str, Any]] = None, method: str = 'GET') -> Dict[str, Any]:
        """
        Call API with multi-source fallback chain.
        Routes to the appropriate fallback based on endpoint type.
        """
        if 'get_stock_chart' in endpoint:
            return self._get_chart_with_fallbacks(query or {})
        elif 'get_stock_insights' in endpoint:
            return self._get_insights_with_fallbacks(query or {})
        else:
            # For other endpoints, try Manus API Hub directly
            return self._call_manus_api(endpoint, query, method)
    
    def get_stock_chart(self, symbol: str, interval: str = '1d', range_period: str = '1y') -> Dict[str, Any]:
        """Get stock chart data with full fallback chain."""
        query = {
            'symbol': symbol,
            'region': 'US',
            'interval': interval,
            'range': range_period,
            'includeAdjustedClose': True
        }
        return self._get_chart_with_fallbacks(query)
    
    def get_stock_insights(self, symbol: str) -> Dict[str, Any]:
        """Get stock insights/fundamentals with fallback chain."""
        query = {'symbol': symbol, 'region': 'US'}
        return self._get_insights_with_fallbacks(query)
    
    # =========================================================================
    # CHART DATA - Multi-source fallback chain
    # =========================================================================
    
    def _get_chart_with_fallbacks(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Try each data source in order until one succeeds.
        All sources return Yahoo Finance-compatible format.
        """
        symbol = query.get('symbol', '')
        errors = []
        
        # Source 1: Polygon.io (most reliable)
        if self.polygon_client:
            try:
                result = self._get_chart_polygon(query)
                if result and result.get('chart', {}).get('result'):
                    self.last_source = 'polygon'
                    print(f"✓ Chart data from Polygon.io for {symbol}", file=sys.stderr)
                    return result
            except Exception as e:
                errors.append(f"Polygon: {e}")
                print(f"Polygon chart failed for {symbol}: {e}", file=sys.stderr)
        
        # Source 2: Finnhub
        if self.FINNHUB_API_KEY:
            try:
                result = self._get_chart_finnhub(query)
                if result and result.get('chart', {}).get('result'):
                    self.last_source = 'finnhub'
                    print(f"✓ Chart data from Finnhub for {symbol}", file=sys.stderr)
                    return result
            except Exception as e:
                errors.append(f"Finnhub: {e}")
                print(f"Finnhub chart failed for {symbol}: {e}", file=sys.stderr)
        
        # Source 3: yfinance
        if YFINANCE_AVAILABLE:
            try:
                result = self._get_chart_yfinance(query)
                if result and result.get('chart', {}).get('result'):
                    self.last_source = 'yfinance'
                    print(f"✓ Chart data from yfinance for {symbol}", file=sys.stderr)
                    return result
            except Exception as e:
                errors.append(f"yfinance: {e}")
                print(f"yfinance chart failed for {symbol}: {e}", file=sys.stderr)
        
        # Source 4: Twelve Data
        if self.twelvedata_client:
            try:
                result = self._get_chart_twelvedata(query)
                if result and result.get('chart', {}).get('result'):
                    self.last_source = 'twelvedata'
                    print(f"✓ Chart data from Twelve Data for {symbol}", file=sys.stderr)
                    return result
            except Exception as e:
                errors.append(f"TwelveData: {e}")
                print(f"Twelve Data chart failed for {symbol}: {e}", file=sys.stderr)
        
        # Source 5: Manus API Hub (last resort)
        try:
            result = self._call_manus_api('YahooFinance/get_stock_chart', query)
            if result and result.get('chart', {}).get('result'):
                self.last_source = 'manus_api_hub'
                print(f"✓ Chart data from Manus API Hub for {symbol}", file=sys.stderr)
                return result
        except Exception as e:
            errors.append(f"ManusAPI: {e}")
            print(f"Manus API Hub chart failed for {symbol}: {e}", file=sys.stderr)
        
        # All sources failed
        print(f"ALL DATA SOURCES FAILED for {symbol}: {'; '.join(errors)}", file=sys.stderr)
        return {}
    
    # -------------------------------------------------------------------------
    # Polygon.io chart implementation
    # -------------------------------------------------------------------------
    def _get_chart_polygon(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Get chart data from Polygon.io REST API."""
        symbol = query.get('symbol')
        interval = query.get('interval', '1d')
        range_period = query.get('range', '1y')
        
        # Convert range to date range
        end_date = datetime.now()
        range_to_days = {
            '1d': 1, '5d': 5, '1mo': 30, '3mo': 90,
            '6mo': 180, '1y': 365, '2y': 730, '5y': 1825, '10y': 3650
        }
        days_back = range_to_days.get(range_period, 365)
        start_date = end_date - timedelta(days=days_back)
        
        # Convert interval to Polygon timespan/multiplier
        interval_map = {
            '1m': (1, 'minute'), '5m': (5, 'minute'), '15m': (15, 'minute'),
            '30m': (30, 'minute'), '1h': (1, 'hour'), '1d': (1, 'day'),
            '1wk': (1, 'week'), '1mo': (1, 'month')
        }
        multiplier, timespan = interval_map.get(interval, (1, 'day'))
        
        # Fetch from Polygon
        aggs = list(self.polygon_client.list_aggs(
            symbol, multiplier, timespan,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d'),
            limit=50000
        ))
        
        if not aggs:
            raise ValueError(f"Polygon returned no data for {symbol}")
        
        # Convert to Yahoo Finance format
        timestamps = [int(a.timestamp / 1000) for a in aggs]  # Polygon uses ms
        opens = [float(a.open) for a in aggs]
        highs = [float(a.high) for a in aggs]
        lows = [float(a.low) for a in aggs]
        closes = [float(a.close) for a in aggs]
        volumes = [int(a.volume) if a.volume else 0 for a in aggs]
        
        return {
            'chart': {
                'result': [{
                    'meta': {
                        'currency': 'USD',
                        'symbol': symbol,
                        'regularMarketPrice': closes[-1],
                        'previousClose': closes[-2] if len(closes) > 1 else closes[-1],
                        'dataSource': 'polygon'
                    },
                    'timestamp': timestamps,
                    'indicators': {
                        'quote': [{
                            'open': opens,
                            'high': highs,
                            'low': lows,
                            'close': closes,
                            'volume': volumes
                        }]
                    }
                }]
            }
        }
    
    # -------------------------------------------------------------------------
    # Finnhub chart implementation
    # -------------------------------------------------------------------------
    def _get_chart_finnhub(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Get chart data from Finnhub stock candles API."""
        symbol = query.get('symbol')
        interval = query.get('interval', '1d')
        range_period = query.get('range', '1y')
        
        # Convert range to timestamps
        end_ts = int(datetime.now().timestamp())
        range_to_days = {
            '1d': 1, '5d': 5, '1mo': 30, '3mo': 90,
            '6mo': 180, '1y': 365, '2y': 730, '5y': 1825
        }
        days_back = range_to_days.get(range_period, 365)
        start_ts = int((datetime.now() - timedelta(days=days_back)).timestamp())
        
        # Convert interval to Finnhub resolution
        resolution_map = {
            '1m': '1', '5m': '5', '15m': '15', '30m': '30',
            '1h': '60', '1d': 'D', '1wk': 'W', '1mo': 'M'
        }
        resolution = resolution_map.get(interval, 'D')
        
        url = 'https://finnhub.io/api/v1/stock/candle'
        params = {
            'symbol': symbol,
            'resolution': resolution,
            'from': start_ts,
            'to': end_ts,
            'token': self.FINNHUB_API_KEY
        }
        
        resp = self.session.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        
        if data.get('s') != 'ok' or not data.get('c'):
            raise ValueError(f"Finnhub returned status: {data.get('s', 'no_data')}")
        
        return {
            'chart': {
                'result': [{
                    'meta': {
                        'currency': 'USD',
                        'symbol': symbol,
                        'regularMarketPrice': data['c'][-1],
                        'previousClose': data['c'][-2] if len(data['c']) > 1 else data['c'][-1],
                        'dataSource': 'finnhub'
                    },
                    'timestamp': data['t'],
                    'indicators': {
                        'quote': [{
                            'open': data['o'],
                            'high': data['h'],
                            'low': data['l'],
                            'close': data['c'],
                            'volume': data['v']
                        }]
                    }
                }]
            }
        }
    
    # -------------------------------------------------------------------------
    # yfinance chart implementation
    # -------------------------------------------------------------------------
    def _get_chart_yfinance(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Get chart data using yfinance library."""
        symbol = query.get('symbol')
        interval = query.get('interval', '1d')
        range_period = query.get('range', '1y')
        
        interval_map = {
            '1m': '1m', '2m': '2m', '5m': '5m', '15m': '15m', '30m': '30m',
            '60m': '60m', '90m': '90m', '1h': '1h', '1d': '1d', '5d': '5d',
            '1wk': '1wk', '1mo': '1mo', '3mo': '3mo'
        }
        yf_interval = interval_map.get(interval, '1d')
        
        period_map = {
            '1d': '1d', '5d': '5d', '1mo': '1mo', '3mo': '3mo',
            '6mo': '6mo', '1y': '1y', '2y': '2y', '5y': '5y',
            '10y': '10y', 'ytd': 'ytd', 'max': 'max'
        }
        period = period_map.get(range_period, '1y')
        
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, interval=yf_interval)
        
        if hist.empty:
            raise ValueError(f"yfinance returned empty data for {symbol}")
        
        timestamps = [int(ts.timestamp()) for ts in hist.index]
        
        return {
            'chart': {
                'result': [{
                    'meta': {
                        'currency': 'USD',
                        'symbol': symbol,
                        'regularMarketPrice': float(hist['Close'].iloc[-1]),
                        'previousClose': float(hist['Close'].iloc[-2]) if len(hist) > 1 else float(hist['Close'].iloc[-1]),
                        'dataSource': 'yfinance'
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
    
    # -------------------------------------------------------------------------
    # Twelve Data chart implementation
    # -------------------------------------------------------------------------
    def _get_chart_twelvedata(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Get chart data using Twelve Data API."""
        symbol = query.get('symbol')
        interval = query.get('interval', '1d')
        range_period = query.get('range', '1y')
        
        range_to_days = {
            '1d': 1, '5d': 5, '1mo': 22, '3mo': 66,
            '6mo': 126, '1y': 252, '2y': 504, '5y': 1260
        }
        outputsize = range_to_days.get(range_period, 252)
        
        interval_map = {
            '1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min',
            '1h': '1h', '1d': '1day', '1wk': '1week', '1mo': '1month'
        }
        td_interval = interval_map.get(interval, '1day')
        
        df = self.twelvedata_client.get_time_series(
            symbol, interval=td_interval, outputsize=outputsize
        )
        
        if df.empty:
            raise ValueError(f"Twelve Data returned empty data for {symbol}")
        
        quote = self.twelvedata_client.get_quote(symbol)
        timestamps = [int(ts.timestamp()) for ts in df.index]
        
        return {
            'chart': {
                'result': [{
                    'meta': {
                        'currency': quote.get('currency', 'USD'),
                        'symbol': symbol,
                        'regularMarketPrice': float(df['close'].iloc[-1]),
                        'previousClose': float(df['close'].iloc[-2]) if len(df) > 1 else float(df['close'].iloc[-1]),
                        'dataSource': 'twelvedata'
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
    
    # =========================================================================
    # INSIGHTS DATA - Multi-source fallback chain
    # =========================================================================
    
    def _get_insights_with_fallbacks(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Get stock insights/fundamentals with fallback chain."""
        symbol = query.get('symbol', '')
        
        # Source 1: FMP (Financial Modeling Prep)
        try:
            result = self._get_insights_fmp(symbol)
            if result and result.get('companyName'):
                print(f"✓ Insights from FMP for {symbol}", file=sys.stderr)
                return result
        except Exception as e:
            print(f"FMP insights failed for {symbol}: {e}", file=sys.stderr)
        
        # Source 2: Finnhub
        if self.FINNHUB_API_KEY:
            try:
                result = self._get_insights_finnhub(symbol)
                if result and result.get('companyName'):
                    print(f"✓ Insights from Finnhub for {symbol}", file=sys.stderr)
                    return result
            except Exception as e:
                print(f"Finnhub insights failed for {symbol}: {e}", file=sys.stderr)
        
        # Source 3: yfinance
        if YFINANCE_AVAILABLE:
            try:
                result = self._get_insights_yfinance(symbol)
                if result and result.get('companyName'):
                    print(f"✓ Insights from yfinance for {symbol}", file=sys.stderr)
                    return result
            except Exception as e:
                print(f"yfinance insights failed for {symbol}: {e}", file=sys.stderr)
        
        # Source 4: Manus API Hub
        try:
            result = self._call_manus_api('YahooFinance/get_stock_insights', query)
            if result:
                return result
        except Exception as e:
            print(f"Manus API Hub insights failed for {symbol}: {e}", file=sys.stderr)
        
        return {}
    
    def _get_insights_fmp(self, symbol: str) -> Dict[str, Any]:
        """Get insights from Financial Modeling Prep."""
        url = f"https://financialmodelingprep.com/api/v3/profile/{symbol}"
        resp = self.session.get(url, params={'apikey': self.FMP_API_KEY}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        if not data or not isinstance(data, list) or len(data) == 0:
            raise ValueError("FMP returned empty profile")
        
        profile = data[0]
        return {
            'symbol': symbol,
            'companyName': profile.get('companyName', symbol),
            'sector': profile.get('sector', 'Unknown'),
            'industry': profile.get('industry', 'Unknown'),
            'marketCap': profile.get('mktCap', 0),
            'peRatio': profile.get('pe', 0) or 0,
            'dividendYield': profile.get('lastDiv', 0) or 0,
            'beta': profile.get('beta', 1.0) or 1.0,
            'fiftyTwoWeekHigh': profile.get('range', '0-0').split('-')[-1] if profile.get('range') else 0,
            'fiftyTwoWeekLow': profile.get('range', '0-0').split('-')[0] if profile.get('range') else 0,
            'description': profile.get('description', ''),
            'exchange': profile.get('exchangeShortName', ''),
            'country': profile.get('country', ''),
            'dataSource': 'fmp'
        }
    
    def _get_insights_finnhub(self, symbol: str) -> Dict[str, Any]:
        """Get insights from Finnhub."""
        # Company profile
        url = 'https://finnhub.io/api/v1/stock/profile2'
        resp = self.session.get(url, params={
            'symbol': symbol, 'token': self.FINNHUB_API_KEY
        }, timeout=10)
        resp.raise_for_status()
        profile = resp.json()
        
        if not profile or not profile.get('name'):
            raise ValueError("Finnhub returned empty profile")
        
        # Basic financials
        url2 = 'https://finnhub.io/api/v1/stock/metric'
        resp2 = self.session.get(url2, params={
            'symbol': symbol, 'metric': 'all', 'token': self.FINNHUB_API_KEY
        }, timeout=10)
        metrics = resp2.json().get('metric', {}) if resp2.ok else {}
        
        return {
            'symbol': symbol,
            'companyName': profile.get('name', symbol),
            'sector': profile.get('finnhubIndustry', 'Unknown'),
            'industry': profile.get('finnhubIndustry', 'Unknown'),
            'marketCap': profile.get('marketCapitalization', 0) * 1_000_000 if profile.get('marketCapitalization') else 0,
            'peRatio': metrics.get('peNormalizedAnnual', 0) or 0,
            'dividendYield': metrics.get('dividendYieldIndicatedAnnual', 0) or 0,
            'beta': metrics.get('beta', 1.0) or 1.0,
            'fiftyTwoWeekHigh': metrics.get('52WeekHigh', 0) or 0,
            'fiftyTwoWeekLow': metrics.get('52WeekLow', 0) or 0,
            'exchange': profile.get('exchange', ''),
            'country': profile.get('country', ''),
            'dataSource': 'finnhub'
        }
    
    def _get_insights_yfinance(self, symbol: str) -> Dict[str, Any]:
        """Get insights from yfinance."""
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        if not info or not info.get('longName'):
            raise ValueError("yfinance returned empty info")
        
        return {
            'symbol': symbol,
            'companyName': info.get('longName', symbol),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'marketCap': info.get('marketCap', 0),
            'peRatio': info.get('trailingPE', 0) or 0,
            'dividendYield': info.get('dividendYield', 0) or 0,
            'beta': info.get('beta', 1.0) or 1.0,
            'fiftyTwoWeekHigh': info.get('fiftyTwoWeekHigh', 0),
            'fiftyTwoWeekLow': info.get('fiftyTwoWeekLow', 0),
            'dataSource': 'yfinance'
        }
    
    # =========================================================================
    # Manus API Hub (legacy, kept as last resort)
    # =========================================================================
    
    def _call_manus_api(self, endpoint: str, query: Optional[Dict[str, Any]] = None, method: str = 'GET') -> Dict[str, Any]:
        """Call Manus API Hub endpoint directly."""
        url = f"{self.base_url}/{endpoint}"
        
        if method.upper() == 'GET':
            response = self.session.get(url, params=query, timeout=15)
        else:
            response = self.session.post(url, json=query, timeout=15)
        
        response.raise_for_status()
        data = response.json()
        
        if not data:
            raise ValueError("Manus API returned empty data")
        
        return data
