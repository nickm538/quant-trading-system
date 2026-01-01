"""
Twelve Data API Client
======================

Institutional-grade data provider integration for SadieAI trading system.
Provides real-time and historical market data as a fallback/supplement to existing sources.

API Documentation: https://twelvedata.com/docs
Coverage: 1M+ instruments across stocks, forex, ETFs, crypto, and more.

Author: SadieAI Trading System
Date: January 2026
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TwelveDataClient:
    """
    Client for Twelve Data API - institutional-grade market data provider.
    
    Features:
    - Real-time and historical price data
    - Technical indicators (50+)
    - Fundamentals data
    - Market movers
    - Rate limit handling with exponential backoff
    """
    
    # Hardcoded API key as requested
    API_KEY = "5e7a5daaf41d46a8966963106ebef210"
    BASE_URL = "https://api.twelvedata.com"
    
    # Rate limiting settings
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0  # seconds
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Twelve Data client.
        
        Args:
            api_key: Optional API key override (uses hardcoded key by default)
        """
        self.api_key = api_key or self.API_KEY
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'apikey {self.api_key}'
        })
        logger.info("✓ TwelveDataClient initialized")
    
    def _make_request(
        self,
        endpoint: str,
        params: Dict[str, Any] = None,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """
        Make API request with retry logic and error handling.
        
        Args:
            endpoint: API endpoint (e.g., '/time_series')
            params: Query parameters
            timeout: Request timeout in seconds
            
        Returns:
            JSON response as dictionary
        """
        url = f"{self.BASE_URL}{endpoint}"
        params = params or {}
        params['apikey'] = self.api_key
        
        for attempt in range(self.MAX_RETRIES):
            try:
                response = self.session.get(url, params=params, timeout=timeout)
                data = response.json()
                
                # Check for API errors
                if 'code' in data and data.get('status') == 'error':
                    error_code = data.get('code')
                    error_msg = data.get('message', 'Unknown error')
                    
                    if error_code == 429:  # Rate limit
                        wait_time = self.RETRY_DELAY * (2 ** attempt)
                        logger.warning(f"Rate limited. Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        continue
                    elif error_code == 401:
                        logger.error("Invalid API key")
                        raise ValueError("Invalid Twelve Data API key")
                    else:
                        logger.error(f"API error {error_code}: {error_msg}")
                        return {'error': True, 'code': error_code, 'message': error_msg}
                
                return data
                
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout (attempt {attempt + 1}/{self.MAX_RETRIES})")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY)
                    continue
                return {'error': True, 'message': 'Request timeout'}
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed: {str(e)}")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY)
                    continue
                return {'error': True, 'message': str(e)}
        
        return {'error': True, 'message': 'Max retries exceeded'}
    
    # ==================== PRICE DATA ====================
    
    def get_price(self, symbol: str) -> Dict[str, Any]:
        """
        Get the latest price for a symbol.
        
        Args:
            symbol: Stock ticker (e.g., 'AAPL')
            
        Returns:
            Dictionary with 'price' key or error
        """
        logger.info(f"Fetching latest price for {symbol} from Twelve Data")
        
        response = self._make_request('/price', {'symbol': symbol})
        
        if 'error' in response:
            return response
        
        if 'price' in response:
            return {
                'symbol': symbol,
                'price': float(response['price']),
                'source': 'twelvedata'
            }
        
        return {'error': True, 'message': 'No price data returned'}
    
    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get comprehensive quote data for a symbol.
        
        Args:
            symbol: Stock ticker (e.g., 'AAPL')
            
        Returns:
            Dictionary with quote data including open, high, low, close, volume
        """
        logger.info(f"Fetching quote for {symbol} from Twelve Data")
        
        response = self._make_request('/quote', {'symbol': symbol})
        
        if 'error' in response:
            return response
        
        if 'symbol' in response:
            return {
                'symbol': response.get('symbol'),
                'name': response.get('name'),
                'exchange': response.get('exchange'),
                'currency': response.get('currency'),
                'open': float(response.get('open', 0)),
                'high': float(response.get('high', 0)),
                'low': float(response.get('low', 0)),
                'close': float(response.get('close', 0)),
                'volume': int(response.get('volume', 0)),
                'previous_close': float(response.get('previous_close', 0)),
                'change': float(response.get('change', 0)),
                'percent_change': float(response.get('percent_change', 0)),
                'average_volume': int(response.get('average_volume', 0)),
                'fifty_two_week': response.get('fifty_two_week', {}),
                'is_market_open': response.get('is_market_open', False),
                'timestamp': response.get('timestamp'),
                'source': 'twelvedata'
            }
        
        return {'error': True, 'message': 'No quote data returned'}
    
    def get_time_series(
        self,
        symbol: str,
        interval: str = '1day',
        outputsize: int = 252,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        adjust: str = 'splits'
    ) -> pd.DataFrame:
        """
        Get historical time series data.
        
        Args:
            symbol: Stock ticker (e.g., 'AAPL')
            interval: Time interval ('1min', '5min', '15min', '30min', '1h', '1day', '1week', '1month')
            outputsize: Number of data points (max 5000)
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            adjust: Adjustment type ('all', 'splits', 'dividends', 'none')
            
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Fetching time series for {symbol} ({interval}, {outputsize} points) from Twelve Data")
        
        params = {
            'symbol': symbol,
            'interval': interval,
            'outputsize': min(outputsize, 5000),
            'adjust': adjust,
            'order': 'asc'  # Oldest first for proper DataFrame indexing
        }
        
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        
        response = self._make_request('/time_series', params)
        
        if 'error' in response:
            logger.error(f"Time series error: {response.get('message')}")
            return pd.DataFrame()
        
        if 'values' not in response:
            logger.warning(f"No time series data returned for {symbol}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(response['values'])
        
        if df.empty:
            return df
        
        # Convert columns to proper types
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype(int)
        
        # Rename columns to match existing system convention
        df.columns = [col.lower() for col in df.columns]
        
        logger.info(f"✓ Retrieved {len(df)} data points for {symbol}")
        
        return df
    
    def get_intraday_data(
        self,
        symbol: str,
        interval: str = '5min',
        outputsize: int = 390  # Full trading day at 5min intervals
    ) -> pd.DataFrame:
        """
        Get intraday price data.
        
        Args:
            symbol: Stock ticker
            interval: Intraday interval ('1min', '5min', '15min', '30min')
            outputsize: Number of data points
            
        Returns:
            DataFrame with intraday OHLCV data
        """
        return self.get_time_series(symbol, interval=interval, outputsize=outputsize)
    
    # ==================== TECHNICAL INDICATORS ====================
    
    def get_technical_indicator(
        self,
        symbol: str,
        indicator: str,
        interval: str = '1day',
        outputsize: int = 100,
        **kwargs
    ) -> pd.DataFrame:
        """
        Get technical indicator data.
        
        Args:
            symbol: Stock ticker
            indicator: Indicator name (e.g., 'rsi', 'macd', 'bbands', 'sma', 'ema')
            interval: Time interval
            outputsize: Number of data points
            **kwargs: Additional indicator parameters (e.g., time_period=14 for RSI)
            
        Returns:
            DataFrame with indicator values
        """
        logger.info(f"Fetching {indicator.upper()} for {symbol} from Twelve Data")
        
        params = {
            'symbol': symbol,
            'interval': interval,
            'outputsize': min(outputsize, 5000),
            'order': 'asc',
            **kwargs
        }
        
        response = self._make_request(f'/{indicator}', params)
        
        if 'error' in response:
            logger.error(f"Indicator error: {response.get('message')}")
            return pd.DataFrame()
        
        if 'values' not in response:
            logger.warning(f"No indicator data returned for {symbol}")
            return pd.DataFrame()
        
        df = pd.DataFrame(response['values'])
        
        if df.empty:
            return df
        
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        
        # Convert numeric columns
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def get_rsi(self, symbol: str, interval: str = '1day', time_period: int = 14, outputsize: int = 100) -> pd.DataFrame:
        """Get RSI indicator."""
        return self.get_technical_indicator(symbol, 'rsi', interval, outputsize, time_period=time_period)
    
    def get_macd(self, symbol: str, interval: str = '1day', outputsize: int = 100) -> pd.DataFrame:
        """Get MACD indicator."""
        return self.get_technical_indicator(symbol, 'macd', interval, outputsize)
    
    def get_bbands(self, symbol: str, interval: str = '1day', time_period: int = 20, outputsize: int = 100) -> pd.DataFrame:
        """Get Bollinger Bands."""
        return self.get_technical_indicator(symbol, 'bbands', interval, outputsize, time_period=time_period)
    
    def get_sma(self, symbol: str, interval: str = '1day', time_period: int = 20, outputsize: int = 100) -> pd.DataFrame:
        """Get Simple Moving Average."""
        return self.get_technical_indicator(symbol, 'sma', interval, outputsize, time_period=time_period)
    
    def get_ema(self, symbol: str, interval: str = '1day', time_period: int = 20, outputsize: int = 100) -> pd.DataFrame:
        """Get Exponential Moving Average."""
        return self.get_technical_indicator(symbol, 'ema', interval, outputsize, time_period=time_period)
    
    def get_atr(self, symbol: str, interval: str = '1day', time_period: int = 14, outputsize: int = 100) -> pd.DataFrame:
        """Get Average True Range."""
        return self.get_technical_indicator(symbol, 'atr', interval, outputsize, time_period=time_period)
    
    def get_adx(self, symbol: str, interval: str = '1day', time_period: int = 14, outputsize: int = 100) -> pd.DataFrame:
        """Get Average Directional Index."""
        return self.get_technical_indicator(symbol, 'adx', interval, outputsize, time_period=time_period)
    
    def get_stoch(self, symbol: str, interval: str = '1day', outputsize: int = 100) -> pd.DataFrame:
        """Get Stochastic Oscillator."""
        return self.get_technical_indicator(symbol, 'stoch', interval, outputsize)
    
    # ==================== MARKET DATA ====================
    
    def get_market_movers(self, direction: str = 'gainers', outputsize: int = 20) -> List[Dict]:
        """
        Get top market movers (gainers or losers).
        
        Args:
            direction: 'gainers' or 'losers'
            outputsize: Number of results
            
        Returns:
            List of market movers with symbol, change, percent_change
        """
        logger.info(f"Fetching market {direction} from Twelve Data")
        
        response = self._make_request('/market_movers/stocks', {
            'direction': direction,
            'outputsize': outputsize
        })
        
        if 'error' in response:
            return []
        
        if 'values' in response:
            return response['values']
        
        return []
    
    def get_earnings_calendar(self, symbol: str = None) -> List[Dict]:
        """
        Get earnings calendar.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            List of upcoming earnings
        """
        params = {}
        if symbol:
            params['symbol'] = symbol
        
        response = self._make_request('/earnings_calendar', params)
        
        if 'error' in response:
            return []
        
        if 'earnings' in response:
            return response['earnings']
        
        return []
    
    # ==================== UTILITY METHODS ====================
    
    def get_stock_data_for_analysis(
        self,
        symbol: str,
        lookback_days: int = 252
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Get comprehensive stock data for analysis (price + quote).
        
        This is the main method for integration with the trading system.
        
        Args:
            symbol: Stock ticker
            lookback_days: Number of days of historical data
            
        Returns:
            Tuple of (price_dataframe, quote_dict)
        """
        logger.info(f"Fetching comprehensive data for {symbol} from Twelve Data")
        
        # Get historical data
        df = self.get_time_series(symbol, interval='1day', outputsize=lookback_days)
        
        # Get current quote
        quote = self.get_quote(symbol)
        
        if df.empty:
            logger.warning(f"No historical data available for {symbol}")
        
        if 'error' in quote:
            logger.warning(f"No quote data available for {symbol}")
            quote = {}
        
        return df, quote
    
    def health_check(self) -> bool:
        """
        Check if the API is accessible and the key is valid.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            response = self.get_price('AAPL')
            return 'price' in response
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False


# ==================== STANDALONE TEST ====================

def main():
    """Test the Twelve Data client."""
    print("\n" + "=" * 60)
    print("TWELVE DATA CLIENT TEST")
    print("=" * 60)
    
    client = TwelveDataClient()
    
    # Test health check
    print("\n1. Health Check...")
    if client.health_check():
        print("   ✓ API is healthy")
    else:
        print("   ✗ API health check failed")
        return
    
    # Test price
    print("\n2. Latest Price (AAPL)...")
    price = client.get_price('AAPL')
    if 'price' in price:
        print(f"   ✓ AAPL Price: ${price['price']:.2f}")
    else:
        print(f"   ✗ Error: {price.get('message')}")
    
    # Test quote
    print("\n3. Quote (AAPL)...")
    quote = client.get_quote('AAPL')
    if 'symbol' in quote:
        print(f"   ✓ {quote['name']}")
        print(f"     Open: ${quote['open']:.2f}")
        print(f"     High: ${quote['high']:.2f}")
        print(f"     Low: ${quote['low']:.2f}")
        print(f"     Close: ${quote['close']:.2f}")
        print(f"     Volume: {quote['volume']:,}")
        print(f"     Change: {quote['percent_change']:.2f}%")
    else:
        print(f"   ✗ Error: {quote.get('message')}")
    
    # Test time series
    print("\n4. Time Series (AAPL, 30 days)...")
    df = client.get_time_series('AAPL', interval='1day', outputsize=30)
    if not df.empty:
        print(f"   ✓ Retrieved {len(df)} data points")
        print(f"     Date range: {df.index[0]} to {df.index[-1]}")
        print(f"     Latest close: ${df['close'].iloc[-1]:.2f}")
    else:
        print("   ✗ No data returned")
    
    # Test RSI
    print("\n5. RSI Indicator (AAPL)...")
    rsi = client.get_rsi('AAPL', outputsize=10)
    if not rsi.empty:
        print(f"   ✓ Latest RSI: {rsi['rsi'].iloc[-1]:.2f}")
    else:
        print("   ✗ No RSI data returned")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
