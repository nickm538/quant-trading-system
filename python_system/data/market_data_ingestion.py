"""
Market Data Ingestion Module
Handles real-time and historical market data acquisition from multiple sources
"""

import sys
sys.path.append('/opt/.manus/.sandbox-runtime')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
import yfinance as yf
from data_api import ApiClient
import json
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MarketDataIngestion:
    """
    Comprehensive market data ingestion system supporting multiple data sources
    """
    
    def __init__(self, cache_dir: str = '/home/ubuntu/quant_trading_system/data/cache'):
        """
        Initialize the market data ingestion system
        
        Args:
            cache_dir: Directory for caching data
        """
        self.api_client = ApiClient()
        self.cache_dir = cache_dir
        self.data_cache = {}
        
        # Create cache directory if it doesn't exist
        import os
        os.makedirs(cache_dir, exist_ok=True)
        
        logger.info("MarketDataIngestion initialized")
    
    def get_stock_data(
        self,
        symbol: str,
        interval: str = '1d',
        period: str = '2y',
        include_extended: bool = True
    ) -> pd.DataFrame:
        """
        Get comprehensive stock data from Yahoo Finance via Manus API
        
        Args:
            symbol: Stock ticker symbol
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 1d, 1wk, 1mo)
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            include_extended: Include pre/post market data
            
        Returns:
            DataFrame with OHLCV data and metadata
        """
        cache_key = f"{symbol}_{interval}_{period}"
        
        try:
            logger.info(f"Fetching data for {symbol} ({interval}, {period})")
            
            response = self.api_client.call_api('YahooFinance/get_stock_chart', query={
                'symbol': symbol,
                'region': 'US',
                'interval': interval,
                'range': period,
                'includeAdjustedClose': True,
                'events': 'div,split'
            })
            
            if not response or 'chart' not in response or 'result' not in response['chart']:
                logger.error(f"Invalid response for {symbol}")
                return pd.DataFrame()
            
            result = response['chart']['result'][0]
            meta = result['meta']
            timestamps = result['timestamp']
            quotes = result['indicators']['quote'][0]
            
            # Create DataFrame
            df = pd.DataFrame({
                'timestamp': pd.to_datetime(timestamps, unit='s'),
                'open': quotes['open'],
                'high': quotes['high'],
                'low': quotes['low'],
                'close': quotes['close'],
                'volume': quotes['volume']
            })
            
            # Add adjusted close if available
            if 'adjclose' in result['indicators']:
                df['adj_close'] = result['indicators']['adjclose'][0]['adjclose']
            else:
                df['adj_close'] = df['close']
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            # Remove rows with all NaN values
            df.dropna(how='all', inplace=True)
            
            # Add metadata as attributes
            df.attrs['symbol'] = symbol
            df.attrs['currency'] = meta.get('currency', 'USD')
            df.attrs['exchange'] = meta.get('exchangeName', 'Unknown')
            df.attrs['timezone'] = meta.get('timezone', 'America/New_York')
            df.attrs['current_price'] = meta.get('regularMarketPrice', None)
            df.attrs['previous_close'] = meta.get('previousClose', None)
            
            # Cache the data
            self.data_cache[cache_key] = df
            
            logger.info(f"Successfully fetched {len(df)} data points for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_multiple_stocks(
        self,
        symbols: List[str],
        interval: str = '1d',
        period: str = '2y',
        max_workers: int = 5
    ) -> Dict[str, pd.DataFrame]:
        """
        Get data for multiple stocks in parallel
        
        Args:
            symbols: List of stock ticker symbols
            interval: Data interval
            period: Time period
            max_workers: Maximum number of parallel workers
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(self.get_stock_data, symbol, interval, period): symbol
                for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    df = future.result()
                    if not df.empty:
                        results[symbol] = df
                        logger.info(f"Successfully fetched data for {symbol}")
                    else:
                        logger.warning(f"No data returned for {symbol}")
                except Exception as e:
                    logger.error(f"Error fetching {symbol}: {str(e)}")
        
        return results
    
    def get_stock_insights(self, symbol: str) -> Dict:
        """
        Get comprehensive stock insights including technical indicators and fundamentals
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with stock insights
        """
        try:
            logger.info(f"Fetching insights for {symbol}")
            
            response = self.api_client.call_api('YahooFinance/get_stock_insights', query={
                'symbol': symbol
            })
            
            if response:
                logger.info(f"Successfully fetched insights for {symbol}")
                return response
            else:
                logger.warning(f"No insights returned for {symbol}")
                return {}
                
        except Exception as e:
            logger.error(f"Error fetching insights for {symbol}: {str(e)}")
            return {}
    
    def get_intraday_data(
        self,
        symbol: str,
        interval: str = '1m',
        days: int = 5
    ) -> pd.DataFrame:
        """
        Get high-frequency intraday data
        
        Args:
            symbol: Stock ticker symbol
            interval: Minute-level interval (1m, 2m, 5m, 15m, 30m, 60m)
            days: Number of days of data (max 7 for minute data)
            
        Returns:
            DataFrame with intraday data
        """
        period_map = {1: '1d', 2: '2d', 5: '5d', 7: '7d'}
        period = period_map.get(days, '5d')
        
        return self.get_stock_data(symbol, interval=interval, period=period)
    
    def get_historical_data_yfinance(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = '5y'
    ) -> pd.DataFrame:
        """
        Get historical data using yfinance library (backup method)
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            period: Period if dates not specified
            
        Returns:
            DataFrame with historical data
        """
        try:
            logger.info(f"Fetching historical data for {symbol} using yfinance")
            
            ticker = yf.Ticker(symbol)
            
            if start_date and end_date:
                df = ticker.history(start=start_date, end=end_date)
            else:
                df = ticker.history(period=period)
            
            # Standardize column names
            df.columns = df.columns.str.lower()
            df.rename(columns={'adj close': 'adj_close'}, inplace=True)
            
            logger.info(f"Successfully fetched {len(df)} data points for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_fundamental_data(self, symbol: str) -> Dict:
        """
        Get fundamental data for a stock
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with fundamental data
        """
        try:
            logger.info(f"Fetching fundamental data for {symbol}")
            
            ticker = yf.Ticker(symbol)
            
            fundamentals = {
                'info': ticker.info,
                'financials': ticker.financials.to_dict() if hasattr(ticker, 'financials') and ticker.financials is not None else {},
                'balance_sheet': ticker.balance_sheet.to_dict() if hasattr(ticker, 'balance_sheet') and ticker.balance_sheet is not None else {},
                'cash_flow': ticker.cashflow.to_dict() if hasattr(ticker, 'cashflow') and ticker.cashflow is not None else {},
                'earnings': ticker.earnings.to_dict() if hasattr(ticker, 'earnings') and ticker.earnings is not None else {},
                'recommendations': ticker.recommendations.to_dict() if hasattr(ticker, 'recommendations') and ticker.recommendations is not None else {}
            }
            
            logger.info(f"Successfully fetched fundamental data for {symbol}")
            return fundamentals
            
        except Exception as e:
            logger.error(f"Error fetching fundamental data for {symbol}: {str(e)}")
            return {}
    
    def get_options_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Get options chain data for a stock
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with calls and puts DataFrames
        """
        try:
            logger.info(f"Fetching options data for {symbol}")
            
            ticker = yf.Ticker(symbol)
            
            # Get available expiration dates
            expirations = ticker.options
            
            if not expirations:
                logger.warning(f"No options data available for {symbol}")
                return {}
            
            # Get options for the nearest expiration
            opt = ticker.option_chain(expirations[0])
            
            options_data = {
                'calls': opt.calls,
                'puts': opt.puts,
                'expiration': expirations[0],
                'all_expirations': list(expirations)
            }
            
            logger.info(f"Successfully fetched options data for {symbol}")
            return options_data
            
        except Exception as e:
            logger.error(f"Error fetching options data for {symbol}: {str(e)}")
            return {}
    
    def get_market_indices(self) -> Dict[str, pd.DataFrame]:
        """
        Get data for major market indices
        
        Returns:
            Dictionary with index data
        """
        indices = {
            'SPY': 'S&P 500',
            'QQQ': 'NASDAQ 100',
            'DIA': 'Dow Jones',
            'IWM': 'Russell 2000',
            'VIX': 'Volatility Index'
        }
        
        logger.info("Fetching market indices data")
        return self.get_multiple_stocks(list(indices.keys()), period='1y')
    
    def save_to_cache(self, symbol: str, data: pd.DataFrame, suffix: str = '') -> str:
        """
        Save data to cache file
        
        Args:
            symbol: Stock symbol
            data: DataFrame to save
            suffix: Optional suffix for filename
            
        Returns:
            Path to saved file
        """
        filename = f"{symbol}_{suffix}_{datetime.now().strftime('%Y%m%d')}.parquet"
        filepath = f"{self.cache_dir}/{filename}"
        
        try:
            data.to_parquet(filepath)
            logger.info(f"Saved data to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving data to cache: {str(e)}")
            return ""
    
    def load_from_cache(self, symbol: str, suffix: str = '', date: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from cache file
        
        Args:
            symbol: Stock symbol
            suffix: Optional suffix for filename
            date: Date string (YYYYMMDD), defaults to today
            
        Returns:
            Cached DataFrame or empty DataFrame if not found
        """
        if date is None:
            date = datetime.now().strftime('%Y%m%d')
        
        filename = f"{symbol}_{suffix}_{date}.parquet"
        filepath = f"{self.cache_dir}/{filename}"
        
        try:
            if os.path.exists(filepath):
                df = pd.read_parquet(filepath)
                logger.info(f"Loaded data from {filepath}")
                return df
            else:
                logger.warning(f"Cache file not found: {filepath}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading data from cache: {str(e)}")
            return pd.DataFrame()


def test_market_data_ingestion():
    """
    Test the market data ingestion system
    """
    print("=" * 80)
    print("Testing Market Data Ingestion System")
    print("=" * 80)
    
    ingestion = MarketDataIngestion()
    
    # Test 1: Single stock data
    print("\n1. Testing single stock data (AAPL)...")
    aapl_data = ingestion.get_stock_data('AAPL', interval='1d', period='1y')
    if not aapl_data.empty:
        print(f"✓ Successfully fetched {len(aapl_data)} data points")
        print(f"  Date range: {aapl_data.index[0]} to {aapl_data.index[-1]}")
        print(f"  Columns: {list(aapl_data.columns)}")
        print(f"  Current price: ${aapl_data.attrs.get('current_price', 'N/A')}")
        print("\nLast 5 rows:")
        print(aapl_data.tail())
    else:
        print("✗ Failed to fetch data")
    
    # Test 2: Multiple stocks
    print("\n2. Testing multiple stocks (AAPL, MSFT, GOOGL)...")
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    multi_data = ingestion.get_multiple_stocks(symbols, period='6mo')
    print(f"✓ Successfully fetched data for {len(multi_data)}/{len(symbols)} stocks")
    for symbol, df in multi_data.items():
        print(f"  {symbol}: {len(df)} data points")
    
    # Test 3: Stock insights
    print("\n3. Testing stock insights (AAPL)...")
    insights = ingestion.get_stock_insights('AAPL')
    if insights:
        print(f"✓ Successfully fetched insights")
        print(f"  Keys: {list(insights.keys())[:5]}...")
    else:
        print("✗ No insights returned")
    
    # Test 4: Fundamental data
    print("\n4. Testing fundamental data (AAPL)...")
    fundamentals = ingestion.get_fundamental_data('AAPL')
    if fundamentals and 'info' in fundamentals:
        info = fundamentals['info']
        print(f"✓ Successfully fetched fundamental data")
        print(f"  Company: {info.get('longName', 'N/A')}")
        print(f"  Sector: {info.get('sector', 'N/A')}")
        print(f"  Market Cap: ${info.get('marketCap', 0):,.0f}")
        print(f"  P/E Ratio: {info.get('trailingPE', 'N/A')}")
        print(f"  Beta: {info.get('beta', 'N/A')}")
    else:
        print("✗ Failed to fetch fundamental data")
    
    # Test 5: Market indices
    print("\n5. Testing market indices...")
    indices = ingestion.get_market_indices()
    print(f"✓ Successfully fetched {len(indices)} indices")
    for symbol, df in indices.items():
        if not df.empty:
            print(f"  {symbol}: {len(df)} data points")
    
    print("\n" + "=" * 80)
    print("Market Data Ingestion Tests Complete")
    print("=" * 80)


if __name__ == "__main__":
    test_market_data_ingestion()
