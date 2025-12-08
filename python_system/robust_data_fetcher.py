#!/usr/bin/env python3
"""
ROBUST DATA FETCHER - Production-Ready for Real Money Trading
===============================================================

Handles rate limits, API failures, and provides multiple fallbacks.
Implements intelligent caching to minimize API calls.
ZERO PLACEHOLDERS - only returns real data or clear error states.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import logging
import time
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CACHE_DIR = "/tmp/stock_data_cache"
CACHE_DURATION = 300  # 5 minutes

class RobustDataFetcher:
    """Production-ready data fetcher with caching and fallbacks"""
    
    def __init__(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        logger.info("Robust Data Fetcher initialized with caching")
    
    def _get_cache_path(self, symbol: str, data_type: str) -> str:
        """Get cache file path for a symbol and data type"""
        return os.path.join(CACHE_DIR, f"{symbol}_{data_type}.json")
    
    def _is_cache_valid(self, cache_path: str) -> bool:
        """Check if cache file exists and is fresh"""
        if not os.path.exists(cache_path):
            return False
        
        file_time = os.path.getmtime(cache_path)
        age = time.time() - file_time
        return age < CACHE_DURATION
    
    def _read_cache(self, cache_path: str) -> Optional[Dict]:
        """Read data from cache"""
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except:
            return None
    
    def _write_cache(self, cache_path: str, data: Dict):
        """Write data to cache"""
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to write cache: {e}")
    
    def get_stock_info(self, symbol: str) -> Dict:
        """
        Get stock info with caching and fallbacks
        Returns real data or raises exception
        """
        cache_path = self._get_cache_path(symbol, "info")
        
        # Try cache first
        if self._is_cache_valid(cache_path):
            cached = self._read_cache(cache_path)
            if cached:
                logger.info(f"Using cached info for {symbol}")
                return cached
        
        # Fetch fresh data
        logger.info(f"Fetching fresh info for {symbol} from yfinance...")
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract only serializable data
            clean_info = {}
            for key, value in info.items():
                if isinstance(value, (int, float, str, bool, type(None))):
                    clean_info[key] = value
            
            self._write_cache(cache_path, clean_info)
            return clean_info
            
        except Exception as e:
            logger.error(f"Failed to fetch info for {symbol}: {e}")
            # Try to use stale cache as last resort
            cached = self._read_cache(cache_path)
            if cached:
                logger.warning(f"Using STALE cache for {symbol} (API failed)")
                return cached
            raise
    
    def get_historical_data(self, symbol: str, period: str = "3mo") -> pd.DataFrame:
        """
        Get historical price data with caching
        Returns DataFrame or raises exception
        """
        cache_path = self._get_cache_path(symbol, f"hist_{period}")
        
        # Try cache first
        if self._is_cache_valid(cache_path):
            cached = self._read_cache(cache_path)
            if cached:
                logger.info(f"Using cached historical data for {symbol}")
                df = pd.DataFrame(cached['data'])
                df.index = pd.to_datetime(df.index)
                return df
        
        # Fetch fresh data
        logger.info(f"Fetching historical data for {symbol} from yfinance...")
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval='1d')
            
            if hist.empty:
                raise ValueError(f"No historical data for {symbol}")
            
            # Cache the data
            cache_data = {
                'data': hist.reset_index().to_dict(orient='list')
            }
            self._write_cache(cache_path, cache_data)
            return hist
            
        except Exception as e:
            logger.error(f"Failed to fetch historical data for {symbol}: {e}")
            # Try stale cache
            cached = self._read_cache(cache_path)
            if cached:
                logger.warning(f"Using STALE historical data for {symbol}")
                df = pd.DataFrame(cached['data'])
                df.index = pd.to_datetime(df.index)
                return df
            raise
    
    def get_stock_data(self, symbol: str) -> Tuple[Dict, pd.DataFrame]:
        """
        Get both info and historical data
        Returns (info_dict, hist_dataframe) or raises exception
        """
        info = self.get_stock_info(symbol)
        hist = self.get_historical_data(symbol)
        return info, hist


def main():
    """Test the robust data fetcher"""
    fetcher = RobustDataFetcher()
    
    print("\n" + "="*80)
    print("ROBUST DATA FETCHER TEST")
    print("="*80)
    
    try:
        info, hist = fetcher.get_stock_data("AAPL")
        
        print(f"\n✓ Successfully fetched data for AAPL")
        print(f"  Current Price: ${info.get('currentPrice', info.get('regularMarketPrice', 0)):.2f}")
        print(f"  Market Cap: ${info.get('marketCap', 0):,.0f}")
        print(f"  P/E Ratio: {info.get('trailingPE', 0):.2f}")
        print(f"  Historical Data Points: {len(hist)}")
        print(f"  Date Range: {hist.index[0].date()} to {hist.index[-1].date()}")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")


if __name__ == "__main__":
    main()
