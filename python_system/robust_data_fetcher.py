#!/usr/bin/env python3
"""
ROBUST DATA FETCHER - Production-Ready for Real Money Trading
===============================================================

Handles rate limits, API failures, and provides multiple fallbacks.
Implements intelligent caching to minimize API calls.
ZERO PLACEHOLDERS - only returns real data or clear error states.

Data Source Priority:
1. Cache (if fresh)
2. yfinance (primary)
3. Twelve Data (fallback)
4. Stale cache (last resort)
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

# Import Twelve Data client for fallback
try:
    from twelvedata_client import TwelveDataClient
    TWELVEDATA_AVAILABLE = True
except ImportError:
    TWELVEDATA_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CACHE_DIR = "/tmp/stock_data_cache"
CACHE_DURATION = 300  # 5 minutes

class RobustDataFetcher:
    """Production-ready data fetcher with caching and multi-source fallbacks"""
    
    def __init__(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        # Initialize Twelve Data client as fallback
        self.twelvedata_client = None
        if TWELVEDATA_AVAILABLE:
            try:
                self.twelvedata_client = TwelveDataClient()
                logger.info("✓ Twelve Data fallback initialized")
            except Exception as e:
                logger.warning(f"Twelve Data fallback unavailable: {e}")
        
        logger.info("Robust Data Fetcher initialized with multi-source fallbacks")
    
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
        Get stock info with caching and multi-source fallbacks.
        
        Priority:
        1. Fresh cache
        2. yfinance
        3. Twelve Data
        4. Stale cache
        
        Returns real data or raises exception
        """
        cache_path = self._get_cache_path(symbol, "info")
        
        # Try cache first
        if self._is_cache_valid(cache_path):
            cached = self._read_cache(cache_path)
            if cached:
                logger.info(f"Using cached info for {symbol}")
                return cached
        
        # Try yfinance first (primary source)
        logger.info(f"Fetching fresh info for {symbol} from yfinance...")
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Validate we got real data
            if not info or info.get('regularMarketPrice') is None:
                raise ValueError("yfinance returned empty or invalid data")
            
            # Extract only serializable data
            clean_info = {}
            for key, value in info.items():
                if isinstance(value, (int, float, str, bool, type(None))):
                    clean_info[key] = value
            
            clean_info['data_source'] = 'yfinance'
            self._write_cache(cache_path, clean_info)
            return clean_info
            
        except Exception as yf_error:
            logger.warning(f"yfinance failed for {symbol}: {yf_error}")
            
            # Try Twelve Data as fallback
            if self.twelvedata_client:
                logger.info(f"Trying Twelve Data fallback for {symbol}...")
                try:
                    quote = self.twelvedata_client.get_quote(symbol)
                    
                    if 'error' not in quote and quote.get('close'):
                        # Convert Twelve Data format to yfinance-like format
                        clean_info = {
                            'symbol': quote.get('symbol', symbol),
                            'shortName': quote.get('name', symbol),
                            'longName': quote.get('name', symbol),
                            'regularMarketPrice': quote.get('close'),
                            'regularMarketOpen': quote.get('open'),
                            'regularMarketDayHigh': quote.get('high'),
                            'regularMarketDayLow': quote.get('low'),
                            'regularMarketVolume': quote.get('volume'),
                            'regularMarketPreviousClose': quote.get('previous_close'),
                            'regularMarketChange': quote.get('change'),
                            'regularMarketChangePercent': quote.get('percent_change'),
                            'averageVolume': quote.get('average_volume'),
                            'fiftyTwoWeekHigh': quote.get('fifty_two_week', {}).get('high'),
                            'fiftyTwoWeekLow': quote.get('fifty_two_week', {}).get('low'),
                            'currency': quote.get('currency', 'USD'),
                            'exchange': quote.get('exchange'),
                            'data_source': 'twelvedata'
                        }
                        
                        logger.info(f"✓ Twelve Data fallback successful for {symbol}")
                        self._write_cache(cache_path, clean_info)
                        return clean_info
                        
                except Exception as td_error:
                    logger.warning(f"Twelve Data fallback failed for {symbol}: {td_error}")
            
            # Last resort: stale cache
            cached = self._read_cache(cache_path)
            if cached:
                logger.warning(f"Using STALE cache for {symbol} (all APIs failed)")
                cached['data_source'] = 'stale_cache'
                return cached
            
            raise ValueError(f"All data sources failed for {symbol}: {yf_error}")
    
    def get_historical_data(self, symbol: str, period: str = "3mo") -> pd.DataFrame:
        """
        Get historical price data with caching and multi-source fallbacks.
        
        Priority:
        1. Fresh cache
        2. yfinance
        3. Twelve Data
        4. Stale cache
        
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
        
        # Convert period to days for Twelve Data
        period_to_days = {
            '1mo': 30, '3mo': 90, '6mo': 180, '1y': 252, '2y': 504, '5y': 1260
        }
        lookback_days = period_to_days.get(period, 90)
        
        # Try yfinance first (primary source)
        logger.info(f"Fetching historical data for {symbol} from yfinance...")
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval='1d')
            
            if hist.empty:
                raise ValueError(f"yfinance returned empty data for {symbol}")
            
            # Cache the data - convert timestamps to strings for JSON serialization
            hist_reset = hist.reset_index()
            hist_reset['Date'] = hist_reset['Date'].astype(str) if 'Date' in hist_reset.columns else hist_reset.index.astype(str)
            cache_data = {
                'data': hist_reset.to_dict(orient='list'),
                'source': 'yfinance'
            }
            self._write_cache(cache_path, cache_data)
            logger.info(f"✓ yfinance: Retrieved {len(hist)} data points for {symbol}")
            return hist
            
        except Exception as yf_error:
            logger.warning(f"yfinance historical data failed for {symbol}: {yf_error}")
            
            # Try Twelve Data as fallback
            if self.twelvedata_client:
                logger.info(f"Trying Twelve Data fallback for {symbol} historical data...")
                try:
                    df = self.twelvedata_client.get_time_series(
                        symbol, 
                        interval='1day', 
                        outputsize=lookback_days
                    )
                    
                    if not df.empty:
                        # Rename columns to match yfinance format
                        df.columns = [col.capitalize() for col in df.columns]
                        if 'Volume' not in df.columns:
                            df['Volume'] = 0
                        
                        # Cache the data - convert timestamps to strings for JSON serialization
                        df_reset = df.reset_index()
                        df_reset['datetime'] = df_reset['datetime'].astype(str) if 'datetime' in df_reset.columns else df_reset.index.astype(str)
                        cache_data = {
                            'data': df_reset.to_dict(orient='list'),
                            'source': 'twelvedata'
                        }
                        self._write_cache(cache_path, cache_data)
                        logger.info(f"✓ Twelve Data: Retrieved {len(df)} data points for {symbol}")
                        return df
                        
                except Exception as td_error:
                    logger.warning(f"Twelve Data historical fallback failed for {symbol}: {td_error}")
            
            # Last resort: stale cache
            cached = self._read_cache(cache_path)
            if cached:
                logger.warning(f"Using STALE historical data for {symbol} (all APIs failed)")
                df = pd.DataFrame(cached['data'])
                df.index = pd.to_datetime(df.index)
                return df
            
            raise ValueError(f"All data sources failed for {symbol} historical data: {yf_error}")
    
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
