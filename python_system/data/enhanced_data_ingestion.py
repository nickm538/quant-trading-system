"""
Enhanced Data Ingestion Module - Institutional Grade
Integrates Yahoo Finance and Finnhub for comprehensive market data
Real-time data for actual trading - NO SIMULATIONS
"""

import sys
sys.path.append('/opt/.manus/.sandbox-runtime')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
import yfinance as yf
from data_api import ApiClient
import requests
import json
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedDataIngestion:
    """
    Institutional-grade data ingestion system
    Integrates multiple real-time data sources for actual trading
    """
    
    # Finnhub API credentials (hardcoded as per user request)
    FINNHUB_API_KEY = "d47ssnpr01qk80bicu4gd47ssnpr01qk80bicu50"  # UPDATED KEY
    FINNHUB_BASE_URL = "https://finnhub.io/api/v1"
    
    def __init__(self, cache_dir: str = '/home/ubuntu/quant_trading_system/data/cache'):
        """Initialize the enhanced data ingestion system"""
        self.api_client = ApiClient()
        self.cache_dir = cache_dir
        self.data_cache = {}
        
        os.makedirs(cache_dir, exist_ok=True)
        logger.info("EnhancedDataIngestion initialized with Yahoo Finance + Finnhub")
    
    # ==================== YAHOO FINANCE INTEGRATION ====================
    
    def get_stock_data_yahoo(
        self,
        symbol: str,
        interval: str = '1d',
        period: str = '2y'
    ) -> pd.DataFrame:
        """
        Get comprehensive stock data from Yahoo Finance via Manus API
        REAL DATA - NO SIMULATIONS
        """
        try:
            logger.info(f"Fetching REAL data for {symbol} from Yahoo Finance")
            
            response = self.api_client.call_api('YahooFinance/get_stock_chart', query={
                'symbol': symbol,
                'region': 'US',
                'interval': interval,
                'range': period,
                'includeAdjustedClose': True,
                'events': 'div,split'
            })
            
            if not response or 'chart' not in response:
                logger.error(f"Invalid response for {symbol}")
                return pd.DataFrame()
            
            result = response['chart']['result'][0]
            meta = result['meta']
            timestamps = result['timestamp']
            quotes = result['indicators']['quote'][0]
            
            df = pd.DataFrame({
                'timestamp': pd.to_datetime(timestamps, unit='s'),
                'open': quotes['open'],
                'high': quotes['high'],
                'low': quotes['low'],
                'close': quotes['close'],
                'volume': quotes['volume']
            })
            
            if 'adjclose' in result['indicators']:
                df['adj_close'] = result['indicators']['adjclose'][0]['adjclose']
            else:
                df['adj_close'] = df['close']
            
            df.set_index('timestamp', inplace=True)
            df.dropna(how='all', inplace=True)
            
            # Store metadata
            df.attrs['symbol'] = symbol
            df.attrs['currency'] = meta.get('currency', 'USD')
            df.attrs['exchange'] = meta.get('exchangeName', 'Unknown')
            df.attrs['current_price'] = meta.get('regularMarketPrice', None)
            df.attrs['previous_close'] = meta.get('previousClose', None)
            
            logger.info(f"✓ Fetched {len(df)} REAL data points for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching Yahoo data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    # ==================== FINNHUB INTEGRATION ====================
    
    def _finnhub_request(self, endpoint: str, params: Dict = None) -> Dict:
        """
        Make request to Finnhub API with proper error handling
        """
        if params is None:
            params = {}
        
        params['token'] = self.FINNHUB_API_KEY
        url = f"{self.FINNHUB_BASE_URL}/{endpoint}"
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Finnhub API error for {endpoint}: {str(e)}")
            return {}
    
    def get_stock_profile_finnhub(self, symbol: str) -> Dict:
        """
        Get comprehensive stock profile from Finnhub
        REAL COMPANY DATA
        """
        try:
            logger.info(f"Fetching REAL profile for {symbol} from Finnhub")
            
            # Stock profile endpoint
            profile = self._finnhub_request('stock/profile2', {'symbol': symbol})
            
            if profile:
                logger.info(f"✓ Fetched profile for {symbol}: {profile.get('name', 'N/A')}")
            
            return profile
            
        except Exception as e:
            logger.error(f"Error fetching Finnhub profile for {symbol}: {str(e)}")
            return {}
    
    def get_stock_metrics_finnhub(self, symbol: str) -> Dict:
        """
        Get real-time stock metrics from Finnhub
        REAL MARKET DATA
        """
        try:
            logger.info(f"Fetching REAL metrics for {symbol} from Finnhub")
            
            # Basic financials
            metrics = self._finnhub_request('stock/metric', {
                'symbol': symbol,
                'metric': 'all'
            })
            
            if metrics:
                logger.info(f"✓ Fetched metrics for {symbol}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error fetching Finnhub metrics for {symbol}: {str(e)}")
            return {}
    
    def get_company_news_finnhub(
        self,
        symbol: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None
    ) -> List[Dict]:
        """
        Get real company news from Finnhub
        REAL NEWS DATA FOR CATALYST DETECTION
        """
        try:
            if from_date is None:
                from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            if to_date is None:
                to_date = datetime.now().strftime('%Y-%m-%d')
            
            logger.info(f"Fetching REAL news for {symbol} from Finnhub")
            
            news = self._finnhub_request('company-news', {
                'symbol': symbol,
                'from': from_date,
                'to': to_date
            })
            
            if news:
                logger.info(f"✓ Fetched {len(news)} news articles for {symbol}")
            
            return news if isinstance(news, list) else []
            
        except Exception as e:
            logger.error(f"Error fetching Finnhub news for {symbol}: {str(e)}")
            return []
    
    def get_earnings_calendar_finnhub(self, symbol: str) -> List[Dict]:
        """
        Get earnings calendar from Finnhub
        REAL CATALYST EVENT DATA
        """
        try:
            logger.info(f"Fetching REAL earnings calendar for {symbol} from Finnhub")
            
            # Get earnings calendar
            earnings = self._finnhub_request('calendar/earnings', {'symbol': symbol})
            
            if earnings and 'earningsCalendar' in earnings:
                events = earnings['earningsCalendar']
                logger.info(f"✓ Fetched {len(events)} earnings events for {symbol}")
                return events
            
            return []
            
        except Exception as e:
            logger.error(f"Error fetching earnings calendar for {symbol}: {str(e)}")
            return []
    
    def get_insider_transactions_finnhub(self, symbol: str) -> pd.DataFrame:
        """
        Get insider trading transactions from Finnhub
        REAL INSIDER DATA
        """
        try:
            logger.info(f"Fetching REAL insider transactions for {symbol} from Finnhub")
            
            # Get insider transactions
            transactions = self._finnhub_request('stock/insider-transactions', {
                'symbol': symbol
            })
            
            if transactions and 'data' in transactions:
                df = pd.DataFrame(transactions['data'])
                logger.info(f"✓ Fetched {len(df)} insider transactions for {symbol}")
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching insider transactions for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_recommendation_trends_finnhub(self, symbol: str) -> pd.DataFrame:
        """
        Get analyst recommendation trends from Finnhub
        REAL ANALYST DATA
        """
        try:
            logger.info(f"Fetching REAL analyst recommendations for {symbol} from Finnhub")
            
            recommendations = self._finnhub_request('stock/recommendation', {
                'symbol': symbol
            })
            
            if recommendations:
                df = pd.DataFrame(recommendations)
                logger.info(f"✓ Fetched {len(df)} recommendation periods for {symbol}")
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching recommendations for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_price_target_finnhub(self, symbol: str) -> Dict:
        """
        Get analyst price targets from Finnhub
        REAL PRICE TARGET DATA
        """
        try:
            logger.info(f"Fetching REAL price targets for {symbol} from Finnhub")
            
            targets = self._finnhub_request('stock/price-target', {'symbol': symbol})
            
            if targets:
                logger.info(f"✓ Fetched price targets for {symbol}: Target ${targets.get('targetMean', 'N/A')}")
            
            return targets
            
        except Exception as e:
            logger.error(f"Error fetching price targets for {symbol}: {str(e)}")
            return {}
    
    # ==================== YFINANCE DIRECT INTEGRATION ====================
    
    def get_options_chain_yfinance(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Get REAL options chain data for options analysis
        Critical for 0.3-0.6 delta selection
        """
        try:
            logger.info(f"Fetching REAL options chain for {symbol}")
            
            ticker = yf.Ticker(symbol)
            expirations = ticker.options
            
            if not expirations:
                logger.warning(f"No options data for {symbol}")
                return {}
            
            # Get all expiration dates > 1 week out (as per user requirement)
            min_date = datetime.now() + timedelta(days=7)
            valid_expirations = [
                exp for exp in expirations
                if datetime.strptime(exp, '%Y-%m-%d') > min_date
            ]
            
            if not valid_expirations:
                logger.warning(f"No options with >1 week expiry for {symbol}")
                return {}
            
            options_data = {}
            for expiration in valid_expirations[:10]:  # Limit to first 10 expirations
                try:
                    opt = ticker.option_chain(expiration)
                    
                    # Filter for 0.3-0.6 delta range (medium risk/reward)
                    calls = opt.calls[
                        (opt.calls['delta'] >= 0.3) & 
                        (opt.calls['delta'] <= 0.6)
                    ].copy() if 'delta' in opt.calls.columns else opt.calls.copy()
                    
                    puts = opt.puts[
                        (opt.puts['delta'] >= -0.6) & 
                        (opt.puts['delta'] <= -0.3)
                    ].copy() if 'delta' in opt.puts.columns else opt.puts.copy()
                    
                    # Calculate OI/Volume ratio (critical metric)
                    if not calls.empty:
                        calls['oi_vol_ratio'] = calls['openInterest'] / (calls['volume'] + 1)
                        calls['expiration'] = expiration
                    
                    if not puts.empty:
                        puts['oi_vol_ratio'] = puts['openInterest'] / (puts['volume'] + 1)
                        puts['expiration'] = expiration
                    
                    options_data[expiration] = {
                        'calls': calls,
                        'puts': puts
                    }
                    
                    logger.info(f"✓ Fetched options for {expiration}: {len(calls)} calls, {len(puts)} puts")
                    
                except Exception as e:
                    logger.error(f"Error fetching options for {expiration}: {str(e)}")
                    continue
            
            return options_data
            
        except Exception as e:
            logger.error(f"Error fetching options chain for {symbol}: {str(e)}")
            return {}
    
    def get_fundamental_data_yfinance(self, symbol: str) -> Dict:
        """
        Get REAL fundamental data including cash flow sheets
        """
        try:
            logger.info(f"Fetching REAL fundamental data for {symbol}")
            
            ticker = yf.Ticker(symbol)
            
            fundamentals = {
                'info': ticker.info,
                'financials': ticker.financials.to_dict() if hasattr(ticker, 'financials') and ticker.financials is not None else {},
                'balance_sheet': ticker.balance_sheet.to_dict() if hasattr(ticker, 'balance_sheet') and ticker.balance_sheet is not None else {},
                'cash_flow': ticker.cashflow.to_dict() if hasattr(ticker, 'cashflow') and ticker.cashflow is not None else {},
                'earnings': ticker.earnings.to_dict() if hasattr(ticker, 'earnings') and ticker.earnings is not None else {},
                'quarterly_financials': ticker.quarterly_financials.to_dict() if hasattr(ticker, 'quarterly_financials') and ticker.quarterly_financials is not None else {},
                'quarterly_balance_sheet': ticker.quarterly_balance_sheet.to_dict() if hasattr(ticker, 'quarterly_balance_sheet') and ticker.quarterly_balance_sheet is not None else {},
                'quarterly_cashflow': ticker.quarterly_cashflow.to_dict() if hasattr(ticker, 'quarterly_cashflow') and ticker.quarterly_cashflow is not None else {}
            }
            
            logger.info(f"✓ Fetched comprehensive fundamental data for {symbol}")
            return fundamentals
            
        except Exception as e:
            logger.error(f"Error fetching fundamental data for {symbol}: {str(e)}")
            return {}
    
    # ==================== COMPREHENSIVE DATA AGGREGATION ====================
    
    def get_complete_stock_data(self, symbol: str) -> Dict:
        """
        Get ALL available data for a stock from all sources
        REAL COMPREHENSIVE DATA FOR INSTITUTIONAL ANALYSIS
        """
        logger.info(f"=" * 80)
        logger.info(f"Fetching COMPLETE REAL DATA for {symbol}")
        logger.info(f"=" * 80)
        
        complete_data = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'price_data': {},
            'profile': {},
            'metrics': {},
            'fundamentals': {},
            'options': {},
            'news': [],
            'earnings': [],
            'insider_trades': pd.DataFrame(),
            'analyst_recommendations': pd.DataFrame(),
            'price_targets': {}
        }
        
        # 1. Price data from Yahoo Finance
        complete_data['price_data'] = self.get_stock_data_yahoo(symbol, period='2y')
        
        # 2. Profile from Finnhub
        complete_data['profile'] = self.get_stock_profile_finnhub(symbol)
        
        # 3. Metrics from Finnhub
        complete_data['metrics'] = self.get_stock_metrics_finnhub(symbol)
        
        # 4. Fundamentals from yfinance
        complete_data['fundamentals'] = self.get_fundamental_data_yfinance(symbol)
        
        # 5. Options chain from yfinance
        complete_data['options'] = self.get_options_chain_yfinance(symbol)
        
        # 6. News from Finnhub
        complete_data['news'] = self.get_company_news_finnhub(symbol)
        
        # 7. Earnings calendar from Finnhub
        complete_data['earnings'] = self.get_earnings_calendar_finnhub(symbol)
        
        # 8. Insider trades from Finnhub
        complete_data['insider_trades'] = self.get_insider_transactions_finnhub(symbol)
        
        # 9. Analyst recommendations from Finnhub
        complete_data['analyst_recommendations'] = self.get_recommendation_trends_finnhub(symbol)
        
        # 10. Price targets from Finnhub
        complete_data['price_targets'] = self.get_price_target_finnhub(symbol)
        
        logger.info(f"=" * 80)
        logger.info(f"✓ COMPLETE REAL DATA FETCHED for {symbol}")
        logger.info(f"=" * 80)
        
        return complete_data
    
    def get_multiple_stocks_complete(
        self,
        symbols: List[str],
        max_workers: int = 3
    ) -> Dict[str, Dict]:
        """
        Get complete data for multiple stocks in parallel
        """
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(self.get_complete_stock_data, symbol): symbol
                for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    data = future.result()
                    results[symbol] = data
                    logger.info(f"✓ Complete data fetched for {symbol}")
                except Exception as e:
                    logger.error(f"Error fetching complete data for {symbol}: {str(e)}")
        
        return results


def test_enhanced_data_ingestion():
    """
    Test the enhanced data ingestion system with REAL data
    """
    print("=" * 80)
    print("TESTING ENHANCED DATA INGESTION - REAL DATA ONLY")
    print("=" * 80)
    
    ingestion = EnhancedDataIngestion()
    
    # Test with AAPL
    test_symbol = 'AAPL'
    
    print(f"\n1. Testing Yahoo Finance data for {test_symbol}...")
    price_data = ingestion.get_stock_data_yahoo(test_symbol, period='1y')
    if not price_data.empty:
        print(f"✓ Fetched {len(price_data)} REAL price data points")
        print(f"  Latest close: ${price_data['close'].iloc[-1]:.2f}")
    
    print(f"\n2. Testing Finnhub profile for {test_symbol}...")
    profile = ingestion.get_stock_profile_finnhub(test_symbol)
    if profile:
        print(f"✓ Company: {profile.get('name', 'N/A')}")
        print(f"  Industry: {profile.get('finnhubIndustry', 'N/A')}")
        print(f"  Market Cap: ${profile.get('marketCapitalization', 0):,.0f}M")
    
    print(f"\n3. Testing Finnhub metrics for {test_symbol}...")
    metrics = ingestion.get_stock_metrics_finnhub(test_symbol)
    if metrics:
        print(f"✓ Fetched REAL metrics")
    
    print(f"\n4. Testing options chain for {test_symbol}...")
    options = ingestion.get_options_chain_yfinance(test_symbol)
    if options:
        print(f"✓ Fetched options for {len(options)} expirations")
        for exp, data in list(options.items())[:3]:
            print(f"  {exp}: {len(data['calls'])} calls, {len(data['puts'])} puts (0.3-0.6 delta)")
    
    print(f"\n5. Testing news for {test_symbol}...")
    news = ingestion.get_company_news_finnhub(test_symbol)
    if news:
        print(f"✓ Fetched {len(news)} REAL news articles")
        if news:
            print(f"  Latest: {news[0].get('headline', 'N/A')[:60]}...")
    
    print(f"\n6. Testing COMPLETE data fetch for {test_symbol}...")
    complete_data = ingestion.get_complete_stock_data(test_symbol)
    print(f"✓ Complete data package assembled")
    print(f"  Price data points: {len(complete_data['price_data']) if isinstance(complete_data['price_data'], pd.DataFrame) else 0}")
    print(f"  Options expirations: {len(complete_data['options'])}")
    print(f"  News articles: {len(complete_data['news'])}")
    
    print("\n" + "=" * 80)
    print("ENHANCED DATA INGESTION TEST COMPLETE - ALL REAL DATA")
    print("=" * 80)


if __name__ == "__main__":
    test_enhanced_data_ingestion()
