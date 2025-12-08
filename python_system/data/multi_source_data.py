"""
Multi-Source Data Integration with Fallback Hierarchy
======================================================

Implements robust data fetching with automatic fallback:
1. Finnhub (Primary) - Real-time, comprehensive
2. AlphaVantage (Secondary) - Intraday, technical indicators
3. yfinance (Tertiary) - Options, historical
4. FMP (Backup) - Basic data

Author: Institutional Trading System
Date: 2025-11-20
"""

import sys
sys.path.append('/opt/.manus/.sandbox-runtime')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any
import requests
import json
import time
import logging
from dataclasses import dataclass
import yfinance as yf

logger = logging.getLogger(__name__)

@dataclass
class DataSource:
    """Represents a data source with priority and status"""
    name: str
    priority: int  # 1 = highest
    available: bool = True
    last_error: Optional[str] = None
    error_count: int = 0
    last_success: Optional[datetime] = None


class MultiSourceDataIntegration:
    """
    Robust multi-source data integration with automatic fallback.
    Ensures data integrity and availability for real-money trading.
    """
    
    # API Credentials (hardcoded as per user request)
    FINNHUB_API_KEY = "d3ul051r01qil4aqj8j0d3ul051r01qil4aqj8jg"
    ALPHA_VANTAGE_API_KEY = "UDU3WP1A94ETAIME"
    FMP_API_KEY = "LTecnRjOFtd8bFOTCRLpcncjxrqaZlqq"
    
    # Base URLs
    FINNHUB_BASE = "https://finnhub.io/api/v1"
    ALPHA_VANTAGE_BASE = "https://www.alphavantage.co/query"
    FMP_BASE = "https://financialmodelingprep.com/api/v3"
    
    # Rate limits (calls per minute)
    RATE_LIMITS = {
        'finnhub': 30,  # 30 calls/second = 1800/min
        'alphavantage': 5,  # 5 calls/min (free tier)
        'fmp': 4  # 250 calls/day ≈ 4/min sustained
    }
    
    def __init__(self):
        """Initialize multi-source data integration"""
        self.sources = {
            'finnhub': DataSource('finnhub', priority=1),
            'alphavantage': DataSource('alphavantage', priority=2),
            'yfinance': DataSource('yfinance', priority=3),
            'fmp': DataSource('fmp', priority=4)
        }
        
        # Rate limiting
        self.last_call_time = {
            'finnhub': datetime.now() - timedelta(seconds=10),
            'alphavantage': datetime.now() - timedelta(seconds=60),
            'fmp': datetime.now() - timedelta(seconds=60)
        }
        
        logger.info("Multi-source data integration initialized")
        logger.info(f"  1. Finnhub (Primary)")
        logger.info(f"  2. AlphaVantage (Secondary)")
        logger.info(f"  3. yfinance (Tertiary)")
        logger.info(f"  4. FMP (Backup)")
    
    def _rate_limit_wait(self, source: str):
        """Enforce rate limiting for API calls"""
        if source not in self.last_call_time:
            return
        
        min_interval = {
            'finnhub': 1.0 / 30,  # 30 calls/second
            'alphavantage': 12.0,  # 5 calls/min = 12 sec between calls
            'fmp': 15.0  # 4 calls/min = 15 sec between calls
        }
        
        if source in min_interval:
            elapsed = (datetime.now() - self.last_call_time[source]).total_seconds()
            wait_time = min_interval[source] - elapsed
            
            if wait_time > 0:
                logger.debug(f"Rate limiting {source}: waiting {wait_time:.2f}s")
                time.sleep(wait_time)
        
        self.last_call_time[source] = datetime.now()
    
    # ==================== FINNHUB METHODS ====================
    
    def _finnhub_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make request to Finnhub API with error handling"""
        try:
            self._rate_limit_wait('finnhub')
            
            if params is None:
                params = {}
            params['token'] = self.FINNHUB_API_KEY
            
            url = f"{self.FINNHUB_BASE}/{endpoint}"
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 403:
                logger.warning(f"Finnhub 403 Forbidden for {endpoint} - may require premium plan")
                self.sources['finnhub'].error_count += 1
                return None
            
            if response.status_code == 429:
                logger.warning(f"Finnhub rate limit exceeded")
                self.sources['finnhub'].available = False
                return None
            
            response.raise_for_status()
            self.sources['finnhub'].last_success = datetime.now()
            self.sources['finnhub'].error_count = 0
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Finnhub error for {endpoint}: {str(e)}")
            self.sources['finnhub'].last_error = str(e)
            self.sources['finnhub'].error_count += 1
            return None
    
    def get_quote_finnhub(self, symbol: str) -> Optional[Dict]:
        """Get real-time quote from Finnhub"""
        data = self._finnhub_request('quote', {'symbol': symbol})
        if data and 'c' in data:
            return {
                'price': data['c'],
                'change': data['d'],
                'percent_change': data['dp'],
                'high': data['h'],
                'low': data['l'],
                'open': data['o'],
                'previous_close': data['pc'],
                'timestamp': datetime.now(),
                'source': 'finnhub'
            }
        return None
    
    def get_company_news_finnhub(self, symbol: str, days: int = 30) -> List[Dict]:
        """Get company news from Finnhub"""
        from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        to_date = datetime.now().strftime('%Y-%m-%d')
        
        news = self._finnhub_request('company-news', {
            'symbol': symbol,
            'from': from_date,
            'to': to_date
        })
        
        return news if isinstance(news, list) else []
    
    # ==================== ALPHAVANTAGE METHODS ====================
    
    def _alphavantage_request(self, function: str, params: Dict = None) -> Optional[Dict]:
        """Make request to AlphaVantage API with error handling"""
        try:
            self._rate_limit_wait('alphavantage')
            
            if params is None:
                params = {}
            params['function'] = function
            params['apikey'] = self.ALPHA_VANTAGE_API_KEY
            
            response = requests.get(self.ALPHA_VANTAGE_BASE, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for rate limit message
            if 'Note' in data or 'Information' in data:
                logger.warning(f"AlphaVantage rate limit or info message")
                self.sources['alphavantage'].available = False
                return None
            
            self.sources['alphavantage'].last_success = datetime.now()
            self.sources['alphavantage'].error_count = 0
            
            return data
            
        except Exception as e:
            logger.error(f"AlphaVantage error for {function}: {str(e)}")
            self.sources['alphavantage'].last_error = str(e)
            self.sources['alphavantage'].error_count += 1
            return None
    
    def get_intraday_alphavantage(
        self,
        symbol: str,
        interval: str = '1min'
    ) -> Optional[pd.DataFrame]:
        """Get intraday data from AlphaVantage (1min, 5min, 15min, 30min, 60min)"""
        data = self._alphavantage_request('TIME_SERIES_INTRADAY', {
            'symbol': symbol,
            'interval': interval,
            'outputsize': 'full'
        })
        
        if data and f'Time Series ({interval})' in data:
            ts = data[f'Time Series ({interval})']
            df = pd.DataFrame.from_dict(ts, orient='index')
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df.index = pd.to_datetime(df.index)
            df = df.astype(float)
            df.sort_index(inplace=True)
            return df
        
        return None
    
    def get_news_sentiment_alphavantage(self, symbol: str) -> Optional[Dict]:
        """Get news sentiment from AlphaVantage"""
        data = self._alphavantage_request('NEWS_SENTIMENT', {
            'tickers': symbol,
            'limit': 50
        })
        
        if data and 'feed' in data:
            return {
                'feed': data['feed'],
                'sentiment_score_definition': data.get('sentiment_score_definition', ''),
                'relevance_score_definition': data.get('relevance_score_definition', ''),
                'source': 'alphavantage'
            }
        
        return None
    
    def get_overview_alphavantage(self, symbol: str) -> Optional[Dict]:
        """Get company overview from AlphaVantage"""
        data = self._alphavantage_request('OVERVIEW', {'symbol': symbol})
        
        if data and 'Symbol' in data:
            return {
                'symbol': data.get('Symbol'),
                'name': data.get('Name'),
                'sector': data.get('Sector'),
                'industry': data.get('Industry'),
                'market_cap': float(data.get('MarketCapitalization', 0)),
                'pe_ratio': float(data.get('PERatio', 0)) if data.get('PERatio') != 'None' else None,
                'peg_ratio': float(data.get('PEGRatio', 0)) if data.get('PEGRatio') != 'None' else None,
                'eps': float(data.get('EPS', 0)) if data.get('EPS') != 'None' else None,
                'beta': float(data.get('Beta', 0)) if data.get('Beta') != 'None' else None,
                '52_week_high': float(data.get('52WeekHigh', 0)),
                '52_week_low': float(data.get('52WeekLow', 0)),
                'source': 'alphavantage'
            }
        
        return None
    
    # ==================== FMP METHODS ====================
    
    def _fmp_request(self, endpoint: str, params: Dict = None) -> Optional[Union[Dict, List]]:
        """Make request to FMP API with error handling"""
        try:
            self._rate_limit_wait('fmp')
            
            if params is None:
                params = {}
            params['apikey'] = self.FMP_API_KEY
            
            url = f"{self.FMP_BASE}/{endpoint}"
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            self.sources['fmp'].last_success = datetime.now()
            self.sources['fmp'].error_count = 0
            
            return data
            
        except Exception as e:
            logger.error(f"FMP error for {endpoint}: {str(e)}")
            self.sources['fmp'].last_error = str(e)
            self.sources['fmp'].error_count += 1
            return None
    
    def get_quote_fmp(self, symbol: str) -> Optional[Dict]:
        """Get quote from FMP"""
        data = self._fmp_request(f'quote/{symbol}')
        
        if data and isinstance(data, list) and len(data) > 0:
            quote = data[0]
            return {
                'price': quote.get('price'),
                'change': quote.get('change'),
                'percent_change': quote.get('changesPercentage'),
                'high': quote.get('dayHigh'),
                'low': quote.get('dayLow'),
                'open': quote.get('open'),
                'previous_close': quote.get('previousClose'),
                'volume': quote.get('volume'),
                'timestamp': datetime.now(),
                'source': 'fmp'
            }
        
        return None
    
    def get_profile_fmp(self, symbol: str) -> Optional[Dict]:
        """Get company profile from FMP"""
        data = self._fmp_request(f'profile/{symbol}')
        
        if data and isinstance(data, list) and len(data) > 0:
            profile = data[0]
            return {
                'symbol': profile.get('symbol'),
                'name': profile.get('companyName'),
                'sector': profile.get('sector'),
                'industry': profile.get('industry'),
                'market_cap': profile.get('mktCap'),
                'beta': profile.get('beta'),
                'description': profile.get('description'),
                'source': 'fmp'
            }
        
        return None
    
    # ==================== YFINANCE METHODS ====================
    
    def get_data_yfinance(self, symbol: str, period: str = '2y') -> Optional[pd.DataFrame]:
        """Get historical data from yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            
            if not df.empty:
                df.columns = [c.lower() for c in df.columns]
                self.sources['yfinance'].last_success = datetime.now()
                self.sources['yfinance'].error_count = 0
                return df
            
            return None
            
        except Exception as e:
            logger.error(f"yfinance error for {symbol}: {str(e)}")
            self.sources['yfinance'].last_error = str(e)
            self.sources['yfinance'].error_count += 1
            return None
    
    # ==================== INTELLIGENT FALLBACK SYSTEM ====================
    
    def get_quote_with_fallback(self, symbol: str) -> Dict:
        """
        Get quote with automatic fallback across sources.
        Returns best available data with confidence score.
        """
        results = []
        
        # Try sources in priority order
        sources_by_priority = sorted(
            [(name, source) for name, source in self.sources.items()],
            key=lambda x: x[1].priority
        )
        
        for source_name, source_info in sources_by_priority:
            if not source_info.available:
                continue
            
            try:
                if source_name == 'finnhub':
                    quote = self.get_quote_finnhub(symbol)
                elif source_name == 'fmp':
                    quote = self.get_quote_fmp(symbol)
                elif source_name == 'yfinance':
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    quote = {
                        'price': info.get('currentPrice') or info.get('regularMarketPrice'),
                        'change': info.get('regularMarketChange'),
                        'percent_change': info.get('regularMarketChangePercent'),
                        'high': info.get('dayHigh'),
                        'low': info.get('dayLow'),
                        'open': info.get('open'),
                        'previous_close': info.get('previousClose'),
                        'volume': info.get('volume'),
                        'timestamp': datetime.now(),
                        'source': 'yfinance'
                    }
                else:
                    continue
                
                if quote and quote.get('price'):
                    results.append(quote)
                    logger.info(f"✓ Got quote from {source_name}: ${quote['price']:.2f}")
                
            except Exception as e:
                logger.warning(f"Failed to get quote from {source_name}: {str(e)}")
                continue
        
        if not results:
            raise ValueError(f"Failed to get quote for {symbol} from any source")
        
        # Cross-validate if multiple sources
        if len(results) > 1:
            prices = [r['price'] for r in results]
            mean_price = np.mean(prices)
            std_price = np.std(prices)
            max_deviation = max(abs(p - mean_price) / mean_price for p in prices)
            
            if max_deviation > 0.005:  # 0.5% threshold
                logger.warning(
                    f"Price discrepancy detected: {max_deviation*100:.2f}% "
                    f"(prices: {prices})"
                )
            
            # Use primary source but add validation info
            best_quote = results[0]
            best_quote['cross_validated'] = True
            best_quote['validation_sources'] = len(results)
            best_quote['price_std'] = std_price
            best_quote['confidence'] = max(0, 100 - (max_deviation * 10000))
        else:
            best_quote = results[0]
            best_quote['cross_validated'] = False
            best_quote['validation_sources'] = 1
            best_quote['confidence'] = 90.0  # Single source = 90% confidence
        
        return best_quote
    
    def get_comprehensive_data(self, symbol: str) -> Dict:
        """
        Get comprehensive data from all available sources.
        Implements intelligent fallback and cross-validation.
        """
        logger.info(f"="*80)
        logger.info(f"Fetching comprehensive data for {symbol} with multi-source fallback")
        logger.info(f"="*80)
        
        data = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'quote': None,
            'historical': None,
            'intraday': None,
            'news': [],
            'sentiment': None,
            'profile': None,
            'sources_used': [],
            'data_quality': {}
        }
        
        # 1. Get quote with fallback
        try:
            data['quote'] = self.get_quote_with_fallback(symbol)
            data['sources_used'].append(data['quote']['source'])
        except Exception as e:
            logger.error(f"Failed to get quote: {str(e)}")
        
        # 2. Get historical data (yfinance is best for this)
        try:
            data['historical'] = self.get_data_yfinance(symbol)
            if data['historical'] is not None:
                data['sources_used'].append('yfinance_historical')
        except Exception as e:
            logger.error(f"Failed to get historical data: {str(e)}")
        
        # 3. Get intraday data (AlphaVantage)
        try:
            data['intraday'] = self.get_intraday_alphavantage(symbol, '1min')
            if data['intraday'] is not None:
                data['sources_used'].append('alphavantage_intraday')
        except Exception as e:
            logger.debug(f"Intraday data not available: {str(e)}")
        
        # 4. Get news (Finnhub first, then AlphaVantage)
        try:
            finnhub_news = self.get_company_news_finnhub(symbol)
            if finnhub_news:
                data['news'].extend(finnhub_news)
                data['sources_used'].append('finnhub_news')
        except Exception as e:
            logger.debug(f"Finnhub news not available: {str(e)}")
        
        # 5. Get sentiment (AlphaVantage)
        try:
            data['sentiment'] = self.get_news_sentiment_alphavantage(symbol)
            if data['sentiment']:
                data['sources_used'].append('alphavantage_sentiment')
        except Exception as e:
            logger.debug(f"Sentiment data not available: {str(e)}")
        
        # 6. Get profile (try all sources)
        for source_func, source_name in [
            (lambda: self.get_overview_alphavantage(symbol), 'alphavantage'),
            (lambda: self.get_profile_fmp(symbol), 'fmp')
        ]:
            try:
                profile = source_func()
                if profile:
                    data['profile'] = profile
                    data['sources_used'].append(f'{source_name}_profile')
                    break
            except Exception as e:
                continue
        
        # Calculate data quality score
        data['data_quality'] = {
            'sources_count': len(set(data['sources_used'])),
            'quote_confidence': data['quote'].get('confidence', 0) if data['quote'] else 0,
            'has_historical': data['historical'] is not None,
            'has_intraday': data['intraday'] is not None,
            'news_count': len(data['news']),
            'has_sentiment': data['sentiment'] is not None,
            'has_profile': data['profile'] is not None
        }
        
        logger.info(f"✓ Data collection complete")
        logger.info(f"  Sources used: {', '.join(set(data['sources_used']))}")
        logger.info(f"  Quote confidence: {data['data_quality']['quote_confidence']:.1f}%")
        logger.info(f"  News articles: {data['data_quality']['news_count']}")
        
        return data


# Global instance
multi_source_data = MultiSourceDataIntegration()
