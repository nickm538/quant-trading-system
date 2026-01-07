"""
MarketData.app API Client
Provides institutional-grade real-time options data with full liquidity metrics.
"""

import requests
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from marketdata_cache import get_cache

logger = logging.getLogger(__name__)
cache = get_cache()

class MarketDataClient:
    """
    Client for MarketData.app API - provides real-time options data with:
    - Real bid/ask spreads and sizes
    - Real volume and open interest
    - Real Greeks (delta, gamma, theta, vega)
    - Real implied volatility
    """
    
    BASE_URL = "https://api.marketdata.app/v1"
    API_TOKEN = "dUFVWDVlYmc0Zm53eGxuc0NRWWFWQkNuSTFnWnVrZTBWVlNzWE9ldEhQQT0"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'InstitutionalTradingSystem/1.0'
        })
    
    def get_option_expirations(self, symbol: str) -> List[str]:
        """
        Get all available expiration dates for a symbol.
        CACHED: 24 hours TTL (expirations rarely change)
        
        Args:
            symbol: Stock ticker (e.g., 'AAPL')
            
        Returns:
            List of expiration dates in YYYY-MM-DD format
        """
        # Check cache first
        cache_key = symbol
        cached = cache.get('expirations', cache_key)
        if cached is not None:
            return cached
        
        url = f"{self.BASE_URL}/options/expirations/{symbol}/"
        params = {'token': self.API_TOKEN}
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get('s') == 'ok':
                expirations = data.get('expirations', [])
                logger.info(f"✓ Found {len(expirations)} expiration dates for {symbol}")
                
                # Cache the result
                cache.set('expirations', cache_key, expirations)
                
                return expirations
            else:
                logger.error(f"MarketData API error: {data}")
                return []
                
        except Exception as e:
            logger.error(f"Failed to fetch expirations for {symbol}: {e}")
            return []
    
    def get_options_chain(
        self,
        symbol: str,
        expiration: str,
        side: str = 'call',
        min_delta: float = 0.3,
        max_delta: float = 0.6,
        current_price: float = None
    ) -> List[Dict[str, Any]]:
        """
        Get options chain for specific expiration with full data.
        OPTIMIZED: Pre-filters strikes to reduce API calls from 500+ to ~50
        
        Args:
            symbol: Stock ticker
            expiration: Expiration date (YYYY-MM-DD)
            side: 'call' or 'put'
            min_delta: Minimum delta filter (default 0.3)
            max_delta: Maximum delta filter (default 0.6)
            current_price: Current stock price for strike filtering
            
        Returns:
            List of option contracts with full data (bid, ask, volume, OI, Greeks)
        """
        # First get the option symbols
        url = f"{self.BASE_URL}/options/chain/{symbol}/"
        params = {
            'token': self.API_TOKEN,
            'expiration': expiration,
            'side': side
        }
        
        try:
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            if data.get('s') != 'ok':
                logger.error(f"MarketData API error: {data}")
                return []
            
            option_symbols = data.get('optionSymbol', [])
            logger.info(f"✓ Found {len(option_symbols)} {side} options for {symbol} expiring {expiration}")
            
            # OPTIMIZATION: Filter strikes before fetching quotes
            if current_price:
                filtered_symbols = self._filter_strikes_by_price(option_symbols, current_price, side)
                logger.info(f"  ✓ Filtered to {len(filtered_symbols)} strikes near current price (±20%)")
            else:
                filtered_symbols = option_symbols
            
            # Now get quotes only for filtered options
            options_data = []
            
            for symbol_name in filtered_symbols:
                quote = self.get_option_quote(symbol_name)
                if quote:
                    # Filter by delta
                    delta = abs(quote.get('delta', 0))
                    if min_delta <= delta <= max_delta:
                        options_data.append(quote)
            
            logger.info(f"✓ Retrieved {len(options_data)} options with delta {min_delta}-{max_delta}")
            return options_data
            
        except Exception as e:
            logger.error(f"Failed to fetch options chain for {symbol}: {e}")
            return []
    
    def _filter_strikes_by_price(
        self,
        option_symbols: List[str],
        current_price: float,
        side: str
    ) -> List[str]:
        """
        Filter option symbols to only include strikes within ±20% of current price.
        This dramatically reduces API calls from ~500 to ~50.
        
        Option symbol format: AAPL251219C00280000
        Strike is last 8 digits divided by 1000: 00280000 = $280.00
        """
        filtered = []
        min_strike = current_price * 0.80  # 20% below
        max_strike = current_price * 1.20  # 20% above
        
        for symbol in option_symbols:
            try:
                # Extract strike from OCC symbol (last 8 digits / 1000)
                strike_str = symbol[-8:]
                strike = float(strike_str) / 1000.0
                
                if min_strike <= strike <= max_strike:
                    filtered.append(symbol)
            except:
                continue
        
        return filtered
    
    def get_option_quote(self, option_symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get full quote data for a single option contract.
        CACHED: 30 minutes TTL (this is the most called method - ~100x per analysis)
        
        Args:
            option_symbol: Option symbol (e.g., 'AAPL251219C00280000')
            
        Returns:
            Dict with bid, ask, volume, OI, Greeks, IV, etc.
        """
        # Check cache first (CRITICAL for staying under 100 requests/day)
        cache_key = option_symbol
        cached = cache.get('option_quote', cache_key)
        if cached is not None:
            return cached
        
        url = f"{self.BASE_URL}/options/quotes/{option_symbol}/"
        params = {'token': self.API_TOKEN}
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get('s') != 'ok':
                return None
            
            # Extract data from arrays (API returns arrays for batch compatibility)
            quote_data = {
                'option_symbol': data['optionSymbol'][0] if data.get('optionSymbol') else option_symbol,
                'underlying': data['underlying'][0] if data.get('underlying') else None,
                'strike': data['strike'][0] if data.get('strike') else 0,
                'expiration': data['expiration'][0] if data.get('expiration') else 0,
                'side': data['side'][0] if data.get('side') else 'call',
                'dte': data['dte'][0] if data.get('dte') else 0,
                'bid': data['bid'][0] if data.get('bid') else 0,
                'bid_size': data['bidSize'][0] if data.get('bidSize') else 0,
                'mid': data['mid'][0] if data.get('mid') else 0,
                'ask': data['ask'][0] if data.get('ask') else 0,
                'ask_size': data['askSize'][0] if data.get('askSize') else 0,
                'last': data['last'][0] if data.get('last') else 0,
                'volume': data['volume'][0] if data.get('volume') else 0,
                'open_interest': data['openInterest'][0] if data.get('openInterest') else 0,
                'underlying_price': data['underlyingPrice'][0] if data.get('underlyingPrice') else 0,
                'iv': data['iv'][0] if data.get('iv') else 0,
                'delta': data['delta'][0] if data.get('delta') else 0,
                'gamma': data['gamma'][0] if data.get('gamma') else 0,
                'theta': data['theta'][0] if data.get('theta') else 0,
                'vega': data['vega'][0] if data.get('vega') else 0,
                'intrinsic_value': data['intrinsicValue'][0] if data.get('intrinsicValue') else 0,
                'extrinsic_value': data['extrinsicValue'][0] if data.get('extrinsicValue') else 0,
                'in_the_money': data['inTheMoney'][0] if data.get('inTheMoney') else False,
            }
            
            # Cache the result (30 min TTL)
            cache.set('option_quote', cache_key, quote_data)
            
            return quote_data
            
        except Exception as e:
            logger.warning(f"Failed to get quote for {option_symbol}: {e}")
            return None
    
    def get_stock_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get real-time stock quote.
        CACHED: 5 minutes TTL (more frequent than options)
        
        Args:
            symbol: Stock ticker (e.g., 'AAPL')
            
        Returns:
            Dict with current price, bid, ask, volume, etc.
        """
        # Check cache first
        cache_key = symbol
        cached = cache.get('stock_quote', cache_key)
        if cached is not None:
            return cached
        
        url = f"{self.BASE_URL}/stocks/quotes/{symbol}/"
        params = {'token': self.API_TOKEN}
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get('s') != 'ok':
                return None
            
            quote_data = {
                'symbol': data.get('symbol', [symbol])[0],
                'last': data.get('last', [0])[0],
                'bid': data.get('bid', [0])[0],
                'ask': data.get('ask', [0])[0],
                'mid': data.get('mid', [0])[0],
                'volume': data.get('volume', [0])[0],
                'change': data.get('change', [0])[0],
                'change_pct': data.get('changepct', [0])[0],
                'high': data.get('high', [0])[0],
                'low': data.get('low', [0])[0],
                'open': data.get('open', [0])[0],
                'prev_close': data.get('close', [0])[0],
            }
            
            # Cache the result (5 min TTL)
            cache.set('stock_quote', cache_key, quote_data)
            logger.info(f"✓ Got real-time quote for {symbol}: ${quote_data['last']:.2f}")
            
            return quote_data
            
        except Exception as e:
            logger.warning(f"Failed to get stock quote for {symbol}: {e}")
            return None
    
    def get_stock_candles(
        self,
        symbol: str,
        resolution: str = 'D',
        from_date: str = None,
        to_date: str = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get historical candle data for a stock.
        CACHED: 30 minutes TTL for daily data
        
        Args:
            symbol: Stock ticker
            resolution: Candle resolution (D=daily, W=weekly, M=monthly)
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            
        Returns:
            Dict with OHLCV arrays
        """
        # Build cache key
        cache_key = f"{symbol}_{resolution}_{from_date}_{to_date}"
        cached = cache.get('stock_candles', cache_key)
        if cached is not None:
            return cached
        
        url = f"{self.BASE_URL}/stocks/candles/{resolution}/{symbol}/"
        params = {'token': self.API_TOKEN}
        if from_date:
            params['from'] = from_date
        if to_date:
            params['to'] = to_date
        
        try:
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            if data.get('s') != 'ok':
                return None
            
            candles_data = {
                'symbol': symbol,
                'timestamps': data.get('t', []),
                'open': data.get('o', []),
                'high': data.get('h', []),
                'low': data.get('l', []),
                'close': data.get('c', []),
                'volume': data.get('v', []),
            }
            
            # Cache the result
            cache.set('stock_candles', cache_key, candles_data)
            logger.info(f"✓ Got {len(candles_data['timestamps'])} candles for {symbol}")
            
            return candles_data
            
        except Exception as e:
            logger.warning(f"Failed to get candles for {symbol}: {e}")
            return None
    
    def get_filtered_options(
        self,
        symbol: str,
        min_dte: int = 3,       # Reduced from 7 - allow shorter term
        max_dte: int = 120,     # Extended from 90 - allow longer term
        min_delta: float = 0.15, # Widened from 0.3 - include more OTM
        max_delta: float = 0.85, # Widened from 0.6 - include more ITM
        min_volume: int = 1,     # Reduced from 10 - some good options have low volume
        min_oi: int = 10         # Reduced from 50 - allow newer strikes
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get filtered options for both calls and puts with institutional-grade criteria.
        
        Args:
            symbol: Stock ticker
            min_dte: Minimum days to expiration
            max_dte: Maximum days to expiration
            min_delta: Minimum delta (absolute value)
            max_delta: Maximum delta (absolute value)
            min_volume: Minimum daily volume
            min_oi: Minimum open interest
            
        Returns:
            Dict with 'calls' and 'puts' lists
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Fetching institutional-grade options for {symbol}")
        logger.info(f"Filters: DTE {min_dte}-{max_dte}, Delta {min_delta}-{max_delta}, Vol≥{min_volume}, OI≥{min_oi}")
        logger.info(f"{'='*80}")
        
        # Get all expirations
        expirations = self.get_option_expirations(symbol)
        if not expirations:
            return {'calls': [], 'puts': []}
        
        # Filter expirations by DTE
        filtered_expirations = []
        for exp_date in expirations:
            try:
                exp_dt = datetime.strptime(exp_date, '%Y-%m-%d')
                dte = (exp_dt - datetime.now()).days
                if min_dte <= dte <= max_dte:
                    filtered_expirations.append(exp_date)
            except:
                continue
        
        logger.info(f"✓ Filtered to {len(filtered_expirations)} expirations with DTE {min_dte}-{max_dte}")
        
        all_calls = []
        all_puts = []
        
        # Get current stock price for strike filtering (use MarketData instead of yfinance)
        stock_quote = self.get_stock_quote(symbol)
        if stock_quote and stock_quote['last'] > 0:
            current_price = stock_quote['last']
            logger.info(f"✓ Current price for {symbol}: ${current_price:.2f} (MarketData real-time)")
        else:
            # Fallback to yfinance if MarketData fails
            try:
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                current_price = ticker.info.get('currentPrice', ticker.info.get('regularMarketPrice', 0))
                logger.info(f"✓ Current price for {symbol}: ${current_price:.2f} (yfinance fallback)")
            except:
                current_price = None
                logger.warning(f"⚠ Could not fetch current price for {symbol}, strike filtering disabled")
        
        # Fetch calls and puts for each expiration (LIMIT TO 5 FOR BETTER COVERAGE)
        for exp_date in filtered_expirations[:5]:  # Increased from 3 to 5 expirations
            logger.info(f"\nFetching options for expiration: {exp_date}")
            
            # Get calls (with current_price for strike filtering)
            calls = self.get_options_chain(symbol, exp_date, 'call', min_delta, max_delta, current_price)
            # Filter by volume and OI
            calls = [c for c in calls if c['volume'] >= min_volume and c['open_interest'] >= min_oi]
            all_calls.extend(calls)
            logger.info(f"  ✓ {len(calls)} calls passed liquidity filters")
            
            # Get puts (with current_price for strike filtering)
            puts = self.get_options_chain(symbol, exp_date, 'put', min_delta, max_delta, current_price)
            # Filter by volume and OI
            puts = [p for p in puts if p['volume'] >= min_volume and p['open_interest'] >= min_oi]
            all_puts.extend(puts)
            logger.info(f"  ✓ {len(puts)} puts passed liquidity filters")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"✓ FINAL RESULTS: {len(all_calls)} calls, {len(all_puts)} puts")
        logger.info(f"{'='*80}\n")
        
        return {
            'calls': all_calls,
            'puts': all_puts
        }


# Convenience function for quick access
def get_real_options_data(
    symbol: str,
    min_dte: int = 7,
    max_dte: int = 90,
    min_delta: float = 0.3,
    max_delta: float = 0.6
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Quick access function to get real options data with full liquidity metrics.
    
    Returns:
        Dict with 'calls' and 'puts' containing real bid/ask/volume/OI/Greeks
    """
    client = MarketDataClient()
    return client.get_filtered_options(
        symbol=symbol,
        min_dte=min_dte,
        max_dte=max_dte,
        min_delta=min_delta,
        max_delta=max_delta,
        min_volume=10,  # Minimum 10 contracts/day
        min_oi=50       # Minimum 50 open interest
    )
