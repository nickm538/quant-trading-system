#!/usr/bin/env python3
"""
TAAPI.io API Client
Provides backup/validation for technical indicators
"""

import requests
import logging
from typing import Dict, Any, Optional, List
import os

logger = logging.getLogger(__name__)

# TAAPI.io API configuration
TAAPI_API_KEY = os.getenv(
    'TAAPI_API_KEY',
    'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjbHVlIjoiNjkyYjA3OWM4MDZmZjE2NTFlNGZjOTVhIiwiaWF0IjoxNzY0NjEzODYxLCJleHAiOjMzMjY5MDc3ODYxfQ.Qi2NBgjXnhFx46C3L0RshuBp7srZ9xKqg0MsOYyZJM0'
)
TAAPI_BASE_URL = "https://api.taapi.io"
TAAPI_TIMEOUT = 10  # seconds

class TaapiClient:
    """Client for TAAPI.io technical analysis API"""
    
    def __init__(self, api_key: str = TAAPI_API_KEY):
        """
        Initialize TAAPI.io client
        
        Args:
            api_key: TAAPI.io API key (defaults to environment variable)
        """
        self.api_key = api_key
        self.base_url = TAAPI_BASE_URL
        self.timeout = TAAPI_TIMEOUT
    
    def get_indicator(
        self,
        indicator: str,
        symbol: str,
        interval: str = "1d",
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch technical indicator from TAAPI.io
        
        Args:
            indicator: Indicator name (e.g., 'rsi', 'macd', 'atr')
            symbol: Stock symbol (e.g., 'AAPL')
            interval: Time frame (e.g., '1h', '1d')
            **kwargs: Additional parameters (backtrack, results, period, etc.)
        
        Returns:
            dict: API response with indicator value(s) or None on error
        """
        endpoint = f"{self.base_url}/{indicator}"
        
        params = {
            'symbol': symbol,
            'interval': interval,
            'type': 'stocks',
            'secret': self.api_key,
            **kwargs
        }
        
        try:
            response = requests.get(endpoint, params=params, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            logger.debug(f"TAAPI.io {indicator.upper()} for {symbol}: {result}")
            return result
        except requests.exceptions.HTTPError as e:
            logger.error(f"TAAPI.io HTTP error for {indicator}/{symbol}: {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            logger.error(f"TAAPI.io error for {indicator}/{symbol}: {str(e)}")
            return None
    
    def get_rsi(self, symbol: str, period: int = 14, interval: str = "1d") -> Optional[float]:
        """Get RSI (Relative Strength Index)"""
        result = self.get_indicator('rsi', symbol, interval, period=period)
        return result['value'] if result else None
    
    def get_macd(self, symbol: str, interval: str = "1d") -> Optional[Dict[str, float]]:
        """Get MACD (Moving Average Convergence Divergence)"""
        result = self.get_indicator('macd', symbol, interval)
        if result:
            return {
                'macd': result.get('valueMACD'),
                'signal': result.get('valueMACDSignal'),
                'hist': result.get('valueMACDHist')
            }
        return None
    
    def get_atr(self, symbol: str, period: int = 14, interval: str = "1d") -> Optional[float]:
        """Get ATR (Average True Range)"""
        result = self.get_indicator('atr', symbol, interval, period=period)
        return result['value'] if result else None
    
    def get_adx(self, symbol: str, period: int = 14, interval: str = "1d") -> Optional[float]:
        """Get ADX (Average Directional Index)"""
        result = self.get_indicator('adx', symbol, interval, period=period)
        return result['value'] if result else None
    
    def get_bbands(
        self,
        symbol: str,
        period: int = 20,
        stddev: int = 2,
        interval: str = "1d"
    ) -> Optional[Dict[str, float]]:
        """Get Bollinger Bands"""
        result = self.get_indicator('bbands', symbol, interval, period=period, stddev=stddev)
        if result:
            return {
                'upper': result.get('valueUpperBand'),
                'middle': result.get('valueMiddleBand'),
                'lower': result.get('valueLowerBand')
            }
        return None
    
    def get_ema(self, symbol: str, period: int = 50, interval: str = "1d") -> Optional[float]:
        """Get EMA (Exponential Moving Average)"""
        result = self.get_indicator('ema', symbol, interval, period=period)
        return result['value'] if result else None
    
    def get_sma(self, symbol: str, period: int = 200, interval: str = "1d") -> Optional[float]:
        """Get SMA (Simple Moving Average)"""
        result = self.get_indicator('sma', symbol, interval, period=period)
        return result['value'] if result else None
    
    def get_stoch(self, symbol: str, interval: str = "1d") -> Optional[Dict[str, float]]:
        """Get Stochastic Oscillator"""
        result = self.get_indicator('stoch', symbol, interval)
        if result:
            return {
                'k': result.get('valueK'),
                'd': result.get('valueD')
            }
        return None
    
    def get_cci(self, symbol: str, period: int = 20, interval: str = "1d") -> Optional[float]:
        """Get CCI (Commodity Channel Index)"""
        result = self.get_indicator('cci', symbol, interval, period=period)
        return result['value'] if result else None
    
    def get_willr(self, symbol: str, period: int = 14, interval: str = "1d") -> Optional[float]:
        """Get Williams %R"""
        result = self.get_indicator('willr', symbol, interval, period=period)
        return result['value'] if result else None
    
    def get_obv(self, symbol: str, interval: str = "1d") -> Optional[float]:
        """Get OBV (On Balance Volume)"""
        result = self.get_indicator('obv', symbol, interval)
        return result['value'] if result else None
    
    def get_historical(
        self,
        indicator: str,
        symbol: str,
        count: int = 30,
        interval: str = "1d",
        **kwargs
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get historical indicator values
        
        Args:
            indicator: Indicator name
            symbol: Stock symbol
            count: Number of historical values to fetch
            interval: Time frame
            **kwargs: Additional indicator parameters
        
        Returns:
            List of dicts with 'value' and 'timestamp' keys, or None on error
        """
        result = self.get_indicator(
            indicator,
            symbol,
            interval,
            results=count,
            addResultTimestamp=True,
            **kwargs
        )
        
        if result and 'value' in result and 'timestamp' in result:
            # Combine values and timestamps into list of dicts
            values = result['value']
            timestamps = result['timestamp']
            
            if isinstance(values, list) and isinstance(timestamps, list):
                return [
                    {'value': v, 'timestamp': t}
                    for v, t in zip(values, timestamps)
                ]
        
        return None
    
    def get_bulk_indicators(
        self,
        symbol: str,
        interval: str = "1d",
        indicators: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch multiple indicators in a single bulk API call
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            interval: Time frame (e.g., '1d', '1h')
            indicators: List of indicator configs. If None, uses default comprehensive set.
        
        Returns:
            Dict mapping indicator names to their results, or None on error
        
        Example:
            result = client.get_bulk_indicators('AAPL', '1d')
            # Returns: {
            #   'price': {'value': 278.85},
            #   'rsi': {'value': 67.96},
            #   'macd': {'valueMACD': 3.99, 'valueMACDSignal': 3.96, ...},
            #   ...
            # }
        """
        # Default comprehensive indicator set if not provided
        if indicators is None:
            indicators = [
                {"indicator": "price"},
                {"indicator": "rsi", "period": 14},
                {"indicator": "macd"},
                {"indicator": "atr", "period": 14},
                {"indicator": "adx", "period": 14},
                {"indicator": "bbands", "period": 20, "stddev": 2},
                {"indicator": "ema", "period": 50},
                {"indicator": "sma", "period": 200},
                {"indicator": "stoch"},
                {"indicator": "cci", "period": 20},
                {"indicator": "obv"},
                {"indicator": "mfi", "period": 14},
                {"indicator": "willr", "period": 14},
                {"indicator": "vwap"},
                {"indicator": "supertrend"},
                {"indicator": "psar"},
                {"indicator": "ichimoku"},
                {"indicator": "cmf"},
                {"indicator": "dmi"},
                {"indicator": "volatility", "period": 30}
            ]
        
        # Build bulk query
        payload = {
            "secret": self.api_key,
            "construct": {
                "type": "stocks",
                "symbol": symbol,
                "interval": interval,
                "indicators": indicators
            }
        }
        
        endpoint = f"{self.base_url}/bulk"
        
        try:
            response = requests.post(endpoint, json=payload, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            
            # Parse bulk response into dict keyed by indicator name
            parsed = {}
            if 'data' in result:
                for item in result['data']:
                    # Extract indicator name from id (format: stocks_AAPL_1d_rsi_14_0)
                    item_id = item.get('id', '')
                    parts = item_id.split('_')
                    if len(parts) >= 4:
                        indicator_name = parts[3]  # Extract indicator name
                        # Store the result data (not the wrapper)
                        if 'result' in item and item['result']:
                            parsed[indicator_name] = item['result']
                        elif 'errors' in item and item['errors']:
                            logger.warning(f"Indicator {indicator_name} returned errors: {item['errors']}")
                            parsed[indicator_name] = None
            
            logger.debug(f"TAAPI.io bulk request for {symbol}: {len(parsed)} indicators returned")
            return parsed
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"TAAPI.io bulk HTTP error for {symbol}: {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            logger.error(f"TAAPI.io bulk error for {symbol}: {str(e)}")
            return None
    
    def get_pattern_indicators(
        self,
        symbol: str,
        interval: str = "1d"
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch pattern recognition indicators in a separate bulk call
        Includes 15 key reversal and continuation patterns for NYSE stocks
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            interval: Time frame (e.g., '1d')
        
        Returns:
            Dict mapping pattern names to their results (0 = not detected, 100 = detected)
        """
        pattern_indicators = [
            # Reversal Patterns (Bearish) - Signal potential downtrend
            {"indicator": "3blackcrows"},
            {"indicator": "2crows"},
            {"indicator": "eveningstar"},
            {"indicator": "eveningdojistar"},
            {"indicator": "shootingstar"},
            {"indicator": "darkcloudcover"},
            {"indicator": "hangingman"},
            # Reversal Patterns (Bullish) - Signal potential uptrend
            {"indicator": "3whitesoldiers"},
            {"indicator": "morningstar"},
            {"indicator": "morningdojistar"},
            {"indicator": "hammer"},
            {"indicator": "invertedhammer"},
            {"indicator": "piercing"},
            # Continuation/Indecision patterns
            {"indicator": "doji"},
            {"indicator": "engulfing"}
        ]
        
        # Build bulk query
        payload = {
            "secret": self.api_key,
            "construct": {
                "type": "stocks",
                "symbol": symbol,
                "interval": interval,
                "indicators": pattern_indicators
            }
        }
        
        endpoint = f"{self.base_url}/bulk"
        
        try:
            response = requests.post(endpoint, json=payload, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            
            # Parse bulk response
            parsed = {}
            if 'data' in result:
                for item in result['data']:
                    item_id = item.get('id', '')
                    parts = item_id.split('_')
                    if len(parts) >= 4:
                        indicator_name = parts[3]
                        if 'result' in item and item['result']:
                            parsed[indicator_name] = item['result']
                        elif 'errors' in item and item['errors']:
                            logger.warning(f"Pattern {indicator_name} returned errors: {item['errors']}")
                            parsed[indicator_name] = None
            
            logger.debug(f"TAAPI.io pattern request for {symbol}: {len(parsed)} patterns returned")
            return parsed
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"TAAPI.io pattern HTTP error for {symbol}: {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            logger.error(f"TAAPI.io pattern error for {symbol}: {str(e)}")
            return None
    
    def validate_indicator(
        self,
        indicator: str,
        symbol: str,
        talib_value: float,
        tolerance: float = 0.05,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Validate TA-Lib indicator value against TAAPI.io
        
        Args:
            indicator: Indicator name
            symbol: Stock symbol
            talib_value: Value calculated by TA-Lib
            tolerance: Acceptable difference percentage (default 5%)
            **kwargs: Additional indicator parameters
        
        Returns:
            Dict with validation results:
            {
                'valid': bool,
                'talib_value': float,
                'taapi_value': float,
                'difference': float,
                'difference_pct': float
            }
        """
        taapi_result = self.get_indicator(indicator, symbol, **kwargs)
        
        if not taapi_result:
            return {
                'valid': None,
                'talib_value': talib_value,
                'taapi_value': None,
                'difference': None,
                'difference_pct': None,
                'error': 'TAAPI.io request failed'
            }
        
        # Extract value from TAAPI.io response
        taapi_value = taapi_result.get('value')
        if taapi_value is None:
            # Try other common field names
            taapi_value = taapi_result.get('valueMACD') or taapi_result.get('valueUpperBand')
        
        if taapi_value is None:
            return {
                'valid': None,
                'talib_value': talib_value,
                'taapi_value': None,
                'difference': None,
                'difference_pct': None,
                'error': 'Could not extract value from TAAPI.io response'
            }
        
        # Calculate difference
        difference = abs(talib_value - taapi_value)
        difference_pct = (difference / abs(taapi_value)) * 100 if taapi_value != 0 else 0
        
        # Check if within tolerance
        valid = difference_pct <= (tolerance * 100)
        
        return {
            'valid': valid,
            'talib_value': talib_value,
            'taapi_value': taapi_value,
            'difference': difference,
            'difference_pct': difference_pct
        }


# Global client instance
_client = None

def get_client() -> TaapiClient:
    """Get global TAAPI.io client instance"""
    global _client
    if _client is None:
        _client = TaapiClient()
    return _client


def get_pattern_indicators(
    symbol: str,
    interval: str = "1d"
) -> Optional[Dict[str, Any]]:
    """
    Fetch pattern recognition indicators (15 patterns)
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        interval: Time frame (e.g., '1d')
    
    Returns:
        Dict with pattern results (0 = not detected, 100 = detected) or None on error
    
    Example:
        result = get_pattern_indicators('AAPL', '1d')
        # Returns: {'hammer': {'value': 100}, 'doji': {'value': 0}, ...}
    """
    return get_client().get_pattern_indicators(symbol, interval)


def get_bulk_indicators(
    symbol: str,
    interval: str = "1d",
    indicators: Optional[List[Dict[str, Any]]] = None
) -> Optional[Dict[str, Any]]:
    """
    Fetch multiple indicators in a single bulk API call (up to 20 indicators)
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        interval: Time frame (e.g., '1d', '1h')
        indicators: List of indicator configs, defaults to comprehensive set
    
    Returns:
        Dict with all indicator results or None on error
    
    Example:
        result = get_bulk_indicators('AAPL', '1d')
        # Returns: {'price': {...}, 'rsi': {...}, 'macd': {...}, ...}
    """
    return get_client().get_bulk_indicators(symbol, interval, indicators)


# Convenience functions for quick access
def get_rsi(symbol: str, period: int = 14, interval: str = "1d") -> Optional[float]:
    """Quick access to RSI"""
    return get_client().get_rsi(symbol, period, interval)

def get_macd(symbol: str, interval: str = "1d") -> Optional[Dict[str, float]]:
    """Quick access to MACD"""
    return get_client().get_macd(symbol, interval)

def get_atr(symbol: str, period: int = 14, interval: str = "1d") -> Optional[float]:
    """Quick access to ATR"""
    return get_client().get_atr(symbol, period, interval)

def get_adx(symbol: str, period: int = 14, interval: str = "1d") -> Optional[float]:
    """Quick access to ADX"""
    return get_client().get_adx(symbol, period, interval)

def validate_indicator(
    indicator: str,
    symbol: str,
    talib_value: float,
    tolerance: float = 0.05,
    **kwargs
) -> Dict[str, Any]:
    """Quick access to indicator validation"""
    return get_client().validate_indicator(indicator, symbol, talib_value, tolerance, **kwargs)
