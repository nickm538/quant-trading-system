"""
Risk-Free Rate Fetcher
======================

Fetches the current 10-year Treasury yield for use in Black-Scholes calculations.
Updates dynamically to ensure accurate options pricing and Greeks.

Author: Institutional Trading System
Date: 2025-11-29
"""

import sys
sys.path.append('/opt/.manus/.sandbox-runtime')
from data_api import ApiClient
import logging
from datetime import datetime, timedelta
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache file for risk-free rate
CACHE_FILE = '/tmp/risk_free_rate_cache.json'
CACHE_DURATION_HOURS = 1  # Update every hour


def get_risk_free_rate() -> float:
    """
    Get current 10-year Treasury yield (risk-free rate).
    
    Returns:
        float: Risk-free rate as decimal (e.g., 0.042 for 4.2%)
    """
    # Check cache first
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                cache_data = json.load(f)
                cache_time = datetime.fromisoformat(cache_data['timestamp'])
                
                # Use cached value if less than 1 hour old
                if datetime.now() - cache_time < timedelta(hours=CACHE_DURATION_HOURS):
                    logger.info(f"Using cached risk-free rate: {cache_data['rate']:.4f}")
                    return cache_data['rate']
        except Exception as e:
            logger.warning(f"Failed to read cache: {e}")
    
    # Fetch fresh data from Yahoo Finance
    try:
        logger.info("Fetching current 10-year Treasury yield from Yahoo Finance...")
        client = ApiClient()
        
        # Get ^TNX (10-year Treasury yield index)
        response = client.call_api('YahooFinance/get_stock_chart', query={
            'symbol': '^TNX',
            'region': 'US',
            'interval': '1d',
            'range': '5d',
            'includeAdjustedClose': False
        })
        
        if response and 'chart' in response and 'result' in response['chart']:
            result = response['chart']['result'][0]
            meta = result['meta']
            
            # Get the most recent close price (this is the yield percentage)
            current_yield_pct = meta['regularMarketPrice']
            
            # Convert to decimal (e.g., 4.2% -> 0.042)
            risk_free_rate = current_yield_pct / 100.0
            
            logger.info(f"Current 10-year Treasury yield: {current_yield_pct:.2f}% ({risk_free_rate:.4f})")
            
            # Cache the result
            cache_data = {
                'rate': risk_free_rate,
                'rate_pct': current_yield_pct,
                'timestamp': datetime.now().isoformat(),
                'source': 'Yahoo Finance ^TNX'
            }
            
            with open(CACHE_FILE, 'w') as f:
                json.dump(cache_data, f)
            
            return risk_free_rate
        else:
            raise Exception("No data in response")
            
    except Exception as e:
        logger.error(f"Failed to fetch risk-free rate: {e}")
        logger.warning("Falling back to default rate of 4.2%")
        return 0.042  # Fallback to reasonable default


if __name__ == '__main__':
    # Test the function
    rate = get_risk_free_rate()
    print(f"\nCurrent Risk-Free Rate: {rate:.4f} ({rate*100:.2f}%)")
