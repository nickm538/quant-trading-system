"""
Simple IV Scraper for optioncharts.io
Uses regex to extract IV from the overview paragraph which is in the initial HTML
"""

import requests
import re
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)

class IVScraper:
    """Scrape IV data from optioncharts.io using simple regex"""
    
    def __init__(self):
        self.base_url = "https://optioncharts.io/options/{}"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
    
    def get_iv_data(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Fetch IV data for a symbol from optioncharts.io
        
        Parses the overview paragraph like:
        "AAPL options have an IV of 19.89 % and an IV rank of 8.7%"
        
        Returns:
            {
                'iv_30d': float,  # Implied Volatility (30d) in %
                'iv_rank': float,  # IV Rank in %
            }
        """
        try:
            url = self.base_url.format(symbol.upper())
            response = self.session.get(url, timeout=10)
            
            if response.status_code != 200:
                logger.warning(f"Failed to fetch IV for {symbol}: HTTP {response.status_code}")
                return None
            
            html = response.text
            
            # Parse the overview paragraph which is in the initial HTML:
            # "AAPL options have an IV of 19.89 % and an IV rank of 8.7%"
            iv_match = re.search(r'options have an IV of\s+([\d.]+)\s*%', html)
            iv_rank_match = re.search(r'IV rank of\s+([\d.]+)%', html)
            
            if not iv_match:
                logger.warning(f"Could not parse IV for {symbol}")
                return None
            
            iv_30d = float(iv_match.group(1))
            iv_rank = float(iv_rank_match.group(1)) if iv_rank_match else 0.0
            
            result = {
                'iv_30d': iv_30d,
                'iv_rank': iv_rank,
            }
            
            logger.info(f"✅ Scraped IV for {symbol}: IV={iv_30d}%, Rank={iv_rank}%")
            return result
            
        except Exception as e:
            logger.error(f"Error scraping IV for {symbol}: {e}")
            return None
    
    def get_iv_simple(self, symbol: str) -> Optional[float]:
        """Get just the IV (30d) value, returns None if unavailable"""
        data = self.get_iv_data(symbol)
        return data['iv_30d'] if data else None
