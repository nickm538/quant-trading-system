"""
IV Scraper for optioncharts.io
Fetches real Implied Volatility data since yfinance's IV field is unreliable
"""

import requests
from bs4 import BeautifulSoup
import re
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class IVScraper:
    """Scrape IV data from optioncharts.io"""
    
    def __init__(self):
        self.base_url = "https://optioncharts.io/options/{}"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_iv_data(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Fetch IV data for a symbol from optioncharts.io
        
        Returns:
            {
                'iv_30d': float,  # Implied Volatility (30d) in %
                'iv_rank': float,  # IV Rank in %
                'iv_percentile': float,  # IV Percentile in %
                'hist_vol': float  # Historical Volatility in %
            }
        """
        try:
            url = self.base_url.format(symbol.upper())
            response = self.session.get(url, timeout=10)
            
            if response.status_code != 200:
                logger.warning(f"Failed to fetch IV for {symbol}: HTTP {response.status_code}")
                return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Parse the "Option Overview" section
            # Looking for text like: "IBM options have an IV of 26.90 % and an IV rank of 18.01%"
            overview_text = soup.get_text()
            
            # Extract IV (30d) from overview text
            iv_match = re.search(r'have an IV of\s+([\d.]+)\s*%', overview_text)
            iv_rank_match = re.search(r'IV rank of\s+([\d.]+)%', overview_text)
            
            # Extract from the detailed IV section
            # Look for "Implied Volatility (30d)" followed by percentage
            iv_30d = None
            iv_rank = None
            iv_percentile = None
            hist_vol = None
            
            # Method 1: From overview paragraph
            if iv_match:
                iv_30d = float(iv_match.group(1))
            if iv_rank_match:
                iv_rank = float(iv_rank_match.group(1))
            
            # Method 2: From structured data (more reliable)
            # Find patterns like "Implied Volatility (30d) \n 26.90%"
            iv_30d_match = re.search(r'Implied Volatility \(30d\)\s*(?:\n|\s)+([\d.]+)%', overview_text)
            iv_rank_match2 = re.search(r'IV Rank\s*(?:\n|\s)+([\d.]+)%', overview_text)
            iv_percentile_match = re.search(r'IV Percentile\s*(?:\n|\s)+([\d.]+)%', overview_text)
            hist_vol_match = re.search(r'Historical Volatility\s*(?:\n|\s)+([\d.]+)%', overview_text)
            
            if iv_30d_match:
                iv_30d = float(iv_30d_match.group(1))
            if iv_rank_match2:
                iv_rank = float(iv_rank_match2.group(1))
            if iv_percentile_match:
                iv_percentile = float(iv_percentile_match.group(1))
            if hist_vol_match:
                hist_vol = float(hist_vol_match.group(1))
            
            if iv_30d is None:
                logger.warning(f"Could not parse IV for {symbol}")
                return None
            
            result = {
                'iv_30d': iv_30d,
                'iv_rank': iv_rank or 0.0,
                'iv_percentile': iv_percentile or 0.0,
                'hist_vol': hist_vol or 0.0
            }
            
            logger.info(f"Scraped IV for {symbol}: IV={iv_30d}%, Rank={iv_rank}%")
            return result
            
        except Exception as e:
            logger.error(f"Error scraping IV for {symbol}: {e}")
            return None
    
    def get_iv_simple(self, symbol: str) -> Optional[float]:
        """Get just the IV (30d) value, returns None if unavailable"""
        data = self.get_iv_data(symbol)
        return data['iv_30d'] if data else None
