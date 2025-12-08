"""
MarketData.app API Response Cache
Aggressive caching to stay within 100 requests/day limit.

Cache Strategy:
- Options data: 30 minutes TTL (market moves slowly for options)
- Stock quotes: 5 minutes TTL (more frequent updates needed)
- Expirations: 1 day TTL (rarely changes)
"""

import json
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class MarketDataCache:
    """
    File-based cache for MarketData.app API responses.
    Reduces API calls from ~100 per analysis to ~0 for cached stocks.
    """
    
    def __init__(self, cache_dir: str = "/tmp/marketdata_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # TTL settings (in seconds)
        self.ttl_config = {
            'options_chain': 1800,      # 30 minutes for options data
            'stock_quote': 300,         # 5 minutes for stock prices
            'expirations': 86400,       # 24 hours for expiration dates
            'option_quote': 1800,       # 30 minutes for individual option quotes
        }
    
    def _get_cache_path(self, cache_type: str, key: str) -> Path:
        """Generate cache file path for a given type and key."""
        safe_key = key.replace('/', '_').replace(':', '_')
        return self.cache_dir / f"{cache_type}_{safe_key}.json"
    
    def get(self, cache_type: str, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached data if it exists and hasn't expired.
        
        Args:
            cache_type: Type of cache (options_chain, stock_quote, etc.)
            key: Unique identifier (e.g., 'AAPL_2025-12-19_call')
            
        Returns:
            Cached data or None if expired/missing
        """
        cache_path = self._get_cache_path(cache_type, key)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r') as f:
                cached = json.load(f)
            
            # Check if expired
            ttl = self.ttl_config.get(cache_type, 1800)
            age = time.time() - cached['timestamp']
            
            if age > ttl:
                logger.debug(f"Cache EXPIRED for {cache_type}:{key} (age: {age:.0f}s, TTL: {ttl}s)")
                cache_path.unlink()  # Delete expired cache
                return None
            
            logger.info(f"✓ Cache HIT for {cache_type}:{key} (age: {age:.0f}s, saved API call)")
            return cached['data']
            
        except Exception as e:
            logger.warning(f"Cache read error for {cache_type}:{key}: {e}")
            return None
    
    def set(self, cache_type: str, key: str, data: Dict[str, Any]) -> None:
        """
        Store data in cache with timestamp.
        
        Args:
            cache_type: Type of cache
            key: Unique identifier
            data: Data to cache
        """
        cache_path = self._get_cache_path(cache_type, key)
        
        try:
            cached = {
                'timestamp': time.time(),
                'data': data
            }
            
            with open(cache_path, 'w') as f:
                json.dump(cached, f)
            
            ttl = self.ttl_config.get(cache_type, 1800)
            logger.info(f"✓ Cached {cache_type}:{key} (TTL: {ttl}s)")
            
        except Exception as e:
            logger.warning(f"Cache write error for {cache_type}:{key}: {e}")
    
    def clear_expired(self) -> int:
        """
        Remove all expired cache files.
        
        Returns:
            Number of files removed
        """
        removed = 0
        current_time = time.time()
        
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r') as f:
                    cached = json.load(f)
                
                # Determine TTL based on filename
                cache_type = cache_file.stem.split('_')[0]
                ttl = self.ttl_config.get(cache_type, 1800)
                
                age = current_time - cached['timestamp']
                if age > ttl:
                    cache_file.unlink()
                    removed += 1
                    
            except Exception as e:
                logger.warning(f"Error checking cache file {cache_file}: {e}")
                continue
        
        if removed > 0:
            logger.info(f"✓ Cleared {removed} expired cache files")
        
        return removed
    
    def clear_all(self) -> int:
        """
        Remove all cache files.
        
        Returns:
            Number of files removed
        """
        removed = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
                removed += 1
            except Exception as e:
                logger.warning(f"Error removing cache file {cache_file}: {e}")
        
        logger.info(f"✓ Cleared all cache ({removed} files)")
        return removed
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dict with cache stats (total files, total size, oldest/newest)
        """
        cache_files = list(self.cache_dir.glob("*.json"))
        
        if not cache_files:
            return {
                'total_files': 0,
                'total_size_mb': 0,
                'oldest_age_minutes': 0,
                'newest_age_minutes': 0
            }
        
        total_size = sum(f.stat().st_size for f in cache_files)
        current_time = time.time()
        
        ages = []
        for cache_file in cache_files:
            try:
                with open(cache_file, 'r') as f:
                    cached = json.load(f)
                ages.append(current_time - cached['timestamp'])
            except:
                continue
        
        return {
            'total_files': len(cache_files),
            'total_size_mb': total_size / (1024 * 1024),
            'oldest_age_minutes': max(ages) / 60 if ages else 0,
            'newest_age_minutes': min(ages) / 60 if ages else 0
        }


# Global cache instance
_cache = MarketDataCache()

def get_cache() -> MarketDataCache:
    """Get the global cache instance."""
    return _cache
