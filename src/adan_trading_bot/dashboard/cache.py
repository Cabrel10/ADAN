"""
Data cache layer for ADAN Dashboard

Provides TTL-based caching with staleness detection and fallback values.
"""

from typing import Optional, Any, Dict
from datetime import datetime, timedelta
import logging


logger = logging.getLogger(__name__)


class DataCache:
    """
    Cache layer for dashboard data with TTL and staleness detection.
    
    Features:
    - TTL-based expiration
    - Staleness detection
    - Fallback to last known value
    - Per-key configuration
    """
    
    def __init__(self, default_ttl_seconds: int = 90):
        """
        Initialize data cache.
        
        Args:
            default_ttl_seconds: Default time-to-live for cached values
        """
        self.default_ttl = timedelta(seconds=default_ttl_seconds)
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.staleness_threshold = timedelta(seconds=90)
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """
        Store value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Optional custom TTL for this key
        """
        ttl = timedelta(seconds=ttl_seconds) if ttl_seconds else self.default_ttl
        
        self.cache[key] = {
            'value': value,
            'timestamp': datetime.now(),
            'ttl': ttl,
            'is_stale': False,
        }
        
        logger.debug(f"📦 Cached {key} (TTL: {ttl.total_seconds()}s)")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve value from cache.
        
        Returns:
            Cached value if valid, None if expired or not found
        """
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        age = datetime.now() - entry['timestamp']
        
        # Check if expired
        if age > entry['ttl']:
            logger.debug(f"⏰ Cache expired for {key} (age: {age.total_seconds():.1f}s)")
            del self.cache[key]
            return None
        
        # Check if stale
        if age > self.staleness_threshold:
            entry['is_stale'] = True
            logger.debug(f"⚠️  Cache stale for {key} (age: {age.total_seconds():.1f}s)")
        else:
            entry['is_stale'] = False
        
        return entry['value']
    
    def get_with_staleness(self, key: str) -> tuple[Optional[Any], bool]:
        """
        Retrieve value and staleness status.
        
        Returns:
            Tuple of (value, is_stale)
        """
        if key not in self.cache:
            return None, False
        
        entry = self.cache[key]
        age = datetime.now() - entry['timestamp']
        
        # Check if expired
        if age > entry['ttl']:
            del self.cache[key]
            return None, False
        
        # Check if stale
        is_stale = age > self.staleness_threshold
        entry['is_stale'] = is_stale
        
        return entry['value'], is_stale
    
    def is_stale(self, key: str) -> bool:
        """
        Check if cached value is stale.
        
        Args:
            key: Cache key
        
        Returns:
            True if value is stale or not found
        """
        if key not in self.cache:
            return True
        
        entry = self.cache[key]
        age = datetime.now() - entry['timestamp']
        
        # Check if expired
        if age > entry['ttl']:
            return True
        
        # Check if stale
        return age > self.staleness_threshold
    
    def get_age_seconds(self, key: str) -> Optional[float]:
        """
        Get age of cached value in seconds.
        
        Args:
            key: Cache key
        
        Returns:
            Age in seconds, or None if not found
        """
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        age = datetime.now() - entry['timestamp']
        return age.total_seconds()
    
    def clear(self, key: Optional[str] = None) -> None:
        """
        Clear cache entries.
        
        Args:
            key: Specific key to clear, or None to clear all
        """
        if key is None:
            self.cache.clear()
            logger.debug("🗑️  Cleared all cache")
        else:
            if key in self.cache:
                del self.cache[key]
                logger.debug(f"🗑️  Cleared cache for {key}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        total_entries = len(self.cache)
        stale_entries = sum(1 for entry in self.cache.values() if entry['is_stale'])
        
        return {
            'total_entries': total_entries,
            'stale_entries': stale_entries,
            'fresh_entries': total_entries - stale_entries,
            'keys': list(self.cache.keys()),
        }


class CachedDataCollector:
    """
    Wrapper around data collector that adds caching layer.
    
    Provides transparent caching with configurable TTLs per data type.
    """
    
    def __init__(self, data_collector, cache: Optional[DataCache] = None):
        """
        Initialize cached data collector.
        
        Args:
            data_collector: Underlying data collector
            cache: Optional DataCache instance
        """
        self.collector = data_collector
        self.cache = cache or DataCache()
        
        # Configure TTLs for different data types
        self.ttls = {
            'portfolio': 5,      # 5 seconds
            'positions': 5,      # 5 seconds
            'trades': 30,        # 30 seconds
            'signal': 2,         # 2 seconds
            'market': 2,         # 2 seconds
            'health': 10,        # 10 seconds
        }
    
    def connect(self) -> bool:
        """Connect underlying collector"""
        return self.collector.connect()
    
    def disconnect(self) -> bool:
        """Disconnect underlying collector"""
        return self.collector.disconnect()
    
    def is_connected(self) -> bool:
        """Check if underlying collector is connected"""
        return self.collector.is_connected()
    
    def get_portfolio_state(self):
        """Get portfolio state with caching"""
        cache_key = 'portfolio'
        
        # Try cache first
        cached = self.cache.get(cache_key)
        if cached is not None:
            logger.debug(f"📦 Using cached portfolio state")
            return cached
        
        # Fetch fresh data
        logger.debug(f"🔄 Fetching fresh portfolio state")
        data = self.collector.get_portfolio_state()
        
        # Cache it
        self.cache.set(cache_key, data, self.ttls['portfolio'])
        
        return data
    
    def get_open_positions(self):
        """Get open positions with caching"""
        cache_key = 'positions'
        
        # Try cache first
        cached = self.cache.get(cache_key)
        if cached is not None:
            logger.debug(f"📦 Using cached positions")
            return cached
        
        # Fetch fresh data
        logger.debug(f"🔄 Fetching fresh positions")
        data = self.collector.get_open_positions()
        
        # Cache it
        self.cache.set(cache_key, data, self.ttls['positions'])
        
        return data
    
    def get_closed_trades(self, limit: int = 5):
        """Get closed trades with caching"""
        cache_key = f'trades_{limit}'
        
        # Try cache first
        cached = self.cache.get(cache_key)
        if cached is not None:
            logger.debug(f"📦 Using cached trades")
            return cached
        
        # Fetch fresh data
        logger.debug(f"🔄 Fetching fresh trades")
        data = self.collector.get_closed_trades(limit=limit)
        
        # Cache it
        self.cache.set(cache_key, data, self.ttls['trades'])
        
        return data
    
    def get_current_signal(self):
        """Get current signal with caching"""
        cache_key = 'signal'
        
        # Try cache first
        cached = self.cache.get(cache_key)
        if cached is not None:
            logger.debug(f"📦 Using cached signal")
            return cached
        
        # Fetch fresh data
        logger.debug(f"🔄 Fetching fresh signal")
        data = self.collector.get_current_signal()
        
        # Cache it if not None
        if data is not None:
            self.cache.set(cache_key, data, self.ttls['signal'])
        
        return data
    
    def get_market_context(self):
        """Get market context with caching"""
        cache_key = 'market'
        
        # Try cache first
        cached = self.cache.get(cache_key)
        if cached is not None:
            logger.debug(f"📦 Using cached market context")
            return cached
        
        # Fetch fresh data
        logger.debug(f"🔄 Fetching fresh market context")
        data = self.collector.get_market_context()
        
        # Cache it if not None
        if data is not None:
            self.cache.set(cache_key, data, self.ttls['market'])
        
        return data
    
    def get_system_health(self):
        """Get system health with caching"""
        cache_key = 'health'
        
        # Try cache first
        cached = self.cache.get(cache_key)
        if cached is not None:
            logger.debug(f"📦 Using cached health")
            return cached
        
        # Fetch fresh data
        logger.debug(f"🔄 Fetching fresh health")
        data = self.collector.get_system_health()
        
        # Cache it
        self.cache.set(cache_key, data, self.ttls['health'])
        
        return data
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.cache.get_stats()
    
    def clear_cache(self, key: Optional[str] = None) -> None:
        """Clear cache"""
        self.cache.clear(key)
