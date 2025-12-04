"""
Centralized caching utilities for performance optimization
"""
from functools import lru_cache
from cachetools import TTLCache
import hashlib
from typing import List, Dict, Optional
import threading

class EmbeddingCache:
    """Thread-safe LRU cache for embeddings"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: Dict[str, List[float]] = {}
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
    
    def _hash_text(self, text: str) -> str:
        """Create hash key for text"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get(self, text: str) -> Optional[List[float]]:
        """Get cached embedding if exists"""
        key = self._hash_text(text)
        with self._lock:
            if key in self._cache:
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None
    
    def set(self, text: str, embedding: List[float]):
        """Cache embedding with LRU eviction"""
        key = self._hash_text(text)
        with self._lock:
            # Simple LRU: if full, remove oldest (first) item
            if len(self._cache) >= self.max_size:
                # Remove first item (oldest in insertion order for Python 3.7+)
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = embedding
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": f"{hit_rate:.2f}%",
                "size": len(self._cache),
                "max_size": self.max_size
            }


class UserStatsCache:
    """TTL cache for user statistics"""
    
    def __init__(self, ttl_seconds: int = 60, max_size: int = 100):
        self._cache = TTLCache(maxsize=max_size, ttl=ttl_seconds)
        self._lock = threading.Lock()
    
    def get(self, user_name: str) -> Optional[Dict]:
        """Get cached stats if exists and not expired"""
        with self._lock:
            return self._cache.get(user_name)
    
    def set(self, user_name: str, stats: Dict):
        """Cache user stats with TTL"""
        with self._lock:
            self._cache[user_name] = stats
    
    def invalidate(self, user_name: str):
        """Invalidate cache for specific user"""
        with self._lock:
            self._cache.pop(user_name, None)


class SystemMessageCache:
    """Cache for system messages per user"""
    
    def __init__(self, max_size: int = 100):
        self._cache: Dict[str, str] = {}
        self._lock = threading.Lock()
        self.max_size = max_size
    
    def get(self, user_name: str, context_summary: str = "") -> Optional[str]:
        """Get cached system message"""
        key = f"{user_name}::{context_summary}"
        with self._lock:
            return self._cache.get(key)
    
    def set(self, user_name: str, context_summary: str, message: str):
        """Cache system message"""
        key = f"{user_name}::{context_summary}"
        with self._lock:
            if len(self._cache) >= self.max_size:
                # Remove oldest
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = message
