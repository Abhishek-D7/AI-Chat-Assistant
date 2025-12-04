"""
Test suite for latency optimizations
"""
import asyncio
import time
from unittest.mock import MagicMock, patch, AsyncMock
from app.cache import EmbeddingCache, UserStatsCache, SystemMessageCache

def test_embedding_cache():
    """Test embedding cache hit/miss behavior"""
    cache = EmbeddingCache(max_size=10)
    
    # Test cache miss
    result = cache.get("test message")
    assert result is None
    
    # Test cache set and hit
    embedding = [0.1, 0.2, 0.3]
    cache.set("test message", embedding)
    result = cache.get("test message")
    assert result == embedding
    
    # Test cache stats
    stats = cache.get_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert "50.00%" in stats["hit_rate"]
    
    print("✅ test_embedding_cache passed")

def test_embedding_cache_lru_eviction():
    """Test LRU eviction when cache is full"""
    cache = EmbeddingCache(max_size=3)
    
    # Fill cache
    cache.set("msg1", [1.0])
    cache.set("msg2", [2.0])
    cache.set("msg3", [3.0])
    
    # Add one more - should evict oldest (msg1)
    cache.set("msg4", [4.0])
    
    # msg1 should be evicted
    assert cache.get("msg1") is None
    # Others should still be there
    assert cache.get("msg2") == [2.0]
    assert cache.get("msg3") == [3.0]
    assert cache.get("msg4") == [4.0]
    
    print("✅ test_embedding_cache_lru_eviction passed")

def test_user_stats_cache_ttl():
    """Test TTL expiration for user stats cache"""
    cache = UserStatsCache(ttl_seconds=1, max_size=10)
    
    # Set stats
    stats = {"total_turns": 10, "intents": {"general": 5}}
    cache.set("user1", stats)
    
    # Should be cached
    result = cache.get("user1")
    assert result == stats
    
    # Wait for TTL to expire
    time.sleep(1.1)
    
    # Should be expired
    result = cache.get("user1")
    assert result is None
    
    print("✅ test_user_stats_cache_ttl passed")

def test_system_message_cache():
    """Test system message caching"""
    cache = SystemMessageCache(max_size=5)
    
    # Test cache miss
    result = cache.get("user1", "context1")
    assert result is None
    
    # Test cache set and hit
    message = "You are an AI assistant for user1"
    cache.set("user1", "context1", message)
    result = cache.get("user1", "context1")
    assert result == message
    
    # Different context should be different cache entry
    result = cache.get("user1", "context2")
    assert result is None
    
    print("✅ test_system_message_cache passed")

async def test_persistence_embedding_cache_integration():
    """Test that persistence uses embedding cache"""
    from app.persistence import ChatPersistence
    
    # Mock Milvus client to avoid real DB
    with patch('app.persistence.MilvusClient'):
        persistence = ChatPersistence()
        
        # Mock the embedding model
        persistence.embedding_model = MagicMock()
        persistence.embedding_model.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]
        persistence.client = MagicMock()  # Enable client
        
        # First call - should generate embedding
        persistence.store_conversation("user1", "id1", "session1", "hello", "hi", "general")
        assert persistence.embedding_model.encode.call_count == 1
        
        # Second call with same message - should use cache
        persistence.store_conversation("user1", "id1", "session2", "hello", "hi", "general")
        # Should still be 1 (not called again)
        assert persistence.embedding_model.encode.call_count == 1
        
        # Check cache stats
        stats = persistence.get_cache_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
    
    print("✅ test_persistence_embedding_cache_integration passed")

if __name__ == "__main__":
    async def run_tests():
        print("Running latency optimization tests...\n")
        
        test_embedding_cache()
        test_embedding_cache_lru_eviction()
        test_user_stats_cache_ttl()
        test_system_message_cache()
        await test_persistence_embedding_cache_integration()
        
        print("\n🎉 All tests passed!")
    
    asyncio.run(run_tests())
