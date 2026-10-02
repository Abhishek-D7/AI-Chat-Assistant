from pinecone import Pinecone, ServerlessSpec
from datetime import datetime, timedelta
import os
import logging
import time
import uuid
from typing import List, Dict, Optional
from app.ml.embeddings import HuggingFaceEmbeddingClient
from app.cache import EmbeddingCache
from app.config import Config

logger = logging.getLogger(__name__)

class ChatPersistence:
    def __init__(self):
        print("DEBUG: Starting ChatPersistence init with Pinecone")
        self.api_key = Config.PINECONE_API_KEY
        self.index_name = Config.PINECONE_INDEX_NAME
        
        # --- Buffer Configuration with TTL ---
        self.write_buffer: Dict[str, List[Dict]] = {}  # Stores data per session_id
        self.buffer_timestamps: Dict[str, float] = {}  # Track buffer creation time
        self.BATCH_SIZE = 5
        self.BUFFER_TTL = Config.BUFFER_TTL
        
        # Initialize Memory Cache
        self.embedding_cache = EmbeddingCache(max_size=Config.EMBEDDING_CACHE_SIZE)

        if not self.api_key:
            logger.error("❌ PINECONE_API_KEY not set")
            self.index = None
            return

        try:
            self.pc = Pinecone(api_key=self.api_key)
            self.embedding_model = HuggingFaceEmbeddingClient(
                token=Config.HF_TOKEN,
                model_name="BAAI/bge-large-en-v1.5"
            )
            self._ensure_index()
            self.index = self.pc.Index(self.index_name)
            logger.info(f"✅ Pinecone initialized successfully with index '{self.index_name}'")
        except Exception as e:
            logger.error(f"❌ Pinecone init failed: {e}")
            logger.warning("⚠️ Falling back to in-memory storage (no persistence)")
            self.index = None

    def _ensure_index(self):
        existing_indexes = [index_info["name"] for index_info in self.pc.list_indexes()]
        if self.index_name not in existing_indexes:
            alt_name = "chat-assistent" if self.index_name == "chat-assistant" else "chat-assistant"
            if alt_name in existing_indexes:
                logger.info(f"Using existing Pinecone index '{alt_name}' instead of '{self.index_name}'")
                self.index_name = alt_name
                return
            print(f"DEBUG: Creating Pinecone index '{self.index_name}'...")
            self.pc.create_index(
                name=self.index_name,
                dimension=1024,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print(f"DEBUG: Index '{self.index_name}' created.")

        # In-memory history and stats for the current application lifecycle
        # (Chat context is kept completely out of Pinecone, dedicated 100% for RAG)
        self.in_memory_history: Dict[str, List[Dict]] = {}
        self.in_memory_stats: Dict[str, Dict] = {}

    def store_conversation(self, user_name: str, user_id: str, session_id: str, user_message: str, bot_response: str, intent: str = "general") -> bool:
        """Store conversation turn in-memory only (Pinecone is reserved strictly for document RAG)."""
        try:
            record = {
                "user_name": user_name,
                "user_id": user_id,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "user_message": user_message[:2000],
                "bot_response": bot_response[:2000],
                "intent": intent
            }
            if user_name not in self.in_memory_history:
                self.in_memory_history[user_name] = []
            self.in_memory_history[user_name].append(record)
            
            # Update user stats
            if user_name not in self.in_memory_stats:
                self.in_memory_stats[user_name] = {"total_turns": 0, "intents": {}, "last_active": "Never"}
            
            stats = self.in_memory_stats[user_name]
            stats["total_turns"] += 1
            stats["intents"][intent] = stats["intents"].get(intent, 0) + 1
            stats["last_active"] = record["timestamp"]
            
            return True
        except Exception as e:
            logger.error(f"❌ In-memory storage error: {e}")
            return False

    def flush_session(self, session_id: str) -> bool:
        """No-op: Chat context is not written to Pinecone."""
        return True

    def get_user_history(self, user_name: str, limit: int = 20, days: int = 30) -> List[Dict]:
        """Fetch user chat history from in-memory store without querying Pinecone."""
        user_records = self.in_memory_history.get(user_name, [])
        return user_records[-limit:] if user_records else []

    def get_user_stats(self, user_name: str) -> Dict:
        """Fetch user activity stats from in-memory store without querying Pinecone."""
        if user_name in self.in_memory_stats:
            return self.in_memory_stats[user_name]
        return {"total_turns": 0, "intents": {}, "last_active": "Never"}

    def cleanup_stale_buffers(self):
        """No-op cleanup."""
        pass
    
    def get_cache_stats(self) -> Dict:
        return self.embedding_cache.get_stats()

    def similarity_search(
        self,
        query: str,
        top_k: int = 4,
        doc_type: Optional[str] = None,
        min_score: float = 0.25
    ) -> List[Dict]:
        """
        Embed query using Hugging Face Inference API and perform cosine similarity
        search against the Pinecone vector database.
        
        Args:
            query: The question or query string to search for
            top_k: Maximum number of most similar results to return
            doc_type: Optional filter by metadata 'type' (e.g. 'document_chunk')
            min_score: Cosine similarity score threshold (0.0 to 1.0)
            
        Returns:
            List of matching records with id, score, text, and metadata
        """
        if not query or not query.strip():
            return []
            
        query_text = query.strip()
        
        try:
            # 1. Embed query (using EmbeddingCache for high performance)
            query_vector = self.embedding_cache.get(query_text)
            if query_vector is None:
                query_vector = self.embedding_model.embed_query(query_text)
                self.embedding_cache.set(query_text, query_vector)
                
            if not self.index:
                logger.warning("⚠️ Pinecone index not connected. Returning empty search results.")
                return []
                
            # 2. Build filter
            pinecone_filter = None
            if doc_type:
                pinecone_filter = {"type": {"$eq": doc_type}}
                
            # 3. Query Pinecone vector database
            res = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
                filter=pinecone_filter
            )
            
            matches = res.get("matches", [])
            results = []
            
            for match in matches:
                score = float(match.get("score", 0.0))
                if score < min_score:
                    continue
                    
                meta = match.get("metadata", {})
                text_content = (
                    meta.get("text")
                    or meta.get("bot_response")
                    or meta.get("user_message")
                    or ""
                )
                source_name = meta.get("source") or meta.get("type", "vector_knowledge_base")
                
                results.append({
                    "id": match.get("id"),
                    "score": score,
                    "text": text_content,
                    "source": source_name,
                    "metadata": meta
                })
                
            logger.info(f"🔍 Vector similarity search for '{query_text[:40]}' found {len(results)} matches")
            return results
            
        except Exception as e:
            logger.error(f"❌ Similarity search error: {e}", exc_info=True)
            return []