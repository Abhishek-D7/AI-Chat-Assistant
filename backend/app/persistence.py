from pinecone import Pinecone, ServerlessSpec
from datetime import datetime, timedelta
import os
import logging
import time
import uuid
from typing import List, Dict, Optional
# pyrefly: ignore [missing-import]
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
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
            self.embedding_model = HuggingFaceInferenceAPIEmbeddings(
                api_key=Config.HF_TOKEN,
                model_name="BAAI/bge-large-en-v1.5"
            )
            self._ensure_index()
            self.index = self.pc.Index(self.index_name)
            logger.info("✅ Pinecone initialized successfully")
        except Exception as e:
            logger.error(f"❌ Pinecone init failed: {e}")
            logger.warning("⚠️ Falling back to in-memory storage (no persistence)")
            self.index = None

    def _ensure_index(self):
        existing_indexes = [index_info["name"] for index_info in self.pc.list_indexes()]
        if self.index_name not in existing_indexes:
            print(f"DEBUG: Creating Pinecone index '{self.index_name}'...")
            self.pc.create_index(
                name=self.index_name,
                dimension=1024,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print(f"DEBUG: Index '{self.index_name}' created.")

    def store_conversation(self, user_name: str, user_id: str, session_id: str, user_message: str, bot_response: str, intent: str = "general") -> bool:
        if not self.index:
            return False

        try:
            combined_text = f"{user_message} {bot_response}"
            embedding = self.embedding_cache.get(combined_text)
            if embedding is None:
                embedding = self.embedding_model.embed_query(combined_text)
                self.embedding_cache.set(combined_text, embedding)
            
            user_message_truncated = user_message[:2000]
            bot_response_truncated = bot_response[:2000]

            record_id = str(uuid.uuid4())
            metadata = {
                "user_name": user_name[:255],
                "user_id": user_id[:255],
                "session_id": session_id[:255],
                "timestamp": datetime.now().isoformat(),
                "user_message": user_message_truncated,
                "bot_response": bot_response_truncated,
                "intent": intent[:100]
            }
            
            data_row = {
                "id": record_id,
                "values": embedding,
                "metadata": metadata
            }

            if session_id not in self.write_buffer:
                self.write_buffer[session_id] = []
                self.buffer_timestamps[session_id] = time.time()

            self.write_buffer[session_id].append(data_row)
            current_buffer_size = len(self.write_buffer[session_id])

            if current_buffer_size >= self.BATCH_SIZE:
                return self.flush_session(session_id)
            
            return True

        except Exception as e:
            logger.error(f"❌ Storage error: {e}")
            return False

    def flush_session(self, session_id: str) -> bool:
        if not self.index:
            return False
        
        if session_id not in self.write_buffer or not self.write_buffer[session_id]:
            return True 

        try:
            data_to_insert = self.write_buffer[session_id]
            self.index.upsert(vectors=data_to_insert)
            
            logger.info(f"💾 Batch Saved: {len(data_to_insert)} records for session {session_id}")
            del self.write_buffer[session_id]
            self.buffer_timestamps.pop(session_id, None)
            return True
        except Exception as e:
            logger.error(f"❌ Failed to flush buffer: {e}")
            return False

    def get_user_history(self, user_name: str, limit: int = 20, days: int = 30) -> List[Dict]:
        if not self.index:
            return []
        try:
            dummy_vector = [0.0] * 1024
            res = self.index.query(
                vector=dummy_vector,
                filter={"user_name": {"$eq": user_name}},
                top_k=100,
                include_metadata=True
            )
            
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            results = []
            for match in res.get("matches", []):
                meta = match.get("metadata", {})
                if meta.get("timestamp", "") >= cutoff_date:
                    results.append(meta)
            
            results = sorted(results, key=lambda x: x.get('timestamp', ''))
            return results[-limit:] if results else []
        except Exception as e:
            logger.error(f"❌ Retrieval error: {e}")
            return []

    def get_user_stats(self, user_name: str) -> Dict:
        if not self.index:
            return {}
        try:
            dummy_vector = [0.0] * 1024
            res = self.index.query(
                vector=dummy_vector,
                filter={"user_name": {"$eq": user_name}},
                top_k=1000,
                include_metadata=True
            )
            
            matches = res.get("matches", [])
            if not matches:
                return {"total_turns": 0, "intents": {}, "last_active": "Never"}
            
            results = [m.get("metadata", {}) for m in matches]
            results = sorted(results, key=lambda x: x.get('timestamp', ''))
            
            intents = {}
            for r in results:
                intent = r.get('intent', 'general')
                intents[intent] = intents.get(intent, 0) + 1
            
            return {
                "total_turns": len(results),
                "intents": intents,
                "last_active": results[-1].get('timestamp') if results else "Never"
            }
        except Exception as e:
            logger.error(f"❌ Stats error: {e}")
            return {}
    
    def cleanup_stale_buffers(self):
        if not self.buffer_timestamps:
            return
        
        current_time = time.time()
        stale_sessions = []
        
        for session_id, timestamp in self.buffer_timestamps.items():
            if current_time - timestamp > self.BUFFER_TTL:
                stale_sessions.append(session_id)
        
        for session_id in stale_sessions:
            success = self.flush_session(session_id)
            if not success:
                self.write_buffer.pop(session_id, None)
                self.buffer_timestamps.pop(session_id, None)
    
    def get_cache_stats(self) -> Dict:
        return self.embedding_cache.get_stats()