from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
from sentence_transformers import SentenceTransformer
from datetime import datetime, timedelta
import os
import logging
import time
from typing import List, Dict, Optional
from app.cache import EmbeddingCache
from app.config import Config

logger = logging.getLogger(__name__)

class ChatPersistence:
    def __init__(self):
        print("DEBUG: Starting ChatPersistence init")
        self.uri = os.getenv("ZILLIZ_URI")
        self.token = os.getenv("ZILLIZ_TOKEN")
        self.collection_name = os.getenv("ZILLIZ_COLLECTION") or "chat_conversations"
        
        # --- Buffer Configuration with TTL ---
        self.write_buffer: Dict[str, List[Dict]] = {}  # Stores data per session_id
        self.buffer_timestamps: Dict[str, float] = {}  # Track buffer creation time
        self.BATCH_SIZE = 5
        self.BUFFER_TTL = Config.BUFFER_TTL
        # -------------------------------------
        
        # --- Embedding Cache ---
        self.embedding_cache = EmbeddingCache(max_size=Config.EMBEDDING_CACHE_SIZE)
        # -----------------------
        
        self.collection_loaded = False  # Track if collection is loaded

        print(f"DEBUG: URI={self.uri}, TOKEN={'set' if self.token else 'unset'}, COLLECTION_NAME={self.collection_name}")

        logger.info(f"📊 Initializing Milvus: {self.uri}")
        try:
            if self.token:
                print("DEBUG: Using token for MilvusClient")
                self.client = MilvusClient(
                    uri=self.uri,
                    token=self.token,
                    db_name="default"
                )
            else:
                print("DEBUG: Using MilvusClient without token")
                self.client = MilvusClient(uri=self.uri)

            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("DEBUG: Initialized SentenceTransformer and MilvusClient")

            self._create_collection()
            logger.info("✅ Milvus initialized successfully")
        except Exception as e:
            print(f"DEBUG: Exception in __init__: {e}")
            logger.error(f"❌ Milvus init failed: {e}")
            logger.warning("⚠️ Falling back to in-memory storage (no persistence)")
            self.client = None

    def _create_collection(self):
        print("DEBUG: Enter _create_collection")
        if not self.client:
            print("DEBUG: No Milvus client, skipping collection creation")
            return

        try:
            collections = self.client.list_collections()
            print(f"DEBUG: list_collections = {collections}")

            if self.collection_name in collections:
                print(f"DEBUG: Collection '{self.collection_name}' already exists")
            else:
                print(f"DEBUG: Creating collection '{self.collection_name}'...")
                fields = [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="user_name", dtype=DataType.VARCHAR, max_length=255),
                    FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=255),
                    FieldSchema(name="session_id", dtype=DataType.VARCHAR, max_length=255),
                    FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=255),
                    FieldSchema(name="user_message", dtype=DataType.VARCHAR, max_length=2000),
                    FieldSchema(name="bot_response", dtype=DataType.VARCHAR, max_length=2000),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
                    FieldSchema(name="intent", dtype=DataType.VARCHAR, max_length=100)
                ]
                schema = CollectionSchema(fields=fields, description="Chat conversations storage", enable_dynamic_field=True)
                self.client.create_collection(collection_name=self.collection_name, schema=schema)
                print(f"DEBUG: Collection '{self.collection_name}' created.")

            # Ensure Index Exists
            try:
                print("DEBUG: Attempt to create index...")
                index_params = self.client.prepare_index_params()
                index_params.add_index(
                    field_name="embedding",
                    index_type="AUTOINDEX", 
                    metric_type="COSINE"
                )
                self.client.create_index(
                    collection_name=self.collection_name,
                    index_params=index_params
                )
                print("DEBUG: create_index finished")
            except Exception as e:
                print(f"DEBUG: create_index failed (likely exists): {e}")

            # Load Collection ONCE
            try:
                self.client.load_collection(self.collection_name)
                self.collection_loaded = True
                print(f"DEBUG: Collection '{self.collection_name}' loaded successfully")
                logger.info(f"✅ Collection '{self.collection_name}' loaded and ready")
            except Exception as e:
                print(f"DEBUG: Collection load error: {e}")
                self.collection_loaded = False

        except Exception as e:
            print(f"DEBUG: Collection creation/setup error: {e}")
            self.client = None

    def store_conversation(self, user_name: str, user_id: str, session_id: str, user_message: str, bot_response: str, intent: str = "general") -> bool:
        """
        Buffers conversation data. Writes to DB only when batch size reached.
        """
        print("DEBUG: Enter store_conversation")
        if not self.client:
            return False

        try:
            # 1. Prepare Data with Cached Embedding
            combined_text = f"{user_message} {bot_response}"
            
            # Try cache first
            embedding = self.embedding_cache.get(combined_text)
            if embedding is None:
                # Cache miss - generate and cache
                embedding = self.embedding_model.encode(combined_text).tolist()
                self.embedding_cache.set(combined_text, embedding)
                logger.debug("🔄 Embedding cache miss - generated new")
            else:
                logger.debug("✅ Embedding cache hit")
            
            # Truncate messages to fit schema limits BEFORE creating data_row
            user_message_truncated = user_message[:2000]
            bot_response_truncated = bot_response[:2000]
            
            # Warn if truncation occurred
            if len(user_message) > 2000:
                logger.warning(f"⚠️ User message truncated from {len(user_message)} to 2000 chars")
            if len(bot_response) > 2000:
                logger.warning(f"⚠️ Bot response truncated from {len(bot_response)} to 2000 chars")

            data_row = {
                "user_name": user_name[:255],
                "user_id": user_id[:255],
                "session_id": session_id[:255],
                "timestamp": datetime.now().isoformat(),
                "user_message": user_message_truncated,
                "bot_response": bot_response_truncated,
                "embedding": embedding,
                "intent": intent[:100]
            }

            # 2. Initialize buffer for this session if not exists
            if session_id not in self.write_buffer:
                self.write_buffer[session_id] = []
                self.buffer_timestamps[session_id] = time.time()  # Track creation time

            # 3. Add to Buffer
            self.write_buffer[session_id].append(data_row)
            current_buffer_size = len(self.write_buffer[session_id])
            
            print(f"DEBUG: Buffered message for session {session_id}. Count: {current_buffer_size}/{self.BATCH_SIZE}")
            logger.info(f"⏳ Buffered {current_buffer_size}/{self.BATCH_SIZE} for {user_name}")

            # 4. Check Batch Size
            if current_buffer_size >= self.BATCH_SIZE:
                print(f"DEBUG: Batch size met. Flushing session {session_id}")
                return self.flush_session(session_id)
            
            return True

        except Exception as e:
            print(f"DEBUG: Exception in store_conversation: {e}")
            logger.error(f"❌ Storage error: {e}")
            return False

    def flush_session(self, session_id: str) -> bool:
        """
        Failsafe: Forces the buffer for a specific session to be written to Milvus immediately.
        Call this on 'Disconnect' or 'App Shutdown'.
        """
        if not self.client:
            return False
        
        # Check if buffer exists and has data
        if session_id not in self.write_buffer or not self.write_buffer[session_id]:
            print(f"DEBUG: No buffer to flush for {session_id}")
            return True # Nothing to do, considered success

        try:
            data_to_insert = self.write_buffer[session_id]
            count = len(data_to_insert)
            
            print(f"DEBUG: Flushing {count} records to Milvus for {session_id}...")
            
            # Collection should already be loaded from init, but check just in case
            if not self.collection_loaded:
                self.client.load_collection(self.collection_name)
                self.collection_loaded = True

            # Bulk Insert
            res = self.client.insert(
                collection_name=self.collection_name,
                data=data_to_insert
            )
            
            print(f"DEBUG: Successfully flushed {count} records. Result: {res}")
            logger.info(f"💾 Batch Saved: {count} records for session {session_id}")

            # Clear Buffer and timestamp after successful insert
            del self.write_buffer[session_id]
            self.buffer_timestamps.pop(session_id, None)
            return True

        except Exception as e:
            print(f"DEBUG: Failed to flush session {session_id}: {e}")
            logger.error(f"❌ Failed to flush buffer: {e}")
            return False

    def get_user_history(self, user_name: str, limit: int = 20, days: int = 30) -> List[Dict]:
        # (Existing logic remains unchanged)
        if not self.client:
            return []
        try:
            # Collection already loaded in init, no need to reload
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            filter_expr = f'user_name == "{user_name}"'
            
            results = self.client.query(
                collection_name=self.collection_name,
                filter=filter_expr,
                output_fields=["user_name", "timestamp", "user_message", "bot_response", "intent"],
                limit=limit,
                consistency_level="Strong"
            )
            if results:
                results = sorted(results, key=lambda x: x.get('timestamp', ''))
            return results if results else []
        except Exception as e:
            logger.error(f"❌ Retrieval error: {e}")
            return []

    def get_user_stats(self, user_name: str) -> Dict:
        # (Existing logic with the NoneType fix from previous steps)
        if not self.client:
            return {}
        try:
            # Collection already loaded in init, no need to reload
            filter_expr = f'user_name == "{user_name}"'
            results = self.client.query(
                collection_name=self.collection_name,
                filter=filter_expr,
                output_fields=["user_name", "intent", "timestamp"],
                limit=1000,
                consistency_level="Strong"
            )
            
            if not results:
                return {"total_turns": 0, "intents": {}, "last_active": "Never"}
            
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
        """Remove buffers older than TTL"""
        if not self.buffer_timestamps:
            return
        
        current_time = time.time()
        stale_sessions = []
        
        for session_id, timestamp in self.buffer_timestamps.items():
            if current_time - timestamp > self.BUFFER_TTL:
                stale_sessions.append(session_id)
        
        for session_id in stale_sessions:
            logger.warning(f"🧹 Cleaning stale buffer for session {session_id}")
            # Try to flush, but if it fails, remove the buffer anyway to prevent infinite retries
            success = self.flush_session(session_id)
            if not success:
                # Flush failed - forcefully remove buffer to prevent retry loop
                logger.error(f"⚠️ Flush failed for {session_id}, forcefully removing buffer")
                self.write_buffer.pop(session_id, None)
                self.buffer_timestamps.pop(session_id, None)
    
    def get_cache_stats(self) -> Dict:
        """Get embedding cache statistics"""
        return self.embedding_cache.get_stats()