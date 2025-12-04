"""
================================================================================
app/main.py - Updated for LangGraph Memory
================================================================================

UPDATES:
1. ChatRequest now accepts 'thread_id'.
2. /chat endpoint passes 'thread_id' to processing functions.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, AsyncGenerator
import uuid
import logging
import json
import asyncio
from datetime import datetime
import time
from app.persistence import ChatPersistence
from app.cache import UserStatsCache
from app.config import Config

# --- PERSISTENCE INITIALIZATION AND ASYNC WRAPPERS ---
persistence = ChatPersistence()

async def async_store_conversation(user_name, user_id, session_id, user_message, bot_response, detected_intent):
    """Run store_conversation in a separate thread for non-blocking I/O."""
    await asyncio.to_thread(
        persistence.store_conversation, 
        user_name, user_id, session_id, 
        user_message, bot_response, detected_intent
    )

async def async_flush_session(persistence_instance: ChatPersistence, session_id: str):
    """Run flush_session in a separate thread for non-blocking I/O."""
    await asyncio.to_thread(persistence_instance.flush_session, session_id)
# -------------------------------------------------------------------------

# Import both streaming and non-streaming functions from langgraph_graph
from app.langgraph_graph import (
    process_user_message_with_context,
    process_user_message_with_context_streaming
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================================================================
# FASTAPI SETUP
# ================================================================================

app = FastAPI(
    title="B2B AI Chat API",
    version="Demo with LangGraph Memory",
    description="Dynamic agent-driven chat with LangGraph Memory"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for active streams/cancellation with TTL tracking
active_streams: Dict[str, asyncio.Event] = {}
stream_timestamps: Dict[str, float] = {}  # Track stream creation time

# User stats cache
stats_cache = UserStatsCache(ttl_seconds=Config.STATS_CACHE_TTL)

# ================================================================================
# DATA MODELS
# ================================================================================

class LoginRequest(BaseModel):
    user_name: str
    password: Optional[str] = None

class LoginResponse(BaseModel):
    user_name: str
    user_id: str
    status: str

class ChatRequest(BaseModel):
    user_message: str
    user_name: str
    user_id: str
    stream_enabled: bool = True
    context_summary: str = ""
    thread_id: Optional[str] = None # <--- ADDED for Memory

class ChatResponse(BaseModel):
    user_message: str
    bot_response: str
    user_id: str
    user_name: str
    intent: str
    session_id: str
    thread_id: str

class CancelRequest(BaseModel):
    session_id: str

# ================================================================================
# SSE FORMATTING HELPER
# ================================================================================

def sse_streamer(async_generator: AsyncGenerator[Dict, None]) -> AsyncGenerator[str, None]:
    """Wraps the dictionary output into Server-Sent Events (SSE) format."""
    async def wrapper():
        async for chunk in async_generator:
            try:
                json_data = json.dumps(chunk)
                yield f"data: {json_data}\n\n"
                
                if chunk.get("type") in ["done", "cancelled", "error"]:
                    break
                    
            except Exception as e:
                logger.error(f"❌ SSE formatting error: {e}", exc_info=True)
                yield f"data: {json.dumps({'type': 'error', 'content': 'Internal streaming error'})}\n\n"
                break
    return wrapper()


# ================================================================================
# ROUTES
# ================================================================================

@app.post("/user/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """Handles user login."""
    user_name = request.user_name
    user_id = f"user_{user_name.lower().replace(' ', '_')}"
    logger.info(f"👤 Login attempt: {user_name}")
    logger.info(f"✅ Login success: {user_name} ({user_id})")
    return LoginResponse(
        user_name=user_name,
        user_id=user_id,
        status="success"
    )

@app.post("/chat")
async def generate_chat_response(request: ChatRequest, background_tasks: BackgroundTasks):
    user_message = request.user_message.strip()
    user_name = request.user_name
    user_id = request.user_id
    context_summary = request.context_summary
    
    # 1. Handle IDs
    # session_id: unique for this specific request/turn (used for Milvus logs)
    # thread_id: unique for the conversation thread (used for LangGraph Memory)
    
    session_id = str(uuid.uuid4())
    thread_id = request.thread_id or session_id # If no thread_id provided, start new thread
    
    if not request.stream_enabled:
        # --- NON-STREAMING PATH ---
        logger.info(f"💬 Chat from {user_name}: {user_message} (stream=False, thread={thread_id})")
        
        try:
            result = await asyncio.to_thread(
                process_user_message_with_context,
                user_message, user_id, user_name, context_summary, thread_id
            )
            
            bot_response = result.get("bot_response", "")
            detected_intent = result.get("intent", "general")
            
            await async_store_conversation(
                user_name, user_id, session_id, 
                user_message, bot_response, detected_intent
            )
            
            return ChatResponse(
                user_message=user_message,
                bot_response=bot_response,
                user_id=user_id,
                user_name=user_name,
                intent=detected_intent,
                session_id=session_id,
                thread_id=thread_id
            )
        except Exception as e:
            logger.error(f"❌ Non-streaming chat error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))


    # --- STREAMING PATH ---
    logger.info(f"💬 Chat from {user_name}: {user_message} (stream=True, thread={thread_id})")
    
    cancel_flag = asyncio.Event()
    active_streams[session_id] = cancel_flag
    stream_timestamps[session_id] = time.time()  # Track creation time

    async def stream_and_cleanup():
        # Get the async generator
        raw_stream_generator = process_user_message_with_context_streaming(
            user_message, user_id, user_name, context_summary, 
            cancel_flag=cancel_flag, 
            thread_id=thread_id
        )
        
        sse_generator = sse_streamer(raw_stream_generator)

        full_bot_response = ""
        detected_intent = "general"
        
        try:
            async for chunk_string in sse_generator:
                yield chunk_string

                # Track content and intent for background persistence
                if chunk_string.startswith("data: "):
                    try:
                        data_payload = json.loads(chunk_string[6:].strip())
                        
                        if data_payload.get("type") == "token":
                            full_bot_response += data_payload.get("content", "")
                        elif data_payload.get("type") == "intent":
                            val = data_payload.get("content", "general")
                            if val and val != "general":
                                detected_intent = val
                        
                    except json.JSONDecodeError:
                        pass 

        finally:
            active_streams.pop(session_id, None)
            
            asyncio.create_task(
                async_store_conversation(
                    user_name, user_id, session_id,
                    user_message, full_bot_response, detected_intent
                )
            )
            asyncio.create_task(async_flush_session(persistence, session_id))


    return StreamingResponse(
        stream_and_cleanup(), 
        media_type="text/event-stream"
    )

@app.post("/chat/cancel")
async def cancel_stream(request: CancelRequest):
    session_id = request.session_id
    if session_id in active_streams:
        active_streams[session_id].set()
        return {"status": "cancelled", "session_id": session_id}
    else:
        return {"status": "not_found", "session_id": session_id}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/user/{user_name}/history")
async def get_user_history(user_name: str, limit: int = 20):
    try:
        history = await asyncio.to_thread(persistence.get_user_history, user_name, limit=limit)
        return {"user_name": user_name, "history": history, "count": len(history)}
    except Exception as e:
        logger.error(f"❌ History error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user/{user_name}/stats")
async def get_user_stats(user_name: str):
    """Get user statistics with caching"""
    try:
        # Try cache first
        cached_stats = stats_cache.get(user_name)
        if cached_stats is not None:
            logger.debug(f"✅ Stats cache hit for {user_name}")
            return {"user_name": user_name, **cached_stats}
        
        # Cache miss - fetch from DB
        stats = await asyncio.to_thread(persistence.get_user_stats, user_name)
        
        # Cache the result
        stats_cache.set(user_name, stats)
        
        return {"user_name": user_name, **stats}
    except Exception as e:
        logger.error(f"❌ Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup():
    logger.info("🚀 Low-Latency API Started")
    logger.info("✅ Persistence initialized")
    logger.info("✅ LangGraph Memory: ENABLED")
    
    # Eager graph initialization (prevents 2-5s delay on first request)
    logger.info("🔨 Initializing agent graph...")
    from app.langgraph_graph import get_agent_graph
    get_agent_graph()
    logger.info("✅ Agent graph initialized and ready")
    
    # Start background cleanup tasks
    asyncio.create_task(cleanup_stale_streams())
    asyncio.create_task(cleanup_stale_buffers())
    logger.info("✅ Background cleanup tasks started")

async def cleanup_stale_streams():
    """Background task to clean up abandoned streams"""
    while True:
        await asyncio.sleep(Config.STREAM_CLEANUP_INTERVAL)
        current_time = time.time()
        stale_sessions = []
        
        for session_id, timestamp in stream_timestamps.items():
            if current_time - timestamp > Config.STREAM_TTL:
                stale_sessions.append(session_id)
        
        for session_id in stale_sessions:
            # Cleanup global state and persist conversation
            active_streams.pop(session_id, None)
            stream_timestamps.pop(session_id, None)  # Remove timestamp
            logger.info(f"🧹 Stream {session_id[:8]} cleaned up")
        
        if stale_sessions:
            logger.info(f"🧹 Cleaned {len(stale_sessions)} stale streams")

async def cleanup_stale_buffers():
    """Background task to clean up abandoned buffers"""
    while True:
        await asyncio.sleep(Config.BUFFER_CLEANUP_INTERVAL)
        persistence.cleanup_stale_buffers()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/cache/stats")
def get_cache_stats():
    """Get cache statistics for monitoring"""
    return {
        "embedding_cache": persistence.get_cache_stats(),
        "stats_cache": {
            "size": len(stats_cache._cache),
            "max_size": stats_cache._cache.maxsize
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)