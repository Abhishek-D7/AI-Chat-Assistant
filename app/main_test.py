"""
================================================================================
app/main.py - COMPLETE with History/Stats Endpoints
================================================================================
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
from app.persistence import ChatPersistence

persistence = ChatPersistence()

async def async_store_conversation(user_name, user_id, session_id, user_message, bot_response, detected_intent):
    await asyncio.to_thread(
        persistence.store_conversation,
        user_name, user_id, session_id,
        user_message, bot_response, detected_intent
    )

async def async_flush_session(persistence_instance: ChatPersistence, session_id: str):
    await asyncio.to_thread(persistence_instance.flush_session, session_id)

from app.langgraph_graph import process_user_message_with_context, process_user_message_with_context_streaming

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================================================================
# FASTAPI SETUP
# ================================================================================
app = FastAPI(
    title="B2B AI Chat API - Low Latency",
    version="4.2.1",
    description="Dynamic agent-driven chat with proper SSE streaming and login"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for active streams/cancellation
active_streams: Dict[str, asyncio.Event] = {}

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

class ChatResponse(BaseModel):
    user_message: str
    bot_response: str
    user_id: str
    user_name: str
    intent: str
    session_id: str

class CancelRequest(BaseModel):
    session_id: str

# ================================================================================
# SSE FORMATTING HELPER
# ================================================================================
def sse_streamer(async_generator: AsyncGenerator[Dict, None]) -> AsyncGenerator[str, None]:
    """
    Helper function to wrap the async generator output into
    Server-Sent Events (SSE) format: 'data: {json_payload}\n\n'.
    """
    async def wrapper():
        async for chunk in async_generator:
            try:
                # The chunk is a dictionary like {'type': 'token', 'content': 'Hello'}
                json_data = json.dumps(chunk)
                # CRITICAL FIX: Format as SSE
                yield f"data: {json_data}\n\n"
                
                # Exit the stream once the terminal message is sent
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

@app.get("/user/{user_name}/history")
async def get_user_history(user_name: str, limit: int = 20):
    """Get user chat history"""
    try:
        history = await asyncio.to_thread(
            persistence.get_user_history,
            user_name,
            limit=limit
        )
        return {"user_name": user_name, "history": history, "count": len(history)}
    except Exception as e:
        logger.error(f"❌ History error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user/{user_name}/stats")
async def get_user_stats(user_name: str):
    """Get user statistics"""
    try:
        stats = await asyncio.to_thread(
            persistence.get_user_stats,
            user_name
        )
        return {"user_name": user_name, **stats}
    except Exception as e:
        logger.error(f"❌ Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def generate_stream(request: ChatRequest, background_tasks: BackgroundTasks):
    user_message = request.user_message.strip()
    user_name = request.user_name
    user_id = request.user_id
    context_summary = request.context_summary
    session_id = str(uuid.uuid4())
    
    # STREAMING PATH
    logger.info(f"💬 Chat from {user_name}: {user_message} (stream=True)")
    logger.info(f"🌊 Streaming mode enabled (session: {session_id[:8]})")
    
    cancel_flag = asyncio.Event()
    active_streams[session_id] = cancel_flag
    
    async def stream_and_cleanup():
        raw_stream_generator = process_user_message_with_context_streaming(
            user_message, user_id, user_name, context_summary, cancel_flag=cancel_flag
        )
        
        # Apply the SSE formatting wrapper
        sse_generator = sse_streamer(raw_stream_generator)
        
        full_bot_response = ""
        detected_intent = "general"
        
        try:
            async for chunk_string in sse_generator:
                yield chunk_string
                
                # Track content and intent for persistence
                if chunk_string.startswith("data: "):
                    try:
                        data_payload = json.loads(chunk_string[6:].strip())
                        if data_payload.get("type") == "token":
                            full_bot_response += data_payload.get("content", "")
                        elif data_payload.get("type") == "intent":
                            if detected_intent == "general":
                                detected_intent = data_payload.get("content", "general")
                    except json.JSONDecodeError:
                        pass
        finally:
            # Cleanup global state and persist conversation
            active_streams.pop(session_id, None)
            logger.info(f"🧹 Stream {session_id[:8]} cleaned up")
            
            # Non-blocking persistence
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
    """Cancel active stream"""
    session_id = request.session_id
    if session_id in active_streams:
        active_streams[session_id].set()
        logger.info(f"✅ Cancel flag set for session: {session_id[:8]}")
        return {
            "status": "cancelled",
            "session_id": session_id,
            "message": "Stream cancellation requested"
        }
    else:
        logger.warning(f"⚠️ Session {session_id[:8]} not found in active streams")
        return {
            "status": "not_found",
            "session_id": session_id,
            "message": "Session not active or already completed"
        }

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.on_event("startup")
async def startup():
    logger.info("🚀 Low-Latency API Started")
    logger.info("✅ Persistence initialized")
    logger.info("✅ Streaming: ENABLED (SSE)")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)