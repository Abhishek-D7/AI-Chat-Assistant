"""
app/worker.py
Background worker for async tasks (optional)
"""

from celery import Celery
import os

# Initialize Celery (optional for background tasks)
celery_app = Celery(
    'ai_chat_assistant',
    broker=os.getenv('CELERY_BROKER', 'redis://localhost:6379/0'),
    backend=os.getenv('CELERY_BACKEND', 'redis://localhost:6379/0')
)


@celery_app.task
def store_conversation_async(user_id: str, session_id: str, 
                              user_message: str, bot_response: str):
    """Store conversation to Zilliz asynchronously"""
    from app.persistence import ZillizPersistence
    
    zilliz = ZillizPersistence()
    zilliz.store_conversation(user_id, session_id, user_message, bot_response)
    
    return {"status": "stored", "user_id": user_id}


@celery_app.task
def send_notification(user_id: str, message: str):
    """Send notification (placeholder)"""
    print(f"Notification for {user_id}: {message}")
    return {"status": "sent"}
