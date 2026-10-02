import os
from dotenv import load_dotenv

# Try loading from backend/.env first, then fallback to default search
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_env = os.path.join(os.path.dirname(current_dir), ".env")
if os.path.exists(backend_env):
    load_dotenv(backend_env)
load_dotenv()

class Config:
    """
    Central configuration for the AI Chat Assistant.
    Values can be overridden by environment variables.
    """
    
    # Calendar / Booking Settings
    WORKING_HOURS_START = int(os.getenv("WORKING_HOURS_START", 9))
    WORKING_HOURS_END = int(os.getenv("WORKING_HOURS_END", 18))
    MEETING_DURATION_MINUTES = int(os.getenv("MEETING_DURATION_MINUTES", 60))
    BUFFER_TIME_MINUTES = int(os.getenv("BUFFER_TIME_MINUTES", 15))
    DEFAULT_TIMEZONE = os.getenv("DEFAULT_TIMEZONE", "UTC")
    
    # Google Credentials
    _cred_env = os.getenv("GOOGLE_CREDENTIALS", "client_secret.json")
    if not os.path.isabs(_cred_env) and not os.path.exists(_cred_env):
        _alt_cred = os.path.join(os.path.dirname(current_dir), _cred_env)
        GOOGLE_CREDENTIALS = _alt_cred if os.path.exists(_alt_cred) else _cred_env
    else:
        GOOGLE_CREDENTIALS = _cred_env

    _tok_env = os.getenv("GOOGLE_TOKEN", "token.json")
    if not os.path.isabs(_tok_env) and not os.path.exists(_tok_env):
        _alt_tok = os.path.join(os.path.dirname(current_dir), _tok_env)
        GOOGLE_TOKEN = _alt_tok if os.path.exists(_alt_tok) else _tok_env
    else:
        GOOGLE_TOKEN = _tok_env

    
    # API Keys
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    HF_TOKEN = os.getenv("HF_TOKEN")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "chat-assistant")
    
    # AWS Config
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
    S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "ai-chat-assistant-bucket")
    
    # App Settings
    APP_TITLE = os.getenv("APP_TITLE", "AI Chat Assistant")
    
    # Cache Settings
    EMBEDDING_CACHE_SIZE = int(os.getenv("EMBEDDING_CACHE_SIZE", 1000))
    STATS_CACHE_TTL = int(os.getenv("STATS_CACHE_TTL", 60))  # seconds
    SYSTEM_MESSAGE_CACHE_SIZE = int(os.getenv("SYSTEM_MESSAGE_CACHE_SIZE", 100))
    
    # Cleanup Settings
    STREAM_CLEANUP_INTERVAL = int(os.getenv("STREAM_CLEANUP_INTERVAL", 300))  # 5 minutes
    BUFFER_CLEANUP_INTERVAL = int(os.getenv("BUFFER_CLEANUP_INTERVAL", 300))  # 5 minutes
    STREAM_TTL = int(os.getenv("STREAM_TTL", 300))  # 5 minutes
    BUFFER_TTL = int(os.getenv("BUFFER_TTL", 300))  # 5 minutes
    
    @classmethod
    def get_working_hours(cls):
        return {"start": cls.WORKING_HOURS_START, "end": cls.WORKING_HOURS_END}
