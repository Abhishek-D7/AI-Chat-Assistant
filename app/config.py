import os
from dotenv import load_dotenv

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
    GOOGLE_CREDENTIALS_FILE = os.getenv("GOOGLE_CREDENTIALS_FILE", "client_secret.json")
    GOOGLE_TOKEN_FILE = os.getenv("GOOGLE_TOKEN_FILE", "token.json")
    
    # API Keys
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # App Settings
    APP_TITLE = os.getenv("APP_TITLE", "B2B AI Chat Assistant")
    
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
