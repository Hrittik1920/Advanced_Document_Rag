#config/settings.py
import sys  # Import sys to be able to exit the application
from pydantic import ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
import urllib.parse

class Settings(BaseSettings):
    """
    Manages Applications setting using Pydantic
    Reads the enviroments settings in .env file
    """
    
    LLM_MODEL_NAME: str
    LLM_ENDPOINT: str
    LLM_EMBEDDING_MODEL: str
    DOCUMENTS_DIR: str
    HISTORY_DIR: str
    CROSS_ENCODER_MODEL: str
    QDRANT_URL: str
    CONTEXT_MODEL: str # New setting for the context model
    COLLECTION_NAME: str
    PG_HOST: str
    PG_PORT: int
    PG_DB: str
    PG_USER: str
    PG_PASSWORD: str
    MEMO_COLLECTION: str
    MAX_HISTORY_MESSAGES: int = 6
    MAX_MSG_CHARS: int = 400
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )

try:
    # This is the line that fails
    settings = Settings()
except ValidationError as e:
    print("❌ Configuration Error: Please check your .env file for missing or invalid settings.")
    print(e)
    sys.exit(1)  # Exit the application if settings are invalid