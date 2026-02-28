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