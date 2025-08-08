"""
Application configuration settings
"""

import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # Project Info
    PROJECT_NAME: str = "Sentiment Analysis & Mood Detection API"
    PROJECT_DESCRIPTION: str = "Professional API for training sentiment models and detecting user emotions"
    PROJECT_VERSION: str = "1.0.0"

    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True

    # CORS
    ALLOWED_ORIGINS: List[str] = ["*"]

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    MODELS_DIR: Path = BASE_DIR / "models"
    DATA_DIR: Path = BASE_DIR / "data"

    # ML Model Configuration
    MAX_FEATURES: int = 5000
    NGRAM_RANGE: tuple = (1, 2)
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42

    # Model Training
    ENSEMBLE_MODELS: List[str] = ["logistic_regression", "naive_bayes", "random_forest"]
    MAX_ITER_LOGISTIC: int = 1000
    N_ESTIMATORS_RF: int = 100

    # File Upload
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: List[str] = [".csv"]

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # API Rate Limiting (optional)
    RATE_LIMIT_ENABLED: bool = False
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_PERIOD: int = 60  # seconds

    class Config:
        env_file = ".env"
        case_sensitive = True


# Create settings instance
settings = Settings()

# Create necessary directories
settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)