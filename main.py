"""
Main FastAPI application entry point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from routes import train, predict, chat, health, batch
from core.model_manager import ModelManager
from utils.logger import setup_logger
from config import settings

# Setup logger
logger = setup_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("Starting Sentiment Analysis API...")

    # Load existing model if available
    model_manager = ModelManager()
    loaded = model_manager.load_latest_model()
    if loaded:
        logger.info("Existing model loaded successfully")
    else:
        logger.info("No existing model found. Please train a model using /api/v1/train")

    yield

    # Shutdown
    logger.info("Shutting down Sentiment Analysis API...")


# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_DESCRIPTION,
    version=settings.PROJECT_VERSION,
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(train.router, prefix="/api", tags=["Training"])
app.include_router(predict.router, prefix="/api", tags=["Prediction"])
app.include_router(chat.router, prefix="/api", tags=["Chat"])
app.include_router(batch.router, prefix="/api", tags=["Batch Operations"])


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": settings.PROJECT_NAME,
        "version": settings.PROJECT_VERSION,
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
        "api_prefix": "/api/v1"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )