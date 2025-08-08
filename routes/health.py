"""
Health and model information endpoints
"""

from fastapi import APIRouter
from models.models import HealthCheck, ModelInfo
from core.model_manager import ModelManager
from config import settings
from utils.logger import setup_logger
from datetime import datetime
import time

logger = setup_logger(__name__)
router = APIRouter()

# Track application start time
START_TIME = time.time()


@router.get("/health", response_model=HealthCheck)
async def health_check():
    """
    Health check endpoint to verify API status.

    Returns:
    - API status
    - Model loading status
    - Uptime information
    """

    model_manager = ModelManager()
    uptime = time.time() - START_TIME

    return HealthCheck(
        status="healthy",
        model_loaded=model_manager.is_model_loaded(),
        api_version=settings.PROJECT_VERSION,
        uptime_seconds=uptime,
        timestamp=datetime.now()
    )


@router.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """
    Get detailed information about the currently loaded model.

    Returns:
    - Model ID and type
    - Training date and metrics
    - Supported sentiment classes
    - Model size and features
    """

    model_manager = ModelManager()
    info = model_manager.get_model_info()

    return ModelInfo(
        model_loaded=info.get('model_loaded', False),
        model_id=info.get('model_id'),
        model_type=info.get('model_type'),
        training_date=info.get('training_date'),
        accuracy=info.get('accuracy'),
        classes=info.get('classes'),
        total_samples_trained=info.get('total_samples_trained'),
        features_count=info.get('features_count'),
        model_size_mb=info.get('model_size_mb')
    )


@router.get("/models/list")
async def list_available_models():
    """
    List all available trained models.

    Returns list of models with their metadata.
    """

    model_manager = ModelManager()
    models = model_manager.list_models()

    return {
        "total_models": len(models),
        "models": models,
        "current_model": model_manager.model_metadata.get('model_id')
    }


@router.post("/model/load/{model_id}")
async def load_specific_model(model_id: str):
    """
    Load a specific model by ID.

    Args:
        model_id: The ID of the model to load
    """

    model_manager = ModelManager()
    success = model_manager.load_model(model_id)

    if success:
        return {
            "status": "success",
            "message": f"Model {model_id} loaded successfully",
            "model_info": model_manager.get_model_info()
        }
    else:
        return {
            "status": "error",
            "message": f"Failed to load model {model_id}"
        }


@router.delete("/model/{model_id}")
async def delete_model(model_id: str):
    """
    Delete a specific model.

    Args:
        model_id: The ID of the model to delete
    """

    model_manager = ModelManager()
    success = model_manager.delete_model(model_id)

    if success:
        return {
            "status": "success",
            "message": f"Model {model_id} deleted successfully"
        }
    else:
        return {
            "status": "error",
            "message": f"Failed to delete model {model_id}"
        }