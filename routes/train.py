"""
Training endpoint for model training
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks
from models.models import TrainingStatus
from services.training_service import TrainingService
from utils.logger import setup_logger
from datetime import datetime

logger = setup_logger(__name__)
router = APIRouter()

training_service = TrainingService()


@router.post("/train", response_model=TrainingStatus)
async def train_model(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(..., description="CSV file with 'message' and 'sentiment' columns")
):
    """
    Train a sentiment analysis model from CSV data.

    The CSV file should contain:
    - 'message' or 'text' column: The text to analyze
    - 'sentiment' or 'feeling/feelings' column: The sentiment label

    Example CSV:
    ```
    message,sentiment
    "I love this product!",happy
    "This is terrible",angry
    ```
    """

    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="File must be a CSV. Received: " + file.filename
        )

    try:
        # Read file contents
        contents = await file.read()

        # Start training
        logger.info(f"Starting training with file: {file.filename}")
        result = await training_service.train_model(contents, file.filename)

        return TrainingStatus(
            status=result['status'],
            message=result['message'],
            model_id=result.get('model_id'),
            metrics=result.get('metrics'),
            model_info=result.get('model_info'),
            training_time_seconds=result.get('training_time'),
            timestamp=datetime.now()
        )

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.post("/train/validate")
async def validate_training_data(
        file: UploadFile = File(..., description="CSV file to validate")
):
    """
    Validate training data without actually training the model.
    Useful for checking data format and quality before training.
    """

    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="File must be a CSV"
        )

    try:
        contents = await file.read()
        validation_result = await training_service.validate_training_data(contents)

        return {
            "valid": validation_result['valid'],
            "message": validation_result['message'],
            "statistics": validation_result.get('statistics'),
            "warnings": validation_result.get('warnings', [])
        }

    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")