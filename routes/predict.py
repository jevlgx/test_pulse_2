"""
Prediction endpoint for sentiment analysis
"""

from fastapi import APIRouter, HTTPException
from models.models import PredictionRequest, PredictionResponse
from services.prediction_service import PredictionService
from utils.logger import setup_logger
from datetime import datetime
import time

logger = setup_logger(__name__)
router = APIRouter()

prediction_service = PredictionService()


@router.post("/predict", response_model=PredictionResponse)
async def predict_sentiment(request: PredictionRequest):
    """
    Predict sentiment for a given text.

    Returns sentiment classification with confidence score.
    Optionally returns probability distribution across all classes.
    """

    start_time = time.time()

    try:
        # Perform prediction
        result = await prediction_service.predict_single(
            text=request.text,
            return_probabilities=request.return_probabilities
        )

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        return PredictionResponse(
            text=request.text,
            sentiment=result['sentiment'],
            confidence=result['confidence'],
            probabilities=result.get('probabilities'),
            processing_time_ms=processing_time,
            timestamp=datetime.now()
        )

    except ValueError as e:
        # Model not loaded
        raise HTTPException(
            status_code=503,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post("/predict/explain")
async def predict_with_explanation(request: PredictionRequest):
    """
    Predict sentiment with detailed explanation of the decision.

    Provides additional insights into why the model made its prediction.
    """

    try:
        # Get prediction with explanation
        result = await prediction_service.predict_with_explanation(
            text=request.text
        )

        return {
            "text": request.text,
            "sentiment": result['sentiment'],
            "confidence": result['confidence'],
            "explanation": result['explanation'],
            "important_words": result['important_words'],
            "emotional_indicators": result['emotional_indicators'],
            "timestamp": datetime.now()
        }

    except ValueError as e:
        raise HTTPException(
            status_code=503,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Prediction with explanation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )