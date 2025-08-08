"""
Batch operations endpoint
"""

from fastapi import APIRouter, File, UploadFile, HTTPException
from models.models import BatchPredictionRequest, BatchPredictionResponse
from services.prediction_service import PredictionService
from utils.logger import setup_logger
from datetime import datetime
import pandas as pd
from io import BytesIO
import time

logger = setup_logger(__name__)
router = APIRouter()

prediction_service = PredictionService()


@router.post("/batch/predict", response_model=BatchPredictionResponse)
async def batch_predict_json(request: BatchPredictionRequest):
    """
    Batch prediction from JSON input.

    Process multiple texts in a single request.
    """

    start_time = time.time()

    try:
        # Extract texts from request
        texts = [item.text for item in request.items]
        ids = [item.id for item in request.items if item.id]

        # Perform batch prediction
        results = await prediction_service.predict_batch(
            texts=texts,
            return_probabilities=request.return_probabilities
        )

        # Add IDs if provided
        if ids and len(ids) == len(results):
            for i, result in enumerate(results):
                result['id'] = ids[i]

        # Calculate statistics
        failed_count = sum(1 for r in results if 'error' in r)
        processing_time = (time.time() - start_time) * 1000

        return BatchPredictionResponse(
            total_items=len(request.items),
            processed_items=len(results) - failed_count,
            failed_items=failed_count,
            predictions=results,
            processing_time_ms=processing_time,
            timestamp=datetime.now()
        )

    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@router.post("/batch/predict-csv")
async def batch_predict_csv(
        file: UploadFile = File(..., description="CSV file with text data for batch prediction")
):
    """
    Batch prediction from CSV file.

    CSV should contain either 'message' or 'text' column.
    Optionally can include an 'id' column for tracking.
    """

    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")

    start_time = time.time()

    try:
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(BytesIO(contents))

        # Validate columns
        text_column = None
        if 'message' in df.columns:
            text_column = 'message'
        elif 'text' in df.columns:
            text_column = 'text'
        else:
            raise HTTPException(
                status_code=400,
                detail="CSV must contain 'message' or 'text' column"
            )

        # Get texts
        texts = df[text_column].tolist()

        # Perform batch prediction
        results = await prediction_service.predict_batch(texts=texts, return_probabilities=True)

        # Add results to dataframe
        df['predicted_sentiment'] = [r['sentiment'] for r in results]
        df['confidence'] = [r['confidence'] for r in results]

        # Prepare response
        processing_time = (time.time() - start_time) * 1000

        return {
            "status": "success",
            "total_predictions": len(results),
            "processing_time_ms": processing_time,
            "predictions": df.to_dict('records'),
            "summary": {
                "sentiment_distribution": df['predicted_sentiment'].value_counts().to_dict(),
                "average_confidence": df['confidence'].mean(),
                "min_confidence": df['confidence'].min(),
                "max_confidence": df['confidence'].max()
            },
            "timestamp": datetime.now()
        }

    except Exception as e:
        logger.error(f"CSV batch prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@router.post("/batch/analyze")
async def batch_analyze(
        file: UploadFile = File(..., description="CSV file for comprehensive analysis")
):
    """
    Comprehensive batch analysis with statistics and insights.

    Provides detailed analysis including sentiment distribution,
    confidence metrics, and text statistics.
    """

    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")

    try:
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(BytesIO(contents))

        # Determine text column
        text_column = 'message' if 'message' in df.columns else 'text'
        if text_column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail="CSV must contain 'message' or 'text' column"
            )

        # Get predictions
        texts = df[text_column].tolist()
        results = await prediction_service.predict_batch(texts=texts, return_probabilities=True)

        # Add predictions to dataframe
        df['sentiment'] = [r['sentiment'] for r in results]
        df['confidence'] = [r['confidence'] for r in results]

        # Calculate text statistics
        df['text_length'] = df[text_column].str.len()
        df['word_count'] = df[text_column].str.split().str.len()

        # Sentiment analysis
        sentiment_stats = {}
        for sentiment in df['sentiment'].unique():
            sentiment_df = df[df['sentiment'] == sentiment]
            sentiment_stats[sentiment] = {
                'count': len(sentiment_df),
                'percentage': len(sentiment_df) / len(df) * 100,
                'avg_confidence': sentiment_df['confidence'].mean(),
                'avg_text_length': sentiment_df['text_length'].mean(),
                'avg_word_count': sentiment_df['word_count'].mean()
            }

        # Identify edge cases
        low_confidence = df[df['confidence'] < 0.5]
        very_short = df[df['word_count'] < 3]
        very_long = df[df['word_count'] > 100]

        return {
            "status": "success",
            "total_analyzed": len(df),
            "sentiment_statistics": sentiment_stats,
            "overall_metrics": {
                "average_confidence": df['confidence'].mean(),
                "confidence_std": df['confidence'].std(),
                "average_text_length": df['text_length'].mean(),
                "average_word_count": df['word_count'].mean()
            },
            "edge_cases": {
                "low_confidence_count": len(low_confidence),
                "very_short_texts": len(very_short),
                "very_long_texts": len(very_long)
            },
            "recommendations": _generate_analysis_recommendations(df, sentiment_stats),
            "timestamp": datetime.now()
        }

    except Exception as e:
        logger.error(f"Batch analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")


def _generate_analysis_recommendations(df: pd.DataFrame, sentiment_stats: dict) -> list:
    """Generate recommendations based on analysis"""
    recommendations = []

    # Check confidence levels
    avg_confidence = df['confidence'].mean()
    if avg_confidence < 0.7:
        recommendations.append(
            "Low average confidence detected. Consider improving model training with more diverse data.")

    # Check sentiment balance
    sentiment_counts = df['sentiment'].value_counts()
    if len(sentiment_counts) > 0:
        max_count = sentiment_counts.max()
        min_count = sentiment_counts.min()
        if max_count / min_count > 5:
            recommendations.append("Highly imbalanced sentiment distribution detected in results.")

    # Check text lengths
    very_short_ratio = len(df[df['word_count'] < 3]) / len(df)
    if very_short_ratio > 0.2:
        recommendations.append(
            f"{very_short_ratio:.1%} of texts are very short. Consider minimum text length requirements.")

    if not recommendations:
        recommendations.append("Analysis looks good. Results appear well-distributed with good confidence levels.")

    return recommendations