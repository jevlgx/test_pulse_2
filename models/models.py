from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# Request Models
class PredictionRequest(BaseModel):
    """Request model for single prediction"""
    text: str = Field(..., description="Text to analyze for sentiment", min_length=1)
    return_probabilities: bool = Field(default=False, description="Return probability scores for all classes")


class ChatMessage(BaseModel):
    """Request model for chat interaction"""
    message: str = Field(..., description="User message to analyze and respond to", min_length=1)
    context: Optional[List[str]] = Field(default=None, description="Previous conversation context")


class BatchPredictionItem(BaseModel):
    """Single item in batch prediction"""
    id: Optional[str] = Field(default=None, description="Optional ID for tracking")
    text: str = Field(..., description="Text to analyze")


class BatchPredictionRequest(BaseModel):
    """Request model for batch prediction"""
    items: List[BatchPredictionItem] = Field(..., description="List of texts to analyze")
    return_probabilities: bool = Field(default=False, description="Return probability scores")


# Response Models
class PredictionResponse(BaseModel):
    """Response model for prediction"""
    text: str
    sentiment: str
    confidence: float = Field(..., ge=0, le=1)
    probabilities: Optional[Dict[str, float]] = None
    processing_time_ms: float
    timestamp: datetime


class ChatResponse(BaseModel):
    """Response model for chat interaction"""
    user_message: str
    detected_mood: str
    mood_confidence: float = Field(..., ge=0, le=1)
    suggested_response: str
    emotional_indicators: List[str]
    context_considered: bool
    timestamp: datetime


class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction"""
    total_items: int
    processed_items: int
    failed_items: int
    predictions: List[Dict[str, Any]]
    processing_time_ms: float
    timestamp: datetime


class TrainingStatus(BaseModel):
    """Response model for training status"""
    status: str = Field(..., description="Training status: success, failed, or in_progress")
    message: str
    model_id: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    model_info: Optional[Dict[str, Any]] = None
    training_time_seconds: Optional[float] = None
    timestamp: datetime


class ModelInfo(BaseModel):
    """Response model for model information"""
    model_loaded: bool
    model_id: Optional[str] = None
    model_type: Optional[str] = None
    training_date: Optional[datetime] = None
    accuracy: Optional[float] = Field(default=None, ge=0, le=1)
    classes: Optional[List[str]] = None
    total_samples_trained: Optional[int] = Field(default=None, ge=0)
    features_count: Optional[int] = None
    model_size_mb: Optional[float] = None


class HealthCheck(BaseModel):
    """Response model for health check"""
    status: str
    model_loaded: bool
    api_version: str
    uptime_seconds: float
    timestamp: datetime


class ErrorResponse(BaseModel):
    """Response model for errors"""
    error: str
    detail: str
    timestamp: datetime


class FileUploadResponse(BaseModel):
    """Response model for file upload"""
    filename: str
    size_bytes: int
    rows_detected: int
    columns_detected: List[str]
    message: str