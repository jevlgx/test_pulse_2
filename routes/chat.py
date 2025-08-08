"""
Chat endpoint with mood detection
"""

from fastapi import APIRouter, HTTPException
from models.models import ChatMessage, ChatResponse
from services.chat_service import ChatService
from utils.logger import setup_logger
from datetime import datetime

logger = setup_logger(__name__)
router = APIRouter()

chat_service = ChatService()


@router.post("/chat", response_model=ChatResponse)
async def chat_with_mood_detection(chat_message: ChatMessage):
    """
    Chat endpoint with mood detection and empathetic response generation.

    This endpoint:
    1. Detects the emotional mood in the user's message
    2. Generates an appropriate empathetic response
    3. Provides emotional indicators found in the text
    4. Can consider conversation context if provided
    """

    try:
        # Process chat message
        result = await chat_service.process_chat(
            message=chat_message.message,
            context=chat_message.context
        )

        return ChatResponse(
            user_message=chat_message.message,
            detected_mood=result['mood'],
            mood_confidence=result['confidence'],
            suggested_response=result['response'],
            emotional_indicators=result['indicators'],
            context_considered=result['context_considered'],
            timestamp=datetime.now()
        )

    except Exception as e:
        logger.error(f"Chat processing failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Chat processing failed: {str(e)}"
        )


@router.post("/chat/conversation")
async def manage_conversation(messages: list[ChatMessage]):
    """
    Process a full conversation with mood tracking across messages.

    Analyzes mood transitions and provides conversation insights.
    """

    try:
        # Process conversation
        result = await chat_service.process_conversation(messages)

        return {
            "conversation_analysis": result['analysis'],
            "mood_trajectory": result['mood_trajectory'],
            "dominant_mood": result['dominant_mood'],
            "mood_transitions": result['transitions'],
            "suggestions": result['suggestions'],
            "timestamp": datetime.now()
        }

    except Exception as e:
        logger.error(f"Conversation processing failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Conversation processing failed: {str(e)}"
        )