"""
Chat service for mood detection and conversation management
"""

from typing import Dict, Any, List, Optional
from core.preprocessor import TextPreprocessor
from core.model_manager import ModelManager
from core.mood_detector import MoodDetector
from services.prediction_service import PredictionService
from utils.logger import setup_logger

logger = setup_logger(__name__)


class ChatService:
    """Service for handling chat and mood detection operations"""

    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.model_manager = ModelManager()
        self.mood_detector = MoodDetector()
        self.prediction_service = PredictionService()

    async def process_chat(self, message: str, context: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Process chat message with mood detection

        Args:
            message: User message
            context: Conversation context

        Returns:
            Chat processing result
        """
        try:
            # Detect mood
            if self.model_manager.is_model_loaded():
                # Use trained model
                prediction = await self.prediction_service.predict_single(message, return_probabilities=True)
                mood = prediction['sentiment']
                confidence = prediction['confidence']
            else:
                # Use basic mood detection
                mood, confidence = self.mood_detector.detect_basic_mood(message)

            # Get emotional indicators
            indicators = self.mood_detector.detect_emotional_indicators(message)

            # Generate response
            response = self.mood_detector.generate_response(mood, confidence, context)

            # Check context consideration
            context_considered = context is not None and len(context) > 0

            return {
                'mood': mood,
                'confidence': confidence,
                'response': response,
                'indicators': indicators,
                'context_considered': context_considered
            }

        except Exception as e:
            logger.error(f"Chat processing failed: {str(e)}")
            raise

    async def process_conversation(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Process full conversation with mood tracking

        Args:
            messages: List of message dictionaries

        Returns:
            Conversation analysis
        """
        try:
            mood_trajectory = []
            all_moods = []
            transitions = []

            for i, msg_obj in enumerate(messages):
                # Extract message text
                if isinstance(msg_obj, dict):
                    message = msg_obj.get('message', '')
                else:
                    message = msg_obj.message if hasattr(msg_obj, 'message') else str(msg_obj)

                # Detect mood for each message
                if self.model_manager.is_model_loaded():
                    prediction = await self.prediction_service.predict_single(message)
                    mood = prediction['sentiment']
                    confidence = prediction['confidence']
                else:
                    mood, confidence = self.mood_detector.detect_basic_mood(message)

                mood_trajectory.append({
                    'message_index': i,
                    'mood': mood,
                    'confidence': confidence,
                    'message_preview': message[:50] + '...' if len(message) > 50 else message
                })

                all_moods.append(mood)

                # Track transitions
                if i > 0:
                    prev_mood = mood_trajectory[i - 1]['mood']
                    if prev_mood != mood:
                        transition_note = self.mood_detector.analyze_mood_transition(prev_mood, mood)
                        if transition_note:
                            transitions.append({
                                'from_index': i - 1,
                                'to_index': i,
                                'from_mood': prev_mood,
                                'to_mood': mood,
                                'note': transition_note
                            })

            # Determine dominant mood
            mood_counts = {}
            for mood in all_moods:
                mood_counts[mood] = mood_counts.get(mood, 0) + 1
            dominant_mood = max(mood_counts, key=mood_counts.get)

            # Generate suggestions
            suggestions = self._generate_conversation_suggestions(
                mood_trajectory, dominant_mood, transitions
            )

            return {
                'analysis': {
                    'total_messages': len(messages),
                    'mood_distribution': mood_counts,
                    'mood_stability': self._calculate_mood_stability(all_moods),
                    'emotional_range': len(set(all_moods))
                },
                'mood_trajectory': mood_trajectory,
                'dominant_mood': dominant_mood,
                'transitions': transitions,
                'suggestions': suggestions
            }

        except Exception as e:
            logger.error(f"Conversation processing failed: {str(e)}")
            raise

    def _calculate_mood_stability(self, moods: List[str]) -> float:
        """
        Calculate mood stability score (0-1)

        Args:
            moods: List of moods

        Returns:
            Stability score
        """
        if len(moods) <= 1:
            return 1.0

        changes = sum(1 for i in range(1, len(moods)) if moods[i] != moods[i - 1])
        max_changes = len(moods) - 1
        stability = 1.0 - (changes / max_changes)

        return stability

    def _generate_conversation_suggestions(self, trajectory: List[Dict],
                                           dominant_mood: str,
                                           transitions: List[Dict]) -> List[str]:
        """
        Generate conversation suggestions based on analysis

        Args:
            trajectory: Mood trajectory
            dominant_mood: Most common mood
            transitions: Mood transitions

        Returns:
            List of suggestions
        """
        suggestions = []

        # Suggest based on dominant mood
        mood_suggestions = {
            'sad': "Consider offering more support and empathy in the conversation.",
            'angry': "Try to identify and address the source of frustration.",
            'anxious': "Provide reassurance and help break down concerns into manageable parts.",
            'happy': "Maintain the positive momentum and celebrate successes.",
            'excited': "Channel this enthusiasm into productive actions or plans.",
            'neutral': "Explore deeper to understand underlying feelings or needs."
        }

        if dominant_mood in mood_suggestions:
            suggestions.append(mood_suggestions[dominant_mood])

        # Suggest based on transitions
        if len(transitions) > 2:
            suggestions.append(
                "The conversation shows multiple mood shifts. Consider checking in on emotional well-being.")

        # Check for mood improvement or decline
        if trajectory:
            first_mood = trajectory[0]['mood']
            last_mood = trajectory[-1]['mood']

            positive_moods = {'happy', 'excited'}
            negative_moods = {'sad', 'angry', 'anxious'}

            if first_mood in negative_moods and last_mood in positive_moods:
                suggestions.append("Great progress! The mood has improved throughout the conversation.")
            elif first_mood in positive_moods and last_mood in negative_moods:
                suggestions.append("The mood has declined. Consider what might have triggered this change.")

        return suggestions if suggestions else ["Continue with active listening and empathetic responses."]