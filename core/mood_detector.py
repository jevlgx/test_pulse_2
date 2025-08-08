"""
Mood detection and response generation module
"""

import random
from typing import List, Dict, Tuple, Optional
from utils.logger import setup_logger

logger = setup_logger(__name__)


class MoodDetector:
    """Mood detection and empathetic response generation"""

    def __init__(self):
        """Initialize mood detector with response templates"""
        self.mood_keywords = self._initialize_mood_keywords()
        self.response_templates = self._initialize_response_templates()

    def _initialize_mood_keywords(self) -> Dict[str, List[str]]:
        """Initialize keyword mappings for basic mood detection"""
        return {
            'happy': [
                'happy', 'joy', 'joyful', 'excited', 'love', 'great', 'wonderful',
                'amazing', 'fantastic', 'excellent', 'awesome', 'delighted', 'pleased',
                'cheerful', 'glad', 'thrilled', 'ecstatic', 'elated'
            ],
            'sad': [
                'sad', 'depressed', 'down', 'unhappy', 'miserable', 'crying', 'tears',
                'sorrow', 'grief', 'melancholy', 'dejected', 'heartbroken', 'gloomy',
                'despair', 'hopeless', 'lonely', 'disappointed'
            ],
            'angry': [
                'angry', 'mad', 'furious', 'pissed', 'annoyed', 'frustrated', 'rage',
                'irritated', 'aggravated', 'outraged', 'livid', 'hostile', 'bitter',
                'resentful', 'indignant', 'infuriated'
            ],
            'anxious': [
                'worried', 'anxious', 'nervous', 'scared', 'afraid', 'concerned', 'stressed',
                'tense', 'uneasy', 'apprehensive', 'fearful', 'panicked', 'overwhelmed',
                'restless', 'jittery', 'on edge'
            ],
            'excited': [
                'excited', 'thrilled', 'pumped', 'energized', 'enthusiastic', 'eager',
                'psyched', 'stoked', 'hyped', 'animated', 'exhilarated', 'passionate',
                'fired up', 'anticipating'
            ],
            'neutral': [
                'okay', 'fine', 'alright', 'normal', 'regular', 'usual', 'standard',
                'average', 'moderate', 'so-so', 'indifferent'
            ]
        }

    def _initialize_response_templates(self) -> Dict[str, List[str]]:
        """Initialize response templates for each mood"""
        return {
            'happy': [
                "That's wonderful to hear! Your positive energy is contagious! 😊",
                "I'm so glad you're feeling great! Keep spreading that joy!",
                "Your happiness shines through! What's making you feel so good?",
                "It's beautiful to see such positivity! Keep that smile going!",
                "Your enthusiasm is inspiring! I'm happy for you!"
            ],
            'sad': [
                "I hear you, and it's okay to feel this way. Would you like to talk about it?",
                "I'm sorry you're going through a tough time. Remember, this feeling will pass.",
                "It sounds like you're having a difficult moment. I'm here to listen.",
                "Your feelings are valid. Sometimes it helps to express what's weighing on you.",
                "I understand this is hard. Take your time, and know that you're not alone."
            ],
            'angry': [
                "I understand you're frustrated. Take a deep breath, and let's work through this.",
                "Your feelings are valid. What would help you feel better right now?",
                "I can sense your frustration. Sometimes expressing it helps - I'm here to listen.",
                "It's okay to feel angry. Let's see if we can find a way to ease this tension.",
                "I hear your frustration. Would you like to talk about what's bothering you?"
            ],
            'anxious': [
                "I sense some worry in your message. Remember to breathe - you've got this.",
                "It's natural to feel anxious sometimes. What's concerning you most?",
                "I'm here to support you. Let's tackle your worries one step at a time.",
                "Anxiety can be overwhelming. Try to focus on what you can control right now.",
                "I understand you're feeling worried. Remember, you've handled challenges before."
            ],
            'excited': [
                "Your excitement is infectious! Tell me more about what has you so energized!",
                "That's amazing energy! I love your enthusiasm!",
                "You sound thrilled! What exciting things are happening?",
                "Your passion is inspiring! Keep riding that wave of excitement!",
                "I can feel your excitement from here! That's fantastic!"
            ],
            'neutral': [
                "I'm here to help with whatever you need. How can I assist you today?",
                "Thank you for sharing. Is there anything specific you'd like to discuss?",
                "I appreciate you reaching out. What's on your mind?",
                "I'm listening. Feel free to share more about what you're thinking.",
                "How can I support you today? I'm here to help."
            ]
        }

    def detect_emotional_indicators(self, text: str) -> List[str]:
        """
        Extract emotional indicators from text

        Args:
            text: Input text to analyze

        Returns:
            List of emotional indicators found
        """
        indicators = []
        text_lower = text.lower()

        # Check for mood keywords
        for mood, keywords in self.mood_keywords.items():
            found_keywords = [kw for kw in keywords if kw in text_lower]
            if found_keywords:
                indicators.append(f"{mood.capitalize()} indicators: {', '.join(found_keywords[:3])}")

        # Check for punctuation patterns
        if text.count('!') > 1:
            indicators.append("Multiple exclamation marks (high emotion)")

        if text.count('?') > 1:
            indicators.append("Multiple questions (uncertainty/seeking)")

        if '...' in text or '…' in text:
            indicators.append("Ellipsis usage (hesitation/trailing thought)")

        # Check for caps usage
        words = text.split()
        caps_words = [w for w in words if w.isupper() and len(w) > 1]
        if len(caps_words) > 0:
            indicators.append(f"Caps usage: {', '.join(caps_words[:3])} (strong emotion)")

        # Check for emoticons
        import re
        emoticons = re.findall(r'[:;=][oO\-]?[D\)\]\(\]/\\OpP]', text)
        if emoticons:
            indicators.append(f"Emoticons detected: {' '.join(emoticons)}")

        # Check for repetition
        words_lower = [w.lower() for w in words if len(w) > 3]
        repeated = [w for w in set(words_lower) if words_lower.count(w) > 2]
        if repeated:
            indicators.append(f"Repeated words: {', '.join(repeated)} (emphasis)")

        return indicators if indicators else ["No strong emotional indicators detected"]

    def detect_basic_mood(self, text: str) -> Tuple[str, float]:
        """
        Basic mood detection without ML model

        Args:
            text: Input text to analyze

        Returns:
            Tuple of (mood, confidence)
        """
        text_lower = text.lower()
        mood_scores = {}

        # Calculate scores for each mood
        for mood, keywords in self.mood_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                mood_scores[mood] = score

        # Determine mood based on scores
        if not mood_scores:
            return 'neutral', 0.5

        # Get mood with highest score
        best_mood = max(mood_scores, key=mood_scores.get)
        total_score = sum(mood_scores.values())
        confidence = mood_scores[best_mood] / total_score if total_score > 0 else 0.5

        # Adjust confidence based on text features
        if text.count('!') > 2:
            confidence = min(confidence + 0.1, 1.0)
        if text.isupper():
            confidence = min(confidence + 0.15, 1.0)

        return best_mood, confidence

    def generate_response(self, mood: str, confidence: float, context: Optional[List[str]] = None) -> str:
        """
        Generate empathetic response based on mood

        Args:
            mood: Detected mood
            confidence: Confidence score
            context: Optional conversation context

        Returns:
            Generated response
        """
        # Ensure mood exists in templates
        if mood not in self.response_templates:
            mood = 'neutral'

        # Select base response
        responses = self.response_templates[mood]
        base_response = random.choice(responses)

        # Modify based on confidence
        if confidence < 0.4:
            prefix = "I'm sensing mixed emotions here. "
            base_response = prefix + base_response
        elif confidence > 0.8:
            suffix = " (I'm quite confident about this mood detection)"
            if not base_response.endswith(('!', '?', '.')):
                base_response += '.'
            base_response = base_response[:-1] + suffix + base_response[-1]

        # Consider context if provided
        if context and len(context) > 0:
            if len(context) > 2:
                base_response += " I see we've been talking for a while."

        return base_response

    def analyze_mood_transition(self, previous_mood: str, current_mood: str) -> str:
        """
        Analyze mood transition and provide appropriate response

        Args:
            previous_mood: Previous detected mood
            current_mood: Current detected mood

        Returns:
            Transition analysis message
        """
        transitions = {
            ('sad', 'happy'): "I'm glad to see your mood improving! That's wonderful progress.",
            ('happy', 'sad'): "I notice a shift in your mood. What changed?",
            ('angry', 'happy'): "It's great to see you've worked through that frustration!",
            ('anxious', 'neutral'): "You seem calmer now. That's good to see.",
            ('neutral', 'excited'): "Something exciting happened! I'd love to hear about it!",
        }

        key = (previous_mood, current_mood)
        if key in transitions:
            return transitions[key]
        elif previous_mood != current_mood:
            return f"I notice your mood has shifted from {previous_mood} to {current_mood}."
        else:
            return ""