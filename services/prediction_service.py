"""
Prediction service for sentiment analysis
"""

from typing import Dict, Any, List, Optional
import numpy as np
from core.preprocessor import TextPreprocessor
from core.model_manager import ModelManager
from core.mood_detector import MoodDetector
from utils.logger import setup_logger

logger = setup_logger(__name__)


class PredictionService:
    """Service for handling prediction operations"""

    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.model_manager = ModelManager()
        self.mood_detector = MoodDetector()

    async def predict_single(self, text: str, return_probabilities: bool = False) -> Dict[str, Any]:
        """
        Predict sentiment for a single text

        Args:
            text: Input text
            return_probabilities: Whether to return probability distribution

        Returns:
            Prediction result
        """
        if not self.model_manager.is_model_loaded():
            raise ValueError("No model loaded. Please train a model first using /api/v1/train endpoint")

        try:
            # Preprocess text
            processed_text = self.preprocessor.preprocess(text)

            # Vectorize
            text_vec = self.model_manager.current_vectorizer.transform([processed_text])

            # Predict
            prediction = self.model_manager.current_model.predict(text_vec)[0]
            probabilities = self.model_manager.current_model.predict_proba(text_vec)[0]

            # Decode prediction
            sentiment = self.model_manager.current_label_encoder.inverse_transform([prediction])[0]
            confidence = float(max(probabilities))

            result = {
                'sentiment': sentiment,
                'confidence': confidence
            }

            if return_probabilities:
                result['probabilities'] = {
                    label: float(prob)
                    for label, prob in zip(
                        self.model_manager.current_label_encoder.classes_,
                        probabilities
                    )
                }

            return result

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

    async def predict_batch(self, texts: List[str], return_probabilities: bool = False) -> List[Dict[str, Any]]:
        """
        Predict sentiment for multiple texts

        Args:
            texts: List of input texts
            return_probabilities: Whether to return probability distributions

        Returns:
            List of prediction results
        """
        if not self.model_manager.is_model_loaded():
            raise ValueError("No model loaded. Please train a model first")

        try:
            # Preprocess all texts
            processed_texts = [self.preprocessor.preprocess(text) for text in texts]

            # Vectorize
            texts_vec = self.model_manager.current_vectorizer.transform(processed_texts)

            # Predict
            predictions = self.model_manager.current_model.predict(texts_vec)
            probabilities = self.model_manager.current_model.predict_proba(texts_vec)

            # Decode predictions
            sentiments = self.model_manager.current_label_encoder.inverse_transform(predictions)

            results = []
            for i, (text, sentiment, probs) in enumerate(zip(texts, sentiments, probabilities)):
                result = {
                    'text': text,
                    'sentiment': sentiment,
                    'confidence': float(max(probs))
                }

                if return_probabilities:
                    result['probabilities'] = {
                        label: float(prob)
                        for label, prob in zip(
                            self.model_manager.current_label_encoder.classes_,
                            probs
                        )
                    }

                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Batch prediction failed: {str(e)}")
            raise

    async def predict_with_explanation(self, text: str) -> Dict[str, Any]:
        """
        Predict sentiment with detailed explanation

        Args:
            text: Input text

        Returns:
            Prediction with explanation
        """
        if not self.model_manager.is_model_loaded():
            # Use basic mood detection
            mood, confidence = self.mood_detector.detect_basic_mood(text)
            return {
                'sentiment': mood,
                'confidence': confidence,
                'explanation': 'Using rule-based detection (no trained model available)',
                'important_words': self._extract_important_words_basic(text, mood),
                'emotional_indicators': self.mood_detector.detect_emotional_indicators(text)
            }

        try:
            # Get basic prediction
            prediction_result = await self.predict_single(text, return_probabilities=True)

            # Preprocess text
            processed_text = self.preprocessor.preprocess(text)

            # Get feature importance (approximate)
            text_vec = self.model_manager.current_vectorizer.transform([processed_text])
            feature_names = self.model_manager.current_vectorizer.get_feature_names_out()

            # Get non-zero features
            non_zero_indices = text_vec.nonzero()[1]
            important_features = []

            for idx in non_zero_indices[:10]:  # Top 10 features
                feature = feature_names[idx]
                weight = text_vec[0, idx]
                important_features.append({
                    'word': feature,
                    'importance': float(weight)
                })

            # Sort by importance
            important_features.sort(key=lambda x: x['importance'], reverse=True)
            important_words = [f['word'] for f in important_features[:5]]

            # Generate explanation
            explanation = self._generate_explanation(
                sentiment=prediction_result['sentiment'],
                confidence=prediction_result['confidence'],
                probabilities=prediction_result.get('probabilities', {}),
                important_words=important_words
            )

            return {
                'sentiment': prediction_result['sentiment'],
                'confidence': prediction_result['confidence'],
                'explanation': explanation,
                'important_words': important_words,
                'emotional_indicators': self.mood_detector.detect_emotional_indicators(text)
            }

        except Exception as e:
            logger.error(f"Prediction with explanation failed: {str(e)}")
            raise

    def _extract_important_words_basic(self, text: str, mood: str) -> List[str]:
        """
        Extract important words for basic mood detection

        Args:
            text: Input text
            mood: Detected mood

        Returns:
            List of important words
        """
        text_lower = text.lower()
        mood_keywords = self.mood_detector.mood_keywords.get(mood, [])

        found_words = []
        for keyword in mood_keywords:
            if keyword in text_lower:
                found_words.append(keyword)

        return found_words[:5]  # Return top 5

    def _generate_explanation(self, sentiment: str, confidence: float,
                              probabilities: Dict[str, float],
                              important_words: List[str]) -> str:
        """
        Generate human-readable explanation

        Args:
            sentiment: Predicted sentiment
            confidence: Confidence score
            probabilities: Probability distribution
            important_words: Important words from text

        Returns:
            Explanation string
        """
        explanation_parts = []

        # Confidence explanation
        if confidence > 0.8:
            explanation_parts.append(f"The model is highly confident ({confidence:.1%}) that this is '{sentiment}'.")
        elif confidence > 0.6:
            explanation_parts.append(f"The model predicts '{sentiment}' with moderate confidence ({confidence:.1%}).")
        else:
            explanation_parts.append(f"The model suggests '{sentiment}' but with low confidence ({confidence:.1%}).")

        # Important words
        if important_words:
            explanation_parts.append(f"Key words influencing this prediction: {', '.join(important_words[:3])}.")

        # Alternative possibilities
        if probabilities:
            sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_probs) > 1 and sorted_probs[1][1] > 0.3:
                explanation_parts.append(
                    f"Alternative possibility: '{sorted_probs[1][0]}' ({sorted_probs[1][1]:.1%})."
                )

        return " ".join(explanation_parts)