"""
Training service for model training operations
"""

import pandas as pd
import numpy as np
from io import BytesIO
import time
from typing import Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from core.preprocessor import TextPreprocessor
from core.model_manager import ModelManager
from config import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


class TrainingService:
    """Service for handling model training operations"""

    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.model_manager = ModelManager()

    async def train_model(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Train a sentiment analysis model from CSV data

        Args:
            file_content: CSV file content as bytes
            filename: Original filename

        Returns:
            Training result dictionary
        """
        start_time = time.time()

        try:
            # Read and validate data
            df = pd.read_csv(BytesIO(file_content))
            df = self._prepare_dataframe(df)

            if len(df) < 10:
                return {
                    'status': 'failed',
                    'message': 'Insufficient data. Need at least 10 samples.'
                }

            # Preprocess text
            logger.info(f"Preprocessing {len(df)} samples...")
            df['processed_text'] = df['message'].apply(self.preprocessor.preprocess)

            # Prepare features and labels
            X = df['processed_text']
            y = df['sentiment']

            # Encode labels
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded,
                test_size=settings.TEST_SIZE,
                random_state=settings.RANDOM_STATE,
                stratify=y_encoded
            )

            # Vectorize text
            vectorizer = TfidfVectorizer(
                max_features=settings.MAX_FEATURES,
                ngram_range=settings.NGRAM_RANGE
            )
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)

            # Train ensemble model
            model = self._create_ensemble_model()
            logger.info("Training ensemble model...")
            model.fit(X_train_vec, y_train)

            # Evaluate model
            y_pred = model.predict(X_test_vec)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(
                y_test, y_pred,
                target_names=label_encoder.classes_,
                output_dict=True
            )

            # Prepare metadata
            metadata = {
                'accuracy': accuracy,
                'model_type': 'Ensemble (LR + NB + RF)',
                'classes': list(label_encoder.classes_),
                'total_samples': len(df),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'features_count': len(vectorizer.get_feature_names_out()),
                'classification_report': report
            }

            # Save model
            model_id = self.model_manager.save_model(
                model=model,
                vectorizer=vectorizer,
                label_encoder=label_encoder,
                metadata=metadata
            )

            training_time = time.time() - start_time

            return {
                'status': 'success',
                'message': f'Model trained successfully with {len(df)} samples',
                'model_id': model_id,
                'metrics': {
                    'accuracy': accuracy,
                    'test_samples': len(X_test),
                    'train_samples': len(X_train),
                    'classes': list(label_encoder.classes_),
                    'per_class_metrics': report
                },
                'model_info': {
                    'type': 'Ensemble Classifier',
                    'vectorizer': 'TF-IDF',
                    'features': len(vectorizer.get_feature_names_out())
                },
                'training_time': training_time
            }

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            return {
                'status': 'failed',
                'message': f'Training failed: {str(e)}'
            }

    async def validate_training_data(self, file_content: bytes) -> Dict[str, Any]:
        """
        Validate training data without training

        Args:
            file_content: CSV file content

        Returns:
            Validation result
        """
        try:
            df = pd.read_csv(BytesIO(file_content))

            # Check columns
            required_cols = {'message', 'sentiment'}
            actual_cols = set(df.columns)

            # Try alternative column names
            if 'text' in actual_cols:
                df.rename(columns={'text': 'message'}, inplace=True)
            if 'feeling' in actual_cols or 'feelings' in actual_cols:
                df.rename(columns={'feeling': 'sentiment', 'feelings': 'sentiment'}, inplace=True)

            actual_cols = set(df.columns)

            if not required_cols.issubset(actual_cols):
                return {
                    'valid': False,
                    'message': f'Missing required columns. Need: {required_cols}, Found: {actual_cols}'
                }

            # Clean data
            df = df.dropna()
            df = df[df['message'].str.len() > 0]

            # Calculate statistics
            sentiment_counts = df['sentiment'].value_counts().to_dict()
            avg_text_length = df['message'].str.len().mean()

            # Check for warnings
            warnings = []
            if len(df) < 50:
                warnings.append(f"Low sample count ({len(df)}). Recommend at least 50 samples.")

            # Check class balance
            min_class_size = min(sentiment_counts.values())
            max_class_size = max(sentiment_counts.values())

            if max_class_size / min_class_size > 5:
                warnings.append("Highly imbalanced classes detected. Consider balancing your dataset.")

            for sentiment, count in sentiment_counts.items():
                if count < 10:
                    warnings.append(f"Class '{sentiment}' has only {count} samples. Recommend at least 10.")

            return {
                'valid': True,
                'message': 'Data validation successful',
                'statistics': {
                    'total_samples': len(df),
                    'sentiment_distribution': sentiment_counts,
                    'unique_sentiments': len(sentiment_counts),
                    'average_text_length': round(avg_text_length, 2),
                    'min_text_length': df['message'].str.len().min(),
                    'max_text_length': df['message'].str.len().max()
                },
                'warnings': warnings
            }

        except Exception as e:
            return {
                'valid': False,
                'message': f'Validation failed: {str(e)}'
            }

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare and clean dataframe for training

        Args:
            df: Input dataframe

        Returns:
            Cleaned dataframe
        """
        # Check and rename columns
        if 'text' in df.columns:
            df.rename(columns={'text': 'message'}, inplace=True)
        if 'feeling' in df.columns:
            df.rename(columns={'feeling': 'sentiment'}, inplace=True)
        if 'feelings' in df.columns:
            df.rename(columns={'feelings': 'sentiment'}, inplace=True)

        # Validate required columns
        required_columns = {'message', 'sentiment'}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"CSV must contain 'message' and 'sentiment' columns. Found: {list(df.columns)}")

        # Clean data
        df = df.dropna()
        df = df[df['message'].str.len() > 0]
        df = df[df['sentiment'].str.len() > 0]

        # Remove duplicates
        df = df.drop_duplicates(subset=['message'])

        return df

    def _create_ensemble_model(self):
        """
        Create ensemble voting classifier

        Returns:
            Configured ensemble model
        """
        models = [
            ('lr', LogisticRegression(
                max_iter=settings.MAX_ITER_LOGISTIC,
                random_state=settings.RANDOM_STATE
            )),
            ('nb', MultinomialNB()),
            ('rf', RandomForestClassifier(
                n_estimators=settings.N_ESTIMATORS_RF,
                random_state=settings.RANDOM_STATE,
                n_jobs=-1
            ))
        ]

        return VotingClassifier(estimators=models, voting='soft')