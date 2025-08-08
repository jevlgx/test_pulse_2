"""
Model management module for loading, saving, and managing ML models
"""

import os
import joblib
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List
from config import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


class ModelManager:
    """Manages ML model lifecycle"""

    _instance = None
    _current_model = None
    _current_vectorizer = None
    _current_label_encoder = None
    _model_metadata = {}

    def __new__(cls):
        """Singleton pattern for model manager"""
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize model manager"""
        self.models_dir = settings.MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)

    @property
    def current_model(self):
        """Get current model"""
        return self._current_model

    @property
    def current_vectorizer(self):
        """Get current vectorizer"""
        return self._current_vectorizer

    @property
    def current_label_encoder(self):
        """Get current label encoder"""
        return self._current_label_encoder

    @property
    def model_metadata(self):
        """Get model metadata"""
        return self._model_metadata

    def is_model_loaded(self) -> bool:
        """Check if a model is currently loaded"""
        return self._current_model is not None

    def save_model(self,
                   model: Any,
                   vectorizer: Any,
                   label_encoder: Any,
                   metadata: Dict[str, Any]) -> str:
        """
        Save model and associated components

        Args:
            model: Trained model
            vectorizer: Fitted vectorizer
            label_encoder: Fitted label encoder
            metadata: Model metadata

        Returns:
            Model ID
        """
        try:
            # Generate model ID
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_id = f"model_{timestamp}"

            # Create model directory
            model_path = self.models_dir / model_id
            model_path.mkdir(exist_ok=True)

            # Save components
            joblib.dump(model, model_path / "model.pkl")
            joblib.dump(vectorizer, model_path / "vectorizer.pkl")
            joblib.dump(label_encoder, model_path / "label_encoder.pkl")

            # Add system metadata
            metadata['model_id'] = model_id
            metadata['saved_at'] = datetime.now().isoformat()
            metadata['model_path'] = str(model_path)

            # Save metadata
            with open(model_path / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2, default=str)

            # Update current model
            self._current_model = model
            self._current_vectorizer = vectorizer
            self._current_label_encoder = label_encoder
            self._model_metadata = metadata

            logger.info(f"Model saved successfully: {model_id}")
            return model_id

        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise

    def load_model(self, model_id: str) -> bool:
        """
        Load specific model by ID

        Args:
            model_id: Model identifier

        Returns:
            Success status
        """
        try:
            model_path = self.models_dir / model_id

            if not model_path.exists():
                logger.error(f"Model not found: {model_id}")
                return False

            # Load components
            self._current_model = joblib.load(model_path / "model.pkl")
            self._current_vectorizer = joblib.load(model_path / "vectorizer.pkl")
            self._current_label_encoder = joblib.load(model_path / "label_encoder.pkl")

            # Load metadata
            with open(model_path / "metadata.json", 'r') as f:
                self._model_metadata = json.load(f)

            logger.info(f"Model loaded successfully: {model_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False

    def load_latest_model(self) -> bool:
        """
        Load the most recent model

        Returns:
            Success status
        """
        try:
            # Find all model directories
            model_dirs = [d for d in self.models_dir.iterdir() if d.is_dir()]

            if not model_dirs:
                logger.info("No saved models found")
                return False

            # Get most recent model
            latest_model = max(model_dirs, key=lambda d: d.stat().st_mtime)
            model_id = latest_model.name

            return self.load_model(model_id)

        except Exception as e:
            logger.error(f"Failed to load latest model: {str(e)}")
            return False

    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all available models

        Returns:
            List of model information
        """
        models = []

        try:
            for model_dir in self.models_dir.iterdir():
                if model_dir.is_dir():
                    metadata_path = model_dir / "metadata.json"
                    if metadata_path.exists():
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                            models.append({
                                'model_id': model_dir.name,
                                'created_at': metadata.get('saved_at'),
                                'accuracy': metadata.get('accuracy'),
                                'classes': metadata.get('classes'),
                                'total_samples': metadata.get('total_samples')
                            })

            # Sort by creation date
            models.sort(key=lambda x: x['created_at'], reverse=True)

        except Exception as e:
            logger.error(f"Failed to list models: {str(e)}")

        return models

    def delete_model(self, model_id: str) -> bool:
        """
        Delete a specific model

        Args:
            model_id: Model identifier

        Returns:
            Success status
        """
        try:
            model_path = self.models_dir / model_id

            if not model_path.exists():
                logger.error(f"Model not found: {model_id}")
                return False

            # Remove directory and all contents
            import shutil
            shutil.rmtree(model_path)

            # Clear current model if it's the one being deleted
            if self._model_metadata.get('model_id') == model_id:
                self.clear_current_model()

            logger.info(f"Model deleted: {model_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete model: {str(e)}")
            return False

    def clear_current_model(self):
        """Clear currently loaded model"""
        self._current_model = None
        self._current_vectorizer = None
        self._current_label_encoder = None
        self._model_metadata = {}
        logger.info("Current model cleared")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about current model

        Returns:
            Model information dictionary
        """
        if not self.is_model_loaded():
            return {
                'model_loaded': False,
                'message': 'No model currently loaded'
            }

        # Calculate model size
        model_size = 0
        if self._model_metadata.get('model_path'):
            model_path = Path(self._model_metadata['model_path'])
            if model_path.exists():
                for file in model_path.iterdir():
                    if file.is_file():
                        model_size += file.stat().st_size

        return {
            'model_loaded': True,
            'model_id': self._model_metadata.get('model_id'),
            'model_type': self._model_metadata.get('model_type'),
            'training_date': self._model_metadata.get('saved_at'),
            'accuracy': self._model_metadata.get('accuracy'),
            'classes': self._model_metadata.get('classes'),
            'total_samples_trained': self._model_metadata.get('total_samples'),
            'features_count': self._model_metadata.get('features_count'),
            'model_size_mb': round(model_size / (1024 * 1024), 2)
        }

    def export_model(self, model_id: str, export_path: Path) -> bool:
        """
        Export model to external location

        Args:
            model_id: Model identifier
            export_path: Path to export to

        Returns:
            Success status
        """
        try:
            model_path = self.models_dir / model_id

            if not model_path.exists():
                logger.error(f"Model not found: {model_id}")
                return False

            # Create export directory
            export_path.mkdir(parents=True, exist_ok=True)

            # Copy all model files
            import shutil
            for file in model_path.iterdir():
                if file.is_file():
                    shutil.copy2(file, export_path / file.name)

            logger.info(f"Model exported to: {export_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export model: {str(e)}")
            return False