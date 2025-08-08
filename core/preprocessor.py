"""
Text preprocessing module
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from typing import List, Optional
from utils.logger import setup_logger

logger = setup_logger(__name__)

class TextPreprocessor:
    """Advanced text preprocessing for sentiment analysis"""

    def __init__(self):
        """Initialize preprocessor with NLTK components"""
        self._ensure_nltk_data()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        # Keep some important negation words
        important_words = {'not', 'no', 'nor', 'neither', 'never', 'none', 'nobody',
                          'nothing', 'nowhere', 'hardly', 'scarcely', 'barely'}
        self.stop_words = self.stop_words - important_words

    def _ensure_nltk_data(self):
        """Download required NLTK data if not present"""
        required_data = [
            ('tokenizers/punkt', 'punkt'),
            ('tokenizers/punkt_tab', 'punkt_tab'),
            ('corpora/stopwords', 'stopwords'),
            ('corpora/wordnet', 'wordnet'),
            ('corpora/omw-1.4', 'omw-1.4')
        ]

        for path, name in required_data:
            try:
                nltk.data.find(path)
            except LookupError:
                try:
                    logger.info(f"Downloading NLTK data: {name}")
                    nltk.download(name, quiet=True)
                except Exception as e:
                    # Continue if one fails, as some might be optional
                    logger.warning(f"Could not download {name}: {e}")
                    continue

    def preprocess(self, text: str, preserve_emoticons: bool = True) -> str:
        """
        Preprocess text for sentiment analysis

        Args:
            text: Input text to preprocess
            preserve_emoticons: Whether to preserve emoticons

        Returns:
            Preprocessed text
        """
        if not text:
            return ""

        # Convert to lowercase
        text = text.lower()

        # Preserve emoticons if requested
        emoticon_pattern = r'[:;=][oO\-]?[D\)\]\(\]/\\OpP]'
        emoticons = []
        if preserve_emoticons:
            emoticons = re.findall(emoticon_pattern, text)

        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove mentions and hashtags (but keep the text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)

        # Handle contractions
        text = self._expand_contractions(text)

        # Remove special characters but keep spaces and apostrophes
        text = re.sub(r'[^a-zA-Z\s\']', ' ', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Tokenize
        tokens = word_tokenize(text)

        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]

        # Add back emoticons
        if emoticons:
            tokens.extend(emoticons)

        return ' '.join(tokens).strip()

    def _expand_contractions(self, text: str) -> str:
        """Expand common contractions"""
        contractions = {
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'m": " am",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'s": " is"
        }

        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)

        return text

    def batch_preprocess(self, texts: List[str], preserve_emoticons: bool = True) -> List[str]:
        """
        Preprocess multiple texts

        Args:
            texts: List of texts to preprocess
            preserve_emoticons: Whether to preserve emoticons

        Returns:
            List of preprocessed texts
        """
        return [self.preprocess(text, preserve_emoticons) for text in texts]

    def extract_features(self, text: str) -> dict:
        """
        Extract additional features from text

        Args:
            text: Input text

        Returns:
            Dictionary of features
        """
        features = {
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
            'text_length': len(text),
            'word_count': len(text.split()),
            'has_emoticons': bool(re.search(r'[:;=][oO\-]?[D\)\]\(\]/\\OpP]', text))
        }

        return features