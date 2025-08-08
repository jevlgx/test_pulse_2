"""
Script to download all required NLTK data
Run this before starting the application
"""

import nltk
import ssl

# Handle SSL certificate issues
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


def download_nltk_data():
    """Download all required NLTK data packages"""

    packages = [
        'punkt',
        'punkt_tab',  # New requirement for recent NLTK versions
        'stopwords',
        'wordnet',
        'omw-1.4',
        'averaged_perceptron_tagger'
    ]

    print("Downloading NLTK data packages...")
    print("-" * 40)

    for package in packages:
        try:
            nltk.data.find(package)
            print(f"✓ {package} - already installed")
        except LookupError:
            print(f"⬇ Downloading {package}...")
            try:
                nltk.download(package)
                print(f"✓ {package} - successfully downloaded")
            except Exception as e:
                print(f"✗ {package} - failed: {e}")

    print("-" * 40)
    print("NLTK data download complete!")

    # Verify installation
    print("\nVerifying installation...")
    try:
        from nltk.tokenize import word_tokenize
        test_text = "This is a test sentence."
        tokens = word_tokenize(test_text)
        print(f"✓ Tokenization test successful: {tokens}")
    except Exception as e:
        print(f"✗ Tokenization test failed: {e}")


if __name__ == "__main__":
    download_nltk_data()