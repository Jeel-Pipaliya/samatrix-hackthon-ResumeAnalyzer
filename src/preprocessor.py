import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure required NLTK corpora are downloaded
try:
    _stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', quiet=True)
    _stop_words = set(stopwords.words('english'))

try:
    _lemmatizer = WordNetLemmatizer()
    _lemmatizer.lemmatize("testing")
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    _lemmatizer = WordNetLemmatizer()


def preprocess_text(text: str) -> str:
    """
    Standard preprocessing for resume classification:
    - Lowercase conversion
    - Strip URLs and email addresses
    - Preserve symbols for programming languages (+ for C++, # for C#)
    - Lemmatization & English stopword removal
    - Normalize whitespace
    """
    if not text or not isinstance(text, str):
        return ""

    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove Emails
    text = re.sub(r'\S+@\S+', '', text)
    # Remove punctuation except + and # (essential for C++, C#)
    text = re.sub(r'[^a-z0-9\+#\s]', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenization, Lemmatization, and Stopword Filtering
    tokens = text.split()
    cleaned_tokens = [
        _lemmatizer.lemmatize(word)
        for word in tokens
        if word not in _stop_words and len(word) > 1
    ]

    return ' '.join(cleaned_tokens)
