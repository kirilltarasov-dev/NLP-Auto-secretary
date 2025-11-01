import os
import re
import string
import joblib
import numpy as np
from typing import List, Optional, Dict, Any
import json

# NLTK-based preprocessing to match training pipeline
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "banking77")
VECTORIZER_PATH = os.path.join(DATA_DIR, "tfidf_vectorizer.joblib")
MODEL_PATH = os.path.join(DATA_DIR, "logreg_model.joblib")
LABEL_NAMES_PATH = os.path.join(DATA_DIR, "label_names.json")

_lemmatizer = None
_stop_words = None


def _init_nltk() -> None:
    global _lemmatizer, _stop_words
    if _lemmatizer is not None:
        return
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
        except Exception:
            pass
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        try:
            nltk.download('stopwords', quiet=True)
        except Exception:
            pass
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        try:
            nltk.download('wordnet', quiet=True)
        except Exception:
            pass
    _lemmatizer = WordNetLemmatizer()
    _stop_words = set(stopwords.words('english')) if stopwords.words else set()


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(f"[{string.punctuation}0-9]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_text(text: str) -> str:
    _init_nltk()
    text = clean_text(text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in _stop_words]
    tokens = [_lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def _load_label_names(default_path: str = LABEL_NAMES_PATH) -> Optional[List[str]]:
    try:
        with open(default_path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


class IntentPredictor:
    def __init__(self, vectorizer_path: str = VECTORIZER_PATH, model_path: str = MODEL_PATH,
                 label_names: Optional[List[str]] = None) -> None:
        self.vectorizer = joblib.load(vectorizer_path)
        self.model = joblib.load(model_path)
        self.label_names = label_names if label_names is not None else _load_label_names()

    def predict_intent(self, text: str, top_k: int = 1) -> Dict[str, Any]:
        processed = preprocess_text(text)
        X = self.vectorizer.transform([processed])
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
        else:
            # fallback: decision_function -> pseudo-proba via softmax
            scores = self.model.decision_function(X)[0]
            exp = np.exp(scores - np.max(scores))
            proba = exp / exp.sum()
        top_idx = np.argsort(proba)[::-1][:top_k]
        predictions = []
        for idx in top_idx:
            name = self.label_names[idx] if self.label_names and idx < len(self.label_names) else str(idx)
            predictions.append({"label_id": int(idx), "label": name, "prob": float(proba[idx])})
        return {
            "input": text,
            "processed": processed,
            "top": predictions,
        }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.predict ""Your text here""")
        sys.exit(0)
    text = " ".join(sys.argv[1:])
    predictor = IntentPredictor()
    result = predictor.predict_intent(text, top_k=3)
    print(result)
