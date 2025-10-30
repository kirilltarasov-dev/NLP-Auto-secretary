import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import joblib
import os

# Setup NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load original CSVs
train_df = pd.read_csv("data/banking77/train.csv")
test_df = pd.read_csv("data/banking77/test.csv")

# Text cleaning functions
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}0-9]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_text(text):
    text = clean_text(text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

# Apply preprocessing
train_df['clean_text'] = train_df['text'].apply(preprocess_text)
test_df['clean_text'] = test_df['text'].apply(preprocess_text)

# Save cleaned datasets
train_df.to_csv("data/banking77/train_clean.csv", index=False)
test_df.to_csv("data/banking77/test_clean.csv", index=False)

print("Cleaned texts saved as train_clean.csv and test_clean.csv")


# Split train_df into train and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df['clean_text'], train_df['label'], test_size=0.2, random_state=42, stratify=train_df['label']
)
# Сохраняем индексы train и val относительно исходного train_df
train_indices, val_indices = train_test_split(
    train_df.index, test_size=0.2, random_state=42, stratify=train_df['label']
)
pd.Series(train_indices).to_csv("data/banking77/train_indices.csv", index=False)
pd.Series(val_indices).to_csv("data/banking77/val_indices.csv", index=False)

# Vectorization
vectorizer = TfidfVectorizer()
train_df_tfidf = vectorizer.fit_transform(train_texts)
val_df_tfidf = vectorizer.transform(val_texts)
test_df_tfidf = vectorizer.transform(test_df['clean_text'])
print("TF-IDF vectorization and validation are completed.")


# Saving the vectorizer and matrices
joblib.dump(vectorizer, "data/banking77/tfidf_vectorizer.joblib")
joblib.dump(train_df_tfidf, "data/banking77/train_tfidf.joblib")
joblib.dump(val_df_tfidf, "data/banking77/val_tfidf.joblib")
joblib.dump(test_df_tfidf, "data/banking77/test_tfidf.joblib")

print("TF-IDF vectorization is Saved as tfidf_vectorizer.joblib, val_tfidf.joblib, train_tfidf.joblib, test_tfidf.joblib.")