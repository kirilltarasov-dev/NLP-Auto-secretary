import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
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