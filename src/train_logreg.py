import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
import os

# Пути к данным
DATA_DIR = "data/banking77/"
TRAIN_VEC = os.path.join(DATA_DIR, "train_tfidf.joblib")
VAL_VEC = os.path.join(DATA_DIR, "val_tfidf.joblib")
TRAIN_CSV = os.path.join(DATA_DIR, "train_clean.csv")
TRAIN_IDX = os.path.join(DATA_DIR, "train_indices.csv")
VAL_IDX = os.path.join(DATA_DIR, "val_indices.csv")
MODEL_PATH = os.path.join(DATA_DIR, "logreg_model.joblib")

# Загрузка признаков
print("Loading TF-IDF features...")
X_train = joblib.load(TRAIN_VEC)
X_val = joblib.load(VAL_VEC)

# Загружаем метки и индексы
labels = pd.read_csv(TRAIN_CSV)['label'].values
train_indices = pd.read_csv(TRAIN_IDX).values.ravel()
val_indices = pd.read_csv(VAL_IDX).values.ravel()
train_labels = labels[train_indices]
val_labels = labels[val_indices]

print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")
print(f"Labels train: {len(train_labels)}, val: {len(val_labels)}")

# Модель
logreg = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42)
logreg.fit(X_train, train_labels)

# Предсказания
train_preds = logreg.predict(X_train)
val_preds = logreg.predict(X_val)

# Метрики
print("--- LogisticRegression Evaluation ---")
print("Train:")
print(f"  Accuracy: {accuracy_score(train_labels, train_preds):.4f}")
print(f"  F1-score: {f1_score(train_labels, train_preds, average='weighted'):.4f}")
print("---\nVal:")
print(f"  Accuracy: {accuracy_score(val_labels, val_preds):.4f}")
print(f"  F1-score: {f1_score(val_labels, val_preds, average='weighted'):.4f}")
print("Classification report (val):")
print(classification_report(val_labels, val_preds))

# Сохраним модель
joblib.dump(logreg, MODEL_PATH)
print(f"LogisticRegression model saved to {MODEL_PATH}")
