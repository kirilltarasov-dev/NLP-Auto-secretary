import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

DATA_DIR = "data/banking77/"
TRAIN_VEC = os.path.join(DATA_DIR, "train_tfidf.joblib")
VAL_VEC = os.path.join(DATA_DIR, "val_tfidf.joblib")
TRAIN_CSV = os.path.join(DATA_DIR, "train_clean.csv")
TRAIN_IDX = os.path.join(DATA_DIR, "train_indices.csv")
VAL_IDX = os.path.join(DATA_DIR, "val_indices.csv")

print("Loading data...")
X_train = joblib.load(TRAIN_VEC)
X_val = joblib.load(VAL_VEC)
labels = pd.read_csv(TRAIN_CSV)['label'].values
train_indices = pd.read_csv(TRAIN_IDX).values.ravel()
val_indices = pd.read_csv(VAL_IDX).values.ravel()
y_train = labels[train_indices]
y_val = labels[val_indices]

results = {}

# Naive Bayes (Multinomial)
print("\n=== MultinomialNB ===")
nb = MultinomialNB()
nb.fit(X_train, y_train)

pred_tr = nb.predict(X_train)
pred_val = nb.predict(X_val)
acc_tr = accuracy_score(y_train, pred_tr)
acc_val = accuracy_score(y_val, pred_val)
f1_tr = f1_score(y_train, pred_tr, average='weighted')
f1_val = f1_score(y_val, pred_val, average='weighted')
print(f"Train: acc={acc_tr:.4f} f1={f1_tr:.4f}")
print(f"Val:   acc={acc_val:.4f} f1={f1_val:.4f}")
print("Classification report (val):")
print(classification_report(y_val, pred_val))
results['MultinomialNB'] = {
    'train': {'accuracy': acc_tr, 'f1_weighted': f1_tr},
    'val': {'accuracy': acc_val, 'f1_weighted': f1_val}
}

# RandomForest (требует dense)
print("\n=== RandomForestClassifier ===")
try:
    X_train_dense = X_train.toarray()
    X_val_dense = X_val.toarray()
    rf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)
    rf.fit(X_train_dense, y_train)
    pred_tr = rf.predict(X_train_dense)
    pred_val = rf.predict(X_val_dense)
    acc_tr = accuracy_score(y_train, pred_tr)
    acc_val = accuracy_score(y_val, pred_val)
    f1_tr = f1_score(y_train, pred_tr, average='weighted')
    f1_val = f1_score(y_val, pred_val, average='weighted')
    print(f"Train: acc={acc_tr:.4f} f1={f1_tr:.4f}")
    print(f"Val:   acc={acc_val:.4f} f1={f1_val:.4f}")
    print("Classification report (val):")
    print(classification_report(y_val, pred_val))
    results['RandomForest'] = {
        'train': {'accuracy': acc_tr, 'f1_weighted': f1_tr},
        'val': {'accuracy': acc_val, 'f1_weighted': f1_val}
    }
except MemoryError:
    print("[WARN] Not enough memory to train RandomForest on dense TF-IDF. Skipping.")

# Сохранение кратких метрик
metrics_path = os.path.join(DATA_DIR, 'benchmark_metrics.csv')
rows = []
for model_name, res in results.items():
    rows.append([model_name, 'train', res['train']['accuracy'], res['train']['f1_weighted']])
    rows.append([model_name, 'val', res['val']['accuracy'], res['val']['f1_weighted']])

pd.DataFrame(rows, columns=['model', 'split', 'accuracy', 'f1_weighted']).to_csv(metrics_path, index=False)
print(f"Saved metrics to {metrics_path}")
