import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

DATA_DIR = "data/banking77/"
TRAIN_VEC = os.path.join(DATA_DIR, "train_tfidf.joblib")
VAL_VEC = os.path.join(DATA_DIR, "val_tfidf.joblib")
TEST_VEC = os.path.join(DATA_DIR, "test_tfidf.joblib")
TRAIN_CSV = os.path.join(DATA_DIR, "train_clean.csv")
TEST_CSV = os.path.join(DATA_DIR, "test_clean.csv")
TRAIN_IDX = os.path.join(DATA_DIR, "train_indices.csv")
VAL_IDX = os.path.join(DATA_DIR, "val_indices.csv")
LOGREG_PATH = os.path.join(DATA_DIR, "logreg_model.joblib")
OUT_CSV = os.path.join(DATA_DIR, "full_benchmark_metrics.csv")
OUT_DIR = os.path.join(DATA_DIR, "eval_artifacts")
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading features and labels...")
X_train = joblib.load(TRAIN_VEC)
X_val = joblib.load(VAL_VEC)
X_test = joblib.load(TEST_VEC)

train_labels_all = pd.read_csv(TRAIN_CSV)['label'].values
test_labels = pd.read_csv(TEST_CSV)['label'].values
train_indices = pd.read_csv(TRAIN_IDX).values.ravel()
val_indices = pd.read_csv(VAL_IDX).values.ravel()
y_train = train_labels_all[train_indices]
y_val = train_labels_all[val_indices]
y_test = test_labels

rows = []

def save_confusion_csv(name: str, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    path = os.path.join(OUT_DIR, f"confusion_{name}.csv")
    pd.DataFrame(cm).to_csv(path, index=False)
    print(f"Saved confusion matrix CSV: {path}")


def evaluate_split(model_name: str, split: str, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1_w = f1_score(y_true, y_pred, average='weighted')
    f1_m = f1_score(y_true, y_pred, average='macro')
    rows.append([model_name, split, acc, f1_w, f1_m])
    # report
    report_txt = os.path.join(OUT_DIR, f"report_{model_name}_{split}.txt")
    with open(report_txt, 'w') as f:
        f.write(classification_report(y_true, y_pred))
    print(f"Saved report: {report_txt}")

# 1) LogisticRegression (load)
print("\n=== LogisticRegression (loaded) ===")
logreg = joblib.load(LOGREG_PATH)
for split_name, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
    y_pred = logreg.predict(X)
    evaluate_split("LogReg", split_name, y, y_pred)
    save_confusion_csv(f"LogReg_{split_name}", y, y_pred)

# 2) MultinomialNB (fit)
print("\n=== MultinomialNB ===")
nb = MultinomialNB()
nb.fit(X_train, y_train)
for split_name, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
    y_pred = nb.predict(X)
    evaluate_split("MultinomialNB", split_name, y, y_pred)
    save_confusion_csv(f"NB_{split_name}", y, y_pred)

# 3) RandomForest (fit on dense)
print("\n=== RandomForestClassifier ===")
X_train_d = X_train.toarray()
X_val_d = X_val.toarray()
X_test_d = X_test.toarray()
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train_d, y_train)
for split_name, Xd, y in [("val", X_val_d, y_val), ("test", X_test_d, y_test)]:
    y_pred = rf.predict(Xd)
    evaluate_split("RandomForest", split_name, y, y_pred)
    save_confusion_csv(f"RF_{split_name}", y, y_pred)

# save metrics table
pd.DataFrame(rows, columns=["model", "split", "accuracy", "f1_weighted", "f1_macro"]).to_csv(OUT_CSV, index=False)
print(f"Saved metrics table: {OUT_CSV}")
