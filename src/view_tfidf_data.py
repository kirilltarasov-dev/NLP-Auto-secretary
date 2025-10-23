import joblib
import pandas as pd

# Пути к файлам
VECTORIZER_PATH = "data/banking77/tfidf_vectorizer.joblib"
TRAIN_PATH = "data/banking77/train_tfidf.joblib"
VAL_PATH = "data/banking77/val_tfidf.joblib"
TEST_PATH = "data/banking77/test_tfidf.joblib"

# Загружаем векторизатор
print("Loading TF-IDF vectorizer...")
vectorizer = joblib.load(VECTORIZER_PATH)
print(f"Number of features in vectorizer: {len(vectorizer.get_feature_names_out())}\n")

# Показываем первые 20 слов словаря
print("First 20 words in the TF-IDF vocabulary:")
print(vectorizer.get_feature_names_out()[:20])
print("\n")

# Загружаем матрицы TF-IDF
print("Loading TF-IDF matrices...")
X_train = joblib.load(TRAIN_PATH)
X_val = joblib.load(VAL_PATH)
X_test = joblib.load(TEST_PATH)

print(f"Train matrix shape: {X_train.shape}")
print(f"Validation matrix shape: {X_val.shape}")
print(f"Test matrix shape: {X_test.shape}\n")

# Преобразуем в DataFrame для наглядности (только первые 5 строк, чтобы не перегружать вывод)
df_train = pd.DataFrame(X_train.toarray()[:5], columns=vectorizer.get_feature_names_out())
df_val = pd.DataFrame(X_val.toarray()[:5], columns=vectorizer.get_feature_names_out())
df_test = pd.DataFrame(X_test.toarray()[:5], columns=vectorizer.get_feature_names_out())

print("First 5 rows of the Train TF-IDF matrix:")
print(df_train)
print("\nFirst 5 rows of the Validation TF-IDF matrix:")
print(df_val)
print("\nFirst 5 rows of the Test TF-IDF matrix:")
print(df_test)