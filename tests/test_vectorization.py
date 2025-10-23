import unittest
import joblib
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

class TestVectorization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        train_df = pd.read_csv("data/banking77/train.csv")
        cls.examples = train_df['text'].sample(10, random_state=42).tolist()

    def test_vectorizer_fit(self):
        vec = TfidfVectorizer()
        X = vec.fit_transform(self.examples)
        self.assertEqual(X.shape[0], 10)
        self.assertGreater(X.shape[1], 0)

    def test_tfidf_nonzero(self):
        vec = TfidfVectorizer()
        X = vec.fit_transform(self.examples)
        self.assertTrue((X.sum(axis=1) > 0).all())

    def test_joblib_save_load(self):
        vec = TfidfVectorizer()
        X = vec.fit_transform(self.examples)
        joblib.dump(vec, "temp_vectorizer.joblib")
        loaded_vec = joblib.load("temp_vectorizer.joblib")
        self.assertEqual(len(loaded_vec.get_feature_names_out()), len(vec.get_feature_names_out()))
        os.remove("temp_vectorizer.joblib")
