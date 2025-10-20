import unittest
import pandas as pd
import joblib
import os
from preprocessing import clean_text, preprocess_text, TfidfVectorizer

class TestPreprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the dataset and select 10 random examples
        train_df = pd.read_csv("data/banking77/train.csv")
        cls.examples = train_df['text'].sample(10, random_state=42).tolist()

    # Test for clean text
    def test_clean_text(self):
        for text in self.examples:
            cleaned = clean_text(text)
            self.assertIsInstance(cleaned, str)
            self.assertNotRegex(cleaned, r"[^a-z ]")  # punctuation should be removed

    # Test for preprocessed text
    def test_preprocess_text(self):
        for text in self.examples:
            processed = preprocess_text(text)
            self.assertIsInstance(processed, str)
            self.assertTrue(len(processed.split()) > 0)  # result contains at least one word

    # TF-IDF vectorization test
    def test_vectorizer_fit(self):
        vec = TfidfVectorizer()
        X = vec.fit_transform(self.examples)
        self.assertEqual(X.shape[0], 10)  # 10 lines
        self.assertGreater(X.shape[1], 0)  # at least one sign

    # Test for non-zero TF-IDF values
    def test_tfidf_nonzero(self):
        vec = TfidfVectorizer()
        X = vec.fit_transform(self.examples)
        self.assertTrue((X.sum(axis=1) > 0).all())  # each line has non-zero TF-IDF values

    # Test saving and loading via joblib
    def test_joblib_save_load(self):
        vec = TfidfVectorizer()
        X = vec.fit_transform(self.examples)
        joblib.dump(vec, "temp_vectorizer.joblib")
        loaded_vec = joblib.load("temp_vectorizer.joblib")
        self.assertEqual(len(loaded_vec.get_feature_names_out()), len(vec.get_feature_names_out()))
        os.remove("temp_vectorizer.joblib")  # clean the temporary file


if __name__ == "__main__":
    unittest.main()