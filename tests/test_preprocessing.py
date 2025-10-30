import unittest
from src.preprocessing import preprocess_text
import pandas as pd
import sys

class TestPreprocessText(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        train_df = pd.read_csv("data/banking77/train.csv")
        cls.examples = train_df['text'].sample(10, random_state=42).tolist()

    def test_preprocess_text(self):
        for text in self.examples:
            processed = preprocess_text(text)
            self.assertIsInstance(processed, str)
            self.assertTrue(len(processed.split()) > 0)
    