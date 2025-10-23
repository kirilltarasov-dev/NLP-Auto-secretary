import unittest
from src.preprocessing import clean_text
import pandas as pd
import sys

class TestCleanText(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        train_df = pd.read_csv("data/banking77/train.csv")
        cls.examples = train_df['text'].sample(10, random_state=42).tolist()

    def test_clean_text(self):
        for text in self.examples:
            cleaned = clean_text(text)
            self.assertIsInstance(cleaned, str)
            self.assertNotRegex(cleaned, r"[^a-z ]")