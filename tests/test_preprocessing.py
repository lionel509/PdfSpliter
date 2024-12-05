import unittest
import numpy as np
import pandas as pd
import os
import nltk
from preprocessing.image_preprocessor import ImagePreprocessor
from preprocessing.tabular_preprocessor import TabularPreprocessor
from preprocessing.text_preprocessor import TextPreprocessor

# nltk.download('all')
nltk.download('punkt_tab')

class TestPreprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create mock data for testing
        cls.mock_image_path = "tests/mock_image.jpg"
        cls.mock_tabular_path = "tests/mock_tabular.csv"
        cls.mock_text_data = "This is a sample text for testing the text preprocessor."

        # Create a mock image (100x100 RGB)
        mock_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        from PIL import Image
        Image.fromarray(mock_image).save(cls.mock_image_path)

        # Create a mock tabular dataset
        mock_tabular_data = {
            "numerical_1": [1.0, 2.0, None, 4.0],
            "numerical_2": [5.0, None, 7.0, 8.0],
            "categorical": ["A", "B", None, "C"],
        }
        pd.DataFrame(mock_tabular_data).to_csv(cls.mock_tabular_path, index=False)


        

    @classmethod
    def tearDownClass(cls):
        # Cleanup mock files
        os.remove(cls.mock_image_path)
        os.remove(cls.mock_tabular_path)

    def test_image_preprocessor(self):
        preprocessor = ImagePreprocessor(target_size=(64, 64), normalize=True)
        processed_image = preprocessor.process(self.mock_image_path)

        self.assertEqual(processed_image.shape, (64, 64, 3))
        self.assertTrue((processed_image >= 0).all() and (processed_image <= 1).all())

    def test_tabular_preprocessor(self):
        preprocessor = TabularPreprocessor(scale_features=True, impute_strategy="mean")
        processed_data = preprocessor.process(self.mock_tabular_path)

        # Ensure the processed data is a NumPy array
        self.assertIsInstance(processed_data, np.ndarray)

        # Check the shape of the processed data
        df = pd.read_csv(self.mock_tabular_path)
        num_categorical = len(df["categorical"].dropna().unique()) + 1  # +1 for missing category
        expected_shape = (df.shape[0], len(df.columns) - 1 + num_categorical)
        self.assertEqual(processed_data.shape, expected_shape)

    def test_text_preprocessor(self):
        preprocessor = TextPreprocessor(lower_case=True, remove_stopwords=True)
        processed_text = preprocessor.process(self.mock_text_data)

        self.assertIsInstance(processed_text, list)
        self.assertGreater(len(processed_text), 0)  # Ensure tokens are returned
        self.assertTrue(all(isinstance(token, str) for token in processed_text))

if __name__ == "__main__":
    unittest.main()
