import unittest
import numpy as np
from models.supervised.random_forest_model import RandomForestModel
from models.supervised.svm_model import SVMModel
from models.supervised.neural_network_model import NeuralNetworkModel
from models.unsupervised.kmeans_model import KMeansModel
from models.unsupervised.dbscan_model import DBSCANModel
from models.pretrained.bert_model import BERTModel
from models.pretrained.gpt_model import GPTModel

class TestModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Generate mock data
        cls.mock_data_supervised = np.random.rand(100, 10)  # 100 samples, 10 features
        cls.mock_labels = np.random.randint(0, 2, 100)  # Binary classification labels
        
        cls.mock_data_unsupervised = np.random.rand(50, 5)  # 50 samples, 5 features
        cls.mock_text_data = ["This is a test sentence." for _ in range(10)]  # Mock text data

    def test_random_forest_model(self):
        model = RandomForestModel(n_estimators=10, max_depth=5)
        model.train(self.mock_data_supervised, self.mock_labels)
        predictions = model.predict(self.mock_data_supervised)
        self.assertEqual(len(predictions), len(self.mock_labels))

    def test_svm_model(self):
        model = SVMModel(kernel='linear', C=0.5)
        model.train(self.mock_data_supervised, self.mock_labels)
        predictions = model.predict(self.mock_data_supervised)
        self.assertEqual(len(predictions), len(self.mock_labels))

    def test_neural_network_model(self):
        model = NeuralNetworkModel(hidden_layer_sizes=(50, 25), activation='relu')
        model.train(self.mock_data_supervised, self.mock_labels)
        predictions = model.predict(self.mock_data_supervised)
        self.assertEqual(len(predictions), len(self.mock_labels))

    def test_kmeans_model(self):
        model = KMeansModel(n_clusters=3)
        model.train(self.mock_data_unsupervised)
        predictions = model.predict(self.mock_data_unsupervised)
        self.assertEqual(len(predictions), len(self.mock_data_unsupervised))

    def test_dbscan_model(self):
        model = DBSCANModel(eps=0.3, min_samples=5)
        model.train(self.mock_data_unsupervised)
        predictions = model.predict(self.mock_data_unsupervised)
        self.assertEqual(len(predictions), len(self.mock_data_unsupervised))

    def test_bert_model(self):
        model = BERTModel(model_name="bert-base-uncased", num_labels=2)
        # Mock training (requires labeled data)
        mock_labels = [0, 1] * 5  # Alternating binary labels for 10 sentences
        model.train(self.mock_text_data, mock_labels, epochs=1, learning_rate=1e-5)
        predictions = model.predict(self.mock_text_data)
        self.assertEqual(len(predictions), len(self.mock_text_data))

def test_gpt_model(self):
    """
    Test the GPTModel for text generation.
    Currently skipped to focus on other tests.
    """
    self.skipTest("Skipping GPT model tests for now.")


if __name__ == "__main__":
    unittest.main()
