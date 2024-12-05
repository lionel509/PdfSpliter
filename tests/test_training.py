import os
import joblib
import unittest
from unittest.mock import MagicMock
from training.model_trainer import ModelTrainer, train_all_models
from training.hyperparameter_tuning import HyperparameterTuner, perform_hyperparameter_tuning

# Mock models for testing
class MockModel:
    def train(self, data, labels=None):
        pass

    def predict(self, data):
        return [0] * len(data), [0.5] * len(data)  # Mock predictions and confidence scores

class TestTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mock_data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  # Mock data
        cls.mock_labels = [0, 1, 0]  # Mock labels
        cls.models = {
            'MockRandomForest': MockModel(),
            'MockSVM': MockModel()
        }
        cls.param_grids = {
            'MockRandomForest': {'n_estimators': [50, 100], 'max_depth': [5, 10]},
            'MockSVM': {'C': [0.1, 1], 'kernel': ['linear', 'rbf']}
        }
        cls.data_dir = "tests/mock_data"
        cls.output_dir = "tests/mock_checkpoints"
        os.makedirs(cls.data_dir, exist_ok=True)
        os.makedirs(cls.output_dir, exist_ok=True)

        # Save mock data for training
        for i, data in enumerate(cls.mock_data):
            joblib.dump(data, os.path.join(cls.data_dir, f"data_{i}.pkl"))

    @classmethod
    def tearDownClass(cls):
        # Clean up created directories and files
        if os.path.exists(cls.data_dir):
            for file in os.listdir(cls.data_dir):
                os.remove(os.path.join(cls.data_dir, file))
            os.rmdir(cls.data_dir)

        if os.path.exists(cls.output_dir):
            for file in os.listdir(cls.output_dir):
                os.remove(os.path.join(cls.output_dir, file))
            os.rmdir(cls.output_dir)

    def test_model_training(self):
        """Test the ModelTrainer class."""
        model_name = "MockRandomForest"
        model = self.models[model_name]
        trainer = ModelTrainer(model, model_name, self.output_dir)
        
        for i, data in enumerate(self.mock_data):
            trainer.train(data, self.mock_labels[i])

        trainer.save_model()

        model_path = os.path.join(self.output_dir, f"{model_name}_model.pkl")
        self.assertTrue(os.path.exists(model_path))
        trained_model = joblib.load(model_path)
        self.assertIsInstance(trained_model, MockModel)

    def test_train_all_models(self):
        """Test training all models in bulk."""
        train_all_models(self.models, self.data_dir, self.mock_labels, self.output_dir)

        for model_name in self.models.keys():
            model_path = os.path.join(self.output_dir, f"{model_name}_model.pkl")
            self.assertTrue(os.path.exists(model_path))
            trained_model = joblib.load(model_path)
            self.assertIsInstance(trained_model, MockModel)

    def test_hyperparameter_tuning(self):
        """Test the HyperparameterTuner class."""
        model_name = "MockRandomForest"
        model = self.models[model_name]
        param_grid = self.param_grids[model_name]
        tuner = HyperparameterTuner(model, param_grid, search_type="grid")
        
        best_model = tuner.tune(self.mock_data, self.mock_labels)
        self.assertIsInstance(best_model, MockModel)

    def test_perform_hyperparameter_tuning(self):
        """Test performing hyperparameter tuning for all models."""
        perform_hyperparameter_tuning(
            self.models, 
            self.param_grids, 
            self.data_dir, 
            self.mock_labels, 
            self.output_dir
        )

        for model_name in self.models.keys():
            model_path = os.path.join(self.output_dir, f"{model_name}_best_model.pkl")
            self.assertTrue(os.path.exists(model_path))
            best_model = joblib.load(model_path)
            self.assertIsInstance(best_model, MockModel)

if __name__ == "__main__":
    unittest.main()
