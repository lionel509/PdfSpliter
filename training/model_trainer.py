import os
import joblib
import logging
from utils.logger import setup_logger

logger = setup_logger('model_trainer', 'training/logs/model_trainer.log')

class ModelTrainer:
    def __init__(self, model, model_name, output_dir):
        self.model = model
        self.model_name = model_name
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def train(self, training_data, labels=None):
        """
        Trains the model with the given data.
        :param training_data: Input data for training.
        :param labels: Target labels (optional for supervised learning).
        """
        logger.info(f"Training {self.model_name}...")
        if labels is not None:
            self.model.train(training_data, labels)
        else:
            self.model.train(training_data)
        logger.info(f"{self.model_name} training completed.")

    def save_model(self):
        """
        Saves the trained model to the output directory.
        """
        model_path = os.path.join(self.output_dir, f"{self.model_name}_model.pkl")
        joblib.dump(self.model, model_path)
        logger.info(f"{self.model_name} model saved at {model_path}.")

def train_all_models(models, data_dir, labels=None, output_dir="training/checkpoints"):
    """
    Trains and saves all models defined in the system.
    :param models: Dictionary of model names and instances.
    :param data_dir: Directory containing preprocessed data.
    :param labels: Optional labels for supervised training.
    :param output_dir: Directory to save trained models.
    """
    for model_name, model_instance in models.items():
        trainer = ModelTrainer(model_instance, model_name, output_dir)
        
        for file_name in os.listdir(data_dir):
            file_path = os.path.join(data_dir, file_name)
            training_data = joblib.load(file_path)

            # Labels provided for supervised models
            if labels is not None:
                trainer.train(training_data, labels)
            else:
                trainer.train(training_data)
        
        trainer.save_model()
