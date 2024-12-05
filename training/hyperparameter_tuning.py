import os
import joblib
import logging
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from utils.logger import setup_logger

logger = setup_logger('hyperparameter_tuning', 'training/logs/hyperparameter_tuning.log')

class HyperparameterTuner:
    def __init__(self, model, param_grid, search_type="grid", n_iter=10):
        """
        Initializes the hyperparameter tuner.
        :param model: ML model instance.
        :param param_grid: Dictionary of parameters to tune.
        :param search_type: Type of search ('grid' or 'random').
        :param n_iter: Number of iterations for random search.
        """
        self.model = model
        self.param_grid = param_grid
        self.search_type = search_type
        self.n_iter = n_iter

    def tune(self, training_data, labels):
        """
        Tunes the hyperparameters of the model.
        :param training_data: Input data for training.
        :param labels: Target labels for supervised learning.
        :return: Best model and parameters.
        """
        logger.info(f"Starting {self.search_type} search for hyperparameter tuning...")
        
        if self.search_type == "grid":
            search = GridSearchCV(self.model, self.param_grid, cv=5, verbose=2, n_jobs=-1)
        elif self.search_type == "random":
            search = RandomizedSearchCV(
                self.model, self.param_grid, cv=5, n_iter=self.n_iter, verbose=2, n_jobs=-1
            )
        else:
            raise ValueError("Invalid search_type. Choose 'grid' or 'random'.")
        
        search.fit(training_data, labels)
        logger.info(f"Best parameters found: {search.best_params_}")
        logger.info(f"Best score: {search.best_score_}")
        
        return search.best_estimator_

def perform_hyperparameter_tuning(models, param_grids, data_dir, labels, output_dir="training/checkpoints"):
    """
    Performs hyperparameter tuning for all models.
    :param models: Dictionary of model names and instances.
    :param param_grids: Dictionary of parameter grids for each model.
    :param data_dir: Directory containing preprocessed data.
    :param labels: Labels for supervised models.
    :param output_dir: Directory to save tuned models.
    """
    os.makedirs(output_dir, exist_ok=True)

    for model_name, model_instance in models.items():
        logger.info(f"Hyperparameter tuning for {model_name}...")
        
        if model_name not in param_grids:
            logger.warning(f"No parameter grid provided for {model_name}. Skipping...")
            continue

        tuner = HyperparameterTuner(model_instance, param_grids[model_name])
        
        # Load training data
        training_data = []
        for file_name in os.listdir(data_dir):
            file_path = os.path.join(data_dir, file_name)
            data = joblib.load(file_path)
            if data is not None:
                training_data.append(data)

        if not training_data:
            logger.warning(f"No valid training data found in {data_dir}. Skipping {model_name}...")
            continue

        # Combine data and labels for tuning
        training_data_combined = training_data[0]
        for i in range(1, len(training_data)):
            training_data_combined = training_data_combined.append(training_data[i], ignore_index=True)

        best_model = tuner.tune(training_data_combined, labels)
        
        model_path = os.path.join(output_dir, f"{model_name}_best_model.pkl")
        joblib.dump(best_model, model_path)
        logger.info(f"Tuned {model_name} model saved at {model_path}.")
