import os
import joblib
import logging
import yaml
from models.supervised.random_forest_model import RandomForestModel
from models.supervised.svm_model import SVMModel
from models.supervised.neural_network_model import NeuralNetworkModel
from models.unsupervised.kmeans_model import KMeansModel
from models.unsupervised.dbscan_model import DBSCANModel
from models.pretrained.bert_model import BERTModel
from models.pretrained.gpt_model import GPTModel
from preprocessing.text_preprocessor import TextPreprocessor
from preprocessing.image_preprocessor import ImagePreprocessor
from preprocessing.tabular_preprocessor import TabularPreprocessor
from evaluation.performance_metrics import PerformanceMetrics
from evaluation.dynamic_weight_adjuster import DynamicWeightAdjuster
from utils.logger import setup_logger

# Setup logger
logger = setup_logger('training_pipeline', 'training/logs/training_pipeline.log')

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def preprocess_data(raw_data_dir, preprocessed_data_dir, config):
    """Preprocess raw data for training."""
    logger.info("Preprocessing data...")
    text_processor = TextPreprocessor(config['text'])
    image_processor = ImagePreprocessor(config['image'])
    tabular_processor = TabularPreprocessor(config['tabular'])
    
    for file_name in os.listdir(raw_data_dir):
        file_path = os.path.join(raw_data_dir, file_name)
        if file_name.endswith('.txt'):
            processed_data = text_processor.process(file_path)
        elif file_name.endswith(('.jpg', '.png')):
            processed_data = image_processor.process(file_path)
        elif file_name.endswith('.csv'):
            processed_data = tabular_processor.process(file_path)
        else:
            logger.warning(f"Unsupported file format: {file_name}")
            continue
        
        output_path = os.path.join(preprocessed_data_dir, file_name)
        joblib.dump(processed_data, output_path)
        logger.info(f"Processed data saved to {output_path}")

def train_models(preprocessed_data_dir, model_dir, config):
    """Train all models using the preprocessed data."""
    logger.info("Training models...")
    models = {
        'RandomForest': RandomForestModel(config['random_forest']),
        'SVM': SVMModel(config['svm']),
        'NeuralNetwork': NeuralNetworkModel(config['neural_network']),
        'KMeans': KMeansModel(config['kmeans']),
        'DBSCAN': DBSCANModel(config['dbscan']),
        'BERT': BERTModel(config['bert']),
        'GPT': GPTModel(config['gpt']),
    }
    
    for model_name, model in models.items():
        logger.info(f"Training {model_name}...")
        for file_name in os.listdir(preprocessed_data_dir):
            file_path = os.path.join(preprocessed_data_dir, file_name)
            data = joblib.load(file_path)
            model.train(data)
        
        model_path = os.path.join(model_dir, f"{model_name}_model.pkl")
        joblib.dump(model, model_path)
        logger.info(f"{model_name} model saved to {model_path}")

def evaluate_models(model_dir, preprocessed_data_dir, config):
    """Evaluate trained models and adjust weights dynamically."""
    logger.info("Evaluating models...")
    metrics = PerformanceMetrics()
    weight_adjuster = DynamicWeightAdjuster(config['weight_adjuster'])
    
    for model_file in os.listdir(model_dir):
        model_path = os.path.join(model_dir, model_file)
        model = joblib.load(model_path)
        performance_scores = []
        
        for file_name in os.listdir(preprocessed_data_dir):
            file_path = os.path.join(preprocessed_data_dir, file_name)
            data = joblib.load(file_path)
            predictions, confidence = model.predict(data)
            performance_score = metrics.evaluate(predictions, confidence)
            performance_scores.append(performance_score)
        
        avg_score = sum(performance_scores) / len(performance_scores)
        weight_adjuster.update_weight(model_file, avg_score)
        logger.info(f"{model_file}: Average performance score = {avg_score:.2f}")

def main():
    """Main function to execute the training pipeline."""
    config_path = 'config/training_config.yaml'
    config = load_config(config_path)
    
    raw_data_dir = 'data/raw'
    preprocessed_data_dir = 'data/processed'
    model_dir = 'training/checkpoints'
    
    os.makedirs(preprocessed_data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    preprocess_data(raw_data_dir, preprocessed_data_dir, config['preprocessing'])
    train_models(preprocessed_data_dir, model_dir, config['models'])
    evaluate_models(model_dir, preprocessed_data_dir, config['evaluation'])

if __name__ == '__main__':
    main()
