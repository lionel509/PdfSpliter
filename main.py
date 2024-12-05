import argparse
import os
from preprocessing.text_preprocessor import TextPreprocessor
from preprocessing.image_preprocessor import ImagePreprocessor
from preprocessing.tabular_preprocessor import TabularPreprocessor
from models.supervised.random_forest_model import RandomForestModel
from models.supervised.svm_model import SVMModel
from models.supervised.neural_network_model import NeuralNetworkModel
from models.unsupervised.kmeans_model import KMeansModel
from models.unsupervised.dbscan_model import DBSCANModel
from models.pretrained.bert_model import BERTModel
from models.pretrained.gpt_model import GPTModel
from aggregation.weighted_voting import WeightedVotingAggregator
from evaluation.performance_metrics import PerformanceMetrics
from postprocessing.result_formatter import format_to_json, format_to_plain_text


def preprocess(input_path):
    # Detect input type (text, image, tabular)
    if input_path.endswith(".txt"):
        preprocessor = TextPreprocessor(lower_case=True, remove_stopwords=True)
        with open(input_path, 'r') as file:
            content = file.read()
        return preprocessor.process(content)
    elif input_path.endswith((".jpg", ".png")):
        preprocessor = ImagePreprocessor(target_size=(224, 224), normalize=True)
        return preprocessor.process(input_path)
    elif input_path.endswith(".csv"):
        preprocessor = TabularPreprocessor(scale_features=True, impute_strategy="mean")
        return preprocessor.process(input_path)
    else:
        raise ValueError("Unsupported file format. Supported formats: .txt, .jpg, .png, .csv")


def run_models(data):
    models = {
        "RandomForest": RandomForestModel(n_estimators=100, max_depth=10),
        "SVM": SVMModel(kernel="rbf", C=1.0),
        "NeuralNetwork": NeuralNetworkModel(hidden_layer_sizes=(128, 64, 32)),
        "KMeans": KMeansModel(n_clusters=5),
        "DBSCAN": DBSCANModel(eps=0.5, min_samples=5),
        "BERT": BERTModel(model_name="bert-base-uncased", num_labels=3),
        "GPT": GPTModel(model_name="gpt2"),
    }

    predictions = {}
    confidences = []
    for name, model in models.items():
        try:
            model_predictions = model.predict(data)
            predictions[name] = model_predictions
            confidences.append(0.9)  # Placeholder confidence for demo purposes
        except Exception as e:
            print(f"Model {name} failed: {e}")
    return predictions, confidences


def aggregate(predictions, confidences):
    aggregator = WeightedVotingAggregator()
    return aggregator.aggregate(predictions, confidences)


def evaluate(result):
    metrics = PerformanceMetrics()
    true_labels = [0, 1, 0]  # Placeholder ground truth labels for demo
    predictions = result.get("result", [])
    return metrics.evaluate(true_labels, predictions)


def postprocess(result, output_format):
    if output_format == "json":
        return format_to_json(result)
    elif output_format == "text":
        return format_to_plain_text(result)
    else:
        raise ValueError("Unsupported output format. Supported formats: json, text")


def main(args):
    data = preprocess(args.input)
    predictions, confidences = run_models(data)
    aggregated_result = aggregate(predictions, confidences)
    evaluation_result = evaluate(aggregated_result)
    final_output = postprocess(evaluation_result, args.output)
    print(final_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process, run models, and output results.")
    parser.add_argument("--input", required=True, help="Path to the input file")
    parser.add_argument("--output", required=True, choices=["json", "text"], help="Output format")
    args = parser.parse_args()
    main(args)
