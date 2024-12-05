from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class PerformanceMetrics:
    def evaluate(self, true_labels, predictions):
        """
        Evaluates model performance based on various metrics.
        :param true_labels: Ground truth labels.
        :param predictions: Predicted labels from the model.
        :return: Dictionary of performance metrics.
        """
        metrics = {
            "accuracy": accuracy_score(true_labels, predictions),
            "precision": precision_score(true_labels, predictions, average="weighted", zero_division=0),
            "recall": recall_score(true_labels, predictions, average="weighted", zero_division=0),
            "f1_score": f1_score(true_labels, predictions, average="weighted", zero_division=0)
        }
        return metrics
