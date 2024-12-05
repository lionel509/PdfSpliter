import numpy as np

class DisagreementHandler:
    def __init__(self, threshold=0.2):
        """
        Initializes the disagreement handler.
        :param threshold: Maximum allowable disagreement between models.
        """
        self.threshold = threshold

    def handle_disagreement(self, model_predictions):
        """
        Resolves disagreements between models.
        :param model_predictions: Dictionary of model names and their predictions.
        :return: Resolved prediction or None if no consensus is reached.
        """
        # Aggregate predictions
        aggregated_predictions = {}
        for model, predictions in model_predictions.items():
            for i, prediction in enumerate(predictions):
                if i not in aggregated_predictions:
                    aggregated_predictions[i] = []
                aggregated_predictions[i].append(prediction)

        # Resolve disagreements
        resolved_predictions = []
        for pred_list in aggregated_predictions.values():
            pred_counts = np.unique(pred_list, return_counts=True)
            majority_vote = pred_counts[0][np.argmax(pred_counts[1])]

            # Check if majority confidence exceeds the threshold
            if max(pred_counts[1]) / len(pred_list) > self.threshold:
                resolved_predictions.append(majority_vote)
            else:
                resolved_predictions.append(None)  # No consensus
        
        return resolved_predictions
