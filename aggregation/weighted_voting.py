# weighted_voting.py

from typing import List, Dict
from .aggregator import Aggregator

class WeightedVotingAggregator(Aggregator):
    """
    Implements confidence-weighted voting aggregation.
    """

    def aggregate(self, predictions: List[Dict], confidences: List[float]) -> Dict:
        """
        Aggregate predictions using confidence-weighted voting.

        Args:
            predictions (List[Dict]): List of model predictions.
            confidences (List[float]): List of confidence scores for each model.

        Returns:
            Dict: Aggregated result as a dictionary.
        """
        weighted_scores = {}
        total_confidence = sum(confidences)

        for pred, conf in zip(predictions, confidences):
            label = pred["label"]
            if label not in weighted_scores:
                weighted_scores[label] = 0
            weighted_scores[label] += conf / total_confidence

        # Determine final result by selecting the max-weighted score
        result = max(weighted_scores, key=weighted_scores.get)
        return {"result": result, "details": weighted_scores}
