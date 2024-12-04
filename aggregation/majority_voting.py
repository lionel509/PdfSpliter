# majority_voting.py

from collections import Counter
from typing import List, Dict
from .aggregator import Aggregator

class MajorityVotingAggregator(Aggregator):
    """
    Implements majority voting aggregation.
    """

    def aggregate(self, predictions: List[Dict], confidences: List[float] = None) -> Dict:
        """
        Aggregate predictions using majority voting.

        Args:
            predictions (List[Dict]): List of model predictions.
            confidences (List[float], optional): Ignored in this method.

        Returns:
            Dict: Aggregated result as a dictionary.
        """
        votes = Counter()
        for pred in predictions:
            for key, value in pred.items():
                votes[value] += 1

        # Determine final result by selecting the most common prediction
        result = votes.most_common(1)[0][0]
        return {"result": result, "details": dict(votes)}
