# aggregator.py

from abc import ABC, abstractmethod
from typing import List, Dict

class Aggregator(ABC):
    """
    Abstract base class for all aggregation methods.
    """

    @abstractmethod
    def aggregate(self, predictions: List[Dict], confidences: List[float]) -> Dict:
        """
        Abstract method to aggregate predictions.

        Args:
            predictions (List[Dict]): List of model predictions.
            confidences (List[float]): List of confidence scores for each model.

        Returns:
            Dict: Aggregated result as a dictionary.
        """
        pass
