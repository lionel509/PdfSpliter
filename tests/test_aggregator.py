import unittest
from aggregation import WeightedVotingAggregator, MajorityVotingAggregator

class TestWeightedVotingAggregator(unittest.TestCase):
    def setUp(self):
        self.aggregator = WeightedVotingAggregator()

    def test_weighted_voting(self):
        predictions = [
            {"label": "A"},
            {"label": "A"},
            {"label": "B"}
        ]
        confidences = [0.6, 0.3, 0.1]  # Weights for predictions
        result = self.aggregator.aggregate(predictions, confidences)
        self.assertEqual(result["result"], "A", "Weighted voting failed to select the correct label.")
        self.assertGreater(result["details"]["A"], result["details"]["B"], "Confidence weighting is incorrect.")

class TestMajorityVotingAggregator(unittest.TestCase):
    def setUp(self):
        self.aggregator = MajorityVotingAggregator()

    def test_majority_voting(self):
        predictions = [
            {"label": "A"},
            {"label": "A"},
            {"label": "B"}
        ]
        result = self.aggregator.aggregate(predictions, confidences=None)
        self.assertEqual(result["result"], "A", "Majority voting failed to select the correct label.")

    def test_majority_voting_tie(self):
        predictions = [
            {"label": "A"},
            {"label": "B"},
            {"label": "B"}
        ]
        result = self.aggregator.aggregate(predictions, confidences=None)
        self.assertEqual(result["result"], "B", "Majority voting failed to handle a tie correctly.")

if __name__ == "__main__":
    unittest.main()
