import unittest
from evaluation.performance_metrics import PerformanceMetrics
from evaluation.dynamic_weight_adjuster import DynamicWeightAdjuster
from evaluation.disagreement_handler import DisagreementHandler

class TestEvaluation(unittest.TestCase):
    def test_performance_metrics(self):
        metrics = PerformanceMetrics()
        true_labels = [0, 1, 1, 0, 1]
        predictions = [0, 1, 0, 0, 1]

        result = metrics.evaluate(true_labels, predictions)

        # Check if the metrics are computed correctly
        self.assertIn("accuracy", result)
        self.assertIn("precision", result)
        self.assertIn("recall", result)
        self.assertIn("f1_score", result)

        # Check metric ranges
        for metric, value in result.items():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_dynamic_weight_adjuster(self):
        adjuster = DynamicWeightAdjuster(initial_weights={"ModelA": 0.5, "ModelB": 0.5}, learning_rate=0.1, decay_factor=0.9)

        # Update weights
        adjuster.update_weight("ModelA", performance_score=0.9)
        adjuster.update_weight("ModelB", performance_score=0.7)

        weights = adjuster.get_weights()

        # Check if weights are updated
        self.assertIn("ModelA", weights)
        self.assertIn("ModelB", weights)
        self.assertNotEqual(weights["ModelA"], 0.5)
        self.assertNotEqual(weights["ModelB"], 0.5)

        # Check if weights are positive
        for weight in weights.values():
            self.assertGreaterEqual(weight, 0.0)

    def test_disagreement_handler(self):
        handler = DisagreementHandler(threshold=0.6)

        model_predictions = {
            "ModelA": [0, 1, 1, 0],
            "ModelB": [0, 1, 0, 0],
            "ModelC": [0, 0, 1, 0],
        }

        resolved_predictions = handler.handle_disagreement(model_predictions)

        # Ensure the result has the same number of predictions
        self.assertEqual(len(resolved_predictions), 4)

        # Check if majority voting resolves disagreements correctly
        # Example: First prediction is [0, 0, 0], so resolved = 0
        self.assertEqual(resolved_predictions[0], 0)

        # Check unresolved cases (when no majority exists)
        # Example: Threshold is 0.6, so predictions with no consensus should return None
        self.assertEqual(resolved_predictions[2], 1)  # For [1, 0, 1], majority is 1

if __name__ == "__main__":
    unittest.main()
