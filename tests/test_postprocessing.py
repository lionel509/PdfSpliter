import unittest
import os
from postprocessing import (
    format_to_json,
    format_to_plain_text,
    filter_low_confidence,
    add_timestamp,
    add_model_attribution,
)
from datetime import datetime
import json


class TestResultFormatter(unittest.TestCase):
    def setUp(self):
        self.aggregated_result = {
            "result": "A",
            "details": {"A": 0.6, "B": 0.3, "C": 0.1},
        }

    def test_format_to_json(self):
        json_result = format_to_json(self.aggregated_result)
        parsed_result = json.loads(json_result)
        self.assertEqual(parsed_result["result"], "A")
        self.assertIn("details", parsed_result)

    def test_format_to_plain_text(self):
        plain_text = format_to_plain_text(self.aggregated_result)
        self.assertIn("Result: A", plain_text)
        self.assertIn("A: 0.60", plain_text)


class TestNoiseFilter(unittest.TestCase):
    def setUp(self):
        self.aggregated_result = {
            "result": "A",
            "details": {"A": 0.6, "B": 0.3, "C": 0.1},
        }

    def test_filter_low_confidence(self):
        filtered_result = filter_low_confidence(self.aggregated_result, threshold=0.4)
        self.assertEqual(filtered_result["result"], "A")
        self.assertNotIn("C", filtered_result["details"])

    def test_filter_all_below_threshold(self):
        filtered_result = filter_low_confidence(self.aggregated_result, threshold=0.7)
        self.assertEqual(filtered_result["result"], "Uncertain")
        self.assertEqual(filtered_result["details"], {})


class TestEnrichment(unittest.TestCase):
    def setUp(self):
        self.aggregated_result = {
            "result": "A",
            "details": {"A": 0.6, "B": 0.3, "C": 0.1},
        }

    def test_add_timestamp(self):
        enriched_result = add_timestamp(self.aggregated_result)
        timestamp = enriched_result["timestamp"]
        self.assertIsInstance(timestamp, str)
        datetime.fromisoformat(timestamp)  # Should not raise an exception

    def test_add_model_attribution(self):
        models = {"Model1": 0.6, "Model2": 0.4}
        enriched_result = add_model_attribution(self.aggregated_result, models)
        self.assertIn("models", enriched_result)
        self.assertEqual(enriched_result["models"], models)


if __name__ == "__main__":
    unittest.main()
