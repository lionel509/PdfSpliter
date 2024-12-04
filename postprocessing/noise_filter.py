from typing import Dict

def filter_low_confidence(aggregated_result: Dict, threshold: float = 0.5) -> Dict:
    """
    Filters out predictions with confidence scores below the threshold.

    Args:
        aggregated_result (Dict): The aggregated prediction and its details.
        threshold (float): Minimum confidence level to retain a prediction.

    Returns:
        Dict: Filtered results.
    """
    filtered_details = {key: value for key, value in aggregated_result["details"].items() if value >= threshold}

    if not filtered_details:
        return {"result": "Uncertain", "details": {}}

    # Recalculate the final result
    result = max(filtered_details, key=filtered_details.get)
    return {"result": result, "details": filtered_details}
