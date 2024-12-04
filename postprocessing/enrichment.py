
from typing import Dict
from datetime import datetime

def add_timestamp(aggregated_result: Dict) -> Dict:
    """
    Adds a timestamp to the result.

    Args:
        aggregated_result (Dict): The aggregated prediction and its details.

    Returns:
        Dict: Result with added timestamp.
    """
    aggregated_result["timestamp"] = datetime.now().isoformat()
    return aggregated_result

def add_model_attribution(aggregated_result: Dict, models: Dict[str, float]) -> Dict:
    """
    Adds model attribution data to the result.

    Args:
        aggregated_result (Dict): The aggregated prediction and its details.
        models (Dict[str, float]): Model names and their respective weights or contributions.

    Returns:
        Dict: Result with added model attribution.
    """
    aggregated_result["models"] = models
    return aggregated_result
