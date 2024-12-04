from typing import Dict

def format_to_json(aggregated_result: Dict) -> str:
    """
    Formats the aggregated result into JSON.

    Args:
        aggregated_result (Dict): The aggregated prediction and its details.

    Returns:
        str: JSON-formatted string of the result.
    """
    import json
    return json.dumps(aggregated_result, indent=4)

def format_to_plain_text(aggregated_result: Dict) -> str:
    """
    Formats the aggregated result into plain text.

    Args:
        aggregated_result (Dict): The aggregated prediction and its details.

    Returns:
        str: Plain text representation of the result.
    """
    result = f"Result: {aggregated_result['result']}\nDetails:\n"
    for key, value in aggregated_result['details'].items():
        result += f"  {key}: {value:.2f}\n"
    return result
