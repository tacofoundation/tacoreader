import random
from typing import Dict

import pandas as pd


def tortilla_message() -> str:
    """Get a random tortilla message"""

    tortilla_messages = [
        "Making a tortilla",
        "Making a tortilla 🫓",
        "Cooking a tortilla",
        "Making a tortilla 🫓",
        "Working on a tortilla",
        "Working on a tortilla 🫓",
        "Rolling out a tortilla",
        "Rolling out a tortilla 🫓",
        "Baking a tortilla",
        "Baking a tortilla 🫓",
        "Grilling a tortilla",
        "Grilling a tortilla 🫓",
        "Toasting a tortilla",
        "Toasting a tortilla 🫓",
    ]

    # Randomly accessing a message
    random_message = random.choice(tortilla_messages)
    return random_message


def build_simplified_range_header(df: pd.DataFrame) -> Dict[str, str]:
    """Build a simplified range header from a DataFrame of byte ranges
    That means that consecutive byte ranges will be merged into a single range.

    Args:
        df (pd.DataFrame): A DataFrame with columns "offset" and "length"
            that represent the byte ranges.

    Returns:
        Dict[str, str]: A dictionary with the "Range" header
    """

    # Create a list to hold the simplified byte ranges
    simplified_ranges = []

    # Get the initial offset and length
    current_start = df.iloc[0]["tortilla:offset"]
    current_end = current_start + df.iloc[0]["tortilla:length"]

    # Iterate over the rows starting from the second row
    for i in range(1, len(df)):
        row = df.iloc[i]
        next_start = row["tortilla:offset"]
        next_end = next_start + row["tortilla:length"]

        # If the current range and the next range are consecutive, merge them
        if next_start == current_end:
            current_end = next_end
        else:
            # Otherwise, add the current range to the list and start a new one
            simplified_ranges.append((current_start, current_end))
            current_start = next_start
            current_end = next_end

    # Append the final range
    simplified_ranges.append((current_start, current_end))

    # Create the range header
    range_header = ",".join(
        [f"bytes={start}-{end-1}" for start, end in simplified_ranges]
    )
    headers = {"Range": range_header}

    return headers

def human2bytes(size_str: str) -> int:
    """
    Converts a human-readable size string (e.g., "100MB") into bytes.
    Supported units: KB, MB, GB, TB, PB.
    """
    units = {"KB": 10**3, "MB": 10**6, "GB": 10**9, "TB": 10**12, "PB": 10**15}
    size_str = size_str.strip().upper()

    for unit, multiplier in units.items():
        if size_str.endswith(unit):
            try:
                value = float(size_str[: -len(unit)].strip())
                return int(value * multiplier)
            except ValueError:
                raise ValueError(f"Invalid size value in '{size_str}'.")
    raise ValueError(
        f"Unsupported unit in '{size_str}'. Supported units are: {', '.join(units.keys())}."
    )