import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, List, Union

import pandas as pd

from tacoreader.v1.loader_dataframe import TortillaDataFrame

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def sanity_check(
    sample_metadata: pd.DataFrame,
    read_function: Callable[[Union[str, bytes]], None],
    batch_size: int = 400,
    max_workers: int = 4,
    **kwargs,
) -> List[str]:
    """
    BATCH Perform a sanity check on a given taco file to validate its contents.

    Parameters:
        file (Union[str, pathlib.Path]): Path to the taco file.
        read_function (Callable[[Union[str, bytes]], None]): Function to read the contents of the file.
        max_workers (int): Number of threads for concurrent processing. Default is 4.
        **kwargs: Ignored keyword arguments.

    Returns:
        List[str]: A list of IDs that failed the sanity check.
    """
    # get kwargs arguments
    super_name = kwargs.get("super_name", "")
    if super_name == "":
        logging.info(f"The sanity check is starting for {len(sample_metadata)} items.")

    # Split every 200 entries
    sample_metadata_batches = [
        sample_metadata.iloc[i : i + batch_size]
        for i in range(0, len(sample_metadata), batch_size)
    ]

    failed_ids = []
    for idx in range(len(sample_metadata_batches)):
        failed_ids += sanity_check_batch(
            sample_metadata=TortillaDataFrame(sample_metadata_batches[idx]),
            read_function=read_function,
            max_workers=max_workers,
            super_name=super_name,
        )

    if failed_ids:
        logging.warning(f"Sanity check failed for {len(failed_ids)} items.")
    else:
        if super_name == "":
            logging.info("All items passed the sanity check.")

    return failed_ids


def sanity_check_batch(
    sample_metadata: pd.DataFrame,
    read_function: Callable[[Union[str, bytes]], None],
    max_workers: int = 4,
    **kwargs,
) -> List[str]:
    """
    BATCH Perform a sanity check on a given taco file to validate its contents.

    Parameters:
        file (Union[str, pathlib.Path]): Path to the taco file.
        read_function (Callable[[Union[str, bytes]], None]): Function to read the contents of the file.
        max_workers (int): Number of threads for concurrent processing. Default is 4.
        **kwargs: Ignored keyword arguments.

    Returns:
        List[str]: A list of IDs that failed the sanity check.
    """
    super_name = kwargs.get("super_name", "")
    failed_ids = []
    tasks = []

    # Use ProcessPoolExecutor for concurrent processing
    if max_workers is None:
        for idx in range(len(sample_metadata)):
            result = process_entry(idx, sample_metadata, read_function, super_name)
            if result:
                failed_ids.append(result)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            tasks = [
                executor.submit(
                    process_entry, idx, sample_metadata, read_function, super_name
                )
                for idx in range(len(sample_metadata))
            ]
            for future in as_completed(tasks):
                result = future.result()
                if result:
                    failed_ids.append(result)

    return failed_ids


def process_entry(idx, sample_metadata, read_function, super_name):
    """Processes a single metadata entry."""
    result = sample_metadata.read(idx)
    sample_id = super_name + sample_metadata.iloc[idx]["tortilla:id"]
    try:
        if isinstance(result, str) or isinstance(result, bytes):
            read_function(result)
        elif isinstance(result, pd.DataFrame):
            return sanity_check(
                result, read_function, max_workers=None, super_name=sample_id
            )
        else:
            raise ValueError(f"Unsupported return type for entry {sample_id}.")
    except Exception:
        return sample_id
