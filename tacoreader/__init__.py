import importlib.metadata as _metadata

from tacoreader.dataframe import TacoDataFrame
from tacoreader.dataset import TacoDataset
from tacoreader.loader import load
from tacoreader.concat import concat


__version__ = _metadata.version("tacoreader")

__all__ = ["TacoDataFrame", "concat", "load", "TacoDataset"]
