import importlib.metadata as _metadata

from tacoreader.concat import concat
from tacoreader.dataframe import TacoDataFrame
from tacoreader.dataset import TacoDataset
from tacoreader.loader import load

__version__ = _metadata.version("tacoreader")

__all__ = ["TacoDataFrame", "TacoDataset", "concat", "load"]
