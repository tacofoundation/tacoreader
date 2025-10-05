import importlib.metadata as _metadata

from tacoreader._tree import TacoDataFrame
from tacoreader.compiler import compile
from tacoreader.loader import load

__version__ = _metadata.version("tacoreader")

__all__ = ["TacoDataFrame", "compile", "load"]
