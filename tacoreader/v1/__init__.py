import importlib.metadata as _metadata

from tacoreader.v1.compile import compile
from tacoreader.v1.loader_dataframe import load
from tacoreader.v1.loader_metadata import load_metadata
from tacoreader.v1.sanity import sanity_check
from tacoreader.v1.TortillaDataFrame import TortillaDataFrame

__all__ = ["load", "load_metadata", "compile", "sanity_check", "TortillaDataFrame"]

__version__ = _metadata.version("tacoreader")
