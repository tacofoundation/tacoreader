from tacoreader.compile import compile
from tacoreader.loader_dataframe import load
from tacoreader.loader_metadata import load_metadata
from tacoreader.sanity import sanity_check
from tacoreader.TortillaDataFrame import TortillaDataFrame

__all__ = ["load", "load_metadata", "compile", "sanity_check", "TortillaDataFrame"]

import importlib.metadata as _metadata

__version__ = _metadata.version("tacoreader")
