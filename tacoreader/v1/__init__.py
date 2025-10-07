import importlib

from tacoreader.v1.compile import compile
from tacoreader.v1.loader_dataframe import load
from tacoreader.v1.loader_metadata import load_metadata
from tacoreader.v1.sanity import sanity_check
from tacoreader.v1.TortillaDataFrame import TortillaDataFrame

__all__ = ["load", "load_metadata", "compile", "sanity_check", "TortillaDataFrame"]

# Hard dependency check (raise immediately if missing)
_REQUIRED_DEPS = ["requests", "tqdm", "fsspec"]

for _pkg in _REQUIRED_DEPS:
    try:
        importlib.import_module(_pkg)
    except ModuleNotFoundError as e:
        raise ImportError(
            f"Missing required dependency '{_pkg}'. "
            f"Install it with: pip install {_pkg}"
        ) from e
