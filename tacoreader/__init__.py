import asyncio
import importlib.metadata as _metadata

from tacoreader._tree import TacoDataFrame
from tacoreader.compiler import compile
from tacoreader.loader import load


def _ensure_async_compatible() -> None:
    """
    Auto-apply nest_asyncio patch if running event loop detected.

    Enables asyncio.run() to work in Jupyter/Colab environments.
    nest_asyncio has internal protection against multiple applications.
    """
    try:
        asyncio.get_running_loop()
        # Loop detected - apply patch
        from tacoreader import _nest_asyncio

        _nest_asyncio.apply()
    except RuntimeError:
        # No running loop - normal Python, no patch needed
        pass


# Auto-apply compatibility on import
_ensure_async_compatible()

__version__ = _metadata.version("tacoreader")

__all__ = ["TacoDataFrame", "compile", "load"]
