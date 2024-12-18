from tacoreader.load import load, load_metadata, TortillaDataFrame
from tacoreader.compile import compile
import pandas as pd
import geopandas as gpd

__all__ = ["load", "compile", "load_metadata"]
__version__ = "0.4.3"

# TODO: I can't implement a better solution for this
# because the TortillaDataFrame constructor enter in conflict 
# with the geopandas constructor.
_original_concat = pd.concat
_original_merge = pd.merge


# Custom pd.concat to handle TortillaDataFrame
# Custom pd.concat to handle TortillaDataFrame
def tortilla_concat(objs, *args, **kwargs):
    if all(isinstance(obj, TortillaDataFrame) for obj in objs):
        result = pd.concat(objs, *args, **kwargs)
        return result.__finalize__(objs[0])
    return pd.concat(objs, *args, **kwargs)

# Monkey patch the pd.concat function
def custom_concat(objs, *args, **kwargs):
    result = _original_concat(objs, *args, **kwargs)

    # Check if any input is a TortillaDataFrame
    if any(isinstance(obj, TortillaDataFrame) for obj in objs):
        result = TortillaDataFrame(gpd.GeoDataFrame(result))
    return result

def custom_merge(left, right, *args, **kwargs):
    result = _original_merge(left, right, *args, **kwargs)

    # Check if any input is a TortillaDataFrame
    if isinstance(left, TortillaDataFrame) or isinstance(right, TortillaDataFrame):
        result = TortillaDataFrame(gpd.GeoDataFrame(result))
    return result

pd.concat = custom_concat   
pd.merge = custom_merge
