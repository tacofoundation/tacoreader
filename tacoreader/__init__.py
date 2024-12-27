from tacoreader.loader_dataframe import load, TortillaDataFrame
from tacoreader.loader_metadata import load_metadata
from tacoreader.compile import compile
from tacoreader.sanity import sanity_check
import pandas as pd
import geopandas as gpd

__all__ = ["load", "compile", "load_metadata", "sanity_check", "TortillaDataFrame"]
__version__ = "0.5.0-beta1"

# TODO: I can't implement a better solution for this
# because the TortillaDataFrame constructor enter in conflict 
# with the geopandas constructor. So, I'm using a monkey patch
# If you have a better solution, please let me know. :)
_original_concat = pd.concat
_original_merge = pd.merge

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