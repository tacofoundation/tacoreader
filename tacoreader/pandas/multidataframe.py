import pandas as pd
from typing import Any
from typing_extensions import Self
from collections.abc import Sequence
from tacoreader.pandas import stats

class MultiDataFrame(pd.DataFrame):
    """
    Hierarchical DataFrame with chainable navigation.
    """
    
    _metadata = ['_auxiliary_dfs']
    
    def __init__(
        self, 
        data: Any = None, 
        auxiliary_dataframes: Sequence[Any] | None = None, 
        *args, 
        **kwargs
    ) -> None:
        
        # Create DataFrame directly without modifying data
        super().__init__(data, *args, **kwargs)
        
        # Create auxiliary chain
        if auxiliary_dataframes is None:
            self._auxiliary_dfs = []
        elif isinstance(auxiliary_dataframes, list):
            self._auxiliary_dfs = []
            for aux_data in auxiliary_dataframes:
                aux_mdf = MultiDataFrame(aux_data)
                self._auxiliary_dfs.append(aux_mdf)
        else:
            self._auxiliary_dfs = [MultiDataFrame(auxiliary_dataframes)]
    
    @property
    def _constructor(self):
        return MultiDataFrame
    
    @property
    def _constructor_sliced(self):
        return pd.Series
    
    def read(self, identifier: int | str) -> Self | str:
        """Navigate or return VSI path."""
        
        # Get row and position
        if isinstance(identifier, int):
            if identifier >= len(self):
                raise IndexError(f"Index {identifier} out of bounds")
            row = self.iloc[identifier]
            position = identifier
        else:
            # Find by ID
            matches = self[self['id'] == identifier]
            if matches.empty:
                raise ValueError(f"ID '{identifier}' not found")
            row = matches.iloc[0]
            position = matches.index[0]
        
        if row['type'] == 'TORTILLA':
            # Navigate to auxiliary
            if not self._auxiliary_dfs:
                raise ValueError("No auxiliary level available")
            
            next_level = self._auxiliary_dfs[0]
            
            # Filter by position using DataFrame column directly
            if 'internal:position' in next_level.columns:
                filtered_df = next_level[next_level['internal:position'] == position].copy()
                
                if filtered_df.empty:
                    raise ValueError(f"No data for position {position}")
                
                # Create new MultiDataFrame with filtered data
                result = MultiDataFrame(filtered_df.reset_index(drop=True))
                
                # Copy auxiliary chain
                result._auxiliary_dfs = next_level._auxiliary_dfs
                
                return result
            else:
                return next_level
        
        else:
            # Terminal file - return VSI path
            if 'internal:gdal_vsi' in self.columns:
                vsi_path = self.iloc[position]['internal:gdal_vsi']
                if pd.notna(vsi_path):
                    return str(vsi_path)
            
            raise ValueError(f"No VSI path for position {position}")
    
    def get_auxiliary_df(self, index: int) -> Self:
        """Get auxiliary level by index."""
        if not (0 <= index < len(self._auxiliary_dfs)):
            raise IndexError(f"Auxiliary index {index} out of bounds")
        return self._auxiliary_dfs[index]
    
    def get_auxiliary_count(self) -> int:
        """Number of auxiliary levels."""
        return len(self._auxiliary_dfs)
    
    # =============================================================================
    # STATS AGGREGATION METHODS (delegate to stats.py)
    # =============================================================================

    def aggregate_min(self, band: int | None = None) -> list[float] | float:
        """Aggregate minimum values across all samples."""        
        return stats.aggregate_min(self, band)

    def aggregate_max(self, band: int | None = None) -> list[float] | float:
        """Aggregate maximum values across all samples."""        
        return stats.aggregate_max(self, band)

    def aggregate_mean(self, band: int | None = None) -> list[float] | float:
        """Aggregate mean values across all samples."""        
        return stats.aggregate_mean(self, band)

    def aggregate_std_approximation(self, band: int | None = None, warn: bool = True) -> list[float] | float:
        """Approximate standard deviation aggregation."""        
        return stats.aggregate_std_approximation(self, band, warn)

    def aggregate_valid_pct(self, band: int | None = None) -> list[float] | float:
        """Aggregate valid pixel percentages."""        
        return stats.aggregate_valid_pct(self, band)

    def aggregate_p25_approximation(self, band: int | None = None, warn: bool = True) -> list[float] | float:
        """Approximate 25th percentile aggregation."""        
        return stats.aggregate_p25_approximation(self, band, warn)

    def aggregate_p50_approximation(self, band: int | None = None, warn: bool = True) -> list[float] | float:
        """Approximate 50th percentile (median) aggregation."""        
        return stats.aggregate_p50_approximation(self, band, warn)

    def aggregate_p75_approximation(self, band: int | None = None, warn: bool = True) -> list[float] | float:
        """Approximate 75th percentile aggregation."""        
        return stats.aggregate_p75_approximation(self, band, warn)

    def aggregate_p95_approximation(self, band: int | None = None, warn: bool = True) -> list[float] | float:
        """Approximate 95th percentile aggregation."""        
        return stats.aggregate_p95_approximation(self, band, warn)

    def aggregate_categorical(self, band: int | None = None) -> list[list[float]] | list[float]:
        """Aggregate categorical probability distributions."""        
        return stats.aggregate_categorical(self, band)    
    
