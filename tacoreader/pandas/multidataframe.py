import pandas as pd
import numpy as np
from typing import Any
from typing_extensions import Self
from collections.abc import Sequence


class MultiDataFrame(pd.DataFrame):
    """
    Hierarchical DataFrame with chainable navigation and filtering preservation.
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
    
    @property
    def _constructor_expanddim(self):
        """Ensure operations return MultiDataFrame when expanding dimensions."""
        return MultiDataFrame
    
    def __getitem__(self, key):
        """Override [] operator to preserve hierarchy in all cases."""
        
        # Boolean indexing: dataset[dataset["col"] == "value"]
        if isinstance(key, (pd.Series, np.ndarray)) and hasattr(key, 'dtype') and key.dtype == bool:
            return self._filter_with_boolean_mask(key)
        
        # Column selection: dataset["col"] or dataset[["col1", "col2"]]
        elif isinstance(key, (str, list)):
            result = super().__getitem__(key)
            # If result is DataFrame, preserve auxiliaries
            if isinstance(result, pd.DataFrame):
                return self._wrap_result(result, preserve_auxiliaries=True)
            return result
        
        # Other indexing (slice, integer, etc.)
        else:
            result = super().__getitem__(key)
            if isinstance(result, pd.DataFrame):
                return self._wrap_result(result, preserve_auxiliaries=True)
            return result
    
    def _build_position_mapping(self, boolean_mask: pd.Series) -> dict:
        """Build mapping between original and filtered positions."""
        # Get original indices that pass the mask
        if hasattr(boolean_mask, 'index'):
            # Align mask with DataFrame index
            aligned_mask = boolean_mask.reindex(self.index, fill_value=False)
        else:
            aligned_mask = boolean_mask
            
        original_positions = self.index[aligned_mask].tolist()
        new_positions = list(range(len(original_positions)))
        
        return {
            'original_to_new': dict(zip(original_positions, new_positions)),
            'new_to_original': dict(zip(new_positions, original_positions)),
            'filtered_indices': original_positions
        }
    
    def _update_auxiliary_positions(self, aux_df: pd.DataFrame, position_mapping: dict) -> pd.DataFrame:
        """Update internal:position references after primary filtering."""
        if 'internal:position' not in aux_df.columns:
            return aux_df.copy()
        
        # Map old positions to new positions
        aux_copy = aux_df.copy()
        original_to_new = position_mapping['original_to_new']
        
        # Update positions, set to -1 if parent was filtered out
        aux_copy['internal:position'] = aux_copy['internal:position'].map(
            lambda pos: original_to_new.get(pos, -1) if pd.notna(pos) else -1
        )
        
        # Remove rows where parent was filtered out
        valid_aux = aux_copy[aux_copy['internal:position'] != -1]
        return valid_aux.reset_index(drop=True)
    
    def _filter_with_boolean_mask(self, mask: pd.Series) -> 'MultiDataFrame':
        """Filter preserving hierarchical structure."""
        
        # Apply mask to primary DataFrame
        filtered_primary = super().__getitem__(mask).copy()
        
        if filtered_primary.empty:
            # Return empty MultiDataFrame with empty auxiliaries
            empty_auxiliaries = []
            for aux_df in self._auxiliary_dfs:
                # Create empty DataFrame with same columns as original auxiliary
                empty_aux = pd.DataFrame(columns=aux_df.columns)
                empty_auxiliaries.append(empty_aux)
            
            return MultiDataFrame(filtered_primary, auxiliary_dataframes=empty_auxiliaries)
        
        # Build position mapping
        position_mapping = self._build_position_mapping(mask)
        
        # Update auxiliary DataFrames
        updated_auxiliaries = []
        for aux_df in self._auxiliary_dfs:
            updated_aux = self._update_auxiliary_positions(aux_df, position_mapping)
            updated_auxiliaries.append(updated_aux)
        
        # Create new MultiDataFrame with updated data
        # Use pandas reset_index directly to avoid our prohibition
        reset_primary = pd.DataFrame(filtered_primary).reset_index(drop=True)
        result = MultiDataFrame(
            reset_primary,
            auxiliary_dataframes=updated_auxiliaries
        )
        
        # Validate result integrity
        result._validate_auxiliary_integrity()
        
        return result
    
    def _wrap_result(self, result: pd.DataFrame, preserve_auxiliaries: bool = False) -> 'MultiDataFrame':
        """Wrap pandas DataFrame results to maintain MultiDataFrame type."""
        
        if not preserve_auxiliaries:
            # Return without auxiliaries
            return MultiDataFrame(result)
        
        if len(result) == len(self) and result.index.equals(self.index):
            # Same length and index - preserve auxiliaries as-is
            return MultiDataFrame(result, auxiliary_dataframes=self._auxiliary_dfs)
        
        elif len(result) <= len(self):
            # Subset of data - need to filter auxiliaries
            try:
                # Create boolean mask based on result index
                mask = self.index.isin(result.index)
                return self._filter_with_boolean_mask(mask)
            except Exception:
                # Fallback: return without auxiliaries if filtering fails
                return MultiDataFrame(result)
        
        # Default: return without auxiliaries for complex cases
        return MultiDataFrame(result)
    
    def _validate_auxiliary_integrity(self) -> bool:
        """Validate that auxiliary position references are valid."""
        for i, aux_df in enumerate(self._auxiliary_dfs):
            if 'internal:position' in aux_df.columns and not aux_df.empty:
                max_pos = aux_df['internal:position'].max()
                if pd.notna(max_pos) and max_pos >= len(self):
                    raise ValueError(
                        f"Auxiliary level {i} has invalid position reference: {max_pos} "
                        f"(primary DataFrame has {len(self)} rows)"
                    )
        return True
    
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
                result._auxiliary_dfs = self._auxiliary_dfs[1:] if len(self._auxiliary_dfs) > 1 else []
                
                return result
            else:
                return next_level
        
        else:
            # Terminal file - return VSI path
            if 'internal:gdal_vsi' in row:
                vsi_path = row['internal:gdal_vsi']
                if pd.notna(vsi_path):
                    return str(vsi_path)
            
            raise ValueError(f"No VSI path for row with id '{identifier}'")
    
    def get_auxiliary_df(self, index: int) -> Self:
        """Get auxiliary level by index."""
        if not (0 <= index < len(self._auxiliary_dfs)):
            raise IndexError(f"Auxiliary index {index} out of bounds")
        return self._auxiliary_dfs[index]
    
    def get_auxiliary_count(self) -> int:
        """Number of auxiliary levels."""
        return len(self._auxiliary_dfs)
    
    # Additional pandas integration methods
    
    def query(self, expr: str, **kwargs) -> 'MultiDataFrame':
        """pandas.query() with hierarchy preservation."""
        result = super().query(expr, **kwargs)
        return self._wrap_result(result, preserve_auxiliaries=True)
    
    def drop(self, *args, **kwargs) -> 'MultiDataFrame':
        """Drop rows/columns with hierarchy preservation."""
        result = super().drop(*args, **kwargs)
        return self._wrap_result(result, preserve_auxiliaries=True)
    
    def reset_index(self, *args, **kwargs):
        """Reset index is prohibited in MultiDataFrame to preserve hierarchical integrity."""
        if self._auxiliary_dfs:
            raise ValueError(
                "reset_index() is not allowed on MultiDataFrame with auxiliaries as it breaks "
                "hierarchical position references. Convert to pandas DataFrame first:\n"
                "  df = pd.DataFrame(multi_df)\n"
                "  df.reset_index(...)"
            )
        
        # Allow reset_index only if no auxiliaries
        if kwargs.get('inplace', False):
            super().reset_index(*args, **kwargs)
            return None
        else:
            result = super().reset_index(*args, **kwargs)
            return MultiDataFrame(result)
    
    def copy(self, deep: bool = True) -> 'MultiDataFrame':
        """Copy MultiDataFrame with auxiliaries."""
        copied_primary = super().copy(deep=deep)
        
        if deep:
            copied_auxiliaries = [aux_df.copy(deep=True) for aux_df in self._auxiliary_dfs]
        else:
            copied_auxiliaries = self._auxiliary_dfs.copy()
        
        return MultiDataFrame(copied_primary, auxiliary_dataframes=copied_auxiliaries)
    
    def head(self, n: int = 5) -> 'MultiDataFrame':
        """Return first n rows with filtered auxiliaries."""
        head_result = super().head(n)
        return self._wrap_result(head_result, preserve_auxiliaries=True)
    
    def tail(self, n: int = 5) -> 'MultiDataFrame':
        """Return last n rows with filtered auxiliaries."""
        tail_result = super().tail(n)
        return self._wrap_result(tail_result, preserve_auxiliaries=True)
    
    def sample(self, *args, **kwargs) -> 'MultiDataFrame':
        """Sample rows with filtered auxiliaries."""
        sample_result = super().sample(*args, **kwargs)
        return self._wrap_result(sample_result, preserve_auxiliaries=True)
    
    def __repr__(self) -> str:
        """Enhanced representation showing auxiliary levels."""
        base_repr = super().__repr__()
        aux_info = f"\nAuxiliary levels: {len(self._auxiliary_dfs)}"
        
        if self._auxiliary_dfs:
            aux_counts = [len(aux) for aux in self._auxiliary_dfs]
            aux_info += f" (sizes: {aux_counts})"
        
        return base_repr + aux_info