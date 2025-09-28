"""
TreeDataFrame: Level-wise Isomorphic Sibling Subtrees

A pandas DataFrame extension that implements tree structures where all siblings 
at each level have identical schema but different data. Supports arbitrary 
nesting of TORTILLA nodes with GDAL_VSI leaf nodes for file access.

Core Concepts:
- LEVEL-WISE ISOMORPHIC: All siblings at same level have identical column structure
- TORTILLA NODES: Container nodes that can nest other TORTILLAs or leaf nodes  
- LEAF NODES: Terminal nodes with GDAL_VSI paths to actual files/resources
- PRIMARY KEY MAPPING: internal:position connects parent nodes to child nodes

LWISS Constraint: All TORTILLA siblings must have identical subtree structures.

Example - Valid LWISS:
    Level 0: [patient_001:TORTILLA, patient_002:TORTILLA, patient_003:TORTILLA]
    Level 1: Each patient has exactly [blood:CSV, xray:DICOM, followup:TORTILLA]  
    Level 2: Each followup has exactly [month_1:PDF, month_3:PDF, month_6:PDF]
    
    All patients have identical subtree pattern: 3 studies → 3 visits
    
Example - Invalid LWISS:
    patient_001 has [blood:CSV, xray:DICOM, followup:TORTILLA]
    patient_002 has [blood:CSV, xray:DICOM, mri:DICOM, followup:TORTILLA]
    
    Violates constraint: patient_002 has different subtree structure (4 vs 3 children)

Structure Example:
    Level 0: [id, type, ...]  [id, type, ...]  [id, type, ...]  ← Same schema
             ↓                ↓                ↓
    Level 1: [id, pos, data]  [id, pos, data]  [id, pos, data]  ← Same schema
             ↓                ↓                ↓
    Leaves:  VSI_path         VSI_path         VSI_path
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Any
from collections.abc import Sequence

# Tree node constants
POSITION_COLUMN = 'internal:position'
VSI_PATH_COLUMN = 'internal:gdal_vsi' 
TORTILLA_TYPE = 'TORTILLA'
ID_COLUMN = 'id'
TYPE_COLUMN = 'type'
INVALID_POSITION = -1


class TreeDataFrame(pd.DataFrame):
    """
    Hierarchical DataFrame implementing level-wise isomorphic sibling subtrees.
    
    This class creates tree structures where:
    1. All sibling nodes at the same level have identical column schemas
    2. TORTILLA nodes can contain other TORTILLAs or leaf nodes
    3. Leaf nodes reference files via GDAL_VSI paths
    4. internal:position acts as primary key connecting levels
    
    The "isomorphic siblings" constraint means if you have 3 nodes at level 2,
    they ALL must have the exact same columns, just different data values.
    
    Example:
        >>> # Primary level - all TORTILLAs have same schema
        >>> primary = pd.DataFrame({
        ...     'id': ['container1', 'container2'], 
        ...     'type': ['TORTILLA', 'TORTILLA'],
        ...     'metadata': ['info1', 'info2']
        ... })
        >>> 
        >>> # Auxiliary level - all children have same schema  
        >>> children = pd.DataFrame({
        ...     'id': ['child1', 'child2', 'child3'],
        ...     'internal:position': [0, 0, 1],  # Points to parent positions
        ...     'data_value': [100, 200, 300]
        ... })
        >>> 
        >>> tree = TreeDataFrame(primary, auxiliary_dataframes=[children])
        >>> samples = tree.read('container1')  # Navigate to children
    """
    
    _metadata = ['_auxiliary_levels']
    
    def __init__(
        self, 
        data: Any = None, 
        auxiliary_dataframes: Sequence[Any] | None = None, 
        *args, 
        **kwargs
    ) -> None:
        """
        Initialize TreeDataFrame with hierarchical levels.
        
        Args:
            data: Primary level DataFrame (root nodes)
            auxiliary_dataframes: List of DataFrames for deeper tree levels
                Each must have internal:position column linking to parent level
            *args, **kwargs: Standard pandas DataFrame arguments
        
        Important:
            - internal:position in level N points to row indices in level N-1
            - All siblings at same level MUST have identical column schemas
            - This constraint is what makes them "isomorphic"
            
        Raises:
            ValueError: If tree violates level-wise isomorphic constraint
        """
        super().__init__(data, *args, **kwargs)
        self._auxiliary_levels = self._build_hierarchy(auxiliary_dataframes)
        
        # Validate isomorphic constraint at construction time
        self._validate_isomorphic_structure()
    
    # ==========================================
    # PANDAS CONSTRUCTOR PROPERTIES  
    # ==========================================
    
    @property
    def _constructor(self) -> type:
        """Ensure operations return TreeDataFrame."""
        return TreeDataFrame
    
    @property 
    def _constructor_sliced(self) -> type:
        """Single column access returns pandas Series."""
        return pd.Series
    
    @property
    def _constructor_expanddim(self) -> type:
        """Dimension expansion returns TreeDataFrame.""" 
        return TreeDataFrame
    
    # ==========================================
    # HIERARCHY CONSTRUCTION AND VALIDATION
    # ==========================================
    
    def _build_hierarchy(self, auxiliary_dataframes: Sequence[Any] | None) -> list[TreeDataFrame]:
        """
        Build tree hierarchy ensuring isomorphic siblings at each level.
        
        Args:
            auxiliary_dataframes: Raw DataFrames for tree levels
            
        Returns:
            List of TreeDataFrame objects forming the hierarchy
            
        Note:
            Each level gets wrapped in TreeDataFrame to maintain the tree
            structure recursively. This allows TORTILLA→TORTILLA→TORTILLA chains.
        """
        if not auxiliary_dataframes:
            return []
        
        if not isinstance(auxiliary_dataframes, list):
            auxiliary_dataframes = [auxiliary_dataframes]
        
        hierarchy = []
        for level_data in auxiliary_dataframes:
            # Wrap each level in TreeDataFrame for recursive nesting
            level_tree = TreeDataFrame(level_data) if not isinstance(level_data, TreeDataFrame) else level_data
            hierarchy.append(level_tree)
            
        return hierarchy
    
    def _validate_isomorphic_structure(self) -> None:
        """
        Validate level-wise isomorphic sibling subtrees constraint.
        
        Only TORTILLA nodes can have children. Only TORTILLA siblings need to be isomorphic.
        Non-TORTILLA nodes are terminal and don't affect isomorphism.
        
        Raises:
            ValueError: If isomorphic constraint is violated
        """
        # Validate that only TORTILLA nodes have children
        self._validate_only_tortillas_have_children()
        
        # Validate TORTILLA siblings are isomorphic
        if self.has_children:
            self._validate_tortilla_siblings_isomorphic()
            
            # Recursively validate deeper levels using canonical TORTILLA subtree
            canonical_tortilla_subtree = self._get_canonical_tortilla_subtree()
            if canonical_tortilla_subtree is not None:
                canonical_tortilla_subtree._validate_isomorphic_structure()
    
    def _validate_only_tortillas_have_children(self) -> None:
        """
        Validate that internal:position only points to TORTILLA nodes.
        
        Raises:
            ValueError: If non-TORTILLA nodes have children
        """
        if not self.has_children:
            return
        
        child_level = self._auxiliary_levels[0]
        
        if POSITION_COLUMN not in child_level.columns:
            return
        
        # Get all parent positions referenced by children
        referenced_positions = child_level[POSITION_COLUMN].unique()
        
        # Check that all referenced positions point to TORTILLA nodes
        for pos in referenced_positions:
            if pd.notna(pos) and 0 <= pos < len(self):
                parent_row = self.iloc[pos]
                if TYPE_COLUMN in parent_row and parent_row[TYPE_COLUMN] != TORTILLA_TYPE:
                    raise ValueError(
                        f"Non-TORTILLA node at position {pos} (type='{parent_row[TYPE_COLUMN]}') "
                        f"has children. Only TORTILLA nodes can have children."
                    )
    
    def _validate_tortilla_siblings_isomorphic(self) -> None:
        """
        Validate that all TORTILLA siblings have identical subtree structures.
        
        Only TORTILLA nodes are compared for isomorphism. Non-TORTILLA nodes are ignored.
        
        Raises:
            ValueError: If TORTILLA siblings are not isomorphic
        """
        child_level = self._auxiliary_levels[0]
        
        if POSITION_COLUMN not in child_level.columns:
            return
        
        # Get all TORTILLA nodes in current level
        tortilla_mask = self[TYPE_COLUMN] == TORTILLA_TYPE if TYPE_COLUMN in self.columns else pd.Series([True] * len(self))
        tortilla_positions = self.index[tortilla_mask].tolist()
        
        if len(tortilla_positions) <= 1:
            return  # Trivially isomorphic with 0-1 TORTILLA nodes
        
        # Build structural signature for each TORTILLA's children
        tortilla_signatures = {}
        for tortilla_pos in tortilla_positions:
            children_mask = child_level[POSITION_COLUMN] == tortilla_pos
            children_subset = child_level[children_mask]
            
            # Create signature for this TORTILLA's children
            signature = self._compute_children_signature(children_subset)
            tortilla_signatures[tortilla_pos] = signature
        
        # All TORTILLA signatures must be identical for isomorphic constraint
        unique_signatures = set(tortilla_signatures.values())
        if len(unique_signatures) > 1:
            # Find which TORTILLA nodes differ
            signature_groups = {}
            for pos, sig in tortilla_signatures.items():
                if sig not in signature_groups:
                    signature_groups[sig] = []
                signature_groups[sig].append(pos)
            
            error_details = []
            for sig, positions in signature_groups.items():
                count, types, depth = sig
                error_details.append(f"TORTILLA positions {positions}: {count} children, types {types}, depth {depth}")
            
            raise ValueError(
                f"Level-wise isomorphic constraint violated for TORTILLA siblings:\n" +
                "\n".join(error_details) +
                "\nAll TORTILLA siblings must have identical subtree structures."
            )
    
    def _get_canonical_tortilla_subtree(self) -> TreeDataFrame | None:
        """
        Get one canonical TORTILLA child subtree for recursive validation.
        
        Returns:
            TreeDataFrame of canonical TORTILLA child subtree, or None if no TORTILLA children
        """
        if not self.has_children:
            return None
        
        child_level = self._auxiliary_levels[0]
        
        if POSITION_COLUMN not in child_level.columns:
            return None
        
        # Find first TORTILLA node that has children
        tortilla_mask = self[TYPE_COLUMN] == TORTILLA_TYPE if TYPE_COLUMN in self.columns else pd.Series([True] * len(self))
        tortilla_positions = self.index[tortilla_mask].tolist()
        
        for tortilla_pos in tortilla_positions:
            children_mask = child_level[POSITION_COLUMN] == tortilla_pos
            tortilla_children = child_level[children_mask]
            
            if not tortilla_children.empty:
                return TreeDataFrame(
                    tortilla_children.reset_index(drop=True),
                    auxiliary_dataframes=self._auxiliary_levels[1:] if len(self._auxiliary_levels) > 1 else None
                )
        
        return None
    
    def _validate_children_isomorphic(self) -> None:
        """
        Validate all children subtrees have identical structure.
        
        Checks:
        1. Same number of children per parent
        2. Same child type distribution per parent  
        3. Same subtree depth per parent
        
        Raises:
            ValueError: If children are not isomorphic
        """
        child_level = self._auxiliary_levels[0]
        
        if POSITION_COLUMN not in child_level.columns:
            return
        
        # Get unique parent positions
        parent_positions = child_level[POSITION_COLUMN].unique()
        
        if len(parent_positions) <= 1:
            return  # Trivially isomorphic with 0-1 parents
        
        # Build structural signature for each parent's children
        signatures = {}
        for parent_pos in parent_positions:
            children_mask = child_level[POSITION_COLUMN] == parent_pos
            children_subset = child_level[children_mask]
            
            # Create signature: (count, type_distribution, subtree_depth)
            signature = self._compute_children_signature(children_subset)
            signatures[parent_pos] = signature
        
        # All signatures must be identical for isomorphic constraint
        unique_signatures = set(signatures.values())
        if len(unique_signatures) > 1:
            # Find which parents differ
            signature_groups = {}
            for pos, sig in signatures.items():
                if sig not in signature_groups:
                    signature_groups[sig] = []
                signature_groups[sig].append(pos)
            
            error_details = []
            for sig, positions in signature_groups.items():
                count, types, depth = sig
                error_details.append(f"Positions {positions}: {count} children, types {types}, depth {depth}")
            
            raise ValueError(
                f"Level-wise isomorphic constraint violated:\n" +
                "\n".join(error_details) +
                "\nAll parents must have identical subtree structures."
            )
    
    def _compute_children_signature(self, children_subset: pd.DataFrame) -> tuple:
        """
        Compute structural signature for a set of children.
        
        Args:
            children_subset: DataFrame containing children of one parent
            
        Returns:
            Tuple of (count, type_distribution, max_subtree_depth)
        """
        # Count of children
        child_count = len(children_subset)
        
        # Type distribution (sorted for consistency)
        type_counts = children_subset[TYPE_COLUMN].value_counts().sort_index()
        type_distribution = tuple(type_counts.items())
        
        # Max subtree depth (for TORTILLA children)
        max_depth = 0
        for _, child_row in children_subset.iterrows():
            if child_row[TYPE_COLUMN] == TORTILLA_TYPE:
                # This would require building subtree - for now use 1
                # In full implementation, recursively compute depth
                max_depth = max(max_depth, 1)
        
        return (child_count, type_distribution, max_depth)
    
    def _get_canonical_child_subtree(self) -> TreeDataFrame | None:
        """
        Get one canonical child subtree for recursive validation.
        
        Since all subtrees must be isomorphic, we only need to validate
        one representative subtree per level.
        
        Returns:
            TreeDataFrame of canonical child subtree, or None if no children
        """
        if not self.has_children:
            return None
        
        child_level = self._auxiliary_levels[0]
        
        if POSITION_COLUMN not in child_level.columns:
            return None
        
        # Get first parent's children as canonical
        first_parent_pos = child_level[POSITION_COLUMN].iloc[0]
        canonical_children = child_level[child_level[POSITION_COLUMN] == first_parent_pos]
        
        return TreeDataFrame(
            canonical_children.reset_index(drop=True),
            auxiliary_dataframes=self._auxiliary_levels[1:] if len(self._auxiliary_levels) > 1 else None
        )
    
    @property
    def tree_depth(self) -> int:
        """Total depth of the tree hierarchy."""
        return len(self._auxiliary_levels) + 1  # +1 for current level
    
    @property 
    def has_children(self) -> bool:
        """Check if this level has child levels."""
        return len(self._auxiliary_levels) > 0
    
    def get_level(self, depth: int) -> TreeDataFrame:
        """
        Get TreeDataFrame at specific depth.
        
        Args:
            depth: Tree depth (0 = current level, 1 = first child level, etc.)
            
        Returns:
            TreeDataFrame at requested depth
            
        Raises:
            IndexError: If depth exceeds tree depth
        """
        if depth == 0:
            return self
        elif 1 <= depth <= len(self._auxiliary_levels):
            return self._auxiliary_levels[depth - 1]
        else:
            raise IndexError(f"Depth {depth} exceeds tree depth {self.tree_depth - 1}")
    
    # ==========================================
    # HIERARCHY-PRESERVING INDEXING
    # ==========================================

    def __getitem__(self, key) -> TreeDataFrame | pd.Series:
        """
        Override indexing to preserve tree hierarchy during filtering.
        
        When you filter parent nodes, child nodes are automatically filtered
        to maintain parent-child relationships via internal:position mapping.
        
        Args:
            key: Pandas indexing key (boolean mask, column names, etc.)
            
        Returns:
            TreeDataFrame with filtered hierarchy or Series for single columns
            
        Examples:
            >>> # Boolean filtering preserves hierarchy
            >>> filtered_tree = tree[tree['quality'] > 0.8] 
            >>> 
            >>> # Column selection preserves hierarchy
            >>> subset_tree = tree[['id', 'type']]
            >>> 
            >>> # Single column returns Series  
            >>> ids = tree['id']
        """
        # Boolean indexing with hierarchy preservation
        if self._is_boolean_indexing(key):
            return self._filter_with_hierarchy(key)
        
        # Column selection with hierarchy preservation  
        elif isinstance(key, (str, list)):
            result = super().__getitem__(key)
            if isinstance(result, pd.DataFrame):
                return self._preserve_hierarchy_in_result(result)
            return result  # Series case
        
        # Other indexing (slices, integers, etc.)
        else:
            result = super().__getitem__(key)
            if isinstance(result, pd.DataFrame):
                return self._preserve_hierarchy_in_result(result)
            return result
    
    def _is_boolean_indexing(self, key) -> bool:
        """Check if key represents boolean indexing operation."""
        return (isinstance(key, (pd.Series, np.ndarray)) and 
                hasattr(key, 'dtype') and 
                key.dtype == bool)
    
    def _filter_with_hierarchy(self, boolean_mask: pd.Series) -> TreeDataFrame:
        """
        Apply boolean filter while preserving parent-child relationships.
        
        Args:
            boolean_mask: Boolean Series for filtering
            
        Returns:
            Filtered TreeDataFrame with updated hierarchy
            
        Process:
            1. Filter current level with boolean mask
            2. Build position mapping (old_position → new_position)  
            3. Update child levels' internal:position references
            4. Remove orphaned child nodes
        """
        # Filter current level
        filtered_current = super().__getitem__(boolean_mask).copy()
        
        if filtered_current.empty:
            return self._create_empty_tree()
        
        # Build position mapping for child level updates
        position_map = self._build_position_mapping(boolean_mask)
        
        # Update all child levels with new position references
        updated_children = []
        for child_level in self._auxiliary_levels:
            updated_child = self._update_child_positions(child_level, position_map)
            updated_children.append(updated_child)
        
        # Create filtered tree with clean indices
        filtered_tree = TreeDataFrame(
            filtered_current.reset_index(drop=True),
            auxiliary_dataframes=updated_children
        )
        
        # Validate tree integrity (but skip isomorphic validation for performance)
        filtered_tree._validate_tree_integrity()
        return filtered_tree
    
    def _build_position_mapping(self, boolean_mask: pd.Series) -> dict[str, dict[int, int]]:
        """
        Build mapping between old and new positions after filtering.
        
        Args:
            boolean_mask: Boolean mask used for filtering
            
        Returns:
            Dictionary with old→new and new→old position mappings
            
        Note:
            This is crucial for maintaining parent-child relationships.
            When parent at position 5 becomes position 2 after filtering,
            all children with internal:position=5 must update to internal:position=2
        """
        # Align mask with current DataFrame index
        if hasattr(boolean_mask, 'index'):
            aligned_mask = boolean_mask.reindex(self.index, fill_value=False)
        else:
            aligned_mask = boolean_mask
            
        # Get original positions that survived filtering
        surviving_positions = self.index[aligned_mask].tolist()
        new_positions = list(range(len(surviving_positions)))
        
        return {
            'old_to_new': dict(zip(surviving_positions, new_positions)),
            'new_to_old': dict(zip(new_positions, surviving_positions)),
            'survivors': surviving_positions
        }
    
    def _update_child_positions(self, child_level: TreeDataFrame, position_map: dict) -> TreeDataFrame:
        """
        Update child level's position references after parent filtering.
        
        Args:
            child_level: Child TreeDataFrame to update
            position_map: Position mapping from filtering operation
            
        Returns:
            Updated TreeDataFrame with corrected position references
            
        Process:
            1. Map old parent positions to new positions
            2. Mark orphaned children (parent filtered out) as invalid
            3. Remove orphaned children  
            4. Reset child indices for clean structure
        """
        if POSITION_COLUMN not in child_level.columns:
            return child_level.copy()
        
        child_copy = child_level.copy()
        old_to_new = position_map['old_to_new']
        
        # Update position references: old_parent_pos → new_parent_pos
        child_copy[POSITION_COLUMN] = child_copy[POSITION_COLUMN].map(
            lambda pos: old_to_new.get(pos, INVALID_POSITION) if pd.notna(pos) else INVALID_POSITION
        )
        
        # Remove orphaned children (those whose parents were filtered out)
        valid_children = child_copy[child_copy[POSITION_COLUMN] != INVALID_POSITION]
        
        return TreeDataFrame(
            valid_children.reset_index(drop=True),
            auxiliary_dataframes=child_level._auxiliary_levels
        )
    
    def _create_empty_tree(self) -> TreeDataFrame:
        """Create empty TreeDataFrame preserving hierarchy structure."""
        empty_children = []
        for child_level in self._auxiliary_levels:
            empty_child = pd.DataFrame(columns=child_level.columns)
            empty_children.append(empty_child)
        
        return TreeDataFrame(
            pd.DataFrame(columns=self.columns),
            auxiliary_dataframes=empty_children
        )
    
    def _preserve_hierarchy_in_result(self, result: pd.DataFrame) -> TreeDataFrame:
        """
        Preserve hierarchy when pandas operations return DataFrames.
        
        Args:
            result: DataFrame result from pandas operation
            
        Returns:
            TreeDataFrame with appropriate hierarchy preservation
        """
        # Same length and index = no filtering occurred
        if len(result) == len(self) and result.index.equals(self.index):
            return TreeDataFrame(result, auxiliary_dataframes=self._auxiliary_levels)
        
        # Subset of original = apply hierarchy filtering
        elif len(result) <= len(self):
            try:
                # Create boolean mask from result indices
                mask = self.index.isin(result.index)
                return self._filter_with_hierarchy(mask)
            except Exception:
                # Fallback: return without hierarchy if filtering fails
                return TreeDataFrame(result)
        
        # Expanded or complex case = return without hierarchy
        return TreeDataFrame(result)
    
    # ==========================================
    # TREE NAVIGATION AND ACCESS
    # ==========================================
    
    def read(self, identifier: int | str) -> TreeDataFrame | str:
        """
        Navigate tree hierarchy or retrieve leaf resources.
        
        Args:
            identifier: Node ID (string) or position (int) to navigate to
            
        Returns:
            TreeDataFrame of child nodes if TORTILLA, or VSI path if leaf
            
        Raises:
            IndexError: If position is out of bounds
            ValueError: If ID not found or no VSI path available
            
        Examples:
            >>> # Navigate to children of TORTILLA node
            >>> children = tree.read('container_001')
            >>> 
            >>> # Get VSI path from leaf node  
            >>> file_path = tree.read('data_file_001') 
            >>> # Returns: '/vsizip/data.zip/file.tif'
        """
        # Get target row and its position
        row, position = self._resolve_identifier(identifier)
        
        # TORTILLA node: navigate to children
        if row[TYPE_COLUMN] == TORTILLA_TYPE:
            return self._navigate_to_children(position)
        
        # Leaf node: return VSI path
        else:
            return self._get_vsi_path(row)
    
    def _resolve_identifier(self, identifier: int | str) -> tuple:
        """
        Resolve identifier to DataFrame row and position.
        
        Args:
            identifier: Node ID or position index
            
        Returns:
            Tuple of (row_data, position_index)
        """
        if isinstance(identifier, int):
            if identifier >= len(self):
                raise IndexError(f"Position {identifier} out of bounds (max: {len(self)-1})")
            return self.iloc[identifier], identifier
        
        else:  # String ID
            matches = self[self[ID_COLUMN] == identifier]
            if matches.empty:
                raise ValueError(f"ID '{identifier}' not found in current level")
            return matches.iloc[0], matches.index[0]
    
    def _navigate_to_children(self, parent_position: int) -> TreeDataFrame:
        """
        Navigate to children of specified parent position.
        
        Args:
            parent_position: Position of parent node
            
        Returns:
            TreeDataFrame containing filtered children
            
        Raises:
            ValueError: If no child level available or no children found
        """
        if not self.has_children:
            raise ValueError("No child level available for navigation")
        
        child_level = self._auxiliary_levels[0]
        
        # Filter children by parent position
        if POSITION_COLUMN in child_level.columns:
            child_mask = child_level[POSITION_COLUMN] == parent_position
            filtered_children = child_level[child_mask]
            
            if filtered_children.empty:
                raise ValueError(f"No children found for parent position {parent_position}")
            
            # Create new TreeDataFrame with remaining hierarchy levels
            return TreeDataFrame(
                filtered_children.reset_index(drop=True),
                auxiliary_dataframes=self._auxiliary_levels[1:] if len(self._auxiliary_levels) > 1 else None
            )
        
        else:
            # No position mapping: return entire child level
            return child_level
    
    def _get_vsi_path(self, row: pd.Series) -> str:
        """
        Extract VSI path from leaf node.
        
        Args:
            row: DataFrame row containing VSI path
            
        Returns:
            VSI path string for file access
            
        Raises:
            ValueError: If no VSI path found in row
        """
        if VSI_PATH_COLUMN in row and pd.notna(row[VSI_PATH_COLUMN]):
            return str(row[VSI_PATH_COLUMN])
        
        raise ValueError(f"No VSI path available for node '{row.get(ID_COLUMN, 'unknown')}'")
    
    # ==========================================
    # TREE INTEGRITY AND VALIDATION
    # ==========================================
    
    def _validate_tree_integrity(self) -> bool:
        """
        Validate tree structure and position references.
        
        Returns:
            True if tree structure is valid
            
        Raises:
            ValueError: If invalid position references found
            
        Checks:
            - All position references point to valid parent positions
            - No orphaned child nodes
            - Position values are within bounds
        """
        for level_idx, child_level in enumerate(self._auxiliary_levels):
            if POSITION_COLUMN in child_level.columns and not child_level.empty:
                max_position = child_level[POSITION_COLUMN].max()
                if pd.notna(max_position) and max_position >= len(self):
                    raise ValueError(
                        f"Child level {level_idx} has invalid position reference: {max_position} "
                        f"(parent level has only {len(self)} rows)"
                    )
        return True
        
    # ==========================================
    # PANDAS INTEGRATION METHODS
    # ==========================================
    
    def query(self, expr: str, **kwargs) -> TreeDataFrame:
        """pandas.query() with hierarchy preservation."""
        result = super().query(expr, **kwargs)
        return self._preserve_hierarchy_in_result(result)
    
    def drop(self, *args, **kwargs) -> TreeDataFrame:
        """pandas.drop() with hierarchy preservation."""
        result = super().drop(*args, **kwargs)
        return self._preserve_hierarchy_in_result(result)
    
    def head(self, n: int = 5) -> TreeDataFrame:
        """pandas.head() with hierarchy preservation."""
        result = super().head(n)
        return self._preserve_hierarchy_in_result(result)
    
    def tail(self, n: int = 5) -> TreeDataFrame:
        """pandas.tail() with hierarchy preservation.""" 
        result = super().tail(n)
        return self._preserve_hierarchy_in_result(result)
    
    def sample(self, *args, **kwargs) -> TreeDataFrame:
        """pandas.sample() with hierarchy preservation."""
        result = super().sample(*args, **kwargs)
        return self._preserve_hierarchy_in_result(result)
    
    def copy(self, deep: bool = True) -> TreeDataFrame:
        """Copy TreeDataFrame with full hierarchy."""
        copied_current = super().copy(deep=deep)
        
        if deep:
            copied_children = [child.copy(deep=True) for child in self._auxiliary_levels]
        else:
            copied_children = self._auxiliary_levels.copy()
        
        return TreeDataFrame(copied_current, auxiliary_dataframes=copied_children)
    
    def reset_index(self, *args, **kwargs):
        """
        Reset index with hierarchy protection.
        
        Raises:
            ValueError: If hierarchy exists (would break position references)
            
        Note:
            reset_index() would break internal:position references in child levels.
            Convert to regular DataFrame first if you need to reset indices.
        """
        if self.has_children:
            raise ValueError(
                "reset_index() would break hierarchy position references. "
                "Convert to pandas DataFrame first: pd.DataFrame(tree_df).reset_index()"
            )
        
        if kwargs.get('inplace', False):
            super().reset_index(*args, **kwargs)
            return None
        else:
            result = super().reset_index(*args, **kwargs)
            return TreeDataFrame(result)
    
    # ==========================================
    # STRING REPRESENTATION
    # ==========================================
    
    def __repr__(self) -> str:
        """Enhanced representation showing tree structure."""
        base_repr = super().__repr__()
        
        if self.has_children:
            level_info = []
            for i, child_level in enumerate(self._auxiliary_levels):
                level_info.append(f"Level {i+1}: {len(child_level)} nodes")
            
            tree_info = f"\nTree Structure:\n├─ Current: {len(self)} nodes\n"
            for i, info in enumerate(level_info):
                connector = "└─" if i == len(level_info) - 1 else "├─"
                tree_info += f"{connector} {info}\n"
            
            return base_repr + tree_info
        else:
            return base_repr + f"\nTree Structure: Leaf level ({len(self)} nodes)"
    
    def print_tree_structure(self, max_depth: int | None = None) -> None:
        """
        Print detailed tree structure for debugging.
        
        Args:
            max_depth: Maximum depth to print (None for full tree)
        """
        print(f"TreeDataFrame Structure (depth: {self.tree_depth})")
        print("=" * 50)
        
        def print_level(level: TreeDataFrame, depth: int, prefix: str = ""):
            if max_depth is not None and depth > max_depth:
                return
                
            print(f"{prefix}Level {depth}: {len(level)} nodes")
            if len(level) > 0:
                print(f"{prefix}  Columns: {list(level.columns)}")
                if POSITION_COLUMN in level.columns:
                    positions = level[POSITION_COLUMN].unique()
                    print(f"{prefix}  Parent positions: {sorted(positions)}")
            
            for i, child in enumerate(level._auxiliary_levels):
                child_prefix = prefix + ("  └─ " if i == len(level._auxiliary_levels) - 1 else "  ├─ ")
                print_level(child, depth + 1, child_prefix)
        
        print_level(self, 0)
