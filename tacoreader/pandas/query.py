import pandas as pd
from tacoreader.pandas.multidataframe import MultiDataFrame


def query(
    multi_df: MultiDataFrame,
    level: int,
    id: str,
    columns: str | list[str] | dict[str, list[str]] | None = None
) -> pd.DataFrame:
    """
    Query files with hierarchical JOINs from MultiDataFrame.
    
    Args:
        multi_df: MultiDataFrame with hierarchical data
        level: Target level (>= 1) where to find matches
        id: ID to match at target level  
        columns: JOIN/Column selection:
            - None (default): only internal:* from target level (excludes internal:position)
            - "left": LEFT JOIN - only PARENT levels (excludes target)
            - "right": RIGHT JOIN - only TARGET level  
            - "full": FULL JOIN - from root TO target (includes target)
            - list[str]: specific field selections with conflict validation
            - dict[str, list[str]]: field specifications by level
    
    Returns:
        DataFrame with hierarchical data and unique hierarchical IDs
        
    Raises:
        ValueError: If level invalid, ID not found, or field conflicts detected
    """
    
    # Validate level
    if level < 1:
        raise ValueError("level must be >= 1")
    if level > multi_df.get_auxiliary_count():
        raise ValueError(f"level {level} not available, max: {multi_df.get_auxiliary_count()}")
    
    # Get target matches
    target_df = multi_df.get_auxiliary_df(level - 1)
    target_matches = target_df[target_df['id'] == id].copy().reset_index(drop=True)
    
    if target_matches.empty:
        # Provide helpful error with available IDs
        available_ids = target_df['id'].unique().tolist()
        if len(available_ids) <= 10:
            available_str = f"Available IDs: {available_ids}"
        else:
            available_str = f"Available IDs: {available_ids[:10]}... ({len(available_ids)} total)"
        
        raise ValueError(f"ID '{id}' not found in level {level}. {available_str}")
    
    # Route to appropriate execution strategy
    if columns is None:
        return _execute_default(multi_df, target_matches, level)
    elif isinstance(columns, str):
        return _execute_join(multi_df, target_matches, level, columns.lower())
    elif isinstance(columns, list):
        return _execute_field_list(multi_df, target_matches, level, columns)
    elif isinstance(columns, dict):
        return _execute_field_dict(multi_df, target_matches, level, columns)
    else:
        raise ValueError(f"Invalid columns parameter type: {type(columns)}")


def _execute_default(
    multi_df: MultiDataFrame, 
    target_matches: pd.DataFrame, 
    level: int
) -> pd.DataFrame:
    """Default: only internal:* from target level (excludes internal:position)."""
    
    hierarchical_ids = _build_hierarchical_ids(multi_df, target_matches, level)
    
    result = pd.DataFrame({
        'id': hierarchical_ids,
        'type': target_matches['type'].tolist()
    })
    
    # Add internal:* columns except internal:position
    for col in target_matches.columns:
        if col.startswith('internal:') and col != 'internal:position':
            result[col] = target_matches[col].tolist()
    
    return result


def _execute_join(
    multi_df: MultiDataFrame, 
    target_matches: pd.DataFrame, 
    level: int, 
    join_type: str
) -> pd.DataFrame:
    """Execute LEFT, RIGHT, or FULL JOIN."""
    
    if join_type not in ["left", "right", "full"]:
        raise ValueError(f"Invalid JOIN type: '{join_type}'. Use 'left', 'right', or 'full'")
    
    hierarchical_ids = _build_hierarchical_ids(multi_df, target_matches, level)
    
    result = pd.DataFrame({
        'id': hierarchical_ids,
        'type': target_matches['type'].tolist()
    })
    
    # Determine levels to include
    if join_type == "left":
        levels_to_include = list(range(level))  # 0 to level-1
    elif join_type == "right":
        levels_to_include = [level]  # target only
    else:  # full
        levels_to_include = list(range(level + 1))  # 0 to level
    
    # Get multi-level data
    all_level_data = _get_multi_level_data(multi_df, target_matches, levels_to_include)
    
    # Merge columns (target level wins conflicts)
    added_columns = {'id', 'type'}
    
    for lv in sorted(levels_to_include, reverse=True):
        level_data = all_level_data[lv]
        for col in level_data.columns:
            if col != 'internal:position' and col not in added_columns:
                result[col] = level_data[col].tolist()
                added_columns.add(col)
    
    return result


def _execute_field_list(
    multi_df: MultiDataFrame, 
    target_matches: pd.DataFrame, 
    level: int, 
    requested_fields: list[str]
) -> pd.DataFrame:
    """Execute field list with conflict detection."""
    
    # Build column availability map
    column_map = _build_column_map(multi_df, level)
    
    # Check for field conflicts across levels
    field_conflicts = {}
    missing_fields = []
    
    for field in requested_fields:
        locations = [lv for lv, cols in column_map.items() if field in cols]
        
        if not locations:
            missing_fields.append(field)
        elif len(locations) > 1:
            field_conflicts[field] = locations
    
    # Validate fields
    if missing_fields:
        raise ValueError(f"Fields not found: {missing_fields}")
    
    if field_conflicts:
        conflicts_desc = [f"'{field}' in levels {levels}" for field, levels in field_conflicts.items()]
        raise ValueError(f"Field conflicts detected: {'; '.join(conflicts_desc)}. Use dict specification.")
    
    # Get all data and build result
    all_data = _get_multi_level_data(multi_df, target_matches, list(range(level + 1)))
    hierarchical_ids = _build_hierarchical_ids(multi_df, target_matches, level)
    
    result = pd.DataFrame({
        'id': hierarchical_ids,
        'type': target_matches['type'].tolist()
    })
    
    # Add requested fields
    for field in requested_fields:
        for lv, level_data in all_data.items():
            if field in level_data.columns:
                result[field] = level_data[field].tolist()
                break
    
    return result


def _execute_field_dict(
    multi_df: MultiDataFrame, 
    target_matches: pd.DataFrame, 
    level: int, 
    level_spec: dict[str, list[str]]
) -> pd.DataFrame:
    """Execute field specification by level."""
    
    # Parse and validate level specifications
    requested_levels = {}
    for level_key, fields in level_spec.items():
        if not level_key.startswith('level'):
            raise ValueError(f"Invalid level key: '{level_key}'. Use 'level0', 'level1', etc.")
        
        lv = int(level_key.replace('level', ''))
        if lv > level:
            raise ValueError(f"Level {lv} not available for target level {level}")
        
        requested_levels[lv] = fields
    
    # Get data for requested levels
    all_data = _get_multi_level_data(multi_df, target_matches, list(requested_levels.keys()))
    
    # Validate fields exist in their respective levels
    for lv, fields in requested_levels.items():
        if lv not in all_data:
            continue
        level_data = all_data[lv]
        missing = [f for f in fields if f not in level_data.columns]
        if missing:
            raise ValueError(f"Fields {missing} not found in level {lv}")
    
    # Build result
    hierarchical_ids = _build_hierarchical_ids(multi_df, target_matches, level)
    result = pd.DataFrame({
        'id': hierarchical_ids,
        'type': target_matches['type'].tolist()
    })
    
    # Add requested fields by level
    for lv, fields in requested_levels.items():
        if lv in all_data:
            level_data = all_data[lv]
            for field in fields:
                if field in level_data.columns:
                    result[field] = level_data[field].tolist()
    
    return result


def _get_multi_level_data(
    multi_df: MultiDataFrame, 
    target_matches: pd.DataFrame, 
    levels: list[int]
) -> dict[int, pd.DataFrame]:
    """Get data from multiple levels using position traversal."""
    
    result = {}
    current_positions = target_matches['internal:position'].tolist()
    
    # Add target level if in requested levels
    max_level = max(levels) if levels else 0
    if max_level in levels:
        result[max_level] = target_matches
    
    # Traverse up hierarchy for parent levels
    for lv in sorted([l for l in levels if l < max_level], reverse=True):
        # Get parent DataFrame
        if lv == 0:
            parent_df = multi_df
        else:
            parent_df = multi_df.get_auxiliary_df(lv - 1)
        
        # Build parent data using position lookups
        parent_rows = []
        next_positions = []
        
        for pos in current_positions:
            if pos is not None and 0 <= pos < len(parent_df):
                parent_row = parent_df.iloc[pos].to_dict()
                parent_rows.append(parent_row)
                next_positions.append(parent_row.get('internal:position'))
            else:
                # Missing parent - create null row
                null_row = {col: None for col in parent_df.columns}
                parent_rows.append(null_row)
                next_positions.append(None)
        
        result[lv] = pd.DataFrame(parent_rows)
        current_positions = next_positions
    
    return result


def _build_hierarchical_ids(
    multi_df: MultiDataFrame, 
    target_matches: pd.DataFrame, 
    target_level: int
) -> list[str]:
    """Build hierarchical IDs by traversing up the hierarchy."""
    
    # Collect IDs from all levels
    level_ids = {}
    current_positions = target_matches['internal:position'].tolist()
    
    # Start with target level IDs
    level_ids[target_level] = target_matches['id'].tolist()
    
    # Traverse up hierarchy to collect parent IDs
    for lv in range(target_level - 1, -1, -1):
        if lv == 0:
            parent_df = multi_df
        else:
            parent_df = multi_df.get_auxiliary_df(lv - 1)
        
        parent_ids = []
        next_positions = []
        
        for pos in current_positions:
            if pos is not None and 0 <= pos < len(parent_df):
                parent_row = parent_df.iloc[pos]
                parent_ids.append(str(parent_row['id']))
                next_positions.append(parent_row.get('internal:position'))
            else:
                parent_ids.append('')
                next_positions.append(None)
        
        level_ids[lv] = parent_ids
        current_positions = next_positions
    
    # Build hierarchical ID strings
    hierarchical_ids = []
    for i in range(len(target_matches)):
        id_parts = []
        for lv in range(target_level + 1):
            if lv in level_ids:
                id_parts.append(level_ids[lv][i])
            else:
                id_parts.append('')
        hierarchical_ids.append(':'.join(id_parts))
    
    return hierarchical_ids


def _build_column_map(multi_df: MultiDataFrame, max_level: int) -> dict[int, list[str]]:
    """Build map of available columns by level."""
    
    column_map = {0: list(multi_df.columns)}
    
    for lv in range(1, max_level + 1):
        aux_df = multi_df.get_auxiliary_df(lv - 1)
        column_map[lv] = list(aux_df.columns)
    
    return column_map