"""DuckDB view construction for concatenated datasets.

Uses strategy pattern to handle format-specific view creation (ZIP/FOLDER/TacoCat).
"""

from typing import TYPE_CHECKING

import duckdb
import pyarrow as pa
import pyarrow.compute as pc
from tqdm import tqdm

from tacoreader._constants import (
    COLUMN_ID,
    COLUMN_TYPE,
    LEVEL_TABLE_SUFFIX,
    LEVEL_VIEW_PREFIX,
    METADATA_CURRENT_ID,
    METADATA_GDAL_VSI,
    METADATA_OFFSET,
    METADATA_PARENT_ID,
    METADATA_RELATIVE_PATH,
    METADATA_SIZE,
    METADATA_SOURCE_FILE,
    METADATA_SOURCE_PATH,
    PADDING_PREFIX,
    SAMPLE_TYPE_FILE,
    SAMPLE_TYPE_FOLDER,
    UNION_VIEW_SUFFIX,
)
from tacoreader._exceptions import TacoFormatError
from tacoreader._logging import get_logger

if TYPE_CHECKING:
    from tacoreader.dataset import TacoDataset

logger = get_logger(__name__)


class ViewBuilder:
    """Builds DuckDB views for concatenated datasets."""

    def __init__(
        self,
        db: duckdb.DuckDBPyConnection,
        datasets: list["TacoDataset"],
        all_levels: set[str],
        target_columns_by_level: dict[str, list[str]],
        show_progress: bool = False,
    ):
        """Initialize view builder.

        Args:
            db: DuckDB connection for creating views
            datasets: List of datasets to concatenate
            all_levels: Set of all available level views
            target_columns_by_level: Dict mapping level_key -> final column list
            show_progress: Whether to show tqdm progress bar
        """
        self.db = db
        self.datasets = datasets
        self.all_levels = all_levels
        self.target_columns_by_level = target_columns_by_level
        self.format_type = datasets[0]._format
        self.show_progress = show_progress
        # Track valid current_ids (int64) per dataset per level for hierarchical filtering
        self._valid_ids: dict[int, dict[str, set]] = {i: {} for i in range(len(datasets))}

    def build_all_views(self) -> None:
        """Build all views: UNION ALL + format-specific final views."""
        self._create_union_views()
        self._create_format_views()
        logger.debug("Created all DuckDB views")

    def _create_union_views(self) -> None:
        """Create UNION ALL views for each level.

        Level0: Reads from ds._view_name (respects .sql() filters)
        Level1+: Reads from raw table, filtered by valid parent_ids from previous level
        """
        for level_key in sorted(self.all_levels):
            if level_key not in self.target_columns_by_level:
                continue

            logger.debug(f"  Consolidating {level_key}...")

            target_cols = self.target_columns_by_level[level_key]
            union_parts = []

            # Progress bar for datasets (only if show_progress and multiple datasets)
            dataset_iter = enumerate(self.datasets)
            if self.show_progress:
                dataset_iter = tqdm(
                    list(dataset_iter),
                    desc=f"  {level_key}",
                    leave=False,
                    unit="ds",
                )

            for ds_idx, ds in dataset_iter:
                max_depth = ds.pit_schema.max_depth()
                available_levels = [f"{LEVEL_VIEW_PREFIX}{i}" for i in range(max_depth + 1)]

                if level_key not in available_levels:
                    continue

                # Get arrow table respecting filters
                arrow_table = self._get_level_table(ds, ds_idx, level_key)

                if arrow_table.num_rows == 0:
                    continue  # Skip empty tables

                # Track current_ids for filtering next level
                self._track_valid_ids(ds_idx, level_key, arrow_table)

                # Register in new connection with unique name
                table_name = f"ds{ds_idx}_{level_key}{LEVEL_TABLE_SUFFIX}"
                self.db.register(table_name, arrow_table)

                # Build SELECT with aligned columns + internal:source_path
                source_path = ds._vsi_base_path.rstrip("/")
                current_cols = set(arrow_table.column_names)

                select_parts = self._build_select_parts(target_cols, current_cols)
                select_parts.append(f"'{source_path}' AS \"{METADATA_SOURCE_PATH}\"")

                union_parts.append(f"SELECT {', '.join(select_parts)} FROM {table_name}")

            if not union_parts:
                continue

            # Create consolidated view with UNION ALL
            union_query = " UNION ALL ".join(union_parts)
            self.db.execute(f"CREATE VIEW {level_key}{UNION_VIEW_SUFFIX} AS {union_query}")

            logger.debug(f"    {level_key}: {len(union_parts)} dataset(s), {len(target_cols)} columns")

    def _get_level_table(self, ds: "TacoDataset", ds_idx: int, level_key: str) -> pa.Table:
        """Get arrow table for level, respecting filters.

        Level0: Read from ds._view_name (filtered view)
        Level1+: Read from raw table, filter by valid parent_ids if tracking available
        """
        if level_key == f"{LEVEL_VIEW_PREFIX}0":
            # Level0: Read from filtered view (respects .sql() filters)
            return ds._duckdb.execute(f"SELECT * FROM {ds._view_name}").fetch_arrow_table()
        else:
            # Level1+: Filter by parent_ids from previous level
            level_num = int(level_key.replace(LEVEL_VIEW_PREFIX, ""))
            prev_level = f"{LEVEL_VIEW_PREFIX}{level_num - 1}"
            parent_ids = self._valid_ids[ds_idx].get(prev_level)

            # If no tracking available, read full table (backward compat)
            if parent_ids is None:
                return ds._duckdb.execute(f"SELECT * FROM {level_key}{LEVEL_TABLE_SUFFIX}").fetch_arrow_table()

            return self._get_filtered_children(ds, level_key, parent_ids)

    def _get_filtered_children(self, ds: "TacoDataset", level_key: str, parent_ids: set) -> pa.Table:
        """Get level table filtered to children of valid parents."""
        table = ds._duckdb.execute(f"SELECT * FROM {level_key}{LEVEL_TABLE_SUFFIX}").fetch_arrow_table()

        if not parent_ids:
            # Return empty table with same schema
            return table.slice(0, 0)

        # Filter rows where parent_id is in valid parent_ids (both int64)
        mask = pc.is_in(table.column(METADATA_PARENT_ID), value_set=pa.array(list(parent_ids), type=pa.int64()))
        return table.filter(mask)

    def _track_valid_ids(self, ds_idx: int, level_key: str, table: pa.Table) -> None:
        """Track valid current_ids (int64) for filtering next level's parent_ids."""
        if METADATA_CURRENT_ID in table.column_names:
            self._valid_ids[ds_idx][level_key] = set(table.column(METADATA_CURRENT_ID).to_pylist())

    @staticmethod
    def _build_select_parts(target_cols: list[str], current_cols: set[str]) -> list[str]:
        """Build SELECT column list with NULL filling for missing columns."""
        select_parts = []
        for col in sorted(target_cols):
            if col in current_cols:
                escaped_col = f'"{col}"' if ":" in col or " " in col else col
                select_parts.append(escaped_col)
            else:
                select_parts.append(f'NULL AS "{col}"')
        return select_parts

    def _create_format_views(self) -> None:
        """Create format-specific views with GDAL VSI paths."""
        if self.format_type == "zip":
            self._create_zip_views()
        elif self.format_type == "folder":
            self._create_folder_views()
        elif self.format_type == "tacocat":
            self._create_tacocat_views()
        else:
            raise TacoFormatError(f"Unknown format: {self.format_type}")

    def _create_zip_views(self) -> None:
        """Create final views for ZIP format."""
        for level_key in sorted(self.all_levels):
            if not self._view_exists(level_key):
                continue

            self.db.execute(f"""
                CREATE VIEW {level_key} AS
                SELECT *,
                  '/vsisubfile/' || "{METADATA_OFFSET}" || '_' ||
                  "{METADATA_SIZE}" || ',' || "{METADATA_SOURCE_PATH}" as "{METADATA_GDAL_VSI}"
                FROM {level_key}{UNION_VIEW_SUFFIX}
                WHERE {COLUMN_ID} NOT LIKE '{PADDING_PREFIX}%'
            """)

    def _create_folder_views(self) -> None:
        """Create final views for FOLDER format."""
        for level_key in sorted(self.all_levels):
            if not self._view_exists(level_key):
                continue

            if level_key == f"{LEVEL_VIEW_PREFIX}0":
                self.db.execute(f"""
                    CREATE VIEW {level_key} AS
                    SELECT *,
                      CASE
                        WHEN {COLUMN_TYPE} = '{SAMPLE_TYPE_FOLDER}' THEN "{METADATA_SOURCE_PATH}" || '/DATA/' || {COLUMN_ID} || '/__meta__'
                        WHEN {COLUMN_TYPE} = '{SAMPLE_TYPE_FILE}' THEN "{METADATA_SOURCE_PATH}" || '/DATA/' || {COLUMN_ID}
                        ELSE NULL
                      END as "{METADATA_GDAL_VSI}"
                    FROM {level_key}{UNION_VIEW_SUFFIX}
                    WHERE {COLUMN_ID} NOT LIKE '{PADDING_PREFIX}%'
                """)
            else:
                self.db.execute(f"""
                    CREATE VIEW {level_key} AS
                    SELECT *,
                      CASE
                        WHEN {COLUMN_TYPE} = '{SAMPLE_TYPE_FOLDER}' THEN "{METADATA_SOURCE_PATH}" || '/DATA/' || "{METADATA_RELATIVE_PATH}" || '__meta__'
                        WHEN {COLUMN_TYPE} = '{SAMPLE_TYPE_FILE}' THEN "{METADATA_SOURCE_PATH}" || '/DATA/' || "{METADATA_RELATIVE_PATH}"
                        ELSE NULL
                      END as "{METADATA_GDAL_VSI}"
                    FROM {level_key}{UNION_VIEW_SUFFIX}
                    WHERE {COLUMN_ID} NOT LIKE '{PADDING_PREFIX}%'
                """)

    def _create_tacocat_views(self) -> None:
        """Create final views for TacoCat format."""
        for level_key in sorted(self.all_levels):
            if not self._view_exists(level_key):
                continue

            self.db.execute(f"""
                CREATE VIEW {level_key} AS
                SELECT *,
                  '/vsisubfile/' || "{METADATA_OFFSET}" || '_' ||
                  "{METADATA_SIZE}" || ',' || "{METADATA_SOURCE_PATH}" || '/' || "{METADATA_SOURCE_FILE}" as "{METADATA_GDAL_VSI}"
                FROM {level_key}{UNION_VIEW_SUFFIX}
                WHERE {COLUMN_ID} NOT LIKE '{PADDING_PREFIX}%'
            """)

    def _view_exists(self, level_key: str) -> bool:
        """Check if union view exists for given level."""
        return bool(
            self.db.execute(
                f"SELECT 1 FROM information_schema.tables WHERE table_name = '{level_key}{UNION_VIEW_SUFFIX}'"
            ).fetchone()
        )