"""DuckDB view construction for concatenated datasets.

Uses strategy pattern to handle format-specific view creation (ZIP/FOLDER/TacoCat).
"""

from typing import TYPE_CHECKING

import duckdb

from tacoreader._constants import (
    COLUMN_ID,
    COLUMN_TYPE,
    LEVEL_TABLE_SUFFIX,
    LEVEL_VIEW_PREFIX,
    METADATA_GDAL_VSI,
    METADATA_OFFSET,
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
    ):
        """Initialize view builder.

        Args:
            db: DuckDB connection for creating views
            datasets: List of datasets to concatenate
            all_levels: Set of all available level views
            target_columns_by_level: Dict mapping level_key -> final column list
        """
        self.db = db
        self.datasets = datasets
        self.all_levels = all_levels
        self.target_columns_by_level = target_columns_by_level
        self.format_type = datasets[0]._format

    def build_all_views(self) -> None:
        """Build all views: UNION ALL + format-specific final views."""
        self._create_union_views()
        self._create_format_views()
        logger.debug("Created all DuckDB views")

    def _create_union_views(self) -> None:
        """Create UNION ALL views for each level."""
        for level_key in sorted(self.all_levels):
            if level_key not in self.target_columns_by_level:  # pragma: no cover
                continue

            logger.debug(f"  Consolidating {level_key}...")

            target_cols = self.target_columns_by_level[level_key]
            union_parts = []

            for ds_idx, ds in enumerate(self.datasets):
                max_depth = ds.pit_schema.max_depth()
                available_levels = [f"{LEVEL_VIEW_PREFIX}{i}" for i in range(max_depth + 1)]

                if level_key not in available_levels:  # pragma: no cover
                    continue

                # Extract PyArrow table from original dataset
                arrow_table = ds._duckdb.execute(f"SELECT * FROM {level_key}{LEVEL_TABLE_SUFFIX}").fetch_arrow_table()

                # Register in new connection with unique name
                table_name = f"ds{ds_idx}_{level_key}{LEVEL_TABLE_SUFFIX}"
                self.db.register(table_name, arrow_table)

                # Build SELECT with aligned columns + internal:source_path
                source_path = ds._path.rstrip("/")
                current_cols = set(arrow_table.column_names)

                select_parts = self._build_select_parts(target_cols, current_cols)
                select_parts.append(f"'{source_path}' AS \"{METADATA_SOURCE_PATH}\"")

                union_parts.append(f"SELECT {', '.join(select_parts)} FROM {table_name}")

            if not union_parts:  # pragma: no cover
                continue

            # Create consolidated view with UNION ALL
            union_query = " UNION ALL ".join(union_parts)
            self.db.execute(f"CREATE VIEW {level_key}{UNION_VIEW_SUFFIX} AS {union_query}")

            logger.debug(f"    {level_key}: {len(union_parts)} dataset(s), {len(target_cols)} columns")

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
        else:  # pragma: no cover
            raise TacoFormatError(f"Unknown format: {self.format_type}")

    def _create_zip_views(self) -> None:
        """Create final views for ZIP format.

        VSI: /vsisubfile/offset_size,source_path
        """
        for level_key in sorted(self.all_levels):
            if not self._view_exists(level_key):  # pragma: no cover
                continue

            self.db.execute(
                f"""
                CREATE VIEW {level_key} AS
                SELECT *,
                  '/vsisubfile/' || "{METADATA_OFFSET}" || '_' ||
                  "{METADATA_SIZE}" || ',' || "{METADATA_SOURCE_PATH}" as "{METADATA_GDAL_VSI}"
                FROM {level_key}{UNION_VIEW_SUFFIX}
                WHERE {COLUMN_ID} NOT LIKE '{PADDING_PREFIX}%'
            """
            )

    def _create_folder_views(self) -> None:
        """Create final views for FOLDER format.

        VSI: source_path/DATA/id (level0) or source_path/DATA/relative_path (level1+)
        """
        for level_key in sorted(self.all_levels):
            if not self._view_exists(level_key):  # pragma: no cover
                continue

            if level_key == f"{LEVEL_VIEW_PREFIX}0":
                self.db.execute(
                    f"""
                    CREATE VIEW {level_key} AS
                    SELECT *,
                      CASE
                        WHEN {COLUMN_TYPE} = '{SAMPLE_TYPE_FOLDER}' THEN "{METADATA_SOURCE_PATH}" || '/DATA/' || {COLUMN_ID} || '/__meta__'
                        WHEN {COLUMN_TYPE} = '{SAMPLE_TYPE_FILE}' THEN "{METADATA_SOURCE_PATH}" || '/DATA/' || {COLUMN_ID}
                        ELSE NULL
                      END as "{METADATA_GDAL_VSI}"
                    FROM {level_key}{UNION_VIEW_SUFFIX}
                    WHERE {COLUMN_ID} NOT LIKE '{PADDING_PREFIX}%'
                """
                )
            else:
                self.db.execute(
                    f"""
                    CREATE VIEW {level_key} AS
                    SELECT *,
                      CASE
                        WHEN {COLUMN_TYPE} = '{SAMPLE_TYPE_FOLDER}' THEN "{METADATA_SOURCE_PATH}" || '/DATA/' || "{METADATA_RELATIVE_PATH}" || '__meta__'
                        WHEN {COLUMN_TYPE} = '{SAMPLE_TYPE_FILE}' THEN "{METADATA_SOURCE_PATH}" || '/DATA/' || "{METADATA_RELATIVE_PATH}"
                        ELSE NULL
                      END as "{METADATA_GDAL_VSI}"
                    FROM {level_key}{UNION_VIEW_SUFFIX}
                    WHERE {COLUMN_ID} NOT LIKE '{PADDING_PREFIX}%'
                """
                )

    def _create_tacocat_views(self) -> None:
        """Create final views for TacoCat format.

        VSI: /vsisubfile/offset_size,source_path/source_file

        TacoCat stores multiple .tacozip files in one directory.
        source_path = path to .tacocat directory
        source_file = individual .tacozip filename (from parquet)
        """
        for level_key in sorted(self.all_levels):
            if not self._view_exists(level_key):  # pragma: no cover
                continue

            self.db.execute(
                f"""
                CREATE VIEW {level_key} AS
                SELECT *,
                  '/vsisubfile/' || "{METADATA_OFFSET}" || '_' ||
                  "{METADATA_SIZE}" || ',' || "{METADATA_SOURCE_PATH}" || '/' || "{METADATA_SOURCE_FILE}" as "{METADATA_GDAL_VSI}"
                FROM {level_key}{UNION_VIEW_SUFFIX}
                WHERE {COLUMN_ID} NOT LIKE '{PADDING_PREFIX}%'
            """
            )

    def _view_exists(self, level_key: str) -> bool:
        """Check if union view exists for given level."""
        return bool(
            self.db.execute(
                f"SELECT 1 FROM information_schema.tables WHERE table_name = '{level_key}{UNION_VIEW_SUFFIX}'"
            ).fetchone()
        )
