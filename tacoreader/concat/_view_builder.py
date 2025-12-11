"""
DuckDB view construction for concatenated datasets.

Uses strategy pattern to handle format-specific view creation (ZIP/FOLDER/TacoCat).
"""

from pathlib import Path
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
    PADDING_PREFIX,
    SAMPLE_TYPE_FILE,
    SAMPLE_TYPE_FOLDER,
    TACOCAT_FOLDER_NAME,
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
        """
        Initialize view builder.

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
        self.root_path = datasets[0]._root_path

    def build_all_views(self) -> None:
        """Build all views: UNION ALL + format-specific final views."""
        # Step 1: Create UNION ALL views (consolidate metadata)
        self._create_union_views()

        # Step 2: Create format-specific final views (add GDAL VSI paths)
        self._create_format_views()

        logger.debug("Created all DuckDB views")

    def _create_union_views(self) -> None:
        """Create UNION ALL views for each level."""
        for level_key in sorted(self.all_levels):
            if level_key not in self.target_columns_by_level:
                continue

            logger.debug(f"  Consolidating {level_key}...")

            target_cols = self.target_columns_by_level[level_key]
            union_parts = []

            for ds_idx, ds in enumerate(self.datasets):
                max_depth = ds.pit_schema.max_depth()
                available_levels = [
                    f"{LEVEL_VIEW_PREFIX}{i}" for i in range(max_depth + 1)
                ]

                if level_key not in available_levels:
                    continue

                # Extract PyArrow table from original dataset
                arrow_table = ds._duckdb.execute(
                    f"SELECT * FROM {level_key}{LEVEL_TABLE_SUFFIX}"
                ).fetch_arrow_table()

                # Register in new connection with unique name
                table_name = f"ds{ds_idx}_{level_key}{LEVEL_TABLE_SUFFIX}"
                self.db.register(table_name, arrow_table)

                # Build SELECT with aligned columns + internal:source_file
                source_file = Path(ds._path).name
                current_cols = set(arrow_table.column_names)

                select_parts = self._build_select_parts(target_cols, current_cols)
                select_parts.append(f"'{source_file}' AS \"{METADATA_SOURCE_FILE}\"")

                union_parts.append(
                    f"SELECT {', '.join(select_parts)} FROM {table_name}"
                )

            if not union_parts:
                continue

            # Create consolidated view with UNION ALL
            union_query = " UNION ALL ".join(union_parts)
            self.db.execute(
                f"CREATE VIEW {level_key}{UNION_VIEW_SUFFIX} AS {union_query}"
            )

            logger.debug(
                f"    {level_key}: {len(union_parts)} dataset(s), {len(target_cols)} columns"
            )

    @staticmethod
    def _build_select_parts(
        target_cols: list[str], current_cols: set[str]
    ) -> list[str]:
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
        # Strategy pattern: delegate to format-specific method
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

            self.db.execute(
                f"""
                CREATE VIEW {level_key} AS
                SELECT *,
                  '/vsisubfile/' || "{METADATA_OFFSET}" || '_' ||
                  "{METADATA_SIZE}" || ',{self.root_path}' as "{METADATA_GDAL_VSI}"
                FROM {level_key}{UNION_VIEW_SUFFIX}
                WHERE {COLUMN_ID} NOT LIKE '{PADDING_PREFIX}%'
            """
            )

    def _create_folder_views(self) -> None:
        """Create final views for FOLDER format."""
        root = self.root_path if self.root_path.endswith("/") else self.root_path + "/"

        for level_key in sorted(self.all_levels):
            if not self._view_exists(level_key):
                continue

            if level_key == f"{LEVEL_VIEW_PREFIX}0":
                self.db.execute(
                    f"""
                    CREATE VIEW {level_key} AS
                    SELECT *,
                      CASE
                        WHEN {COLUMN_TYPE} = '{SAMPLE_TYPE_FOLDER}' THEN '{root}DATA/' || {COLUMN_ID} || '/__meta__'
                        WHEN {COLUMN_TYPE} = '{SAMPLE_TYPE_FILE}' THEN '{root}DATA/' || {COLUMN_ID}
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
                        WHEN {COLUMN_TYPE} = '{SAMPLE_TYPE_FOLDER}' THEN '{root}DATA/' || "{METADATA_RELATIVE_PATH}" || '__meta__'
                        WHEN {COLUMN_TYPE} = '{SAMPLE_TYPE_FILE}' THEN '{root}DATA/' || "{METADATA_RELATIVE_PATH}"
                        ELSE NULL
                      END as "{METADATA_GDAL_VSI}"
                    FROM {level_key}{UNION_VIEW_SUFFIX}
                    WHERE {COLUMN_ID} NOT LIKE '{PADDING_PREFIX}%'
                """
                )

    def _create_tacocat_views(self) -> None:
        """Create final views for TacoCat format."""
        base_path = self._extract_base_path()

        for level_key in sorted(self.all_levels):
            if not self._view_exists(level_key):
                continue

            self.db.execute(
                f"""
                CREATE VIEW {level_key} AS
                SELECT *,
                  '/vsisubfile/' || "{METADATA_OFFSET}" || '_' ||
                  "{METADATA_SIZE}" || ',{base_path}' || "{METADATA_SOURCE_FILE}"
                  as "{METADATA_GDAL_VSI}"
                FROM {level_key}{UNION_VIEW_SUFFIX}
                WHERE {COLUMN_ID} NOT LIKE '{PADDING_PREFIX}%'
            """
            )

    def _view_exists(self, level_key: str) -> bool:
        """Check if union view exists for given level."""
        return bool(
            self.db.execute(
                f"SELECT 1 FROM information_schema.tables "
                f"WHERE table_name = '{level_key}{UNION_VIEW_SUFFIX}'"
            ).fetchone()
        )

    def _extract_base_path(self) -> str:
        """
        Extract base directory for TacoCat format.

        Removes .tacocat folder suffix to get parent directory containing
        both .tacocat/ and source .tacozip files.

        Examples:
            /vsis3/bucket/data/.tacocat → /vsis3/bucket/data/
            /vsis3/bucket/data/.tacocat/ → /vsis3/bucket/data/
            /local/path/.tacocat → /local/path/
        """
        # Remove trailing slash if present
        clean_path = self.root_path.rstrip("/")

        # Remove .tacocat folder name
        if clean_path.endswith(f"/{TACOCAT_FOLDER_NAME}"):
            base_path = clean_path[: -(len(TACOCAT_FOLDER_NAME) + 1)]
        elif clean_path.endswith(TACOCAT_FOLDER_NAME):
            base_path = clean_path[: -len(TACOCAT_FOLDER_NAME)]
        else:
            base_path = clean_path

        # Ensure trailing slash for path concatenation
        if not base_path.endswith("/"):
            base_path += "/"

        return base_path
