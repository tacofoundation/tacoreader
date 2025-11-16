"""
Concatenate multiple TACO datasets into single dataset.

Creates in-memory DuckDB with consolidated metadata using UNION ALL views.
All operations remain lazy - no temp files or disk materialization.
"""

from typing import TYPE_CHECKING
from pathlib import Path
import warnings

import duckdb

from tacoreader._constants import PROTECTED_COLUMNS, PADDING_PREFIX
from tacoreader._logging import get_logger

if TYPE_CHECKING:
    from tacoreader.dataset import TacoDataset
    from tacoreader.schema import PITSchema

logger = get_logger(__name__)


def _get_available_levels(dataset: "TacoDataset") -> list[str]:
    """Get list of available level views from pit_schema.max_depth()."""
    max_depth = dataset.pit_schema.max_depth()
    return [f"level{i}" for i in range(max_depth + 1)]


def _validate_column_compatibility(
    datasets: list["TacoDataset"], mode: str = "intersection"
) -> dict[str, set[str]]:
    """
    Validate column compatibility between datasets.

    Args:
        mode: Column handling strategy:
            - "intersection": Keep only common columns (DEFAULT)
            - "fill_missing": Fill missing columns with NULL
            - "strict": Fail if columns differ

    Returns:
        Dict mapping level_key to set of final columns
    """
    all_levels = set()
    for ds in datasets:
        all_levels.update(_get_available_levels(ds))

    level_columns = {}

    for level_key in sorted(all_levels):
        level_columns[level_key] = []

        for ds in datasets:
            available_levels = _get_available_levels(ds)
            if level_key in available_levels:
                df = ds._duckdb.execute(f"SELECT * FROM {level_key}").pl()
                level_columns[level_key].append(set(df.columns))

    final_columns = {}

    for level_key, column_sets in level_columns.items():
        common_cols = set.intersection(*column_sets)
        all_cols = set.union(*column_sets)

        # Columnas críticas dependen del nivel
        if level_key == "level0":
            critical_cols = PROTECTED_COLUMNS - {
                "internal:source_file",
                "internal:relative_path",
            }
        else:
            critical_cols = PROTECTED_COLUMNS - {"internal:source_file"}

        missing_critical = critical_cols - common_cols
        if missing_critical:
            problematic = []
            for i, cols in enumerate(column_sets):
                missing = critical_cols - cols
                if missing:
                    problematic.append(f"  - Dataset {i}: missing {sorted(missing)}")

            raise ValueError(
                f"Cannot concat: Critical columns missing in {level_key}\n"
                f"\n"
                f"Required columns for navigation:\n"
                f"  {sorted(critical_cols)}\n"
                f"\n"
                f"Missing in some datasets:\n"
                f"  {sorted(missing_critical)}\n"
                f"\n"
                f"Problems found:\n" + "\n".join(problematic) + f"\n"
                f"These columns are required for .read() and hierarchical navigation."
            )

        if mode == "strict":
            if len(set(map(frozenset, column_sets))) > 1:
                all_unique_cols = []
                for i, cols in enumerate(column_sets):
                    all_unique_cols.append(f"  Dataset {i}: {sorted(cols)}")

                raise ValueError(
                    f"Cannot concat in strict mode: Column mismatch in {level_key}\n"
                    f"\n"
                    f"Columns per dataset:\n" + "\n".join(all_unique_cols) + f"\n"
                    f"\n"
                    f"Only in some datasets: {sorted(all_cols - common_cols)}\n"
                    f"Common to all: {sorted(common_cols)}\n"
                    f"\n"
                    f"Solutions:\n"
                    f"  1. Use column_mode='intersection' (default) to keep only common columns\n"
                    f"  2. Use column_mode='fill_missing' to fill missing columns with NULL\n"
                    f"  3. Align columns with SQL before concat:\n"
                    f"     ds1 = ds1.sql('SELECT {', '.join(sorted(common_cols)[:3])} ... FROM data')"
                )
            final_columns[level_key] = common_cols

        elif mode == "intersection":
            final_columns[level_key] = common_cols

            if all_cols != common_cols:
                dropped = all_cols - common_cols
                column_sources = {}
                for col in dropped:
                    sources = []
                    for i, cols in enumerate(column_sets):
                        if col in cols:
                            sources.append(i)
                    column_sources[col] = sources

                details = []
                for col, sources in sorted(column_sources.items()):
                    if len(sources) < len(column_sets):
                        details.append(f"  - '{col}' (only in dataset(s) {sources})")

                warnings.warn(
                    f"\n"
                    f"concat() dropped {len(dropped)} column(s) from {level_key}\n"
                    f"\n"
                    f"Reason: Using column_mode='intersection' (default behavior)\n"
                    f"        Only columns present in ALL datasets are kept.\n"
                    f"\n"
                    f"Dropped columns:\n" + "\n".join(details) + f"\n"
                    f"\n"
                    f"Kept columns ({len(common_cols)}): {sorted(common_cols)}\n"
                    f"\n"
                    f"To keep all columns (fill missing with NULL):\n"
                    f"   concat([ds1, ds2], column_mode='fill_missing')",
                    UserWarning,
                    stacklevel=3,
                )

        elif mode == "fill_missing":
            final_columns[level_key] = all_cols

            if all_cols != common_cols:
                missing = all_cols - common_cols
                column_gaps = {}
                for col in missing:
                    gaps = []
                    for i, cols in enumerate(column_sets):
                        if col not in cols:
                            gaps.append(i)
                    column_gaps[col] = gaps

                details = []
                for col, gaps in sorted(column_gaps.items()):
                    details.append(
                        f"  - '{col}' (missing in dataset(s) {gaps}, will fill with NULL)"
                    )

                warnings.warn(
                    f"\n"
                    f"concat() filling missing columns in {level_key} with NULL\n"
                    f"\n"
                    f"Reason: Using column_mode='fill_missing'\n"
                    f"        All columns from all datasets are kept.\n"
                    f"\n"
                    f"Columns being filled:\n" + "\n".join(details) + f"\n"
                    f"\n"
                    f"Total columns: {len(all_cols)} (common: {len(common_cols)}, filled: {len(missing)})\n"
                    f"\n"
                    f"To avoid NULLs, use column_mode='intersection' (drops columns not in all datasets)",
                    UserWarning,
                    stacklevel=3,
                )

        else:
            raise ValueError(
                f"Invalid column_mode: '{mode}'\n"
                f"Valid options: 'intersection' (default), 'fill_missing', 'strict'"
            )

    return final_columns


def concat(
    datasets: list["TacoDataset"], column_mode: str = "intersection"
) -> "TacoDataset":
    """
    Concatenate multiple datasets into single dataset with lazy SQL.

    Creates in-memory DuckDB with consolidated metadata using UNION ALL views.
    By default, only keeps columns present in ALL datasets (intersection mode).

    Args:
        datasets: List of TacoDataset instances (minimum 2)
        column_mode: Column handling strategy (default "intersection")
            - "intersection": Keep only common columns (DEFAULT, safest)
            - "fill_missing": Keep all columns, fill missing with NULL
            - "strict": Fail if columns differ

    Returns:
        TacoDataset with consolidated data and lazy SQL

    Examples:
        # Default: intersection mode (keeps only common columns)
        dataset = concat([ds1, ds2, ds3])

        # Fill missing: keeps all columns, fills with NULL
        dataset = concat([ds1, ds2, ds3], column_mode="fill_missing")

        # Strict: fails if columns differ
        dataset = concat([ds1, ds2, ds3], column_mode="strict")

        # Manual alignment (most control)
        common = "id, type, cloud_cover, internal:offset, internal:size"
        ds1_aligned = ds1.sql(f"SELECT {common} FROM data")
        dataset = concat([ds1_aligned, ds2, ds3])
    """
    if len(datasets) < 2:
        raise ValueError(f"Need at least 2 datasets to concat, got {len(datasets)}")

    logger.info(f"Concatenating {len(datasets)} datasets...")

    # Validar PIT schemas
    reference_schema = datasets[0].pit_schema
    for i, ds in enumerate(datasets[1:], 1):
        if not reference_schema.is_compatible(ds.pit_schema):
            raise ValueError(
                f"Dataset {i} has incompatible schema. "
                f"All datasets must share same hierarchy structure.\n"
                f"Reference: {reference_schema}\n"
                f"Dataset {i}: {ds.pit_schema}"
            )

    logger.debug("All schemas compatible")

    # Validar columnas y obtener target columns
    logger.debug(f"Validating columns (mode={column_mode})...")

    target_columns_by_level = _validate_column_compatibility(datasets, mode=column_mode)

    for level_key, cols in target_columns_by_level.items():
        logger.debug(f"  {level_key}: {len(cols)} columns")

    # Consolidar schemas
    consolidated_schema = _merge_schemas([ds.pit_schema for ds in datasets])

    logger.debug("Consolidating levels in-memory...")

    # Crear nueva conexión DuckDB
    db = duckdb.connect(":memory:")

    # Cargar extensión espacial
    try:
        db.execute("INSTALL spatial")
        db.execute("LOAD spatial")
        logger.debug("Loaded DuckDB spatial extension")
    except Exception as e:
        logger.debug(f"Spatial extension not available: {e}")

    # Obtener todos los levels disponibles
    all_levels = set()
    for ds in datasets:
        all_levels.update(_get_available_levels(ds))

    # Para cada level, registrar tablas y crear UNION ALL view
    first_format = datasets[0]._format

    for level_key in sorted(all_levels):
        logger.debug(f"  Consolidating {level_key}...")

        target_cols = target_columns_by_level[level_key]
        union_parts = []

        for ds_idx, ds in enumerate(datasets):
            available_levels = _get_available_levels(ds)
            if level_key not in available_levels:
                continue

            # Extraer PyArrow table del dataset original
            arrow_table = ds._duckdb.execute(
                f"SELECT * FROM {level_key}_table"
            ).fetch_arrow_table()

            # Registrar en nueva conexión con nombre único
            table_name = f"ds{ds_idx}_{level_key}_table"
            db.register(table_name, arrow_table)

            # Construir SELECT con columnas alineadas + internal:source_file
            source_file = Path(ds._path).name

            # Obtener columnas actuales de esta tabla
            current_cols = set(arrow_table.column_names)

            # Construir lista de columnas con alineación
            select_parts = []
            for col in sorted(target_cols):
                if col in current_cols:
                    # Escapar nombres con caracteres especiales
                    escaped_col = f'"{col}"' if ":" in col or " " in col else col
                    select_parts.append(escaped_col)
                else:
                    # Columna faltante - rellenar con NULL
                    select_parts.append(f'NULL AS "{col}"')

            # Agregar internal:source_file
            select_parts.append(f"'{source_file}' AS \"internal:source_file\"")

            union_parts.append(f"SELECT {', '.join(select_parts)} FROM {table_name}")

        if not union_parts:
            continue

        # Crear view consolidada con UNION ALL
        union_query = " UNION ALL ".join(union_parts)
        db.execute(f"CREATE VIEW {level_key}_union AS {union_query}")

        logger.debug(
            f"    {level_key}: {len(union_parts)} dataset(s), {len(target_cols)} columns"
        )

    # Crear views finales con internal:gdal_vsi según formato
    if first_format == "zip":
        root_path = datasets[0]._root_path

        for level_key in sorted(all_levels):
            if not db.execute(
                f"SELECT 1 FROM information_schema.tables WHERE table_name = '{level_key}_union'"
            ).fetchone():
                continue

            db.execute(
                f"""
                CREATE VIEW {level_key} AS 
                SELECT *,
                  '/vsisubfile/' || "internal:offset" || '_' || 
                  "internal:size" || ',{root_path}' as "internal:gdal_vsi"
                FROM {level_key}_union
                WHERE id NOT LIKE '{PADDING_PREFIX}%'
            """
            )

    elif first_format == "folder":
        root_path = datasets[0]._root_path
        root = root_path if root_path.endswith("/") else root_path + "/"

        for level_key in sorted(all_levels):
            if not db.execute(
                f"SELECT 1 FROM information_schema.tables WHERE table_name = '{level_key}_union'"
            ).fetchone():
                continue

            if level_key == "level0":
                db.execute(
                    f"""
                    CREATE VIEW {level_key} AS 
                    SELECT *,
                      CASE 
                        WHEN type = 'FOLDER' THEN '{root}DATA/' || id || '/__meta__'
                        WHEN type = 'FILE' THEN '{root}DATA/' || id
                        ELSE NULL
                      END as "internal:gdal_vsi"
                    FROM {level_key}_union
                    WHERE id NOT LIKE '{PADDING_PREFIX}%'
                """
                )
            else:
                db.execute(
                    f"""
                    CREATE VIEW {level_key} AS 
                    SELECT *,
                      CASE 
                        WHEN type = 'FOLDER' THEN '{root}DATA/' || "internal:relative_path" || '__meta__'
                        WHEN type = 'FILE' THEN '{root}DATA/' || "internal:relative_path"
                        ELSE NULL
                      END as "internal:gdal_vsi"
                    FROM {level_key}_union
                    WHERE id NOT LIKE '{PADDING_PREFIX}%'
                """
                )

    elif first_format == "tacocat":
        base_path = datasets[0]._root_path

        for level_key in sorted(all_levels):
            if not db.execute(
                f"SELECT 1 FROM information_schema.tables WHERE table_name = '{level_key}_union'"
            ).fetchone():
                continue

            db.execute(
                f"""
                CREATE VIEW {level_key} AS 
                SELECT *,
                  '/vsisubfile/' || "internal:offset" || '_' || 
                  "internal:size" || ',{base_path}' || "internal:source_file"
                  as "internal:gdal_vsi"
                FROM {level_key}_union
                WHERE id NOT LIKE '{PADDING_PREFIX}%'
            """
            )

    # Crear 'data' view
    db.execute("CREATE VIEW data AS SELECT * FROM level0")

    total_samples = consolidated_schema.root["n"]
    logger.info(
        f"Concatenated {len(datasets)} datasets ({total_samples:,} total samples)"
    )

    # Construir TacoDataset
    from tacoreader.dataset import TacoDataset

    dataset = TacoDataset.model_construct(
        id=datasets[0].id,
        version=datasets[0].version,
        description=datasets[0].description,
        tasks=datasets[0].tasks,
        extent=datasets[0].extent,
        providers=datasets[0].providers,
        licenses=datasets[0].licenses,
        title=datasets[0].title,
        curators=datasets[0].curators,
        keywords=datasets[0].keywords,
        pit_schema=consolidated_schema,
        _path="<concatenated>",
        _format=first_format,
        _collection=datasets[0]._collection,
        _duckdb=db,
        _view_name="data",
        _root_path=datasets[0]._root_path,
    )

    return dataset


def _merge_schemas(schemas: list["PITSchema"]) -> "PITSchema":
    """Merge compatible schemas by summing n values."""
    if not schemas:
        raise ValueError("Need at least one schema to merge")

    reference = schemas[0]
    merged_dict = reference.to_dict()

    merged_dict["root"]["n"] = sum(s.root["n"] for s in schemas)

    for depth_str in merged_dict["hierarchy"]:
        for pattern_idx in range(len(merged_dict["hierarchy"][depth_str])):
            total_n = sum(s.hierarchy[depth_str][pattern_idx]["n"] for s in schemas)
            merged_dict["hierarchy"][depth_str][pattern_idx]["n"] = total_n

    from tacoreader.schema import PITSchema

    return PITSchema(merged_dict)
