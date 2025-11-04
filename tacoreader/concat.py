from typing import TYPE_CHECKING

import duckdb
import polars as pl

if TYPE_CHECKING:
    from tacoreader.dataset import TacoDataset
    from tacoreader.schema import PITSchema


def concat(datasets: list["TacoDataset"], verbose: bool = False) -> "TacoDataset":
    """
    Concatenate multiple TACODatasets into single TacoDataset with lazy SQL interface.

    Creates in-memory DuckDB database with consolidated metadata files.
    All operations are lazy until .collect() or .data.collect() is called.

    Args:
        datasets: List of TacoDataset instances (minimum 2)
        verbose: Show loading progress (default False)

    Returns:
        TacoDataset with consolidated data and lazy SQL query capability

    Raises:
        ValueError: If fewer than 2 datasets
        ValueError: If schemas are incompatible

    Examples:
        >>> ds1 = load("taco_001.zip")
        >>> ds2 = load("taco_002.zip")
        >>> ds3 = load("taco_003.zip")
        >>>
        >>> dataset = concat([ds1, ds2, ds3])
        >>>
        >>> peru = dataset.sql("SELECT * FROM data WHERE country = 'Peru'")
        >>> df = peru.data.collect()
    """
    import tempfile
    import uuid
    from pathlib import Path

    if len(datasets) < 2:
        raise ValueError(f"Need at least 2 datasets to concat, got {len(datasets)}")

    if verbose:
        print(f"Validating {len(datasets)} datasets...")

    reference_schema = datasets[0].pit_schema
    for i, ds in enumerate(datasets[1:], 1):
        if not reference_schema.is_compatible(ds.pit_schema):
            raise ValueError(
                f"Dataset {i} has incompatible schema. "
                f"All datasets must share same hierarchy structure.\n"
                f"Reference: {reference_schema}\n"
                f"Dataset {i}: {ds.pit_schema}"
            )

    if verbose:
        print("All schemas compatible")

    consolidated_schema = _merge_schemas([ds.pit_schema for ds in datasets])

    if verbose:
        print("\nConsolidating levels...")

    cache_dir = Path(tempfile.gettempdir()) / f"tacoreader-concat-{uuid.uuid4()}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_levels = set()
    for ds in datasets:
        all_levels.update(ds._consolidated_files.keys())

    consolidated_files = {}

    for level_key in sorted(all_levels):
        if verbose:
            print(f"  Consolidating {level_key}...")

        level_dfs = []
        for ds in datasets:
            if level_key in ds._consolidated_files:
                file_path = ds._consolidated_files[level_key]

                if file_path.endswith(".avro"):
                    df = pl.read_avro(file_path)
                else:
                    df = pl.read_parquet(file_path)

                level_dfs.append(df)

        if level_dfs:
            consolidated_level = pl.concat(level_dfs)
            output_path = cache_dir / f"{level_key}.parquet"
            consolidated_level.write_parquet(output_path)
            consolidated_files[level_key] = str(output_path)

            if verbose:
                print(f"    {level_key}: {len(consolidated_level):,} rows")

    db = duckdb.connect(":memory:")

    first_format = datasets[0]._format

    if first_format == "zip":
        root_path = datasets[0]._root_path

        for level_key, file_path in consolidated_files.items():
            db.execute(
                f"""
                CREATE VIEW {level_key} AS 
                SELECT *,
                  '/vsisubfile/' || "internal:offset" || '_' || 
                  "internal:size" || ',{root_path}' as "internal:gdal_vsi"
                FROM read_parquet('{file_path}')
                WHERE id NOT LIKE '__TACOPAD__%'
            """
            )

    elif first_format == "folder":
        root_path = datasets[0]._root_path
        root = root_path if root_path.endswith("/") else root_path + "/"

        for level_key, file_path in consolidated_files.items():
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
                    FROM read_parquet('{file_path}')
                    WHERE id NOT LIKE '__TACOPAD__%'
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
                    FROM read_parquet('{file_path}')
                    WHERE id NOT LIKE '__TACOPAD__%'
                """
                )

    elif first_format == "tacocat":
        base_path = datasets[0]._root_path

        for level_key, file_path in consolidated_files.items():
            db.execute(
                f"""
                CREATE VIEW {level_key} AS 
                SELECT *,
                  '/vsisubfile/' || "internal:offset" || '_' || 
                  "internal:size" || ',{base_path}' || "internal:source_file"
                  as "internal:gdal_vsi"
                FROM read_parquet('{file_path}')
                WHERE id NOT LIKE '__TACOPAD__%'
            """
            )

    db.execute("CREATE VIEW data AS SELECT * FROM level0")

    if verbose:
        total_samples = consolidated_schema.root["n"]
        print(
            f"\nConcatenated {len(datasets)} datasets ({total_samples:,} total samples)\n"
        )

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
        _consolidated_files=consolidated_files,
        _duckdb=db,
        _view_name="data",
        _root_path=datasets[0]._root_path,
    )

    return dataset


def _merge_schemas(schemas: list["PITSchema"]) -> "PITSchema":
    """
    Merge compatible schemas by summing n values.

    Args:
        schemas: List of compatible PITSchemas

    Returns:
        New PITSchema with summed n values

    Raises:
        ValueError: If schemas list is empty
    """
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