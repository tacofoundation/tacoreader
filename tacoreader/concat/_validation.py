"""Dataset validation for concatenation.

Validates:
- RSUT compliance (no level1+ JOINs that break structural homogeneity)
- Backend compatibility (all must use same DataFrame backend)
- Format compatibility (all must use same storage format)
- Schema compatibility (all must have compatible PIT structure)
"""

from typing import TYPE_CHECKING

from tacoreader._exceptions import TacoBackendError, TacoSchemaError
from tacoreader._logging import get_logger

if TYPE_CHECKING:
    from tacoreader.dataset import TacoDataset

logger = get_logger(__name__)


def validate_datasets(datasets: list["TacoDataset"]) -> None:
    """Run all validations on datasets before concatenation.

    Raises:
        TacoSchemaError: If any dataset is not RSUT compliant (has level1+ JOINs)
        TacoBackendError: If backends are incompatible
        TacoSchemaError: If formats or schemas are incompatible
    """
    _validate_rsut_compliance(datasets)
    _validate_backends(datasets)
    _validate_formats(datasets)
    _validate_schemas(datasets)
    logger.debug("All validations passed")


def _validate_rsut_compliance(datasets: list["TacoDataset"]) -> None:
    """Ensure no dataset has level1+ JOINs that break RSUT.

    RSUT Invariant 3 (Structural Homogeneity): All level0 FOLDERs must be
    structurally equivalent - same children, same IDs, same types.

    Queries involving level1+ (JOINs, filters on children) break this invariant
    because they can cause different level0 samples to have different effective
    children, violating structural equivalence.

    Only level0-only queries preserve RSUT compliance.
    """
    non_compliant = []

    for i, ds in enumerate(datasets):
        if ds._has_level1_joins:
            non_compliant.append((i, ds._path, ds._joined_levels))

    if non_compliant:
        details = []
        for idx, path, levels in non_compliant:
            details.append(f"  Dataset {idx}: {path}")
            details.append(f"    Joined levels: {sorted(levels)}")

        raise TacoSchemaError(
            "Cannot concat: Some datasets are not RSUT compliant.\n"
            "\n"
            "The following datasets have queries involving level1+ tables:\n"
            + "\n".join(details) + "\n"
            "\n"
            "RSUT Invariant 3 (Structural Homogeneity) requires all level0 FOLDERs\n"
            "to have identical children. Queries on level1+ break this invariant.\n"
            "\n"
            "Only level0 queries are allowed before concat.\n"
            "Filter by child properties after concat if needed."
        )

    logger.debug("RSUT compliance validated")


def _validate_backends(datasets: list["TacoDataset"]) -> None:
    """Ensure all datasets use same DataFrame backend."""
    backends = {ds._dataframe_backend for ds in datasets}

    if len(backends) == 1:
        return

    backend_info = [f"  Dataset {i}: {ds._dataframe_backend} (from {ds._path})" for i, ds in enumerate(datasets)]

    raise TacoBackendError(
        f"Cannot concatenate datasets with different DataFrame backends.\n"
        f"\n"
        f"Found {len(backends)} different backends: {sorted(backends)}\n"
        f"\n"
        f"Backend per dataset:\n" + "\n".join(backend_info) + "\n"
        "\n"
        "All datasets must use the same backend.\n"
        "Solution: Set backend before loading datasets:\n"
        "  tacoreader.use('pyarrow')  # or 'polars', 'pandas'\n"
        "  ds1 = tacoreader.load('data1.taco')\n"
        "  ds2 = tacoreader.load('data2.taco')\n"
        "  result = tacoreader.concat([ds1, ds2])"
    )


def _validate_formats(datasets: list["TacoDataset"]) -> None:
    """Ensure all datasets use same storage format."""
    formats = {ds._format for ds in datasets}

    if len(formats) == 1:
        return

    format_info = [f"  Dataset {i}: {ds._format} (from {ds._path})" for i, ds in enumerate(datasets)]

    raise TacoSchemaError(
        f"Cannot concatenate datasets with different formats.\n"
        f"\n"
        f"Found {len(formats)} different formats: {sorted(formats)}\n"
        f"\n"
        f"Format per dataset:\n" + "\n".join(format_info) + "\n"
        "\n"
        "All datasets must use the same format (zip/folder/tacocat).\n"
        "Mixing formats breaks GDAL VSI path construction and navigation."
    )

def _validate_schemas(datasets: list["TacoDataset"]) -> None:
    """Ensure all datasets have compatible PIT schemas."""
    reference_schema = datasets[0].pit_schema

    for i, ds in enumerate(datasets[1:], 1):
        if not reference_schema.is_compatible(ds.pit_schema):
            raise TacoSchemaError(
                f"Dataset {i} has incompatible schema. "
                f"All datasets must share same hierarchy structure.\n"
                f"Reference: {reference_schema}\n"
                f"Dataset {i}: {ds.pit_schema}"
            )

    logger.debug("All schemas compatible")