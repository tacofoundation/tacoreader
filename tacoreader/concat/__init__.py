"""
Concatenate multiple TACO datasets into single dataset.

Public API:
    concat(datasets, column_mode) -> TacoDataset
        Concatenates multiple datasets with lazy SQL evaluation.
        Supports three column modes: intersection (default), fill_missing, strict.

Internal modules (not exported):
    _orchestrator: Main concatenation orchestration (4-phase pipeline)
    _validation: Dataset compatibility validation (backends, formats, schemas)
    _columns: Column compatibility and resolution (intersection/fill_missing/strict)
    _view_builder: DuckDB view construction with format-specific strategies

Architecture:
    concat() orchestrates a 4-phase process:
    1. Validation: Check dataset compatibility (_validation.py)
    2. Preparation: Resolve columns and merge schemas (_columns.py)
    3. Construction: Build DuckDB UNION ALL views (_view_builder.py)
    4. Finalization: Create consolidated TacoDataset (_orchestrator.py)
"""

from tacoreader.concat._orchestrator import concat

__all__ = ["concat"]
