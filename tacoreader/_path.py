"""Path resolution for TACO datasets.

Unified handling of local/remote paths with format auto-detection.
"""

from tacoreader._constants import COLLECTION_JSON, TACOZIP_EXTENSIONS
from tacoreader._exceptions import TacoFormatError
from tacoreader._format import _file_exists, is_remote
from tacoreader.dataset import TacoDataset
from tacoreader.storage import create_backend


class TacoPath:
    """Resolve path: location + kind + loading."""

    def __init__(self, path: str, base_path: str | None = None):
        self.original = path.rstrip("/")
        self.remote = is_remote(path)
        self.base_path = base_path
        self.kind, self.resolved = self._detect()

    def _detect(self) -> tuple[str, str]:
        # TacoCat explicit
        if self.original.endswith(".tacocat"):
            return "tacocat", self.original

        # TacoZip explicit
        if self.original.endswith(TACOZIP_EXTENSIONS):
            return "zip", self.original

        # Directory with .tacocat inside
        tacocat_path = f"{self.original}/.tacocat"
        if _file_exists(tacocat_path, COLLECTION_JSON):
            return "tacocat", tacocat_path

        # Folder with COLLECTION.json
        if _file_exists(self.original, COLLECTION_JSON):
            return "folder", self.original

        raise TacoFormatError(
            f"COLLECTION.json not found in {self.original}\n"
            f"Expected: .tacozip file, .tacocat folder, or directory with COLLECTION.json"
        )

    def load(self, **opts) -> TacoDataset:
        backend = create_backend(self.kind)
        dataset = backend.load(self.resolved, **opts)

        # Apply base_path override for TacoCat
        if self.base_path is not None and self.kind == "tacocat":
            dataset = self._apply_base_path(dataset, backend, self.base_path)

        return dataset

    def _apply_base_path(self, dataset: TacoDataset, backend, base_path: str) -> TacoDataset:
        """Override vsi_base_path for TacoCat datasets."""
        from tacoreader._constants import DEFAULT_VIEW_NAME, LEVEL_VIEW_PREFIX
        from tacoreader._vsi import to_vsi_root

        base_vsi = to_vsi_root(base_path)
        if not base_vsi.endswith("/"):
            base_vsi += "/"

        max_depth = dataset.pit_schema.max_depth()

        # Drop existing views
        dataset._duckdb.execute(f"DROP VIEW IF EXISTS {DEFAULT_VIEW_NAME}")
        for i in range(max_depth + 1):
            dataset._duckdb.execute(f"DROP VIEW IF EXISTS {LEVEL_VIEW_PREFIX}{i}")

        # Recreate views with new vsi_base_path
        level_ids = list(range(max_depth + 1))
        backend.setup_duckdb_views(dataset._duckdb, level_ids, base_vsi)

        # Recreate 'data' view
        dataset._duckdb.execute(f"CREATE VIEW {DEFAULT_VIEW_NAME} AS SELECT * FROM {LEVEL_VIEW_PREFIX}0")
        dataset._vsi_base_path = base_vsi

        return dataset
