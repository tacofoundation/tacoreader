"""TacoCat remote cache management.

Caches remote TacoCat metadata (level*.parquet, COLLECTION.json) to disk
for faster subsequent loads. Uses ETag/Content-Length for validation.

Cache structure:
    ~/.cache/tacoreader/tacocat/{url_hash}/
        ├── .meta.json
        ├── level0.parquet
        ├── level1.parquet
        └── COLLECTION.json
"""

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from platformdirs import user_cache_path

from tacoreader._constants import (
    CACHE_DIR_NAME,
    CACHE_ENV_VAR,
    CACHE_HASH_LENGTH,
    CACHE_META_FILENAME,
    CACHE_TACOCAT_SUBDIR,
)
from tacoreader._logging import get_logger

logger = get_logger(__name__)


def get_cache_dir() -> Path:
    """Get tacoreader cache directory.

    Resolution:
        1. TACOREADER_CACHE_DIR env var (if set)
        2. Platform-specific via platformdirs:
           - Linux:   ~/.cache/tacoreader
           - macOS:   ~/Library/Caches/tacoreader
           - Windows: C:/Users/<user>/AppData/Local/tacoreader/Cache
    """
    env_override = os.environ.get(CACHE_ENV_VAR)
    if env_override:
        return Path(env_override)
    return user_cache_path(CACHE_DIR_NAME)


def get_tacocat_cache_dir() -> Path:
    """Get TacoCat cache directory."""
    return get_cache_dir() / CACHE_TACOCAT_SUBDIR


def url_to_cache_key(url: str) -> str:
    """Convert URL to cache directory name (16-char hash)."""
    # Normalize: remove trailing slash
    normalized = url.rstrip("/")
    return hashlib.sha256(normalized.encode()).hexdigest()[:CACHE_HASH_LENGTH]


def get_cached_path(url: str) -> Path:
    """Get cache directory path for URL."""
    return get_tacocat_cache_dir() / url_to_cache_key(url)


def read_cache_meta(url: str) -> dict | None:
    """Read .meta.json for cached URL. Returns None if not exists."""
    meta_path = get_cached_path(url) / CACHE_META_FILENAME
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def write_cache_meta(url: str, etag: str | None, size: int | None, files: list[str]) -> None:
    """Write .meta.json for cached URL."""
    cache_dir = get_cached_path(url)
    cache_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "url": url,
        "etag": etag,
        "size": size,
        "cached_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }

    meta_path = cache_dir / CACHE_META_FILENAME
    meta_path.write_text(json.dumps(meta, indent=2))


def is_cache_valid(url: str, remote_etag: str | None, remote_size: int | None) -> bool:
    """Check if cached data is still valid by comparing ETag/size."""
    meta = read_cache_meta(url)
    if meta is None:
        logger.debug(f"Cache meta not found for {url}")
        return False

    # ETag match (primary)
    if remote_etag and meta.get("etag"):
        valid = remote_etag == meta["etag"]
        logger.debug(f"ETag validation: remote={remote_etag}, cached={meta.get('etag')}, valid={valid}")
        return valid

    # Size match (fallback)
    if remote_size and meta.get("size"):
        valid = remote_size == meta["size"]
        logger.debug(f"Size validation: remote={remote_size}, cached={meta.get('size')}, valid={valid}")
        return valid

    # No remote metadata available - use cache if exists (trust local)
    if remote_etag is None and remote_size is None and meta:
        logger.debug("No remote metadata, trusting local cache")
        return True

    logger.debug(f"Cannot validate cache: remote_etag={remote_etag}, remote_size={remote_size}")
    return False


def get_remote_metadata(url: str) -> dict | None:
    """HEAD request to get ETag and Content-Length from remote.

    Uses COLLECTION.json as reference file for validation.

    Returns:
        {"etag": str | None, "size": int | None} or None if failed
    """
    import obstore as obs

    from tacoreader._remote_io import _create_store

    try:
        # HEAD to COLLECTION.json (always exists in .tacocat)
        store = _create_store(url)
        head = obs.head(store, "COLLECTION.json")
        return {
            "etag": getattr(head, "e_tag", None) or getattr(head, "etag", None),
            "size": getattr(head, "size", None),
        }
    except Exception as e:
        logger.debug(f"HEAD request failed for {url}: {e}")
        return None


def load_from_cache(url: str) -> dict[str, bytes] | None:
    """Load cached files for URL.

    Returns:
        Dict of {filename: bytes} or None if cache miss
    """
    meta = read_cache_meta(url)
    if meta is None:
        logger.debug(f"No cache meta for: {url}")
        return None

    cache_dir = get_cached_path(url)
    files_list = meta.get("files", [])

    if not files_list:
        logger.debug(f"Cache meta has no files list: {cache_dir}")
        return None

    files = {}
    for filename in files_list:
        file_path = cache_dir / filename
        if not file_path.exists():
            logger.debug(f"Cache incomplete, missing: {filename}")
            return None
        files[filename] = file_path.read_bytes()

    logger.info(f"Loaded {len(files)} files from cache: {cache_dir}")
    return files


def save_to_cache(url: str, files: dict[str, bytes], etag: str | None, size: int | None) -> None:
    """Save files to cache."""
    cache_dir = get_cached_path(url)
    cache_dir.mkdir(parents=True, exist_ok=True)

    filenames = []
    for filename, data in files.items():
        file_path = cache_dir / filename
        file_path.write_bytes(data)
        filenames.append(filename)
        logger.debug(f"Cached file: {file_path} ({len(data)} bytes)")

    write_cache_meta(url, etag, size, filenames)
    logger.info(f"Saved {len(files)} files to cache: {cache_dir}")


def invalidate_cache(url: str) -> None:
    """Delete cached data for URL."""
    cache_dir = get_cached_path(url)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        logger.debug(f"Invalidated cache: {cache_dir}")


def clear_tacocat_cache() -> None:
    """Clear all TacoCat disk cache."""
    cache_dir = get_tacocat_cache_dir()
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        logger.info(f"Cleared TacoCat cache: {cache_dir}")


def get_cache_stats() -> dict:
    """Get cache statistics."""
    cache_dir = get_tacocat_cache_dir()
    if not cache_dir.exists():
        return {"entries": 0, "size_mb": 0.0}

    total_size = 0
    entries = 0

    for entry in cache_dir.iterdir():
        if entry.is_dir():
            entries += 1
            for f in entry.rglob("*"):
                if f.is_file():
                    total_size += f.stat().st_size

    return {
        "entries": entries,
        "size_mb": round(total_size / (1024 * 1024), 2),
        "path": str(cache_dir),
    }
