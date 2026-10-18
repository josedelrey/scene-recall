"""Source identity and resource measurements for saved scientific runs."""

import hashlib
import sys
from pathlib import Path

import scene_recall


def peak_rss() -> float | None:
    if not sys.platform.startswith("linux"):
        return None
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_hash(*additional_paths: Path) -> str:
    """Hash package Python sources, then any additional entry-point contents."""
    digest = hashlib.sha256()
    root = Path(scene_recall.__file__).parent
    for path in sorted(root.rglob("*")):
        if path.suffix == ".py":
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    for path in additional_paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()
