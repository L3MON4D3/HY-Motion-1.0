"""Locate bundled runtime data independently of the process working directory."""

from pathlib import Path


def asset_path(relative_path: str) -> Path:
    """Resolve data in an installed wheel or an uninstalled source checkout."""
    package_root = Path(__file__).resolve().parent
    for root in (package_root, package_root.parent):
        candidate = root / relative_path
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"HY-Motion runtime asset is missing: {relative_path}")
