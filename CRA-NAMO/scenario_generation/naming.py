"""Human-readable artifact names for seeded experiments."""

from __future__ import annotations


def experiment_id(seed: int) -> str:
    """Return a filename-safe, sortable experiment number."""
    value = int(seed)
    if value < 0:
        return f"experiment_neg_{abs(value):04d}"
    return f"experiment_{value:04d}"


def map_stem(manifest: dict) -> str:
    """Name a generated map while retaining a short collision guard."""
    fingerprint = str(manifest.get("fingerprint", "unknown"))[:8]
    return (f"{experiment_id(manifest['seed'])}_{manifest['profile']}"
            f"_map_{fingerprint}")


def run_stem(manifest: dict, strategy: str) -> str:
    """Name one strategy run of a generated experiment."""
    fingerprint = str(manifest.get("fingerprint", "unknown"))[:8]
    return (f"{experiment_id(manifest['seed'])}_{manifest['profile']}"
            f"_{strategy}_{fingerprint}")
