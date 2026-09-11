"""Content fingerprints for generated manifests."""

from __future__ import annotations

import hashlib
import json


def fingerprint_manifest(manifest: dict) -> str:
    payload = dict(manifest)
    payload.pop("fingerprint", None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

