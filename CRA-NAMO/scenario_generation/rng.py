"""Stable named random streams.

Python's process-randomised ``hash`` is deliberately not used: a saved seed must
mean the same thing in another process and after unrelated samplers are added.
"""

from __future__ import annotations

import hashlib
import random


class SeedStreams:
    def __init__(self, seed: int, attempt: int = 0):
        self.seed = int(seed)
        self.attempt = int(attempt)

    def derived_seed(self, namespace: str) -> int:
        raw = f"ca-namo:v1:{self.seed}:{self.attempt}:{namespace}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(raw).digest()[:16], "big")

    def rng(self, namespace: str) -> random.Random:
        return random.Random(self.derived_seed(namespace))

