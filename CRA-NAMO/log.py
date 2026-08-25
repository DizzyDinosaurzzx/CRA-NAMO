"""Fold consecutive duplicate console messages."""

from __future__ import annotations

from typing import Optional

_pending: Optional[str] = None   # last line printed, still open to repeats
_repeats = 0                     # how many times it has repeated since


def emit(line: str) -> None:
    """Print *line*, folding an immediate repeat of the previous one into a count."""
    global _pending, _repeats
    if line == _pending:
        _repeats += 1
        return
    flush()
    print(line)
    _pending = line


def flush() -> None:
    """Close off an open run of repeats, so later output cannot land inside it."""
    global _pending, _repeats
    if _repeats:
        times = "time" if _repeats == 1 else "times"
        print(f"  ... last line repeated {_repeats} more {times}")
    _pending = None
    _repeats = 0
