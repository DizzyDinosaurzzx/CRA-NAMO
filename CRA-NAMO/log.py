"""Console output, with consecutive duplicate lines folded into a count.

Dependency-free like `geometry`, so anything may log without creating a cycle —
`config.Config.log` and the SE(2) planner both come through here, which is the
point: the folding only works if every line goes through one place.

A long run says the same thing hundreds of times over. `plan_move_se2` asks the
planner about one obstacle once per candidate drop pose, and every rejection
prints identically, so a single failed manipulation can bury the rest of the
transcript. Folding those runs makes the log as long as the story rather than as
long as the search.

Only *consecutive* repeats fold, so nothing is ever reordered and no line is
withheld: the first occurrence prints the moment it happens and a live run still
reads as progress. The tail of a run is only known to be over once something
else is printed, hence `flush` — call it before printing by any other means.
"""

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
