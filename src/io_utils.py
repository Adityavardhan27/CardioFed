"""Output-safety helpers.

Research artifacts (checkpoints, metrics, figures) must never be replaced by
accident. Every script that writes results calls `guard_outputs` before doing
any work, so a run aborts immediately if it would overwrite an existing file,
unless the caller explicitly passes `overwrite=True` (the `--overwrite` flag).
"""

import os


class OutputExistsError(FileExistsError):
    """Raised when a script would overwrite an existing artifact."""


def guard_outputs(paths, overwrite=False):
    """Abort if any output path already exists (unless overwrite=True).

    Creates the parent directories of every path that is allowed.
    """
    paths = [os.fspath(p) for p in paths]
    existing = [p for p in paths if os.path.exists(p)]

    if existing and not overwrite:
        listing = "\n  ".join(existing)
        raise OutputExistsError(
            "Refusing to overwrite existing research artifact(s):\n  "
            f"{listing}\n"
            "Choose a different output location, or pass --overwrite "
            "if replacing them is intentional."
        )

    for p in paths:
        parent = os.path.dirname(p)
        if parent:
            os.makedirs(parent, exist_ok=True)

    return paths
