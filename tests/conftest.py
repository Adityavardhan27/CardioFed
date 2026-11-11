import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # scripts use repository-relative paths

# Large files (data arrays, checkpoints) are stored with Git LFS. A clone made
# without git-lfs, or with GIT_LFS_SKIP_SMUDGE=1, contains small text "pointer"
# files under the same names. They exist but are not usable data, so a plain
# os.path.exists() check would wrongly run (and crash) the data-dependent tests.
LFS_SIGNATURE = b"version https://git-lfs"


def is_real_file(path):
    """True if `path` exists and is real content, not a Git-LFS pointer."""
    try:
        with open(os.path.join(ROOT, path), "rb") as handle:
            return handle.read(len(LFS_SIGNATURE)) != LFS_SIGNATURE
    except OSError:
        return False


REQUIRED_DATA = (
    "data/processed/X.npy",
    "data/processed/y.npy",
    "data/processed/ecg_ids.npy",
    "data/processed/folds.npy",
    "data/raw/ptb-xl/ptbxl_database.csv",
)

HAS_DATA = all(is_real_file(p) for p in REQUIRED_DATA)

requires_data = pytest.mark.skipif(
    not HAS_DATA,
    reason="PTB-XL processed data not available (missing, or only Git-LFS "
           "pointer files: run `git lfs pull`)",
)
