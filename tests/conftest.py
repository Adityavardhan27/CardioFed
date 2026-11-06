import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # scripts use repository-relative paths

HAS_DATA = all(
    os.path.exists(os.path.join(ROOT, p)) for p in (
        "data/processed/X.npy", "data/processed/y.npy",
        "data/raw/ptb-xl/ptbxl_database.csv",
    )
)

requires_data = pytest.mark.skipif(
    not HAS_DATA, reason="PTB-XL processed data not available"
)
