
"""Download PTB-XL 1.0.3 (metadata + 100 Hz records) from PhysioNet.

Resumable: files that already exist are skipped. The two metadata files are
downloaded first if they are missing.

    python download_ptbxl.py                    # everything (100 Hz)
    python download_ptbxl.py --metadata-only    # only the two CSV files
"""

import argparse
import os
import time
import pandas as pd
import wfdb

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", default="data/raw/ptb-xl")
parser.add_argument("--metadata-only", action="store_true")
ARGS = parser.parse_args()

# Project paths
ROOT = ARGS.root
METADATA = os.path.join(ROOT, "ptbxl_database.csv")
METADATA_FILES = ["ptbxl_database.csv", "scp_statements.csv"]

BATCH_SIZE = 100       # files (.hea/.dat) per download batch
MAX_RETRIES = 5

os.makedirs(ROOT, exist_ok=True)

# Fetch the metadata first; the record list is read from it.
missing_metadata = [
    name for name in METADATA_FILES
    if not os.path.exists(os.path.join(ROOT, name))
]

if missing_metadata:
    print("Downloading metadata:", ", ".join(missing_metadata))
    wfdb.dl_files("ptb-xl", ROOT, missing_metadata, overwrite=False)

if ARGS.metadata_only:
    print("Metadata available in", ROOT)
    raise SystemExit(0)

# Read official PTB-XL metadata
df = pd.read_csv(METADATA)

records = df["filename_lr"].dropna().astype(str).tolist()

# Build the list of missing header and signal files
missing_files = []

for record in records:
    for extension in [".hea", ".dat"]:
        relative_path = record + extension
        local_path = os.path.join(ROOT, relative_path)

        if not os.path.exists(local_path):
            missing_files.append(relative_path)

total_records = len(records)
total_files = len(missing_files)
total_batches = (total_files + BATCH_SIZE - 1) // BATCH_SIZE

print("PTB-XL 100 Hz downloader")
print("Total records:", total_records)
print("Files still missing:", total_files)
print("Batches remaining:", total_batches)

if total_files == 0:
    print("\nAll ECG files are already downloaded!")

else:
    for start in range(0, total_files, BATCH_SIZE):
        batch = missing_files[start:start + BATCH_SIZE]
        batch_number = start // BATCH_SIZE + 1

        success = False

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                print(
                    f"\nBatch {batch_number}/{total_batches} "
                    f"| Attempt {attempt}/{MAX_RETRIES}",
                    flush=True
                )

                wfdb.dl_files(
                    "ptb-xl",
                    ROOT,
                    batch,
                    keep_subdirs=True,
                    overwrite=False
                )

                success = True
                print("Batch completed.", flush=True)
                break

            except Exception as error:
                print("Download error:", error, flush=True)

                if attempt < MAX_RETRIES:
                    wait = 10 * attempt
                    print(f"Retrying in {wait} seconds...")
                    time.sleep(wait)

        if not success:
            raise RuntimeError(
                f"Batch {batch_number} failed. "
                "Run the script again to resume."
            )

    print("\nALL ECG RECORDINGS DOWNLOADED!")
