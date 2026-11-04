
import os
import time
import pandas as pd
import wfdb

# Project paths
ROOT = "data/raw/ptb-xl"
METADATA = os.path.join(ROOT, "ptbxl_database.csv")

BATCH_SIZE = 100       # 50 ECG records per batch
MAX_RETRIES = 5

os.makedirs(ROOT, exist_ok=True)

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
