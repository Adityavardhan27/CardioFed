
"""Memory-efficient PTB-XL PyTorch dataset."""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ECGDataset(Dataset):

    def __init__(self, processed_dir, indices):
        self.processed_dir = processed_dir
        self.indices = np.asarray(indices, dtype=np.int64)

        # Memory-mapped files: do not load the complete dataset into RAM.
        self.X = np.load(
            os.path.join(processed_dir, "X.npy"),
            mmap_mode="r"
        )

        self.y = np.load(
            os.path.join(processed_dir, "y.npy"),
            mmap_mode="r"
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        idx = self.indices[position]

        # Model input: [12 leads, 1000 time points].
        x = np.array(self.X[idx], dtype=np.float32, copy=True)
        y = np.array(self.y[idx], dtype=np.float32, copy=True)

        x = torch.from_numpy(x).permute(1, 0).contiguous()
        y = torch.from_numpy(y)

        return x, y


def make_loaders(processed_dir="data/processed", batch_size=32):

    folds = np.load(
        os.path.join(processed_dir, "folds.npy")
    )

    # Official PTB-XL fold separation.
    train_idx = np.where(folds <= 8)[0]
    val_idx = np.where(folds == 9)[0]
    test_idx = np.where(folds == 10)[0]

    train_ds = ECGDataset(processed_dir, train_idx)
    val_ds = ECGDataset(processed_dir, val_idx)
    test_ds = ECGDataset(processed_dir, test_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # Only the training labels are used for class weighting.
    y = np.load(
        os.path.join(processed_dir, "y.npy"),
        mmap_mode="r"
    )

    y_train = np.array(y[train_idx], dtype=np.float32)

    print("Training records:", len(train_idx))
    print("Validation records:", len(val_idx))
    print("Test records:", len(test_idx))

    return train_loader, val_loader, test_loader, y_train
