"""
dataset.py – PyTorch Dataset for DeepRadar2022.

Expected directory layout
─────────────────────────
data/DeepRadar/
    X_train.mat   (h5py)
    Y_train.mat   (scipy)
    lbl_train.mat (scipy)
    X_val.mat
    Y_val.mat
    lbl_val.mat
    X_test.mat
    Y_test.mat
    lbl_test.mat

X files are HDF5 (h5py); Y and lbl files are MATLAB v5 (scipy.io).

Shape after loading & transposing:
    X : (N, 1024, 2)   float32   IQ time series
    lbl: (N, 6)                  col-0 = class label (1-indexed)

Preprocessing
─────────────
1. Center-crop 1024 → crop_len (default 128) samples
2. Transpose to (2, crop_len) for PyTorch
3. Normalise each channel independently to [-1, 1]
"""

import os
import numpy as np
import h5py
import scipy.io as sio
import torch
from torch.utils.data import Dataset


# Key names inside the HDF5 files differ per split.
_H5_KEYS = {
    "train": "Xr_train",
    "val":   "X_val",
    "test":  "X_test",
}

_LBL_KEYS = {
    "train": "lbl_train",
    "val":   "lbl_val",
    "test":  "lbl_test",
}


def _load_X(mat_path, split):
    """Load IQ matrix from HDF5 .mat file.  Returns (N, 1024, 2) float32."""
    key = _H5_KEYS[split]
    with h5py.File(mat_path, "r") as f:
        # stored as (2, 1024, N) in MATLAB column-major → transpose
        X = np.array(f[key], dtype=np.float32).T   # → (N, 1024, 2)
    return X


def _load_lbl(mat_path, split):
    """Load label matrix.  Returns (N, 6) ndarray.  col-0 = class (1-indexed)."""
    key = _LBL_KEYS[split]
    data = sio.loadmat(mat_path)[key]               # (N, 6)
    return data.astype(np.int64)


class DeepRadarDataset(Dataset):
    """
    Parameters
    ----------
    data_dir  : path to folder containing the six .mat files
    split     : 'train' | 'val' | 'test'
    crop_len  : centre-crop target length (default 128)
    """

    def __init__(self, data_dir: str, split: str = "train", crop_len: int = 128):
        super().__init__()
        assert split in ("train", "val", "test"), f"Unknown split '{split}'"
        self.crop_len = crop_len

        x_path  = os.path.join(data_dir, f"X_{split}.mat")
        lbl_path = os.path.join(data_dir, f"lbl_{split}.mat")

        print(f"[DeepRadar] Loading X from {x_path} …")
        self.X = _load_X(x_path, split)                 # (N, 1024, 2)

        print(f"[DeepRadar] Loading labels from {lbl_path} …")
        lbl = _load_lbl(lbl_path, split)                 # (N, 6)
        # Convert 1-indexed class → 0-indexed
        self.labels = lbl[:, 0] - 1                      # (N,)  values 0-22

        assert len(self.X) == len(self.labels), "X / label count mismatch"
        print(f"[DeepRadar] {split}: {len(self.X)} samples, "
              f"{int(self.labels.max()) + 1} classes")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        signal = self.X[idx]       # (1024, 2)
        label  = self.labels[idx]  # scalar

        # ── centre-crop ────────────────────────────────────────────────────
        total = signal.shape[0]    # 1024
        start = (total - self.crop_len) // 2
        signal = signal[start: start + self.crop_len]   # (crop_len, 2)

        # ── transpose to (2, crop_len) for PyTorch Conv1d ─────────────────
        signal = signal.T.copy()   # (2, crop_len)

        # ── normalise each channel to [-1, 1] ──────────────────────────────
        for ch in range(signal.shape[0]):
            mx = np.abs(signal[ch]).max()
            if mx > 0:
                signal[ch] /= mx

        signal_t = torch.from_numpy(signal)             # (2, 128) float32
        label_t  = torch.tensor(label, dtype=torch.long)
        return signal_t, label_t