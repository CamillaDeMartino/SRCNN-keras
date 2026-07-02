from pathlib import Path
import random

import h5py
import numpy as np
import matplotlib.pyplot as plt


TRAIN_H5 = Path("/home/c.demartino/datasets/processed/train.h5")
VALIDATION_H5 = Path("/home/c.demartino/datasets/processed/validation.h5")
OUTPUT_DIR = Path("assets/debug_h5_residual")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



def main() -> None:
    with h5py.File(TRAIN_H5, "r") as h:
        X = h["data"]
        Y = h["label"]

        print("Data shape:", X.shape)
        print("Label shape:", Y.shape)

        print("\n=== INPUT DATA ===")
        print("min:", np.min(X))
        print("max:", np.max(X))
        print("mean:", np.mean(X))
        print("std:", np.std(X))

        print("\n=== RESIDUAL LABEL ===")
        print("min:", np.min(Y))
        print("max:", np.max(Y))
        print("mean:", np.mean(Y))
        print("std:", np.std(Y))
        print("mean abs:", np.mean(np.abs(Y)))


    with h5py.File(VALIDATION_H5, "r") as h:
        data = h["data"][:]
        label = h["label"][:]

        print("Data shape:", data.shape)
        print("Label shape:", label.shape)

        print("\n=== INPUT DATA ===")
        print("min:", data.min())
        print("max:", data.max())
        print("mean:", data.mean())
        print("std:", data.std())

        print("\n=== RESIDUAL LABEL ===")
        print("min:", label.min())
        print("max:", label.max())
        print("mean:", label.mean())
        print("std:", label.std())
        print("mean abs:", np.mean(np.abs(label)))


if __name__ == "__main__":
    main()