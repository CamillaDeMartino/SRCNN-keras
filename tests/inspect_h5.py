from pathlib import Path
import random

import h5py
import numpy as np
import matplotlib.pyplot as plt


TRAIN_H5 = Path("/home/c.demartino/datasets/processed/validation.h5")
OUTPUT_DIR = Path("assets/debug_h5_residual")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_PATCHES = 20
SEED = 42
CONV_SIDE = 6


def normalize_for_preview(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    mn = img.min()
    mx = img.max()

    if mx <= mn:
        return np.zeros_like(img)

    return (img - mn) / (mx - mn)


def residual_for_preview(residual: np.ndarray) -> np.ndarray:
    """
    Visual preview only.
    Residual has negative and positive values.
    We map 0 residual around gray.
    """
    residual = residual.astype(np.float32)
    max_abs = np.max(np.abs(residual))

    if max_abs == 0:
        return np.full_like(residual, 0.5)

    return np.clip((residual / (2 * max_abs)) + 0.5, 0.0, 1.0)


def save_patch_debug(idx: int, x_nchw: np.ndarray, residual_nchw: np.ndarray) -> float:
    x = np.transpose(x_nchw, (1, 2, 0))
    residual = np.transpose(residual_nchw, (1, 2, 0))

    x_center = x[CONV_SIDE:-CONV_SIDE, CONV_SIDE:-CONV_SIDE, :]
    reconstructed = np.clip(x_center + residual, 0.0, 1.0)

    residual_abs = np.abs(residual).mean(axis=2)
    residual_mae = float(np.mean(np.abs(residual)))

    x_preview = normalize_for_preview(x)
    x_center_preview = normalize_for_preview(x_center)
    residual_preview = residual_for_preview(residual)
    reconstructed_preview = normalize_for_preview(reconstructed)

    fig, ax = plt.subplots(1, 5, figsize=(20, 4))

    ax[0].imshow(x_preview)
    ax[0].set_title("Input 32x32")

    ax[1].imshow(x_center_preview)
    ax[1].set_title("Input Center 20x20")

    ax[2].imshow(residual_preview)
    ax[2].set_title("Residual Label\n0 ≈ gray")

    ax[3].imshow(reconstructed_preview)
    ax[3].set_title("Input Center + Residual")

    err = ax[4].imshow(residual_abs)
    ax[4].set_title(f"|Residual|\nmean={residual_mae:.6f}")
    fig.colorbar(err, ax=ax[4], fraction=0.046, pad=0.04)

    for a in ax:
        a.axis("off")

    plt.tight_layout()

    output_path = OUTPUT_DIR / f"patch_{idx:07d}_residual.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    return residual_mae


def main() -> None:
    with h5py.File(TRAIN_H5, "r") as h5:
        data = h5["data"]
        label = h5["label"]

        print("Data shape:", data.shape)
        print("Label shape:", label.shape)

        if data.shape[0] != label.shape[0]:
            raise RuntimeError("Data and label have different number of samples.")

        x_sample = data[:1000]
        y_sample = label[:1000]

        print("\n=== INPUT DATA ===")
        print("min:", x_sample.min())
        print("max:", x_sample.max())
        print("mean:", x_sample.mean())
        print("std:", x_sample.std())

        print("\n=== RESIDUAL LABEL ===")
        print("min:", y_sample.min())
        print("max:", y_sample.max())
        print("mean:", y_sample.mean())
        print("std:", y_sample.std())
        print("mean abs:", np.mean(np.abs(y_sample)))

        total = data.shape[0]

        random.seed(SEED)
        candidate_indices = list(range(total))
        random.shuffle(candidate_indices)

        selected_indices = []

        for idx in candidate_indices:
            x = data[idx]
            y = label[idx]

            # avoid completely flat input and completely flat residual
            if x.max() > 0.05 and np.std(y) > 0.001:
                selected_indices.append(idx)

            if len(selected_indices) >= NUM_PATCHES:
                break

        if not selected_indices:
            raise RuntimeError("No useful residual patches found.")

        residual_maes = []

        print("\nSelected patches:")
        for idx in selected_indices:
            x = data[idx]
            y = label[idx]

            residual_mae = float(np.mean(np.abs(y)))
            residual_maes.append(residual_mae)

            print(
                f"idx={idx} | "
                f"x min/max/mean={x.min():.4f}/{x.max():.4f}/{x.mean():.4f} | "
                f"res min/max/mean/std={y.min():.4f}/{y.max():.4f}/{y.mean():.6f}/{y.std():.6f} | "
                f"mean_abs_res={residual_mae:.6f}"
            )

            save_patch_debug(idx, x, y)

        residual_maes = np.array(residual_maes, dtype=np.float32)

        print("\n=== Summary ===")
        print(f"Patches checked: {len(residual_maes)}")
        print(f"Residual mean abs:   {residual_maes.mean():.6f}")
        print(f"Residual median abs: {np.median(residual_maes):.6f}")
        print(f"Residual min abs:    {residual_maes.min():.6f}")
        print(f"Residual max abs:    {residual_maes.max():.6f}")
        print(f"Saved images in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()