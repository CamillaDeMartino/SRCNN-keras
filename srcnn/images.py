from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

try:
    from .config import *
except Exception:
    from config import *

try:
    from .psnr import compute_psnr
except Exception:
    from psnr import compute_psnr

reference = np.load(ASSETS_DIR / "reference_b.npy")
bicubic = np.load(ASSETS_DIR / "bicubic_b.npy")
srcnn = np.load(ASSETS_DIR / "srcnn_pred_clip_b.npy")

reference = np.clip(reference, 0.0, 1.0)
bicubic = np.clip(bicubic, 0.0, 1.0)
srcnn = np.clip(srcnn, 0.0, 1.0)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].imshow(reference)
axs[0].set_title("Reference HR")
axs[0].axis("off")

axs[1].imshow(bicubic)
axs[1].set_title("Bicubic")
axs[1].axis("off")

axs[2].imshow(srcnn)
axs[2].set_title("SRCNN")
axs[2].axis("off")

plt.tight_layout()

#output_path = ASSETS_DIR / "comparison_reference_bicubic_srcnn4.png"
output_path = ASSETS_DIR / "best_case.png"

plt.savefig(output_path, dpi=200, bbox_inches="tight")
print(f"Saved: {output_path}")

error_bicubic = np.abs(reference - bicubic).mean(axis=2)
error_srcnn = np.abs(reference - srcnn).mean(axis=2)

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].imshow(error_bicubic)
axs[0].set_title("Bicubic Error")
axs[0].axis("off")

axs[1].imshow(error_srcnn)
axs[1].set_title("SRCNN Error")
axs[1].axis("off")

plt.tight_layout()

output_path = ASSETS_DIR / "error_maps_best_case.png"
plt.savefig(output_path, dpi=200, bbox_inches="tight")
print(f"Saved: {output_path}")