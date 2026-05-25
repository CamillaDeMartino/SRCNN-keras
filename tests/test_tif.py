from pathlib import Path
import sys
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

REPORT_PATH = Path("assets/tif_input_shapes.json")
# preview settings
PREVIEW_DIR = Path("assets/previews")
PREVIEW_COUNT = 6

 
# Default dataset folder (used to locate files when JSON contains only filenames)
DATASET_DIR = Path("/storage/internal_02/SRCNN/dataset_rgb")

image_path = Path("/home/cdemartino/projects/SRCNN-keras/LC08_L1GT_148015_20260429_20260429_02_RT_B4.TIF")
img = cv2.imread(str(image_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)

valid_mask = img > 0
valid = img[valid_mask]

print("Original DN")
print("dtype:", img.dtype)
print("shape:", img.shape)
print("min:", img.min())
print("max:", img.max())
print("valid min:", valid.min())
print("valid max:", valid.max())
print("mean valid:", valid.mean())

plt.figure(figsize=(10, 6))
plt.hist(valid.ravel(), bins=256)
plt.title("Histogram of B4 DN values")
plt.xlabel("DN value")
plt.ylabel("Frequency")
plt.grid(True)

output_path = Path("/home/cdemartino/projects/SRCNN-keras/assets/histograms/B4_dn_histogram.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Histogram saved to: {output_path}")


## Normalizzazione 
img_float = np.zeros_like(img, dtype=np.float32)
img_float[valid_mask] = (
    img[valid_mask].astype(np.float32) - valid.min()
) / (valid.max() - valid.min())

print("\nNormalized float32")
print("dtype:", img_float.dtype)
print("min:", img_float.min())
print("max:", img_float.max())
print("mean:", img_float.mean())

plt.figure(figsize=(10, 6))
plt.hist(img_float[valid_mask].ravel(), bins=256)
plt.title("Histogram of valid DN values")
plt.xlabel("DN value")
plt.ylabel("Frequency")

output_path = Path("/home/cdemartino/projects/SRCNN-keras/assets/histograms/B4_nrm_histogram.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Histogram saved to: {output_path}")

