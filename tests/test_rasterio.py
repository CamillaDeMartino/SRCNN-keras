from pathlib import Path
import cv2
import rasterio
import tifffile as tiff
import numpy as np

DATASET_DIR = Path("/storage/internal_02/SRCNN/dataset")

image_path = DATASET_DIR / "LC08_L1TP_182043_20260427_20260427_02_RT_B4.TIF"

img = cv2.imread(str(image_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
print("\n=== OpenCV ===")
print("dtype:", img.dtype)
print("shape:", img.shape)
print("min:", img.min())
print("max:", img.max())
print("mean:", img.mean())

# Rasterio
with rasterio.open(image_path) as src:
    img_rs = src.read()

print("\n=== Rasterio ===")
print("dtype:", img_rs.dtype)
print("shape:", img_rs.shape)
print("min:", img_rs.min())
print("max:", img_rs.max())
print("mean:", img_rs.mean())

img_tiff = tiff.imread(str(image_path))

print("\n=== tifffile ===")
print("dtype:", img_tiff.dtype)
print("shape:", img_tiff.shape)
print("min:", img_tiff.min())
print("max:", img_tiff.max())
print("mean:", img_tiff.mean())
