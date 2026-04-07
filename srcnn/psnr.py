from pathlib import Path

import cv2
import math
import numpy


ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT / "assets"


def compute_psnr(target, ref, max_val = 4095.0):

    target_data = numpy.array(target, dtype=numpy.float32)
    ref_data = numpy.array(ref, dtype=numpy.float32)

    diff = ref_data - target_data
    rmse = math.sqrt(numpy.mean(diff ** 2))

    return 20 * math.log10(max_val / rmse)


if __name__ == "__main__":
    MAX_VAL = 4095.0

    #INPUT (Bicubic)
    im1 = cv2.imread(str(ASSETS_DIR / "input.jpg"), cv2.IMREAD_UNCHANGED)
    #REFERENCE (Ground Truth - HR image)
    im2 = cv2.imread(str(ASSETS_DIR / "butterfly_GT.bmp"), cv2.IMREAD_UNCHANGED)
    #Modello addestrato con Adam (vecchio - non penso serve)
    im3 = cv2.imread(str(ASSETS_DIR / "pre_adam30.jpg"), cv2.IMREAD_UNCHANGED)
    #Modello addestrato con Adam (nuovo)
    im4 = cv2.imread(str(ASSETS_DIR / "pre2.jpg"), cv2.IMREAD_UNCHANGED)

    if im1 is None or im2 is None or im3 is None or im4 is None:
        raise ValueError("One or more images could not be loaded.")

    # Crop valid region because SRCNN with valid convolutions reduces borders
    im1 = im1[6:-6, 6:-6, :]
    im2 = im2[6:-6, 6:-6, :]
    im3 = im3[6:-6, 6:-6, :]
    im4 = im4[6:-6, 6:-6, :]

    print("adam:")
    print(compute_psnr(im2, im3, MAX_VAL))

    print("bicubic:")
    print(compute_psnr(im2, im1, MAX_VAL))

    print("SRCNN:")
    print(compute_psnr(im2, im4, MAX_VAL))
