
import math
import numpy as np
import cv2
import random
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim

try:
    from .config import *
except Exception:
    from config import *

def compute_psnr(target, ref, max_val = 1.0):

    target_data = np.array(target, dtype=np.float32)
    ref_data = np.array(ref, dtype=np.float32)

    # Skip images where the reference is mostly empty (e.g. all black)    
    valid_mask = np.mean(ref_data, axis=2) > 0.001
    if not np.any(valid_mask):
        return float("nan")

    #diff = ref_data - target_data
    #rmse = math.sqrt(np.mean(diff ** 2))    
    mse = np.mean((ref_data[valid_mask] - target_data[valid_mask]) ** 2)

    if mse == 0:
        return float("inf")
        
    return 20 * math.log10(max_val /  math.sqrt(mse))




if __name__ == "__main__":
    reference = np.load(ASSETS_DIR / "reference_w.npy")
    bicubic = np.load(ASSETS_DIR / "bicubic_w.npy")
    srcnn = np.load(ASSETS_DIR / "srcnn_pred_clip_w.npy")

    print("Shapes:")
    print("Reference:", reference.shape)
    print("Bicubic:", bicubic.shape)
    print("SRCNN:", srcnn.shape)
    
    psnr_bicubic = compute_psnr(bicubic, reference, max_val=1.0)
    psnr_srcnn = compute_psnr(srcnn, reference, max_val=1.0)

    print("\nPSNR Bicubic:")
    print(psnr_bicubic)

    print("\nPSNR SRCNN:")
    print(psnr_srcnn)

    print("\nGain:")
    print(psnr_srcnn - psnr_bicubic)

    sk_psnr_bicubic = peak_signal_noise_ratio(
        reference,
        bicubic,
        data_range=1.0
    )

    sk_psnr_srcnn = peak_signal_noise_ratio(
        reference,
        srcnn,
        data_range=1.0
    )

    print("\nskimage PSNR Bicubic:")
    print(sk_psnr_bicubic)

    print("\nskimage PSNR SRCNN:")
    print(sk_psnr_srcnn)

    print("\nskimage Gain:")
    print(sk_psnr_srcnn - sk_psnr_bicubic)

    valid_mask = np.mean(reference, axis=2) > 0.001
    mae_bicubic = np.mean(np.abs(reference[valid_mask] - bicubic[valid_mask]))
    mae_srcnn = np.mean(np.abs(reference[valid_mask] - srcnn[valid_mask]))
    mae_srcnn_vs_bicubic = np.mean(np.abs(srcnn[valid_mask] - bicubic[valid_mask]))

    print("MAE Bicubic vs Ref:", mae_bicubic)
    print("MAE SRCNN vs Ref:", mae_srcnn)
    print("MAE SRCNN vs Bicubic:", mae_srcnn_vs_bicubic)

    ssim_bicubic = ssim(
        reference,
        bicubic,
        channel_axis=2,
        data_range=1.0
    )

    ssim_srcnn = ssim(
        reference,
        srcnn,
        channel_axis=2,
        data_range=1.0
    )

    print("SSIM Bicubic:", ssim_bicubic)
    print("SSIM SRCNN:", ssim_srcnn)
    print("SSIM Gain:", ssim_srcnn - ssim_bicubic)