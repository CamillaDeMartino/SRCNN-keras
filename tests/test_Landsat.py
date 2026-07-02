from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


ROOT = Path("/home/c.demartino/projects/SRCNN-keras")
ASSETS_DIR = ROOT / "assets"


def masked_psnr(target, ref, threshold=0.001, max_val=1.0):
    target = target.astype(np.float32)
    ref = ref.astype(np.float32)

    valid_mask = np.mean(ref, axis=2) > threshold

    if not np.any(valid_mask):
        return float("nan")

    mse = np.mean((ref[valid_mask] - target[valid_mask]) ** 2)

    if mse == 0:
        return float("inf")

    return 20 * np.log10(max_val / np.sqrt(mse))


def valid_fraction(img, threshold=0.001):
    valid_mask = np.mean(img, axis=2) > threshold
    return np.mean(valid_mask)

if __name__ == "__main__":

    reference = np.load(ASSETS_DIR / "reference_b.npy")
    bicubic = np.load(ASSETS_DIR / "bicubic_b.npy")
    srcnn = np.load(ASSETS_DIR / "srcnn_pred_clip_b.npy")

    reference = np.clip(reference, 0, 1)
    bicubic = np.clip(bicubic, 0, 1)
    srcnn = np.clip(srcnn, 0, 1)

    psnr_bic_full = masked_psnr(bicubic, reference)
    psnr_src_full = masked_psnr(srcnn, reference)

    print("\n=== FULL IMAGE MASKED PSNR ===")
    print(f"Bicubic: {psnr_bic_full:.4f} dB")
    print(f"SRCNN:   {psnr_src_full:.4f} dB")
    print(f"Gain:    {psnr_src_full - psnr_bic_full:.4f} dB")
    print(f"Valid fraction full: {valid_fraction(reference) * 100:.2f}%")

    OUT = ASSETS_DIR / "zoom_crops"
    OUT.mkdir(exist_ok=True)

    error_bicubic = np.mean(np.abs(reference - bicubic), axis=2)
    error_srcnn = np.mean(np.abs(reference - srcnn), axis=2)

    gain = error_bicubic - error_srcnn

    margin = 80

    valid_gain = gain.copy()
    valid_gain[:margin, :] = -np.inf
    valid_gain[-margin:, :] = -np.inf
    valid_gain[:, :margin] = -np.inf
    valid_gain[:, -margin:] = -np.inf

    valid_mask = np.mean(reference, axis=2) > 0.001
    valid_gain[~valid_mask] = -np.inf

    y, x = np.unravel_index(np.argmax(valid_gain), valid_gain.shape)

    print("Best internal improvement pixel:", y, x)


    crop_size = 120
    half = crop_size // 2

    y1 = max(0, y - half)
    y2 = min(reference.shape[0], y + half)
    x1 = max(0, x - half)
    x2 = min(reference.shape[1], x + half)

    crop = (slice(y1, y2), slice(x1, x2))

    # # crop agricolo centrale
    # crop = (
    #     slice(120, 280),
    #     slice(60, 220)
    # )

    ref_crop = reference[crop]
    bic_crop = bicubic[crop]
    src_crop = srcnn[crop]

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    ax[0].imshow(ref_crop)
    ax[0].set_title("Reference")

    ax[1].imshow(bic_crop)
    ax[1].set_title("Bicubic")

    ax[2].imshow(src_crop)
    ax[2].set_title("SRCNN")

    for a in ax:
        a.axis("off")

    plt.tight_layout()

    out_file = OUT / "crop_agriculture.png"
    plt.savefig(out_file, dpi=400, bbox_inches="tight")
    plt.close()

    print("Saved:", out_file)

    psnr_bic_crop = masked_psnr(bic_crop, ref_crop)
    psnr_src_crop = masked_psnr(src_crop, ref_crop)

    print("\n=== CROP MASKED PSNR ===")
    print(f"Bicubic crop: {psnr_bic_crop:.4f} dB")
    print(f"SRCNN crop:   {psnr_src_crop:.4f} dB")
    print(f"Gain crop:    {psnr_src_crop - psnr_bic_crop:.4f} dB")
    print(f"Valid fraction crop: {valid_fraction(ref_crop) * 100:.2f}%")