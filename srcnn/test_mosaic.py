from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import numpy as np
import re

try:
    from .config import *
except Exception:
    from config import *

try:
    from .main import predict_model
except Exception:
    from main import predict_model

try:
    from .psnr import compute_psnr
except Exception:
    from psnr import compute_psnr

PATCH_SIZE_HR = 512
PATCH_SIZE_LR = 256


def read_band(path):
    img = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read band: {path}")
    return img.astype(np.uint16)


def extract_scene_and_patch(rgb_name):
    stem = Path(rgb_name).stem
    match = re.match(r"RGB_(.+)_(\d{4})$", stem)
    if not match:
        raise ValueError(f"Invalid RGB file name: {rgb_name}")
    return match.group(1), int(match.group(2))


def find_band(scene_id, band):
    for ext in ["tif", "TIF", "tiff", "TIFF"]:
        path = DATASET_DIR / f"{scene_id}_{band}.{ext}"
        if path.exists():
            print(f"Found band {band} for scene {scene_id}")
            return path

    raise FileNotFoundError(f"Band {band} not found for scene {scene_id}")

def binning2x2(image, offset_x=0, offset_y=0):
    """
    2x2 binning with optional offset.
    """
    image = image[offset_y:, offset_x:]
    n, m = image.shape[:2]

    n_even = (n // 2) * 2
    m_even = (m // 2) * 2
    image = image[:n_even, :m_even].astype(np.float32)

    low_img = (
        image[0::2, 0::2]
        + image[1::2, 0::2]
        + image[0::2, 1::2]
        + image[1::2, 1::2]
    ) / 4.0

    return low_img

def get_patch(image, patch_index, patch_size):
    h, w = image.shape[:2]

    patches_per_row = w // patch_size
    patches_per_col = h // patch_size
    total_patches = patches_per_row * patches_per_col

    if patch_index >= total_patches:
        raise IndexError(
            f"patch_index={patch_index} non valido. "
            f"Patch disponibili: {total_patches}"
        )

    row = patch_index // patches_per_row
    col = patch_index % patches_per_row

    y = row * patch_size
    x = col * patch_size

    patch = image[y:y + patch_size, x:x + patch_size]

    return patch, y, x


def build_bayer_mosaic_from_bands(b4, b5, bg, bg_shifted=None):
    """
    Bayer pattern:
        G R
        B G

    R = B5
    G = B8
    B = B4

    If bg_shifted is provided, it is used for the second G position.
    """
    b4 = b4.astype(np.float32)
    b5 = b5.astype(np.float32)
    bg = bg.astype(np.float32)

    if bg_shifted is None:
        bg_shifted = bg
    else:
        bg_shifted = bg_shifted.astype(np.float32)

    mosaic = np.zeros(b4.shape, dtype=np.float32)

    mosaic[0::2, 0::2] = bg[0::2, 0::2]               # G
    mosaic[0::2, 1::2] = b5[0::2, 1::2]               # R
    mosaic[1::2, 0::2] = b4[1::2, 0::2]               # B
    mosaic[1::2, 1::2] = bg_shifted[1::2, 1::2]       # G shifted if available

    return mosaic


def demosaic_bayer_gr(mosaic):
    """
    OpenCV Bayer demosaicing works better with uint16/uint8.
    Since our data is normalized in [0,1], convert to uint16 and back.
    """
    mosaic = np.clip(mosaic, 0.0, 1.0)

    mosaic_uint16 = (mosaic * 65535.0).astype(np.uint16)

    demosaiced_uint16 = cv2.cvtColor(
        mosaic_uint16,
        cv2.COLOR_BayerGR2RGB
    )

    # OpenCV result has R/B swapped with respect to our cube convention.
    # Align it back to [B5, B6, B4] = [R, G, B].
    demosaiced_uint16 = demosaiced_uint16[..., [2, 1, 0]]

    demosaiced = demosaiced_uint16.astype(np.float32) / 65535.0
    demosaiced = np.clip(demosaiced, 0.0, 1.0)

    return demosaiced, mosaic_uint16, demosaiced_uint16


def run_srcnn_on_mosaic_image(srcnn_model, demosaiced_rgb):
    demosaiced_rgb = np.clip(demosaiced_rgb, 0.0, 1.0).astype(np.float32)

    h, w = demosaiced_rgb.shape[:2]

    lr_img = cv2.resize(
        demosaiced_rgb,
        (w // scale, h // scale),
        interpolation=cv2.INTER_CUBIC
    )

    bicubic_img = cv2.resize(
        lr_img,
        (w, h),
        interpolation=cv2.INTER_CUBIC
    ).astype(np.float32)

    bicubic_img = np.clip(bicubic_img, 0.0, 1.0)

    X = np.expand_dims(bicubic_img, axis=0)

    residual_pred = srcnn_model.predict(X, batch_size=1, verbose=0)[0]

    bicubic_crop = bicubic_img[conv_side:-conv_side, conv_side:-conv_side, :]

    srcnn_pred = bicubic_crop + residual_pred
    srcnn_pred_clip = np.clip(srcnn_pred, 0.0, 1.0)

    return lr_img, bicubic_crop, srcnn_pred, srcnn_pred_clip

def save_mosaic_visualizations(
    out_dir,
    reference,
    lr_rgb_256,
    bicubic_crop,
    srcnn_pred_clip,
    psnr_bicubic,
    psnr_srcnn,
):

    
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    axs[0].imshow(reference)
    axs[0].set_title("Mosaic Reference")
    axs[0].axis("off")

    axs[1].imshow(lr_rgb_256)
    axs[1].set_title(f"Mosaic LR 256")
    axs[1].axis("off")

    axs[2].imshow(bicubic_crop)
    axs[2].set_title(f"Bicubic\nPSNR: {psnr_bicubic:.4f} dB")
    axs[2].axis("off")

    axs[3].imshow(srcnn_pred_clip)
    axs[3].set_title(f"SRCNN\nPSNR: {psnr_srcnn:.4f} dB")
    axs[3].axis("off")

    plt.tight_layout()

    results_path = out_dir / "results_mosaic.png"
    plt.savefig(results_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved: {results_path}")

    # =========================
    # MAPPE DI ERRORE
    # =========================
    error_bicubic = np.abs(reference - bicubic_crop).mean(axis=2)
    error_srcnn = np.abs(reference - srcnn_pred_clip).mean(axis=2)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    im1 = axs[0].imshow(error_bicubic, cmap="viridis")
    axs[0].set_title("Bicubic Error")
    axs[0].axis("off")
    fig.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04)

    im2 = axs[1].imshow(error_srcnn, cmap="viridis")
    axs[1].set_title("SRCNN Error")
    axs[1].axis("off")
    fig.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)

    plt.tight_layout()

    errors_path = out_dir / "error_maps_mosaic.png"
    plt.savefig(errors_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved: {errors_path}")


def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    input_path = TEST_RGB / "RGB_LC08_L1TP_193029_20260424_20260424_02_RT_0069.npy"
    scene_id, patch_index = extract_scene_and_patch(input_path)

    out_dir = ASSETS_DIR / "test_mosaic" / Path(input_path).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Scene:", scene_id)
    print("Patch index:", patch_index)

    b4_path = find_band(scene_id, "B4")
    b5_path = find_band(scene_id, "B5")
    b6_path = find_band(scene_id, "B6")

    b4 = read_band(b4_path)
    b5 = read_band(b5_path)
    b6 = read_band(b6_path)
    print("B4 shape:", b4.shape)
    print("B5 shape:", b5.shape)
    print("B6 shape:", b6.shape)

    # B6 has double resolution: remove first row, then bin 2x2
    #b6_cut = b6[1:, :]
    #b8_512_full = binning2x2(b8_cut, offset_x=0, offset_y=0)

    # HR/reference patches 512x512
    b4_512, _, _ = get_patch(b4, patch_index, PATCH_SIZE_HR)
    b5_512, _, _ = get_patch(b5, patch_index, PATCH_SIZE_HR)
    b6_512, _, _ = get_patch(b6, patch_index, PATCH_SIZE_HR)

    # Normalize HR bands globally
    b4_512 = np.clip(b4_512.astype(np.float32) / 65535.0, 0, 1)
    b5_512 = np.clip(b5_512.astype(np.float32) / 65535.0, 0, 1)
    b6_512 = np.clip(b6_512.astype(np.float32) / 65535.0, 0, 1)

    # HR mosaic reference 512x512
    mosaic_hr = build_bayer_mosaic_from_bands(
        b4=b4_512,
        b5=b5_512,
        bg=b6_512
    )

    mosaic_reference, _, _ = demosaic_bayer_gr(mosaic_hr)

    # LR 256x256 bands
    b4_256_full = binning2x2(b4, offset_x=0, offset_y=0)
    b4_256, _, _ = get_patch(b4_256_full, patch_index, PATCH_SIZE_LR)

    b5_256_full = binning2x2(b5, offset_x=1, offset_y=1)
    b5_256, _, _ = get_patch(b5_256_full, patch_index, PATCH_SIZE_LR)

    # B6 LR: two shifted low-resolution versions
    b6_256_full_a = binning2x2(b6, offset_x=0, offset_y=0)
    b6_256_full_b = binning2x2(b6, offset_x=1, offset_y=1)

    b6_256_a, _, _ = get_patch(b6_256_full_a, patch_index, PATCH_SIZE_LR)
    b6_256_b, _, _ = get_patch(b6_256_full_b, patch_index, PATCH_SIZE_LR)

    b4_256 = np.clip(b4_256.astype(np.float32) / 65535.0, 0, 1)
    b5_256 = np.clip(b5_256.astype(np.float32) / 65535.0, 0, 1)
    b6_256_a = np.clip(b6_256_a.astype(np.float32) / 65535.0, 0, 1)
    b6_256_b = np.clip(b6_256_b.astype(np.float32) / 65535.0, 0, 1)

    # LR mosaic 256x256
    mosaic_lr = build_bayer_mosaic_from_bands(
        b4=b4_256,
        b5=b5_256,
        bg=b6_256_a,
        bg_shifted=None
    )

    lr_rgb_256, _, _ = demosaic_bayer_gr(mosaic_lr)

    # Upscale LR 256 -> 512
    bicubic_img = cv2.resize(
        lr_rgb_256,
        (512, 512),
        interpolation=cv2.INTER_CUBIC
    ).astype(np.float32)

    bicubic_img = np.clip(bicubic_img, 0.0, 1.0)

    srcnn_model = predict_model()
    srcnn_model.load_weights(str(WEIGHTS_DIR / "train_global_norm_2M.h5"))
    print("Model loaded successfully.")

    X = np.expand_dims(bicubic_img, axis=0)
    residual_pred = srcnn_model.predict(X, batch_size=1, verbose=0)[0]

    reference = mosaic_reference[conv_side:-conv_side, conv_side:-conv_side, :]
    bicubic_crop = bicubic_img[conv_side:-conv_side, conv_side:-conv_side, :]

    srcnn_pred = bicubic_crop + residual_pred
    srcnn_pred_clip = np.clip(srcnn_pred, 0.0, 1.0)

    psnr_bicubic = compute_psnr(bicubic_crop, reference, max_val=1.0)
    psnr_srcnn = compute_psnr(srcnn_pred_clip, reference, max_val=1.0)
    
    np.save(out_dir / "mosaic_reference.npy", reference)
    np.save(out_dir / "lr_rgb_256.npy", lr_rgb_256)
    np.save(out_dir / "bicubic_crop.npy", bicubic_crop)
    np.save(out_dir / "srcnn_pred_clip.npy", srcnn_pred_clip)

    print("Saved outputs in:", out_dir)
    print(f"PSNR Bicubic vs MosaicRef: {psnr_bicubic:.4f} dB")
    print(f"PSNR SRCNN vs MosaicRef:   {psnr_srcnn:.4f} dB")
    print(f"Gain:                      {psnr_srcnn - psnr_bicubic:.4f} dB")

    save_mosaic_visualizations(
        out_dir=out_dir,
        reference=reference,
        lr_rgb_256=lr_rgb_256,
        bicubic_crop=bicubic_crop,
        srcnn_pred_clip=srcnn_pred_clip,
        psnr_bicubic=psnr_bicubic,
        psnr_srcnn=psnr_srcnn,
    )


    fig, axs = plt.subplots(1, 4, figsize=(16, 4))

    axs[0].imshow(b5_256, cmap="gray")
    axs[0].set_title("B5_256 (R)")
    axs[0].axis("off")

    axs[1].imshow(b6_256_a, cmap="gray")
    axs[1].set_title("B6_256_a (G1)")
    axs[1].axis("off")

    axs[2].imshow(b4_256, cmap="gray")
    axs[2].set_title("B4_256 (B)")
    axs[2].axis("off")

    axs[3].imshow(b6_256_b, cmap="gray")
    axs[3].set_title("B6_256_b (G2)")
    axs[3].axis("off")

    plt.tight_layout()
    plt.savefig(out_dir / "lr_bands_debug.png", dpi=200, bbox_inches="tight")
    plt.close()

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(b6_512, cmap="gray")
    axs[0].set_title("B6_512")
    axs[0].axis("off")

    axs[1].imshow(cv2.resize(b6_256_a, (512, 512), interpolation=cv2.INTER_NEAREST), cmap="gray")
    axs[1].set_title("B6_256_a upsampled")
    axs[1].axis("off")

    diff = np.abs(b6_512 - cv2.resize(b6_256_a, (512, 512), interpolation=cv2.INTER_LINEAR))
    axs[2].imshow(diff, cmap="hot")
    axs[2].set_title("Abs diff")
    axs[2].axis("off")

    plt.tight_layout()
    plt.savefig(out_dir / "b6_alignment_debug.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("B4_256", b4_256.shape, b4_256.mean(), b4_256.std())
    print("B5_256", b5_256.shape, b5_256.mean(), b5_256.std())
    print("B6_256_a", b6_256_a.shape, b6_256_a.mean(), b6_256_a.std())
    print("B6_256_b", b6_256_b.shape, b6_256_b.mean(), b6_256_b.std())
    print("LR rgb means:", lr_rgb_256.mean(axis=(0, 1)))
    print("Reference means:", reference.mean(axis=(0, 1)))

if __name__ == "__main__":
    main()