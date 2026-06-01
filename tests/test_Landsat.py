from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


ROOT = Path("/home/c.demartino/projects/SRCNN-keras")
ASSETS_DIR = ROOT / "assets"

reference = np.load(ASSETS_DIR / "reference.npy")
bicubic = np.load(ASSETS_DIR / "bicubic.npy")
srcnn = np.load(ASSETS_DIR / "srcnn_pred_clip.npy")

reference = np.clip(reference, 0, 1)
bicubic = np.clip(bicubic, 0, 1)
srcnn = np.clip(srcnn, 0, 1)

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