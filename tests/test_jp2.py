from pathlib import Path
import json
import cv2
import numpy as np

RGB_FOLDER = Path("/storage/internal_02/SRCNN/dataset_rgb")
REPORT_PATH = Path("assets/tif_input_shapes.json")


def find_tif_files(dataset_dir: Path) -> list[Path]:
    return sorted(dataset_dir.rglob("*.tif"))


def load_image(image_path: Path):
    return cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)


def channel_stats(image):
    stats = []

    if image.ndim == 2:
        image = image[:, :, None]

    for c in range(image.shape[2]):
        channel = image[:, :, c]

        stats.append({
            "channel": c,
            "min": int(channel.min()),
            "max": int(channel.max()),
            "mean": float(channel.mean()),
            "zero_pixels": int(np.sum(channel == 0)),
            "zero_ratio": float(np.mean(channel == 0)),
            "nonzero_pixels": int(np.sum(channel != 0)),
        })

    return stats


def main() -> int:
    dataset_dir = RGB_FOLDER

    if not dataset_dir.exists():
        print(f"Dataset directory not found: {dataset_dir}")
        return 1

    tif_files = find_tif_files(dataset_dir)

    if not tif_files:
        print(f"No .tif files found in: {dataset_dir}")
        return 1

    records = []
    suspicious_zero = []
    suspicious_over_4095 = []

    for image_path in tif_files:
        image = load_image(image_path)

        if image is None:
            records.append({
                "file_name": image_path.name,
                "loaded": False,
            })
            continue

        global_min = int(image.min())
        global_max = int(image.max())

        record = {
            "file_name": image_path.name,
            "loaded": True,
            "shape": list(image.shape),
            "dtype": str(image.dtype),
            "global_min": global_min,
            "global_max": global_max,
            "channel_stats": channel_stats(image),
        }

        records.append(record)

        if global_max == 0:
            suspicious_zero.append(image_path.name)

        if global_max > 4095:
            suspicious_over_4095.append(image_path.name)

        print(
            f"{image_path.name} | shape={image.shape} | dtype={image.dtype} | "
            f"min={global_min} | max={global_max}"
        )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with REPORT_PATH.open("w", encoding="utf-8") as fp:
        json.dump(records, fp, indent=2)

    print("\nSummary")
    print(f"Total files: {len(tif_files)}")
    print(f"All-zero patches: {len(suspicious_zero)}")
    print(f"Patches with max > 4095: {len(suspicious_over_4095)}")
    print(f"Report saved to: {REPORT_PATH}")

    if suspicious_zero:
        print("\nAll-zero examples:")
        for name in suspicious_zero[:20]:
            print(name)

    if suspicious_over_4095:
        print("\nOver-4095 examples:")
        for name in suspicious_over_4095[:20]:
            print(name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
