from pathlib import Path
import sys
import numpy
import cv2


DEFAULT_TIF_PATH = Path("LC08_L1TP_188028_20210218_20210302_02_T1_B5.TIF")


def main() -> int:
    tif_path = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else DEFAULT_TIF_PATH

    if not tif_path.exists():
        print(f"File not found: {tif_path}")
        return 1

    image = cv2.imread(str(tif_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Unable to load image: {tif_path}")
        return 1

    print(f"File: {tif_path}")
    print(f"Shape: {image.shape}")
    print(f"Dtype: {image.dtype}")
    print("min:", numpy.min(image))
    print("max:", numpy.max(image))
    print("unique sample:", numpy.unique(image[:100, :100])[:20])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
