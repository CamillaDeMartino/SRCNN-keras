from pathlib import Path
import sys

import cv2
import numpy as np


def find_image_path() -> Path | None:
	if len(sys.argv) > 1:
		return Path(sys.argv[1]).expanduser()

	current_dir = Path.cwd()
	jp2_files = sorted(current_dir.rglob("*.jp2"))
	return jp2_files[0] if jp2_files else None


def main() -> int:
	image_path = find_image_path()
	if image_path is None:
		print("No .jp2 file found. Pass the image path as an argument.")
		return 1

	image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
	if image is None:
		print(f"Failed to open: {image_path}")
		print("OpenCV on this system may not support JP2, or the file may be corrupted.")
		return 1

	print(f"Opened: {image_path}")
	print(f"Shape: {image.shape}")
	print(f"Dtype: {image.dtype}")
	print(f"Bit depth: {image.dtype.itemsize * 8} bits per channel")
	print(f"Min pixel value: {image.min()}")
	print(f"Max pixel value: {image.max()}")
	print(f"Mean pixel value: {image.mean():.2f}")
	print(f"Std pixel value: {image.std():.2f}")

	percentiles = [1, 5, 50, 95, 99]
	percentile_values = {p: float(np.percentile(image, p)) for p in percentiles}
	for p in percentiles:
		print(f"Percentile {p}%: {percentile_values[p]:.2f}")

	if image.dtype == np.uint16:
		flat = image.reshape(-1)
		for bits in (1, 2, 3, 4, 8):
			remainder_mask = (1 << bits) - 1
			zero_ratio = float(np.mean((flat & remainder_mask) == 0))
			print(f"Zero ratio for lower {bits} bit(s): {zero_ratio:.4f}")

		lower_4_zero_ratio = float(np.mean((flat & 0x000F) == 0))
		if lower_4_zero_ratio > 0.95:
			print("Pattern suggests 12-bit data may be stored in a 16-bit container (lower 4 bits mostly zero).")
		elif lower_4_zero_ratio > 0.50:
			print("Pattern suggests partial 12-bit-like storage or mild scaling in a 16-bit container.")
		else:
			print("Pattern does not look like clean 12-bit data stored in a 16-bit container.")

	max_value = int(image.max())
	if max_value <= 255:
		likely_bits = 8
	elif max_value <= 4095:
		likely_bits = 12
	elif max_value <= 65535:
		likely_bits = 16
	else:
		likely_bits = image.dtype.itemsize * 8
	print(f"Likely original bit depth: {likely_bits} bits per channel")

	display_image = image
	if len(display_image.shape) == 2:
		display_image = cv2.cvtColor(display_image, cv2.COLOR_GRAY2BGR)
	max_width = 1400
	max_height = 900

	cv2.namedWindow("JP2 Image", cv2.WINDOW_NORMAL)
	cv2.resizeWindow("JP2 Image", max_width, max_height)
	#cv2.imshow("JP2 Image", display_image)
	#cv2.waitKey(0)
	cv2.destroyAllWindows()
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
