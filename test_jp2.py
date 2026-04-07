from pathlib import Path
import sys
import json

import cv2


DATASET_DIR = Path("/storage/internal_02/SRCNN/dataset")
REPORT_PATH = Path("assets/jp2_input_shapes.json")


def find_jp2_files(dataset_dir: Path) -> list[Path]:
	return sorted(dataset_dir.rglob("*.jp2"))


def load_image(image_path: Path):
	return cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)


def save_shapes_report(records: list[dict], output_path: Path) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding="utf-8") as fp:
		json.dump(records, fp, indent=2)


def main() -> int:
	dataset_dir = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else DATASET_DIR
	report_path = Path(sys.argv[2]).expanduser() if len(sys.argv) > 2 else REPORT_PATH
	if not dataset_dir.exists():
		print(f"Dataset directory not found: {dataset_dir}")
		return 1

	jp2_files = find_jp2_files(dataset_dir)
	if not jp2_files:
		print(f"No .jp2 files found in: {dataset_dir}")
		return 1

	loaded_images = []
	failures = []
	report_records = []

	for idx, image_path in enumerate(jp2_files, start=1):
		image = load_image(image_path)
		if image is None:
			failures.append(image_path)
			print(f"FAILED: {image_path}")
			report_records.append({
				"file_name": image_path.name,
				"shape": None,
				"dtype": None,
			})
			continue

		loaded_images.append((image_path, image))
		print(f"LOADED: {image_path.name} shape={image.shape} dtype={image.dtype}")
		report_records.append({
			"file_name": image_path.name,
			"shape": list(image.shape),
			"dtype": str(image.dtype),
		})

	save_shapes_report(report_records, report_path)
	print(f"JSON report saved to: {report_path}")

	print(f"\nTotal files found: {len(jp2_files)}")
	print(f"Successfully loaded: {len(loaded_images)}")
	print(f"Failed to load: {len(failures)}")

	if failures:
		print("\nUnloaded files:")
		for image_path in failures:
			print(image_path)
		return 1

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
