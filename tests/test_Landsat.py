from pathlib import Path
import sys
import json
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from srcnn.config import RGB_FOLDER, RGB_NORM

REPORT_PATH = Path("assets/dataset_quality_report.json")
EXPECTED_SHAPE = (512, 512, 3)


def check_raw_file(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if img is None:
        return {"ok": False, "error": "cannot_read"}

    return {
        "ok": True,
        "shape": list(img.shape),
        "dtype": str(img.dtype),
        "min": int(img.min()),
        "max": int(img.max()),
        "mean": float(img.mean()),
        "zero_ratio": float(np.mean(img == 0)),
        "valid_shape": img.shape == EXPECTED_SHAPE,
        "valid_dtype": img.dtype == np.uint16,
        "all_zero": bool(img.max() == 0),
        "channel_stats": [
            {
                "channel": c,
                "min": int(img[:, :, c].min()),
                "max": int(img[:, :, c].max()),
                "mean": float(img[:, :, c].mean()),
                "zero_ratio": float(np.mean(img[:, :, c] == 0)),
            }
            for c in range(img.shape[-1])
        ],
    }


def check_norm_file(path: Path):
    arr = np.load(path)

    finite = np.isfinite(arr)

    return {
        "ok": True,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "valid_shape": arr.shape == EXPECTED_SHAPE,
        "valid_dtype": arr.dtype == np.float32,
        "has_nan": bool(np.isnan(arr).any()),
        "has_inf": bool(np.isinf(arr).any()),
        "inside_0_1": bool(arr.min() >= 0.0 and arr.max() <= 1.0),
        "all_zero": bool(arr.max() == 0.0),
        "finite_ratio": float(np.mean(finite)),
        "channel_stats": [
            {
                "channel": c,
                "min": float(arr[:, :, c].min()),
                "max": float(arr[:, :, c].max()),
                "mean": float(arr[:, :, c].mean()),
                "zero_ratio": float(np.mean(arr[:, :, c] == 0)),
            }
            for c in range(arr.shape[-1])
        ],
    }


def main():
    empty_count = 0
    raw_files = sorted(RGB_FOLDER.glob("*.tif"))
    norm_files = sorted(RGB_NORM.glob("*.npy"))

    raw_stems = {p.stem for p in raw_files}
    norm_stems = {p.stem for p in norm_files}

    missing_norm = sorted(raw_stems - norm_stems)
    extra_norm = sorted(norm_stems - raw_stems)

    report = {
        "raw_count": len(raw_files),
        "norm_count": len(norm_files),
        "missing_norm_count": len(missing_norm),
        "extra_norm_count": len(extra_norm),
        "missing_norm_examples": missing_norm[:20],
        "extra_norm_examples": extra_norm[:20],
        "files": [],
    }

    problems = []

    for raw_path in raw_files:
        norm_path = RGB_NORM / f"{raw_path.stem}.npy"

        raw_check = check_raw_file(raw_path)

        if norm_path.exists():
            norm_check = check_norm_file(norm_path)
        else:
            norm_check = {"ok": False, "error": "missing_norm_file"}

        file_record = {
            "name": raw_path.name,
            "raw": raw_check,
            "normalized": norm_check,
        }

        report["files"].append(file_record)

        if not raw_check.get("ok", False):
            problems.append((raw_path.name, "raw_not_readable"))

        elif not raw_check["valid_shape"]:
            problems.append((raw_path.name, "raw_invalid_shape"))

        elif not raw_check["valid_dtype"]:
            problems.append((raw_path.name, "raw_invalid_dtype"))

        elif raw_check["all_zero"]:
            empty_count += 1  # patch vuota: nota ma non considerata errore
        if not norm_check.get("ok", False):
            problems.append((raw_path.name, "norm_missing_or_not_readable"))

        elif not norm_check["valid_shape"]:
            problems.append((raw_path.name, "norm_invalid_shape"))

        elif not norm_check["valid_dtype"]:
            problems.append((raw_path.name, "norm_invalid_dtype"))

        elif norm_check["has_nan"] or norm_check["has_inf"]:
            problems.append((raw_path.name, "norm_nan_or_inf"))

        elif not norm_check["inside_0_1"]:
            problems.append((raw_path.name, "norm_outside_0_1"))

        elif norm_check["all_zero"]:
            problems.append((raw_path.name, "norm_all_zero"))

    report["problem_count"] = len(problems)
    report["problem_examples"] = problems[:50]
    report["empty_patch_count"] = empty_count

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with REPORT_PATH.open("w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)

    print("Dataset quality check completed")
    print(f"Raw files: {len(raw_files)}")
    print(f"Normalized files: {len(norm_files)}")
    print(f"Missing normalized files: {len(missing_norm)}")
    print(f"Extra normalized files: {len(extra_norm)}")
    print(f"Problems found: {len(problems)}")
    print(f"Report saved to: {REPORT_PATH}")

    if problems:
        print("\nProblem examples:")
        for name, reason in problems[:30]:
            print(f"{name}: {reason}")


if __name__ == "__main__":
    main()