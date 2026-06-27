import argparse
import json
import shutil
import sys
from pathlib import Path

# this code rectifies all the img in test_images only, it currently doesn't call gemini. Not the full pipeline

SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(SRC_ROOT))

from detect.yolo_world_save_crops import save_yolo_crops


from preprocess.rectify_from_yolo_box import crop_regions_from_detections

def make_image_output_dirs(image_path, output_root):
    image_path = Path(image_path)
    output_root = Path(output_root)

    image_name = image_path.stem
    image_output_dir = output_root / image_name

    crops_dir = image_output_dir / "crops"
    rectified_dir = image_output_dir / "rectified"

    image_output_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)
    rectified_dir.mkdir(parents=True, exist_ok=True)

    original_copy_path = image_output_dir / image_path.name
    if not original_copy_path.exists():
        shutil.copy(image_path, original_copy_path)

    return {
        "image_output_dir": image_output_dir,
        "crops_dir": crops_dir,
        "rectified_dir": rectified_dir,
        "results_path": image_output_dir / "gemini_results.json"
    }


def get_image_paths(input_dir):
    input_dir = Path(input_dir)

    image_paths = []
    image_paths.extend(input_dir.glob("*.jpg"))
    image_paths.extend(input_dir.glob("*.JPG"))
    image_paths.extend(input_dir.glob("*.png"))
    image_paths.extend(input_dir.glob("*.PNG"))
    image_paths.extend(input_dir.glob("*.jpeg"))
    image_paths.extend(input_dir.glob("*.JPEG"))

    return sorted(image_paths)


def run_single_image(image_path, output_root):
    dirs = make_image_output_dirs(image_path, output_root)

    print("\n==============================")
    print(f"Processing image: {image_path}")
    print("==============================")
    print(f"Image folder: {dirs['image_output_dir']}")
    print(f"Crops folder: {dirs['crops_dir']}")
    print(f"Rectified folder: {dirs['rectified_dir']}")

    # Step 1 later: call YOLO-World here
    crop_paths, detections = save_yolo_crops(
        image_path=image_path,
        output_dir=dirs["crops_dir"]
    )

    # Step 2 later: call OpenCV rectification here
    rectified_paths = crop_regions_from_detections(
        image_path=image_path,
        detections=detections,
        output_dir=dirs["rectified_dir"]
    )

    # Step 3 later: call Gemini here
    gemini_results = []

    results = {
        "input_image": str(image_path),
        "detections": detections,
        "crop_paths": [str(p) for p in crop_paths],
        "rectified_paths": [str(p) for p in rectified_paths],
        "gemini_results": gemini_results
    }

    with open(dirs["results_path"], "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved: {dirs['results_path']}")


def run_pipeline(input_dir, output_root):
    image_paths = get_image_paths(input_dir)

    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_paths)} images in {input_dir}")

    for image_path in image_paths:
        run_single_image(image_path, output_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        default="data/test_images",
        help="Folder containing input images"
    )

    parser.add_argument(
        "--output-root",
        default="outputs/pipeline",
        help="Root folder for pipeline outputs"
    )

    args = parser.parse_args()

    run_pipeline(
        input_dir=args.input_dir,
        output_root=args.output_root
    )