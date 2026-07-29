"""Controlled still-image ablations for the TRB navigation-sign study.

This driver deliberately leaves ``run_multiview_pipeline.py`` unchanged. It
freezes the benchmark and YOLO detections, creates aligned raw/rectified views,
and varies only which views are sent to Gemini.
"""

# python src/pipeline/run_still_ablation.py \
#   --variant original_raw_no_annotation \
#   --cache-dir outputs/still_ablation/full/cache \
#   --results-dir outputs/still_ablation/full/results \
#   --resume

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(SRC_ROOT))

SCHEMA_VERSION = 1
VARIANTS = (
    "original_only",
    "raw_crop_multiview",
    "rectified_crops_only",
    "full_multiview",
)
ADDITIONAL_VARIANTS = (
    "raw_crops_only",
    "original_rectified_no_annotation",
    "original_raw_no_annotation",
)
SUPPORTED_VARIANTS = (*VARIANTS, *ADDITIONAL_VARIANTS)
EXCLUDED_IMAGES = ("IMG_0001.JPG", "IMG_3499.JPG", "IMG_3500.JPG")
EXPECTED_IMAGE_COUNT = 346
EXPECTED_QUESTION_COUNT = 1100
DETECTOR_CONFIGURATION = {
    "implementation": "detect.yolo_world_save_crops.save_yolo_crops",
    "model": "yolov8s-world.pt",
    "confidence": 0.05,
    "duplicate_iou_threshold": 0.6,
}
RECTIFICATION_CONFIGURATION = {
    "implementation": "preprocess.rectify_from_yolo_box.crop_regions_from_detections",
    "model": "MobileSAM vit_t",
    "checkpoint": "weights/mobile_sam/mobile_sam.pt",
}
SCIENTIFIC_LIMITATION = (
    "The original_only -> raw_crop_multiview comparison measures the combined "
    "effect of sign detection, localization, crop magnification, annotation, "
    "and multiview presentation. It does not isolate those subcomponents "
    "individually."
)
VARIANT_METADATA = {
    "original_only": {
        "label": "Original only - no preprocessing baseline",
        "hypothesis": (
            "Sign localization and close-up views improve recognition beyond "
            "Gemini's native ability to inspect the full scene."
        ),
        "components": {
            "original_scene": True,
            "annotated_scene": False,
            "yolo_localization": False,
            "raw_crops": False,
            "rectification": False,
        },
    },
    "raw_crop_multiview": {
        "label": "Original plus raw detected crops - localization without rectification",
        "hypothesis": (
            "Any improvement from full_multiview over this condition is "
            "attributable to rectification rather than detection and crop "
            "magnification alone."
        ),
        "components": {
            "original_scene": True,
            "annotated_scene": True,
            "yolo_localization": True,
            "raw_crops": True,
            "rectification": False,
        },
    },
    "rectified_crops_only": {
        "label": "Rectified crops only - local detail without scene context",
        "hypothesis": (
            "The annotated full-scene image contributes global context beyond "
            "the information in rectified close-up views."
        ),
        "components": {
            "original_scene": False,
            "annotated_scene": False,
            "yolo_localization": True,
            "raw_crops": False,
            "rectification": True,
        },
    },
    "full_multiview": {
        "label": "Original plus rectified crops - complete proposed method",
        "hypothesis": (
            "Combining global scene context with detected, rectified close-up "
            "views produces the strongest navigation-sign understanding."
        ),
        "components": {
            "original_scene": True,
            "annotated_scene": True,
            "yolo_localization": True,
            "raw_crops": False,
            "rectification": True,
        },
    },
    "raw_crops_only": {
        "label": "Raw crops only - localized detail without scene context",
        "hypothesis": (
            "Raw detected crops test localized close-up evidence without the "
            "original scene, annotation, or perspective rectification."
        ),
        "components": {
            "original_scene": False,
            "annotated_scene": False,
            "yolo_localization": True,
            "raw_crops": True,
            "rectification": False,
        },
    },
    "original_rectified_no_annotation": {
        "label": "Original plus rectified crops - no annotation correspondence",
        "hypothesis": (
            "The unannotated scene and rectified close-ups test complementary "
            "global and local evidence without explicit numbered correspondence."
        ),
        "components": {
            "original_scene": True,
            "annotated_scene": False,
            "yolo_localization": True,
            "raw_crops": False,
            "rectification": True,
        },
    },
    "original_raw_no_annotation": {
        "label": "Original plus raw crops - no annotation correspondence",
        "hypothesis": (
            "The unannotated scene and raw close-ups test complementary global "
            "and local evidence without rectification or explicit numbered "
            "correspondence."
        ),
        "components": {
            "original_scene": True,
            "annotated_scene": False,
            "yolo_localization": True,
            "raw_crops": True,
            "rectification": False,
        },
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    os.replace(temporary, path)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def project_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def build_frozen_manifest(
    image_dir: Path,
    qa_file: Path,
    manifest_path: Path,
    *,
    write: bool,
) -> dict[str, Any]:
    qa_data = load_json(qa_file)
    if not isinstance(qa_data, list):
        raise ValueError("QA file must contain a JSON list.")

    seen: set[str] = set()
    images = []
    question_count = 0
    excluded = set(EXCLUDED_IMAGES)

    for qa_item in qa_data:
        image_name = qa_item["imagePath"]
        if image_name in seen:
            raise ValueError(f"Duplicate QA image entry: {image_name}")
        seen.add(image_name)
        if image_name in excluded:
            continue

        image_path = image_dir / image_name
        if not image_path.is_file():
            raise FileNotFoundError(f"QA image is missing: {image_path}")

        ordered_questions = []
        for question_item in qa_item.get("questions", []):
            ordered_questions.append(
                {
                    "question": question_item["question"],
                    "expected": question_item["answer"],
                }
            )
        if not ordered_questions:
            raise ValueError(f"QA image has no questions: {image_name}")

        images.append(
            {
                "imagePath": image_name,
                "sha256": sha256_file(image_path),
                "questions": ordered_questions,
            }
        )
        question_count += len(ordered_questions)

    if len(images) != EXPECTED_IMAGE_COUNT or question_count != EXPECTED_QUESTION_COUNT:
        raise ValueError(
            "Frozen benchmark invariant failed: "
            f"found {len(images)} images/{question_count} questions; expected "
            f"{EXPECTED_IMAGE_COUNT}/{EXPECTED_QUESTION_COUNT}."
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "TRB still-image navigation-sign ablation",
        "image_root": project_relative(image_dir),
        "qa_file": project_relative(qa_file),
        "excluded_images": list(EXCLUDED_IMAGES),
        "ordering": "QA file image order and per-image question order",
        "image_count": len(images),
        "question_count": question_count,
        "images": images,
    }
    manifest["manifest_content_sha256"] = canonical_json_hash(manifest)
    if write:
        write_json_atomic(manifest_path, manifest)
    return manifest


def validate_manifest(
    manifest_path: Path,
    image_dir: Path,
    qa_file: Path,
    *,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    stored_hash = manifest.get("manifest_content_sha256")
    hash_input = {k: v for k, v in manifest.items() if k != "manifest_content_sha256"}
    if stored_hash != canonical_json_hash(hash_input):
        raise ValueError("Manifest content hash does not match its contents.")

    rebuilt = build_frozen_manifest(image_dir, qa_file, manifest_path, write=False)
    if manifest != rebuilt:
        raise ValueError(
            "Frozen manifest differs from the current verified QA/image intersection."
        )

    if verify_hashes:
        for item in manifest["images"]:
            image_path = image_dir / item["imagePath"]
            if sha256_file(image_path) != item["sha256"]:
                raise ValueError(f"Image hash mismatch: {item['imagePath']}")
    return manifest


@lru_cache(maxsize=1)
def cache_configuration() -> dict[str, Any]:
    files = {
        "detector_code": PROJECT_ROOT / "src/detect/yolo_world_save_crops.py",
        "detector_weights": PROJECT_ROOT / "yolov8s-world.pt",
        "rectification_code": PROJECT_ROOT / "src/preprocess/rectify_from_yolo_box.py",
        "rectification_weights": PROJECT_ROOT / "weights/mobile_sam/mobile_sam.pt",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Preprocessing dependency files are missing: {missing}")
    return {
        "detector": DETECTOR_CONFIGURATION,
        "rectification": RECTIFICATION_CONFIGURATION,
        "file_sha256": {name: sha256_file(path) for name, path in files.items()},
    }


def cache_configuration_hash() -> str:
    return canonical_json_hash(cache_configuration())


def detection_index_from_path(path: Path, kind: str) -> int | None:
    patterns = {
        "raw": r"^crop_(\d+)_",
        "rectified": r"^rectified_(\d+)_",
    }
    match = re.match(patterns[kind], path.name)
    return int(match.group(1)) if match else None


def relative_cache_path(path: Path, cache_dir: Path) -> str:
    return path.resolve().relative_to(cache_dir.resolve()).as_posix()


def prepare_image_cache(
    image_item: dict[str, Any], image_dir: Path, cache_dir: Path, *, overwrite: bool
) -> dict[str, Any]:
    image_name = image_item["imagePath"]
    image_path = image_dir / image_name
    image_cache_dir = cache_dir / "images" / Path(image_name).stem
    cache_record_path = image_cache_dir / "detections.json"

    if cache_record_path.is_file() and not overwrite:
        record = load_json(cache_record_path)
        if (
            record.get("image_sha256") == image_item["sha256"]
            and record.get("cache_configuration_sha256") == cache_configuration_hash()
        ):
            return record
        raise ValueError(
            f"Stale preprocessing cache for {image_name}; rerun with --overwrite."
        )

    raw_dir = image_cache_dir / "raw_crops"
    rectified_dir = image_cache_dir / "rectified_crops"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rectified_dir.mkdir(parents=True, exist_ok=True)

    # Heavy vision dependencies are imported only when preprocessing is requested.
    from detect.yolo_world_save_crops import save_yolo_crops
    from preprocess.rectify_from_yolo_box import crop_regions_from_detections
    from src.pipeline.run_multiview_pipeline import create_annotated_original_image

    yolo_result = save_yolo_crops(image_path=image_path, output_dir=raw_dir)
    if not isinstance(yolo_result, tuple) or len(yolo_result) != 2:
        raise RuntimeError(f"YOLO failed to return crops and detections for {image_name}")
    raw_paths, detections = yolo_result

    rectified_paths = crop_regions_from_detections(
        image_path=image_path,
        detections=detections,
        output_dir=rectified_dir,
    )
    annotated_path = image_cache_dir / "annotated_original.jpg"
    create_annotated_original_image(image_path, detections, annotated_path)

    raw_by_index = {
        index: path
        for path in map(Path, raw_paths)
        if (index := detection_index_from_path(path, "raw")) is not None
    }
    rectified_by_index = {
        index: path
        for path in map(Path, rectified_paths)
        if (index := detection_index_from_path(path, "rectified")) is not None
    }

    aligned = []
    for detection in detections:
        index = int(detection["index"])
        raw_path = raw_by_index.get(index)
        rectified_path = rectified_by_index.get(index)
        aligned.append(
            {
                "detection_index": index,
                "box": detection["box"],
                "label": detection["label"],
                "confidence": detection["confidence"],
                "raw_crop_success": raw_path is not None,
                "raw_crop_path": (
                    relative_cache_path(raw_path, cache_dir) if raw_path else None
                ),
                "rectification_success": rectified_path is not None,
                "rectified_crop_path": (
                    relative_cache_path(rectified_path, cache_dir)
                    if rectified_path
                    else None
                ),
            }
        )

    record = {
        "schema_version": SCHEMA_VERSION,
        "imagePath": image_name,
        "image_sha256": image_item["sha256"],
        "cache_configuration_sha256": cache_configuration_hash(),
        "annotated_original_path": relative_cache_path(annotated_path, cache_dir),
        "detection_count": len(detections),
        "raw_crop_count": sum(item["raw_crop_success"] for item in aligned),
        "successful_rectification_count": sum(
            item["rectification_success"] for item in aligned
        ),
        "detections": aligned,
    }
    write_json_atomic(cache_record_path, record)
    return record


def load_image_cache(
    image_item: dict[str, Any], cache_dir: Path
) -> dict[str, Any] | None:
    path = cache_dir / "images" / Path(image_item["imagePath"]).stem / "detections.json"
    if not path.is_file():
        return None
    record = load_json(path)
    if (
        record.get("image_sha256") != image_item["sha256"]
        or record.get("cache_configuration_sha256") != cache_configuration_hash()
    ):
        raise ValueError(f"Stale preprocessing cache: {image_item['imagePath']}")
    return record


def resolve_cache_artifact(cache_dir: Path, relative_path: str | None) -> Path | None:
    if relative_path is None:
        return None
    path = cache_dir / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Cached artifact is missing: {path}")
    return path


@dataclass
class VariantViews:
    paths: list[Path]
    detection_indices: list[int]
    note: str | None


def select_variant_views(
    variant: str,
    image_path: Path,
    cache_record: dict[str, Any] | None,
    cache_dir: Path,
) -> VariantViews:
    if variant == "original_only":
        return VariantViews([image_path], [], None)
    if cache_record is None:
        raise ValueError(f"Variant {variant} requires a preprocessing cache.")

    detections = sorted(cache_record["detections"], key=lambda item: item["detection_index"])

    if variant == "raw_crops_only":
        crops = [item for item in detections if item["raw_crop_success"]]
        paths = [
            resolve_cache_artifact(cache_dir, item["raw_crop_path"])
            for item in crops
        ]
        note = (
            "You are given unrectified close-up views of navigation signs detected "
            "in one scene. The views are ordered by detector box index. Reason "
            "jointly across all supplied sign views."
        )
        return VariantViews(paths, [item["detection_index"] for item in crops], note)

    if variant == "original_raw_no_annotation":
        crops = [item for item in detections if item["raw_crop_success"]]
        raw_paths = [
            resolve_cache_artifact(cache_dir, item["raw_crop_path"])
            for item in crops
        ]
        note = (
            "You are given multiple views of the same scene. Image 1 is the "
            "unmodified original scene without detection annotations. The remaining "
            "images are raw (non-rectified) close-up views of navigation signs "
            "detected in that scene. Use Image 1 for global scene context and the "
            "close-up views for local sign detail. Reason jointly across all "
            "supplied images."
        )
        return VariantViews(
            [image_path] + raw_paths,
            [item["detection_index"] for item in crops],
            note,
        )

    annotated = resolve_cache_artifact(
        cache_dir, cache_record["annotated_original_path"]
    )

    if variant == "raw_crop_multiview":
        crops = [item for item in detections if item["raw_crop_success"]]
        paths = [annotated] + [
            resolve_cache_artifact(cache_dir, item["raw_crop_path"]) for item in crops
        ]
        mapping = "\n".join(
            f"Image {position + 2} corresponds to the annotated box labeled "
            f"Image {item['detection_index'] + 2}."
            for position, item in enumerate(crops)
        )
        note = (
            "You are given multiple views of the same scene. Image 1 is the "
            "original scene annotated with detected navigation-sign boxes. "
            "Each remaining image is an unrectified close-up of a detected box.\n"
            f"{mapping}\nReason jointly across all images."
        )
        return VariantViews(paths, [item["detection_index"] for item in crops], note)

    rectified = [item for item in detections if item["rectification_success"]]
    rectified_paths = [
        resolve_cache_artifact(cache_dir, item["rectified_crop_path"])
        for item in rectified
    ]
    indices = [item["detection_index"] for item in rectified]

    if variant == "rectified_crops_only":
        note = (
            "You are given rectified close-up views of navigation signs detected "
            "in one scene. The views are ordered by detector box index. Reason "
            "jointly across all supplied sign views."
        )
        return VariantViews(rectified_paths, indices, note)

    if variant == "original_rectified_no_annotation":
        note = (
            "You are given multiple views of the same scene. Image 1 is the "
            "unmodified original scene without detection annotations. The remaining "
            "images are rectified close-up views of navigation signs detected in "
            "that scene. Use Image 1 for global scene context and the close-up views "
            "for local sign detail. Reason jointly across all supplied images."
        )
        return VariantViews([image_path] + rectified_paths, indices, note)

    if variant == "full_multiview":
        crop_descriptions = "\n".join(
            f"Image {position + 2} is the rectified version of the region "
            f"labeled Image {item['detection_index'] + 2} in Image 1."
            for position, item in enumerate(rectified)
        )
        note = (
            "You are given multiple views of the SAME scene.\n\n"
            "Image 1 is the ORIGINAL image with bounding boxes drawn around each "
            "detected navigational sign. Each box is labeled Crop 1, Crop 2, etc.\n\n"
            "The remaining images are rectified close-up views of those labeled boxes:\n"
            f"{crop_descriptions}\n\n"
            "Use Image 1 to understand the overall scene layout, spatial relationships, "
            "and which sign each crop belongs to.\n"
            "Use the rectified crop images to read small text and determine arrow "
            "directions more accurately.\n"
            "The rectified images are NOT independent signs. Each one is simply a "
            "higher-quality view of the corresponding labeled region in Image 1.\n"
            "Reason jointly across ALL images before answering the questions."
        )
        return VariantViews([annotated] + rectified_paths, indices, note)

    raise ValueError(f"Unknown variant: {variant}")


def build_prompt(qa_prompt: str, questions: list[str], note: str | None) -> str:
    sections = [qa_prompt]
    if note:
        sections.append(note)
    sections.append("Questions:\n" + "\n".join(f"- {question}" for question in questions))
    return "\n\n".join(sections)


def ask_variant_with_retries(
    qa: Any,
    prompt: str,
    image_paths: list[Path],
) -> dict[str, Any]:
    # Importing the canonical helper also imports the vision stack. Keep it out
    # of manifest and dry-run commands, which intentionally need no ML runtime.
    from src.pipeline.run_multiview_pipeline import get_image_string

    start = time.monotonic()
    attempts = 0
    parse_failures = 0
    errors = []
    raw_response = None
    parsed_response = None

    while True:
        attempts += 1
        try:
            content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            for image_path in image_paths:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": get_image_string(image_path),
                            "detail": "high",
                        },
                    }
                )
            completion = qa.client.chat.completions.create(
                model=qa.model_version,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant capable of understanding "
                            "navigational signs."
                        ),
                    },
                    {"role": "user", "content": content},
                ],
                n=1,
                temperature=0,
            )
            raw_response = completion.choices[0].message.content
            parsed_response = qa.parse_response(raw_response)
            if parsed_response is not None:
                print(f"Gemini multiview succeeded on attempt {attempts}")
                break
            parse_failures += 1
            sleep_time = random.randint(15, 60)
            print(
                f"Attempt {attempts} failed to parse. "
                f"Retrying in {sleep_time} seconds..."
            )
        except Exception as exc:  # API errors must be retained in experiment output.
            errors.append(
                {"attempt": attempts, "error": f"{type(exc).__name__}: {exc}"}
            )
            sleep_time = random.randint(15, 60)
            print(f"Gemini error on attempt {attempts}: {exc}")
            print(f"Retrying Gemini in {sleep_time} seconds...")
        time.sleep(sleep_time)

    return {
        "raw_response": raw_response,
        "parsed_response": parsed_response,
        "gemini_attempts": attempts,
        "parse_failures": parse_failures,
        "errors": errors,
        "latency_seconds": time.monotonic() - start,
    }


def preprocessing_metadata(
    variant: str, cache_record: dict[str, Any] | None, views: VariantViews
) -> dict[str, Any]:
    if cache_record is None:
        return {
            "detection_count": 0,
            "raw_crop_count": 0,
            "successful_rectification_count": 0,
            "view_count": len(views.paths),
            "view_detection_indices": [],
            "cache_available": False,
            "cache_used_for_views": False,
        }
    return {
        "detection_count": cache_record["detection_count"],
        "raw_crop_count": cache_record["raw_crop_count"],
        "successful_rectification_count": cache_record[
            "successful_rectification_count"
        ],
        "view_count": len(views.paths),
        "view_detection_indices": views.detection_indices,
        "cache_available": True,
        "cache_used_for_views": variant != "original_only",
        "cache_configuration_sha256": cache_record[
            "cache_configuration_sha256"
        ],
    }


def experiment_metadata(
    variant: str,
    manifest_path: Path,
    manifest: dict[str, Any],
    model_version: str,
    prompt_file: Path,
    retry_count: int,
) -> dict[str, Any]:
    return {
        "variant": variant,
        **VARIANT_METADATA[variant],
        "scientific_limitations": [SCIENTIFIC_LIMITATION],
        "manifest_path": project_relative(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_content_sha256": manifest["manifest_content_sha256"],
        "model": model_version,
        "temperature": 0,
        "prompt_path": project_relative(prompt_file),
        "prompt_sha256": sha256_file(prompt_file),
        "configured_retry_count": retry_count,
        "retry_policy": (
            "Retry indefinitely until the response parses successfully; the "
            "configured retry count is retained for canonical CLI compatibility "
            "but is not used as a limit."
        ),
        "retry_delay_seconds": "random integer from 15 through 60",
        "logical_request_policy": "one request per image; retries are attempts",
        "voting": False,
        "paddleocr": False,
        "tracking": False,
        "cropwise_calls": False,
        "original_image_fallback": False,
    }


def summarize_results(results: list[dict[str, Any]], expected_images: int) -> dict[str, Any]:
    questions = [question for image in results for question in image["questions"]]
    requests = [image["request"] for image in results]
    return {
        "images_expected": expected_images,
        "images_completed": len(results),
        "total_questions": len(questions),
        "total_correct": sum(bool(question["correct"]) for question in questions),
        "accuracy": (
            sum(bool(question["correct"]) for question in questions) / len(questions)
            if questions
            else 0.0
        ),
        "logical_gemini_requests": sum(
            request["logical_call_count"] for request in requests
        ),
        "gemini_attempts": sum(request["gemini_attempts"] for request in requests),
        "parse_failures": sum(request["parse_failures"] for request in requests),
        "preprocessing_failures": sum(
            request["status"] == "preprocessing_failed" for request in requests
        ),
    }


def run_variant(
    variant: str,
    manifest: dict[str, Any],
    manifest_path: Path,
    image_dir: Path,
    cache_dir: Path,
    results_dir: Path,
    root: Path,
    api_key_path: str,
    model_version: str,
    prompt_file: Path,
    retry_count: int,
    limit: int | None,
    image_filter: str | None,
    resume: bool,
    overwrite: bool,
) -> Path:
    from src.understand.code.rpi_continuous_testing import (
        GeminiDirectionQA,
        is_correct,
        normalize_prediction,
    )

    selected = select_manifest_images(manifest, limit, image_filter)
    output_path = results_dir / f"{variant}.json"
    metadata = experiment_metadata(
        variant, manifest_path, manifest, model_version, prompt_file, retry_count
    )

    completed: list[dict[str, Any]] = []
    if output_path.is_file() and resume and not overwrite:
        previous = load_json(output_path)
        if previous.get("experiment") != metadata:
            raise ValueError(f"Cannot resume incompatible result file: {output_path}")
        completed = previous.get("results", [])
    elif output_path.exists() and not overwrite:
        raise FileExistsError(f"Result exists; use --resume or --overwrite: {output_path}")

    completed_names = {item["imagePath"] for item in completed}
    qa = GeminiDirectionQA(
        root=str(root),
        api_key_path=api_key_path,
        model_version=model_version,
        prompt_file=project_relative(prompt_file),
    )

    for ordinal, image_item in enumerate(selected, start=1):
        image_name = image_item["imagePath"]
        if image_name in completed_names:
            continue
        print(f"[{variant}] {ordinal}/{len(selected)}: {image_name}")
        image_path = image_dir / image_name
        cache_record = load_image_cache(image_item, cache_dir)
        views = select_variant_views(variant, image_path, cache_record, cache_dir)
        questions = [item["question"] for item in image_item["questions"]]

        if not views.paths:
            request = {
                "logical_call_count": 0,
                "gemini_attempts": 0,
                "parse_failures": 0,
                "latency_seconds": 0.0,
                "status": "preprocessing_failed",
                "errors": ["No views available; original-image fallback is disabled."],
                "request_prompt_sha256": None,
            }
            parsed = {}
            raw_response = None
        else:
            full_prompt = build_prompt(qa.qa_prompt, questions, views.note)
            response = ask_variant_with_retries(
                qa,
                full_prompt,
                views.paths,
            )
            parsed = response["parsed_response"] or {}
            raw_response = response["raw_response"]
            request = {
                "logical_call_count": 1,
                "gemini_attempts": response["gemini_attempts"],
                "parse_failures": response["parse_failures"],
                "latency_seconds": response["latency_seconds"],
                "status": (
                    "completed" if response["parsed_response"] is not None else "failed"
                ),
                "errors": response["errors"],
                "request_prompt_sha256": hashlib.sha256(
                    full_prompt.encode("utf-8")
                ).hexdigest(),
            }

        evaluated_questions = []
        for item in image_item["questions"]:
            predicted = normalize_prediction(parsed.get(item["question"], "ERROR"))
            question_result = {
                "question": item["question"],
                "expected": item["expected"],
                "predicted": predicted,
                "correct": is_correct(predicted, item["expected"]),
                "latency_seconds": request["latency_seconds"],
                "gemini_call_count": request["logical_call_count"],
                "parse_failures": request["parse_failures"],
            }
            if predicted == "ERROR":
                question_result["error_response"] = raw_response
            evaluated_questions.append(question_result)

        completed.append(
            {
                "imagePath": image_name,
                "image_sha256": image_item["sha256"],
                "preprocessing": preprocessing_metadata(variant, cache_record, views),
                "request": request,
                "questions": evaluated_questions,
            }
        )
        output = {
            "schema_version": SCHEMA_VERSION,
            "experiment": metadata,
            "summary": summarize_results(completed, len(selected)),
            "results": completed,
        }
        write_json_atomic(output_path, output)
        # Match the canonical runner's pacing between successive images.
        time.sleep(2)
    return output_path


def select_manifest_images(
    manifest: dict[str, Any], limit: int | None, image_filter: str | None
) -> list[dict[str, Any]]:
    images = manifest["images"]
    if image_filter:
        images = [item for item in images if item["imagePath"] == image_filter]
        if not images:
            raise ValueError(f"Image is not in the frozen manifest: {image_filter}")
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive.")
        images = images[:limit]
    return images


def prepare_cache(
    manifest: dict[str, Any],
    image_dir: Path,
    cache_dir: Path,
    limit: int | None,
    image_filter: str | None,
    overwrite: bool,
) -> None:
    selected = select_manifest_images(manifest, limit, image_filter)
    cache_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        cache_dir / "cache_metadata.json",
        {
            "schema_version": SCHEMA_VERSION,
            "configuration_sha256": cache_configuration_hash(),
            **cache_configuration(),
        },
    )
    for ordinal, image_item in enumerate(selected, start=1):
        print(f"[cache] {ordinal}/{len(selected)}: {image_item['imagePath']}")
        prepare_image_cache(image_item, image_dir, cache_dir, overwrite=overwrite)


def dry_run_report(
    variants: list[str],
    manifest: dict[str, Any],
    image_dir: Path,
    cache_dir: Path,
    limit: int | None,
    image_filter: str | None,
    model_version: str,
    prompt_file: Path,
    retry_count: int,
) -> None:
    selected = select_manifest_images(manifest, limit, image_filter)
    cache_hits = 0
    missing_cache = []
    view_totals = {variant: 0 for variant in variants}
    zero_view_images = {variant: [] for variant in variants}

    for item in selected:
        record = load_image_cache(item, cache_dir)
        if record is not None:
            cache_hits += 1
        else:
            missing_cache.append(item["imagePath"])
        for variant in variants:
            if variant != "original_only" and record is None:
                continue
            views = select_variant_views(
                variant, image_dir / item["imagePath"], record, cache_dir
            )
            view_totals[variant] += len(views.paths)
            if not views.paths:
                zero_view_images[variant].append(item["imagePath"])

    report = {
        "mode": "dry-run (no API key loaded; no YOLO, MobileSAM, or Gemini calls)",
        "manifest_images": manifest["image_count"],
        "manifest_questions": manifest["question_count"],
        "selected_images": len(selected),
        "selected_questions": sum(len(item["questions"]) for item in selected),
        "variants": variants,
        "planned_logical_gemini_requests": len(selected) * len(variants),
        "model": model_version,
        "temperature": 0,
        "prompt_sha256": sha256_file(prompt_file),
        "configured_retry_count": retry_count,
        "effective_retry_policy": "indefinite until parse success",
        "retry_delay_seconds": "random integer from 15 through 60",
        "cache_hits": cache_hits,
        "cache_missing_count": len(missing_cache),
        "cache_missing_first_20": missing_cache[:20],
        "view_totals_where_cache_available": view_totals,
        "zero_view_images": zero_view_images,
        "scientific_limitations": [SCIENTIFIC_LIMITATION],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


def build_comparison_summary(results_dir: Path) -> Path:
    outputs = {variant: load_json(results_dir / f"{variant}.json") for variant in VARIANTS}
    manifest_hashes = {
        output["experiment"]["manifest_content_sha256"] for output in outputs.values()
    }
    if len(manifest_hashes) != 1:
        raise ValueError("Variant results do not use the same frozen manifest.")

    manifests = []
    for variant, output in outputs.items():
        keys = [
            (image["imagePath"], question["question"])
            for image in output["results"]
            for question in image["questions"]
        ]
        manifests.append((variant, keys))
    reference = manifests[0][1]
    for variant, keys in manifests[1:]:
        if keys != reference:
            raise ValueError(f"Question manifest mismatch in {variant}.")

    summary = {
        "schema_version": SCHEMA_VERSION,
        "manifest_content_sha256": next(iter(manifest_hashes)),
        "scientific_limitations": [SCIENTIFIC_LIMITATION],
        "variants": {
            variant: {
                "label": VARIANT_METADATA[variant]["label"],
                "hypothesis": VARIANT_METADATA[variant]["hypothesis"],
                **output["summary"],
            }
            for variant, output in outputs.items()
        },
    }
    output_path = results_dir / "comparison_summary.json"
    write_json_atomic(output_path, summary)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=(*SUPPORTED_VARIANTS, "all"))
    parser.add_argument("--build-manifest", action="store_true")
    parser.add_argument("--prepare-cache", action="store_true")
    parser.add_argument("--build-comparison", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest", default="experiments/still_ablation/manifest.json")
    parser.add_argument("--image-dir", default="data/Nav_sign_data")
    parser.add_argument(
        "--qa-file", default="src/understand/qa_test_set/rpi_test_set.json"
    )
    parser.add_argument(
        "--cache-dir", default="outputs/still_ablation/detection_cache"
    )
    parser.add_argument("--results-dir", default="results/still_ablation")
    parser.add_argument("--root", default=".")
    parser.add_argument("--api-key-path", default="keys/gemini_api_key.yaml")
    parser.add_argument("--model-version", default="gemini-3.5-flash")
    parser.add_argument(
        "--prompt-file", default="src/understand/prompts/qa_prompt.txt"
    )
    parser.add_argument("--retry-count", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--image")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    image_dir = Path(args.image_dir)
    qa_file = Path(args.qa_file)
    manifest_path = Path(args.manifest)
    cache_dir = Path(args.cache_dir)
    results_dir = Path(args.results_dir)
    prompt_file = Path(args.prompt_file)

    if args.retry_count < 1:
        raise ValueError("--retry-count must be positive.")

    if args.build_manifest:
        manifest = build_frozen_manifest(
            image_dir, qa_file, manifest_path, write=not args.dry_run
        )
        action = "Validated" if args.dry_run else "Wrote"
        print(
            f"{action} manifest: {manifest_path} "
            f"({manifest['image_count']} images, {manifest['question_count']} questions)"
        )
        if not args.variant and not args.prepare_cache:
            return

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Frozen manifest not found: {manifest_path}. Use --build-manifest first."
        )
    manifest = validate_manifest(manifest_path, image_dir, qa_file)

    if args.build_comparison:
        if args.dry_run:
            print("Dry-run: comparison inputs would be validated; no file written.")
        else:
            print(f"Wrote comparison: {build_comparison_summary(results_dir)}")
        if not args.variant and not args.prepare_cache:
            return

    if args.prepare_cache:
        if args.dry_run:
            selected = select_manifest_images(manifest, args.limit, args.image)
            missing = sum(load_image_cache(item, cache_dir) is None for item in selected)
            print(
                f"Dry-run: preprocessing cache selected={len(selected)}, "
                f"missing={missing}; no YOLO or MobileSAM calls made."
            )
        else:
            prepare_cache(
                manifest,
                image_dir,
                cache_dir,
                args.limit,
                args.image,
                args.overwrite,
            )
        if not args.variant:
            return

    if not args.variant:
        raise ValueError(
            "Specify --variant, --build-manifest, --prepare-cache, or --build-comparison."
        )
    variants = list(VARIANTS) if args.variant == "all" else [args.variant]

    if args.dry_run:
        dry_run_report(
            variants,
            manifest,
            image_dir,
            cache_dir,
            args.limit,
            args.image,
            args.model_version,
            prompt_file,
            args.retry_count,
        )
        return

    if any(variant != "original_only" for variant in variants):
        selected = select_manifest_images(manifest, args.limit, args.image)
        missing = [
            item["imagePath"]
            for item in selected
            if load_image_cache(item, cache_dir) is None
        ]
        if missing:
            raise FileNotFoundError(
                f"Preprocessing cache is missing {len(missing)} selected images. "
                "Run --prepare-cache first."
            )

    for variant in variants:
        output = run_variant(
            variant,
            manifest,
            manifest_path,
            image_dir,
            cache_dir,
            results_dir,
            Path(args.root),
            args.api_key_path,
            args.model_version,
            prompt_file,
            args.retry_count,
            args.limit,
            args.image,
            args.resume,
            args.overwrite,
        )
        print(f"Wrote result: {output}")


if __name__ == "__main__":
    main()
