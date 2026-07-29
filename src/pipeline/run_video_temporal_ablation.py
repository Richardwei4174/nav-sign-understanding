"""Controlled temporal-policy ablations for prerecorded navigation videos.

This runner is intentionally isolated from the production video pipelines. It
creates one synchronous observation cache per video/stride, applies temporal
policies to exactly the same observations, and only then delegates a selected
observation to the unchanged production multiview backend.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(SRC_ROOT))

from src.pipeline.visual_quality import crop_quality_components

SCHEMA_VERSION = 1
POLICIES = (
    "first_match",
    "fixed_window",
    "last_seen",
    "track_lifetime_best_view",
)
FINAL_EXPERIMENT_MATRIX = (
    (3, "track_lifetime_best_view"),
    (6, "first_match"),
    (6, "fixed_window"),
    (6, "last_seen"),
    (6, "track_lifetime_best_view"),
    (12, "track_lifetime_best_view"),
)
YOLO_MODEL = "yolov8s-world.pt"
YOLO_CLASSES = (
    "navigation sign",
    "directional sign",
    "wayfinding sign",
    "sign with arrow",
    "directional arrow sign",
    "hallway directional sign",
    "exit sign",
    "",
)
YOLO_CONFIDENCE = 0.05
YOLO_IOU = 0.3
MATCH_THRESHOLD = 70
FIXED_WINDOW_FRAMES = 5
LAST_SEEN_GRACE_UPDATES = 8
TRACKER_CONFIG = Path("config/bytetrack_navigation.yaml")
OCR_CONFIGURATION = {
    "use_angle_cls": True,
    "lang": "en",
    "use_gpu": True,
    "show_log": False,
}


def extract_target_from_question(question_item: dict[str, Any]) -> str:
    """Dependency-light copy of the frozen recorded runner's exact extraction."""
    if "destination" in question_item:
        return question_item["destination"]
    question = question_item["question"].strip()
    question = re.sub(r"^where\s+is\s+", "", question, flags=re.IGNORECASE)
    return question.rstrip("?").strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def project_relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_qa_entries(
    qa_file: Path, extension_files: tuple[Path, ...] = ()
) -> list[dict[str, Any]]:
    """Load a frozen base manifest plus optional, non-overlapping extensions."""
    entries: list[dict[str, Any]] = []
    seen: dict[str, Path] = {}
    seen_stems: dict[str, str] = {}
    for source in (qa_file, *extension_files):
        data = load_json(source)
        if not isinstance(data, list):
            raise ValueError(f"Video QA must be a list of per-video entries: {source}")
        for index, item in enumerate(data):
            if not isinstance(item, dict) or not isinstance(item.get("videoPath"), str):
                raise ValueError(f"Invalid video entry {index} in {source}")
            video_name = item["videoPath"]
            if video_name in seen:
                raise ValueError(
                    f"Duplicate videoPath {video_name!r} in {seen[video_name]} and {source}"
                )
            normalized_stem = Path(video_name).stem.casefold()
            if normalized_stem in seen_stems:
                raise ValueError(
                    "Duplicate video stem would collide in output directories: "
                    f"{seen_stems[normalized_stem]!r} and {video_name!r}"
                )
            questions = item.get("questions")
            if not isinstance(questions, list) or not questions:
                raise ValueError(f"No questions found for {video_name} in {source}.")
            # Validate reference semantics for the entire merged manifest before
            # any cache preparation, model loading, or inference can begin.
            question_records(questions)
            seen[video_name] = source
            seen_stems[normalized_stem] = video_name
            entries.append({**item, "_manifest_source": str(source)})
    return entries


def load_video_questions(qa_entries: list[dict[str, Any]], video: Path) -> list[dict[str, Any]]:
    matches = [item for item in qa_entries if item.get("videoPath") == video.name]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one QA entry for {video.name}; found {len(matches)}."
        )
    questions = matches[0].get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError(f"No questions found for {video.name}.")
    return questions


def question_records(question_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for index, item in enumerate(question_items):
        # Keep the video annotations aligned with the still-image QA schema:
        # absence is represented by an expected answer of "unknown", not by
        # a separate presence field. ``answer`` is the current repository QA
        # spelling; ``expected`` is accepted for manifest-style annotations.
        if "expected" in item:
            expected = item["expected"]
        elif "answer" in item:
            expected = item["answer"]
        else:
            raise ValueError(
                f"Question {index} is missing an expected/answer field."
            )
        question = item.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Question {index} has empty or invalid text.")
        reference_type = item.get("reference_type")
        if isinstance(expected, list):
            if reference_type != "alternatives":
                raise ValueError(
                    f"Question {index} has a list-valued answer and must declare "
                    'reference_type="alternatives".'
                )
            if (
                not expected
                or any(not isinstance(value, str) or not value.strip() for value in expected)
            ):
                raise ValueError(f"Question {index} has an empty or invalid alternatives list.")
        else:
            if not isinstance(expected, str) or not expected.strip():
                raise ValueError(f"Question {index} has an empty or invalid scalar answer.")
            if reference_type not in (None, "scalar"):
                raise ValueError(
                    f"Question {index} has unsupported reference_type={reference_type!r} "
                    "for a scalar answer."
                )
            reference_type = "scalar"
        target_present = expected != "unknown"
        records.append(
            {
                "question_id": f"question_{index:04d}",
                "question": question,
                "destination": extract_target_from_question(item),
                "expected_direction": expected,
                "reference_type": reference_type,
                "target_present": target_present,
                "presence_source": "inferred_from_expected_answer",
                "question_item": item,
            }
        )
    return records


def normalized_labels(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, (list, tuple, set)):
        if not value or any(not isinstance(item, str) for item in value):
            return set()
        return set(value)
    return set()


def score_reference(prediction: Any, expected: Any, reference_type: str) -> bool:
    """Apply the manuscript's explicit scalar/alternatives set semantics."""
    predicted = normalized_labels(prediction)
    reference = normalized_labels(expected)
    if reference_type == "scalar":
        return len(reference) == 1 and predicted == reference
    if reference_type == "alternatives":
        return bool(predicted) and predicted <= reference
    raise ValueError(f"Unsupported reference_type: {reference_type!r}")


def normalize_cached_question(question: dict[str, Any]) -> dict[str, Any]:
    """Normalize one persisted question without mutating its cache record."""
    normalized = dict(question)
    if "expected_direction" not in normalized:
        raise ValueError("Cached question is missing expected_direction.")
    expected = normalized["expected_direction"]
    reference_type = normalized.get("reference_type")
    if isinstance(expected, list):
        if reference_type != "alternatives":
            raise ValueError(
                "A list-valued cached answer must declare "
                'reference_type="alternatives".'
            )
        if (
            not expected
            or any(
                not isinstance(value, str) or not value.strip()
                for value in expected
            )
        ):
            raise ValueError("Cached alternatives are empty or invalid.")
    else:
        if not isinstance(expected, str) or not expected.strip():
            raise ValueError("Cached scalar answer is empty or invalid.")
        if reference_type not in (None, "scalar"):
            raise ValueError(
                "A scalar cached answer cannot use "
                f"reference_type={reference_type!r}."
            )
        # Legacy caches predate the persisted field. Their validated scalar
        # annotations have always had scalar semantics.
        reference_type = "scalar"
    normalized["reference_type"] = reference_type
    return normalized


def normalize_loaded_cache(cache: dict[str, Any]) -> dict[str, Any]:
    """Return a policy-safe cache view with canonical question semantics."""
    questions = cache.get("questions")
    if not isinstance(questions, list):
        raise ValueError("Observation cache is missing its question records.")
    normalized = dict(cache)
    normalized["questions"] = [
        normalize_cached_question(question) for question in questions
    ]
    return normalized


def stored_question_records(
    question_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return canonical, serializable per-video question records."""
    return [
        {key: value for key, value in item.items() if key != "question_item"}
        for item in question_records(question_items)
    ]


def cache_configuration(
    stride: int, video: Path, question_items: list[dict[str, Any]]
) -> dict[str, Any]:
    tracker_path = PROJECT_ROOT / TRACKER_CONFIG
    return {
        "schema_version": SCHEMA_VERSION,
        "video_filename": video.name,
        "video_sha256": sha256_file(video),
        # Per-video identity lets a frozen cache survive additions to a separate
        # extension manifest while still rejecting edits/reordering within this
        # video's annotations.
        "qa_video_sha256": canonical_json_sha256(question_items),
        "sampling_stride": stride,
        "yolo": {
            "model": YOLO_MODEL,
            "classes": list(YOLO_CLASSES),
            "confidence": YOLO_CONFIDENCE,
            "iou": YOLO_IOU,
            "agnostic_nms": True,
        },
        "tracker_configuration": project_relative(tracker_path),
        "tracker_configuration_sha256": sha256_file(tracker_path),
        "ocr": OCR_CONFIGURATION,
        "match_threshold": MATCH_THRESHOLD,
    }


def validate_cache(
    cache: dict[str, Any],
    expected_configuration: dict[str, Any],
    expected_questions: list[dict[str, Any]],
    path: Path,
) -> None:
    existing_configuration = cache.get("configuration")
    if not isinstance(existing_configuration, dict):
        raise ValueError(f"Missing cache configuration: {path}")
    existing_comparable = dict(existing_configuration)
    expected_comparable = dict(expected_configuration)
    # Legacy caches used the hash of the entire QA file. Compare every
    # non-manifest setting, then verify the embedded per-video question records.
    existing_comparable.pop("qa_sha256", None)
    existing_comparable.pop("qa_video_sha256", None)
    expected_comparable.pop("qa_sha256", None)
    expected_comparable.pop("qa_video_sha256", None)
    if existing_comparable != expected_comparable:
        raise ValueError(f"Incompatible observation cache configuration: {path}")
    stored_questions = cache.get("questions")
    if not isinstance(stored_questions, list) or [
        normalize_cached_question(item) for item in stored_questions
    ] != [
        normalize_cached_question(item) for item in expected_questions
    ]:
        raise ValueError(f"Incompatible per-video annotations in cache: {path}")


def cache_path(output_root: Path, stride: int, video: Path) -> Path:
    return output_root / "cache" / f"stride_{stride}" / video.stem


def observations_path(output_root: Path, stride: int, video: Path) -> Path:
    return cache_path(output_root, stride, video) / "observations.json"


def cache_preparation_action(
    video: Path,
    qa_entries: list[dict[str, Any]],
    output_root: Path,
    stride: int,
    *,
    resume: bool,
    overwrite: bool,
) -> tuple[str, Path]:
    """Return the cache action after applying the real compatibility checks."""
    destination = cache_path(output_root, stride, video)
    output_file = destination / "observations.json"
    question_items = load_video_questions(qa_entries, video)
    stored_questions = stored_question_records(question_items)
    configuration = cache_configuration(stride, video, question_items)

    if output_file.is_file() and resume and not overwrite:
        validate_cache(
            load_json(output_file),
            configuration,
            stored_questions,
            output_file,
        )
        return "reuse", output_file
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Cache output exists; use --resume or --overwrite: {destination}"
        )
    return "create", output_file


def policy_result_path(
    output_root: Path, stride: int, policy: str, video: Path
) -> Path:
    return (
        output_root
        / "runs"
        / f"stride_{stride}"
        / policy
        / video.stem
        / "per_target_results.json"
    )


class StableTrackAssigner:
    """Preserve ByteTrack IDs and give deterministic IDs to unconfirmed boxes."""

    def __init__(self) -> None:
        self.next_temporary_id = 0
        self.previous: list[tuple[str, list[int]]] = []

    @staticmethod
    def iou(first: list[int], second: list[int]) -> float:
        ax1, ay1, ax2, ay2 = first
        bx1, by1, bx2, by2 = second
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        first_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        second_area = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = first_area + second_area - intersection
        return intersection / union if union > 0 else 0.0

    def assign(
        self, boxes: list[tuple[int | None, list[int]]]
    ) -> list[str]:
        assigned = []
        used_previous: set[int] = set()
        current = []
        for tracker_id, bbox in boxes:
            if tracker_id is not None:
                stable_id = f"bytetrack_{tracker_id}"
            else:
                candidates = [
                    (self.iou(bbox, old_bbox), old_index, old_id)
                    for old_index, (old_id, old_bbox) in enumerate(self.previous)
                    if old_index not in used_previous
                ]
                best = max(candidates, default=(0.0, -1, ""))
                if best[0] >= 0.30:
                    _, old_index, stable_id = best
                    used_previous.add(old_index)
                else:
                    stable_id = f"temporary_{self.next_temporary_id}"
                    self.next_temporary_id += 1
            assigned.append(stable_id)
            current.append((stable_id, bbox))
        self.previous = current
        return assigned


def prepare_cache(
    video: Path,
    qa_entries: list[dict[str, Any]],
    output_root: Path,
    stride: int,
    *,
    resume: bool,
    overwrite: bool,
) -> Path:
    destination = cache_path(output_root, stride, video)
    output_file = destination / "observations.json"
    question_items = load_video_questions(qa_entries, video)
    questions = question_records(question_items)
    stored_questions = stored_question_records(question_items)
    action, planned_output = cache_preparation_action(
        video,
        qa_entries,
        output_root,
        stride,
        resume=resume,
        overwrite=overwrite,
    )
    if action == "reuse":
        return planned_output
    configuration = cache_configuration(stride, video, question_items)
    if planned_output != output_file:
        raise AssertionError("Cache planner returned an unexpected output path.")

    # Heavy dependencies remain outside dry-run and policy-only execution.
    import cv2
    from paddleocr import PaddleOCR
    from ultralytics import YOLOWorld
    from src.pipeline.run_stream_video_pipeline import (
        get_avg_confidence,
        get_detection_text,
        run_ocr,
        score_match,
    )

    model = YOLOWorld(YOLO_MODEL)
    model.set_classes(list(YOLO_CLASSES))
    ocr = PaddleOCR(**OCR_CONFIGURATION)
    tracker_path = PROJECT_ROOT / TRACKER_CONFIG
    track_assigner = StableTrackAssigner()

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    reported_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Do not mutate an experiment directory until dependencies, models, OCR,
    # and the source video have all initialized successfully.
    if destination.exists():
        shutil.rmtree(destination)
    frames_dir = destination / "frames"
    crops_dir = destination / "crops"
    frames_dir.mkdir(parents=True)
    crops_dir.mkdir(parents=True)

    start = time.monotonic()
    source_frame_index = 0
    processed_frame_index = 0
    yolo_calls = 0
    detection_count = 0
    ocr_calls = 0
    frames = []

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            decoder_msec = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            if source_frame_index % stride != 0:
                source_frame_index += 1
                continue

            timestamp = (
                decoder_msec / 1000.0
                if decoder_msec > 0.0
                else source_frame_index / reported_fps if reported_fps > 0 else None
            )
            timestamp_method = (
                "decoder_position_msec"
                if decoder_msec > 0.0
                else "source_frame_index_divided_by_reported_fps"
            )
            frame_name = f"source_{source_frame_index:08d}.jpg"
            frame_path = frames_dir / frame_name
            cv2.imwrite(str(frame_path), frame)

            results = model.track(
                source=frame,
                persist=True,
                tracker=str(tracker_path),
                conf=YOLO_CONFIDENCE,
                iou=YOLO_IOU,
                agnostic_nms=True,
                verbose=False,
            )
            yolo_calls += 1
            result = results[0] if results else None
            raw_boxes = []
            if result is not None:
                for box in result.boxes:
                    coords = [int(value) for value in box.xyxy[0].tolist()]
                    tracker_id = int(box.id[0]) if box.id is not None else None
                    raw_boxes.append((tracker_id, coords))
            stable_ids = track_assigner.assign(raw_boxes)

            detections = []
            height, width = frame.shape[:2]
            for detection_index, (box, stable_id) in enumerate(
                zip(result.boxes if result is not None else [], stable_ids)
            ):
                x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                bbox = [x1, y1, x2, y2]
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                tracker_id = int(box.id[0]) if box.id is not None else None
                crop = frame[y1:y2, x1:x2]
                crop_name = (
                    f"source_{source_frame_index:08d}_"
                    f"detection_{detection_index:04d}.jpg"
                )
                crop_path = crops_dir / crop_name
                cv2.imwrite(str(crop_path), crop)

                ocr_lines = run_ocr(ocr, crop)
                ocr_calls += 1
                ocr_text = get_detection_text(ocr_lines)
                average_confidence = get_avg_confidence(ocr_lines)
                matches = {}
                for question in questions:
                    match_score = score_match(question["destination"], ocr_text) if ocr_text else 0
                    matches[question["question_id"]] = {
                        "destination": question["destination"],
                        "match_score": match_score,
                        "qualifies": match_score >= MATCH_THRESHOLD,
                    }

                quality = crop_quality_components(
                    crop, bbox, frame.shape, confidence
                )
                detections.append(
                    {
                        "detection_index": detection_index,
                        "bbox_xyxy": bbox,
                        "yolo_class": class_name,
                        "yolo_confidence": confidence,
                        "tracker_id": tracker_id,
                        "stable_track_id": stable_id,
                        "raw_crop_path": project_relative(crop_path),
                        "ocr_lines": ocr_lines,
                        "ocr_text": ocr_text,
                        "average_ocr_confidence": average_confidence,
                        "target_matches": matches,
                        "visual_quality": quality,
                    }
                )
                detection_count += 1

            frames.append(
                {
                    "source_frame_index": source_frame_index,
                    "source_timestamp_seconds": timestamp,
                    "timestamp_method": timestamp_method,
                    "processed_frame_index": processed_frame_index,
                    "sampling_stride": stride,
                    "full_frame_path": project_relative(frame_path),
                    "detections": detections,
                }
            )
            processed_frame_index += 1
            source_frame_index += 1
    finally:
        capture.release()

    output = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "configuration": configuration,
        "video_metadata": {
            "reported_fps": reported_fps,
            "reported_source_frames": reported_frames,
            "decoded_source_frames": source_frame_index,
            "processed_frames": processed_frame_index,
        },
        "statistics": {
            "yolo_calls": yolo_calls,
            "detections": detection_count,
            "ocr_calls": ocr_calls,
            "cache_generation_seconds": time.monotonic() - start,
        },
        "questions": stored_questions,
        "frames": frames,
    }
    write_json_atomic(output_file, output)
    return output_file


@dataclass(frozen=True)
class Candidate:
    frame: dict[str, Any]
    detection: dict[str, Any]
    match_score: float
    evidence_detection: dict[str, Any] | None = None

    @property
    def ocr_detection(self) -> dict[str, Any]:
        return self.evidence_detection or self.detection

    @property
    def average_ocr_confidence(self) -> float:
        return float(self.ocr_detection["average_ocr_confidence"])

    @property
    def ocr_text(self) -> str:
        return str(self.ocr_detection["ocr_text"])

    @property
    def production_rank(self) -> float:
        return self.match_score + 10 * self.average_ocr_confidence

    @property
    def track_lifetime_rank(self) -> tuple[float, float, float]:
        return (
            float(self.detection["visual_quality"]["final_score"]),
            self.average_ocr_confidence,
            float(self.match_score),
        )


@dataclass(frozen=True)
class TrackLifetimeSelection:
    candidate: Candidate | None
    candidate_count: int
    audit: dict[str, Any]


def qualifying_candidates(
    cache: dict[str, Any], question_id: str
) -> list[Candidate]:
    candidates = []
    for frame in cache["frames"]:
        for detection in frame["detections"]:
            match = detection["target_matches"][question_id]
            if match["qualifies"]:
                candidates.append(Candidate(frame, detection, match["match_score"]))
    return candidates


def better_candidate(current: Candidate | None, proposed: Candidate) -> Candidate:
    # Production behavior replaces only on a strictly greater score; ties keep first.
    if current is None or proposed.production_rank > current.production_rank:
        return proposed
    return current


def select_first_match(cache: dict[str, Any], question_id: str) -> tuple[Candidate | None, int]:
    candidates = qualifying_candidates(cache, question_id)
    return (candidates[0], 1) if candidates else (None, 0)


def select_fixed_window(cache: dict[str, Any], question_id: str) -> tuple[Candidate | None, int]:
    candidates = qualifying_candidates(cache, question_id)
    if not candidates:
        return None, 0
    first_index = candidates[0].frame["processed_frame_index"]
    end_index = first_index + FIXED_WINDOW_FRAMES
    eligible = [
        candidate
        for candidate in candidates
        if candidate.frame["processed_frame_index"] <= end_index
    ]
    best = None
    for candidate in eligible:
        best = better_candidate(best, candidate)
    return best, len(eligible)


def select_last_seen(cache: dict[str, Any], question_id: str) -> tuple[Candidate | None, int]:
    all_qualifying = qualifying_candidates(cache, question_id)
    if not all_qualifying:
        return None, 0
    first = all_qualifying[0]
    bound_track = first.detection["stable_track_id"]
    first_index = first.frame["processed_frame_index"]
    best = None
    candidate_count = 0
    consecutive_misses = 0

    for frame in cache["frames"]:
        if frame["processed_frame_index"] < first_index:
            continue
        bound_detections = [
            detection
            for detection in frame["detections"]
            if detection["stable_track_id"] == bound_track
        ]
        if bound_detections:
            consecutive_misses = 0
        else:
            consecutive_misses += 1
            if consecutive_misses >= LAST_SEEN_GRACE_UPDATES:
                break

        for detection in bound_detections:
            match = detection["target_matches"][question_id]
            if not match["qualifies"]:
                continue
            candidate = Candidate(frame, detection, match["match_score"])
            best = better_candidate(best, candidate)
            candidate_count += 1
    return best, candidate_count


def evaluate_track_lifetime_best_view(
    cache: dict[str, Any], question_id: str
) -> TrackLifetimeSelection:
    """Apply the live best-view policy to frozen cached observations only."""
    track_best_views: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    bound_track: str | None = None
    binding_count = 0
    best_ocr_evidence: tuple[dict[str, Any], float] | None = None
    best_candidate: Candidate | None = None
    candidate_count = 0
    consecutive_misses = 0
    miss_resets = 0
    visual_updates_after_binding = 0
    candidate_improvements_after_initial = 0
    closed_by_grace = False
    misses_at_close: int | None = None
    first_match_frame: int | None = None
    selected_before_first_match = False

    for frame in cache["frames"]:
        bound_track_seen = False

        for detection in frame["detections"]:
            track_id = detection["stable_track_id"]
            quality = float(detection["visual_quality"]["final_score"])
            previous_best = track_best_views.get(track_id)
            if previous_best is None or quality > float(
                previous_best[1]["visual_quality"]["final_score"]
            ):
                track_best_views[track_id] = (frame, detection)
                if bound_track == track_id:
                    visual_updates_after_binding += 1

            match = detection["target_matches"][question_id]
            if bound_track is None:
                if not match["qualifies"]:
                    continue
                bound_track = track_id
                binding_count += 1
                first_match_frame = frame["processed_frame_index"]

            if track_id != bound_track:
                continue

            bound_track_seen = True
            if match["qualifies"]:
                proposed_evidence = (detection, float(match["match_score"]))
                if best_ocr_evidence is None:
                    best_ocr_evidence = proposed_evidence
                else:
                    current_detection, current_match = best_ocr_evidence
                    proposed_rank = (
                        float(detection["average_ocr_confidence"]),
                        float(match["match_score"]),
                    )
                    current_rank = (
                        float(current_detection["average_ocr_confidence"]),
                        float(current_match),
                    )
                    if proposed_rank > current_rank:
                        best_ocr_evidence = proposed_evidence

            if best_ocr_evidence is None:
                continue
            visual_frame, visual_detection = track_best_views[bound_track]
            evidence_detection, evidence_match_score = best_ocr_evidence
            proposed = Candidate(
                visual_frame,
                visual_detection,
                evidence_match_score,
                evidence_detection=evidence_detection,
            )
            candidate_count += 1
            if (
                best_candidate is None
                or proposed.track_lifetime_rank > best_candidate.track_lifetime_rank
            ):
                if best_candidate is not None:
                    candidate_improvements_after_initial += 1
                best_candidate = proposed

        if bound_track is None:
            continue
        if bound_track_seen:
            if consecutive_misses:
                miss_resets += 1
            consecutive_misses = 0
        else:
            consecutive_misses += 1
            if consecutive_misses >= LAST_SEEN_GRACE_UPDATES:
                closed_by_grace = True
                misses_at_close = consecutive_misses
                break

    force_closed_at_video_end = bound_track is not None and not closed_by_grace
    if best_candidate is not None and first_match_frame is not None:
        selected_before_first_match = (
            best_candidate.frame["processed_frame_index"] < first_match_frame
        )
    audit = {
        "bound_track_id": bound_track,
        "binding_count": binding_count,
        "track_switch_count": 0,
        "first_match_processed_frame_index": first_match_frame,
        "selected_processed_frame_index": (
            best_candidate.frame["processed_frame_index"] if best_candidate else None
        ),
        "selected_before_first_match": selected_before_first_match,
        "visual_updates_after_binding": visual_updates_after_binding,
        "candidate_improvements_after_initial": candidate_improvements_after_initial,
        "miss_counter_resets": miss_resets,
        "closed_by_grace": closed_by_grace,
        "misses_at_close": misses_at_close,
        "force_closed_at_video_end": force_closed_at_video_end,
        "candidate_rank": (
            list(best_candidate.track_lifetime_rank) if best_candidate else None
        ),
    }
    return TrackLifetimeSelection(best_candidate, candidate_count, audit)


def select_track_lifetime_best_view(
    cache: dict[str, Any], question_id: str
) -> tuple[Candidate | None, int]:
    selection = evaluate_track_lifetime_best_view(cache, question_id)
    return selection.candidate, selection.candidate_count


SELECTORS = {
    "first_match": select_first_match,
    "fixed_window": select_fixed_window,
    "last_seen": select_last_seen,
    "track_lifetime_best_view": select_track_lifetime_best_view,
}


def save_selected_candidate(
    run_dir: Path, question: dict[str, Any], candidate: Candidate
) -> tuple[Path, Path]:
    target_dir = run_dir / "selected" / question["question_id"]
    target_dir.mkdir(parents=True, exist_ok=True)
    selected_frame = target_dir / "best_frame.jpg"
    selected_crop = target_dir / "best_crop.jpg"
    shutil.copy(PROJECT_ROOT / candidate.frame["full_frame_path"], selected_frame)
    shutil.copy(PROJECT_ROOT / candidate.detection["raw_crop_path"], selected_crop)
    return selected_frame, selected_crop


def result_record(
    question: dict[str, Any],
    candidate: Candidate | None,
    first: Candidate | None,
    candidate_count: int,
) -> dict[str, Any]:
    question = normalize_cached_question(question)
    base = {
        key: question[key]
        for key in (
            "question_id",
            "question",
            "destination",
            "expected_direction",
            "reference_type",
            "target_present",
            "presence_source",
        )
    }
    if candidate is None:
        return {
            **base,
            "candidate_retrieved": False,
            "selected_source_frame_index": None,
            "selected_source_timestamp_seconds": None,
            "selected_processed_frame_index": None,
            "selected_crop_path": None,
            "selected_track_id": None,
            "ocr_text": None,
            "match_score": None,
            "average_ocr_confidence": None,
            "visual_quality_score": None,
            "production_final_score": None,
            "first_match_seconds": None,
            "selection_seconds": None,
            "selection_delay_seconds": None,
            "candidate_count": 0,
            "gemini_prediction": "unknown",
            "correct": question["expected_direction"] == "unknown",
            "logical_gemini_calls": 0,
            "api_attempts": 0,
        }

    first_time = first.frame["source_timestamp_seconds"] if first else None
    selection_time = candidate.frame["source_timestamp_seconds"]
    delay = (
        selection_time - first_time
        if selection_time is not None and first_time is not None
        else None
    )
    return {
        **base,
        "candidate_retrieved": True,
        "selected_source_frame_index": candidate.frame["source_frame_index"],
        "selected_source_timestamp_seconds": selection_time,
        "selected_processed_frame_index": candidate.frame["processed_frame_index"],
        "selected_crop_path": candidate.detection["raw_crop_path"],
        "selected_track_id": candidate.detection["stable_track_id"],
        "ocr_text": candidate.ocr_text,
        "match_score": candidate.match_score,
        "average_ocr_confidence": candidate.average_ocr_confidence,
        "visual_quality_score": candidate.detection["visual_quality"]["final_score"],
        "production_final_score": candidate.production_rank,
        "first_match_seconds": first_time,
        "selection_seconds": selection_time,
        "selection_delay_seconds": delay,
        "candidate_count": candidate_count,
        "gemini_prediction": None,
        "correct": None,
        "logical_gemini_calls": 0,
        "api_attempts": 0,
    }


REVIEW_FIELDS = (
    "video_filename",
    "policy",
    "question_id",
    "question",
    "destination",
    "expected_direction",
    "reference_type",
    "target_present",
    "candidate_retrieved",
    "selected_source_frame_index",
    "selected_source_timestamp_seconds",
    "selected_crop_path",
    "selected_track_id",
    "ocr_text",
    "match_score",
    "visual_quality_score",
    "gemini_prediction",
    "correct",
    "destination_visible_in_selected_crop",
    "selected_correct_physical_sign",
    "review_notes",
)


def write_review_csv(path: Path, video_name: str, policy: str, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "video_filename": video_name,
                    "policy": policy,
                    **result,
                    "destination_visible_in_selected_crop": "",
                    "selected_correct_physical_sign": "",
                    "review_notes": "",
                }
            )


def summarize_policy_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [item for item in results if item.get("correct") is not None]
    correct = sum(bool(item["correct"]) for item in evaluated)
    return {
        "targets": len(results),
        "candidates_retrieved": sum(
            bool(item["candidate_retrieved"]) for item in results
        ),
        "logical_gemini_calls": sum(
            int(item.get("logical_gemini_calls", 0)) for item in results
        ),
        "api_attempts": sum(int(item.get("api_attempts", 0)) for item in results),
        "evaluated_targets": len(evaluated),
        "correct": correct,
        "accuracy": correct / len(evaluated) if evaluated else None,
    }


def aggregate_policy_results(
    stride_dir: Path,
    policy: str,
    expected_videos: list[str],
) -> tuple[Path, Path]:
    """Combine completed per-video outputs without running any pipeline stages."""
    policy_dir = stride_dir / policy
    source_paths = sorted(
        policy_dir.glob("*/per_target_results.json"),
        key=lambda path: path.parent.name.lower(),
    )
    expected = sorted(set(expected_videos), key=str.lower)
    expected_set = set(expected)
    found: list[str] = []
    observed_strides: set[int] = set()
    all_results: list[dict[str, Any]] = []
    per_video: list[dict[str, Any]] = []

    for source_path in source_paths:
        document = load_json(source_path)
        if document.get("policy") != policy:
            raise ValueError(
                f"Policy mismatch in {source_path}: {document.get('policy')!r}"
            )
        video_stride = document.get("stride")
        if video_stride is None:
            raise ValueError(f"Missing stride in {source_path}")
        observed_strides.add(int(video_stride))
        video_name = document.get("video_filename")
        if not video_name:
            raise ValueError(f"Missing video_filename in {source_path}")
        if video_name in found:
            raise ValueError(f"Duplicate result for {video_name} in {policy_dir}")

        results = document.get("results")
        if not isinstance(results, list):
            raise ValueError(f"Invalid results list in {source_path}")
        found.append(video_name)
        source = project_relative(source_path)
        summary = summarize_policy_results(results)
        per_video.append(
            {
                "video_filename": video_name,
                "source_path": source,
                "summary": summary,
            }
        )
        for result in results:
            all_results.append(
                {
                    **result,
                    "video_filename": video_name,
                    "policy": policy,
                    "stride": video_stride,
                    "source_path": source,
                }
            )

    if len(observed_strides) > 1:
        raise ValueError(
            f"Mixed strides in {policy_dir}: {sorted(observed_strides)}"
        )
    stride = next(iter(observed_strides), None)
    found = sorted(found, key=str.lower)
    missing = sorted(expected_set - set(found), key=str.lower)
    unexpected = sorted(set(found) - expected_set, key=str.lower)
    evaluated = [item for item in all_results if item.get("correct") is not None]
    correct = sum(bool(item["correct"]) for item in evaluated)
    incorrect = len(evaluated) - correct
    total = len(all_results)
    video_accuracies = [
        item["summary"]["accuracy"]
        for item in per_video
        if item["summary"]["accuracy"] is not None
    ]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "policy": policy,
        "stride": stride,
        "stride_directory": project_relative(stride_dir),
        "expected_videos": expected,
        "found_videos": found,
        "missing_videos": missing,
        "unexpected_videos": unexpected,
        "total_targets": total,
        "evaluated_targets": len(evaluated),
        "correct": correct,
        "incorrect": incorrect,
        "unretrieved_or_unevaluated": total - len(evaluated),
        "retrieval_count": sum(
            bool(item.get("candidate_retrieved")) for item in all_results
        ),
        "gemini_call_count": sum(
            int(item.get("logical_gemini_calls", 0)) for item in all_results
        ),
        "api_attempt_count": sum(
            int(item.get("api_attempts", 0)) for item in all_results
        ),
        "micro_accuracy_correct_over_evaluated_targets": (
            correct / len(evaluated) if evaluated else None
        ),
        "coverage_evaluated_targets_over_total_targets": (
            len(evaluated) / total if total else None
        ),
        "end_to_end_accuracy_correct_over_total_targets": (
            correct / total if total else None
        ),
        "macro_accuracy_mean_per_video_accuracy": (
            statistics.fmean(video_accuracies) if video_accuracies else None
        ),
        "macro_accuracy_videos_included": len(video_accuracies),
        "per_video": per_video,
    }
    all_results_path = policy_dir / "all_results.json"
    summary_path = policy_dir / "summary.json"
    write_json_atomic(
        all_results_path,
        {
            "schema_version": SCHEMA_VERSION,
            "policy": policy,
            "stride": stride,
            "expected_videos": expected,
            "found_videos": found,
            "results": all_results,
        },
    )
    write_json_atomic(summary_path, summary)
    return all_results_path, summary_path


def aggregate_stride_directory(
    stride_dir: Path,
    policies: tuple[str, ...],
    expected_videos: list[str],
) -> None:
    if not stride_dir.is_dir():
        raise FileNotFoundError(f"Stride run directory not found: {stride_dir}")
    for policy in policies:
        all_results_path, summary_path = aggregate_policy_results(
            stride_dir, policy, expected_videos
        )
        print(f"Wrote aggregate results: {all_results_path}")
        print(f"Wrote aggregate summary: {summary_path}")


def run_policy(
    video: Path,
    qa_entries: list[dict[str, Any]],
    output_root: Path,
    stride: int,
    policy: str,
    *,
    selection_only: bool,
    limit_targets: int | None,
    root: str,
    api_key_path: str,
    model_version: str,
    prompt_file: str,
    resume: bool,
    overwrite: bool,
) -> Path:
    cached_path = observations_path(output_root, stride, video)
    if not cached_path.is_file():
        raise FileNotFoundError(f"Observation cache not found: {cached_path}")
    cache = load_json(cached_path)
    question_items = load_video_questions(qa_entries, video)
    expected_questions = [
        {key: value for key, value in item.items() if key != "question_item"}
        for item in question_records(question_items)
    ]
    expected_configuration = cache_configuration(stride, video, question_items)
    validate_cache(cache, expected_configuration, expected_questions, cached_path)
    cache = normalize_loaded_cache(cache)

    result_path = policy_result_path(output_root, stride, policy, video)
    run_dir = result_path.parent
    if result_path.exists() and not (resume or overwrite):
        raise FileExistsError(
            f"Policy output exists; use --resume or --overwrite: {result_path}"
        )
    if run_dir.exists() and overwrite:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    questions = cache["questions"]
    if limit_targets is not None:
        questions = questions[:limit_targets]
    existing_results = []
    policy_configuration = {
        "cache_configuration_sha256": canonical_json_sha256(
            cache["configuration"]
        ),
        "selection_only": selection_only,
        "model_version": None if selection_only else model_version,
        "prompt_file": None if selection_only else prompt_file,
        "prompt_sha256": (
            None
            if selection_only
            else sha256_file(PROJECT_ROOT / prompt_file)
        ),
    }
    if result_path.is_file() and resume:
        previous = load_json(result_path)
        existing_results = previous.get("results", [])
        if (
            previous.get("video_filename") != video.name
            or previous.get("policy") != policy
            or int(previous.get("stride", -1)) != stride
            or bool(previous.get("selection_only")) != selection_only
        ):
            raise ValueError(f"Cannot resume incompatible policy output: {result_path}")
        question_by_id = {item["question_id"]: item for item in questions}
        for item in existing_results:
            current = question_by_id.get(item.get("question_id"))
            identity = (
                "question",
                "destination",
                "expected_direction",
                "reference_type",
                "target_present",
            )
            if current is None or any(
                (
                    item.get(key, "scalar")
                    if key == "reference_type"
                    else item.get(key)
                )
                != (
                    current.get(key, "scalar")
                    if key == "reference_type"
                    else current.get(key)
                )
                for key in identity
            ):
                raise ValueError(
                    f"Cannot resume changed or unknown question in {result_path}: "
                    f"{item.get('question_id')}"
                )
        previous_configuration = previous.get("policy_configuration")
        if previous_configuration is not None and previous_configuration != policy_configuration:
            raise ValueError(f"Cannot resume changed policy configuration: {result_path}")
    completed = {item["question_id"] for item in existing_results}
    results = list(existing_results)
    missing_question_ids = {
        item["question_id"] for item in questions
    } - completed
    if (
        result_path.is_file()
        and resume
        and missing_question_ids
        and not selection_only
        and previous.get("policy_configuration") is None
    ):
        raise ValueError(
            "Cannot safely resume a partial legacy Gemini result without a "
            f"persisted policy configuration: {result_path}"
        )
    if not missing_question_ids:
        write_review_csv(
            run_dir / "human_review.csv", video.name, policy, results
        )
        return result_path

    qa = None
    if not selection_only:
        from src.understand.code.rpi_continuous_testing import GeminiDirectionQA

        qa = GeminiDirectionQA(
            root=root,
            api_key_path=api_key_path,
            model_version=model_version,
            prompt_file=prompt_file,
        )

    for question in questions:
        if question["question_id"] in completed:
            continue
        first, _ = select_first_match(cache, question["question_id"])
        candidate, candidate_count = SELECTORS[policy](cache, question["question_id"])
        record = result_record(question, candidate, first, candidate_count)

        if candidate is not None:
            selected_frame, selected_crop = save_selected_candidate(
                run_dir, question, candidate
            )
            record["selected_crop_path"] = project_relative(selected_crop)
            record["selected_frame_path"] = project_relative(selected_frame)

            if not selection_only:
                from src.pipeline.run_stream_video_pipeline import build_detection
                from src.pipeline.run_multiview_pipeline import run_multiview_from_detections

                question_item = {
                    "question": question["question"],
                    "answer": question["expected_direction"],
                    "reference_type": question.get("reference_type", "scalar"),
                }
                downstream = run_multiview_from_detections(
                    image_path=selected_frame,
                    output_root=run_dir / "multiview",
                    qa=qa,
                    question_items=[question_item],
                    detections=build_detection(
                        {
                            "bbox_xyxy": candidate.detection["bbox_xyxy"]
                        }
                    ),
                    crop_paths=[selected_crop],
                    image_output_name=question["question_id"],
                )
                evaluated = downstream["results"][0]
                result_json = load_json(
                    run_dir
                    / "multiview"
                    / question["question_id"]
                    / "gemini_results.json"
                )
                record["gemini_prediction"] = evaluated["predicted"]
                record["correct"] = score_reference(
                    evaluated["predicted"],
                    question["expected_direction"],
                    question.get("reference_type", "scalar"),
                )
                record["logical_gemini_calls"] = 1
                record["api_attempts"] = result_json["gemini_attempts"]

        results.append(record)
        output = {
            "schema_version": SCHEMA_VERSION,
            "video_filename": video.name,
            "policy": policy,
            "stride": stride,
            "selection_only": selection_only,
            "policy_configuration": policy_configuration,
            "cache_path": project_relative(cached_path),
            "fixed_window_frames": FIXED_WINDOW_FRAMES,
            "last_seen_grace_updates": LAST_SEEN_GRACE_UPDATES,
            "ranking": (
                "(visual_quality, average_ocr_confidence, match_score)"
                if policy == "track_lifetime_best_view"
                else "match_score + 10 * average_ocr_confidence"
            ),
            "summary": summarize_policy_results(results),
            "results": results,
        }
        write_json_atomic(result_path, output)

    write_review_csv(
        run_dir / "human_review.csv", video.name, policy, results
    )
    return result_path


def selected_videos(args: argparse.Namespace, qa_entries: list[dict[str, Any]]) -> list[Path]:
    annotated_names = {item["videoPath"] for item in qa_entries}
    if args.video:
        videos = [Path(args.video)]
    elif args.all_videos:
        video_dir = Path(args.video_dir)
        directory = video_dir if video_dir.is_absolute() else PROJECT_ROOT / video_dir
        extensions = {".mov", ".mp4", ".avi", ".mkv"}
        videos = sorted(
            (
                path
                for path in directory.iterdir()
                if path.is_file()
                and path.suffix.lower() in extensions
                and path.name in annotated_names
            ),
            key=lambda path: path.name.lower(),
        )
        if not videos:
            raise ValueError(f"No videos found in: {directory}")
    else:
        raise ValueError("Specify --video or --all-videos.")
    resolved = []
    for video in videos:
        path = video if video.is_absolute() else PROJECT_ROOT / video
        if not path.is_file():
            raise FileNotFoundError(f"Video not found: {path}")
        load_video_questions(qa_entries, path)
        resolved.append(path)
    return resolved


def benchmark_inventory(
    qa_entries: list[dict[str, Any]], video_dir: Path, base_manifest: Path
) -> dict[str, Any]:
    extensions = {".mov", ".mp4", ".avi", ".mkv"}
    files = {
        path.name: path
        for path in video_dir.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    }
    annotated = {item["videoPath"]: item for item in qa_entries}
    missing = sorted(set(annotated) - set(files), key=str.lower)
    unannotated = sorted(set(files) - set(annotated), key=str.lower)
    base_text = str(base_manifest)
    original = [
        item for item in qa_entries if item.get("_manifest_source") == base_text
    ]
    extension = [
        item for item in qa_entries if item.get("_manifest_source") != base_text
    ]
    return {
        "files": files,
        "missing_video_files": missing,
        "unannotated_video_files": unannotated,
        "original_entries": original,
        "extension_entries": extension,
    }


def dry_run(
    args: argparse.Namespace,
    videos: list[Path],
    qa_file: Path,
    qa_entries: list[dict[str, Any]],
    output_root: Path,
) -> None:
    video_dir = Path(args.video_dir)
    if not video_dir.is_absolute():
        video_dir = PROJECT_ROOT / video_dir
    inventory = benchmark_inventory(qa_entries, video_dir, qa_file)
    report = {
        "mode": "dry-run; no model loading, preprocessing, writes, or Gemini calls",
        "action": "prepare_cache" if args.prepare_cache else "run_policy",
        "base_manifest": project_relative(qa_file),
        "extension_manifests": [
            project_relative(path) for path in args.qa_extension
        ],
        "original_completed_videos": len(inventory["original_entries"]),
        "original_completed_questions": sum(
            len(item["questions"]) for item in inventory["original_entries"]
        ),
        "new_annotated_videos": len(inventory["extension_entries"]),
        "new_annotated_questions": sum(
            len(item["questions"]) for item in inventory["extension_entries"]
        ),
        "unannotated_video_files": inventory["unannotated_video_files"],
        "missing_video_files": inventory["missing_video_files"],
        "videos": [],
        "frame_stride": args.frame_stride,
        "policies": list(POLICIES) if args.policy == "all" else [args.policy] if args.policy else [],
        "missing_work_by_stride_and_policy": [],
        "output_files_that_would_be_updated": [],
    }
    audit_new_policy = args.policy in ("track_lifetime_best_view", "all")
    for video in videos:
        questions = question_records(load_video_questions(qa_entries, video))
        cached_path = observations_path(output_root, args.frame_stride, video)
        video_report = {
            "video": video.name,
            "questions": len(questions),
            "present": sum(item["target_present"] for item in questions),
            "absent": sum(not item["target_present"] for item in questions),
            "presence_inferred": sum(
                item["presence_source"] != "qa" for item in questions
            ),
            "cache_path": project_relative(cached_path),
            "cache_exists": cached_path.is_file(),
        }
        if args.prepare_cache:
            cache_action, planned_path = cache_preparation_action(
                video,
                qa_entries,
                output_root,
                args.frame_stride,
                resume=getattr(args, "resume", False),
                overwrite=getattr(args, "overwrite", False),
            )
            video_report["cache_action"] = cache_action
            video_report["cache_path"] = project_relative(planned_path)
        if audit_new_policy and cached_path.is_file():
            cache = load_json(cached_path)
            items = load_video_questions(qa_entries, video)
            expected_questions = [
                {key: value for key, value in item.items() if key != "question_item"}
                for item in question_records(items)
            ]
            expected_configuration = cache_configuration(
                args.frame_stride, video, items
            )
            validate_cache(
                cache, expected_configuration, expected_questions, cached_path
            )
            cache = normalize_loaded_cache(cache)
            audits = [
                evaluate_track_lifetime_best_view(cache, item["question_id"]).audit
                for item in questions
            ]
            bound = [item for item in audits if item["binding_count"]]
            grace_closed = [item for item in bound if item["closed_by_grace"]]
            video_report["track_lifetime_best_view_audit"] = {
                "cache_only": True,
                "destinations_bound": len(bound),
                "bind_once_for_every_retrieved_destination": all(
                    item["binding_count"] == 1 for item in bound
                ),
                "never_switches_tracks": all(
                    item["track_switch_count"] == 0 for item in audits
                ),
                "destinations_with_post_match_visual_updates": sum(
                    item["visual_updates_after_binding"] > 0 for item in bound
                ),
                "destinations_improved_after_initial_candidate": sum(
                    item["candidate_improvements_after_initial"] > 0
                    for item in bound
                ),
                "destinations_selecting_pre_match_best_view": sum(
                    item["selected_before_first_match"] for item in bound
                ),
                "observed_miss_counter_resets": sum(
                    item["miss_counter_resets"] for item in bound
                ),
                "grace_closed_destinations": len(grace_closed),
                "all_grace_closures_at_exactly_8_misses": all(
                    item["misses_at_close"] == LAST_SEEN_GRACE_UPDATES
                    for item in grace_closed
                ),
                "video_end_force_closed_destinations": sum(
                    item["force_closed_at_video_end"] for item in bound
                ),
            }
        report["videos"].append(video_report)

    for video in videos:
        items = load_video_questions(qa_entries, video)
        questions = question_records(items)
        for stride, policy in FINAL_EXPERIMENT_MATRIX:
            cached_path = observations_path(output_root, stride, video)
            result_path = policy_result_path(
                output_root, stride, policy, video
            )
            completed_ids: set[str] = set()
            completed_calls = 0
            if result_path.is_file():
                previous = load_json(result_path)
                previous_results = previous.get("results", [])
                completed_ids = {
                    item["question_id"] for item in previous_results
                }
                completed_calls = sum(
                    int(item.get("logical_gemini_calls", 0))
                    for item in previous_results
                )
            missing_questions = [
                item for item in questions if item["question_id"] not in completed_ids
            ]
            missing_calls: int | str
            if cached_path.is_file():
                cache = load_json(cached_path)
                expected_questions = [
                    {key: value for key, value in item.items() if key != "question_item"}
                    for item in questions
                ]
                validate_cache(
                    cache,
                    cache_configuration(stride, video, items),
                    expected_questions,
                    cached_path,
                )
                cache = normalize_loaded_cache(cache)
                missing_calls = sum(
                    SELECTORS[policy](cache, item["question_id"])[0] is not None
                    for item in missing_questions
                )
            else:
                missing_calls = (
                    f"unknown_until_cache; at most {len(missing_questions)}"
                )
            report["missing_work_by_stride_and_policy"].append(
                {
                    "video": video.name,
                    "stride": stride,
                    "policy": policy,
                    "cache": "reused" if cached_path.is_file() else "required",
                    "completed_predictions_reused": len(completed_ids),
                    "completed_gemini_calls_reused": completed_calls,
                    "missing_questions": len(missing_questions),
                    "missing_gemini_calls": missing_calls,
                    "result_file": project_relative(result_path),
                }
            )
            if missing_questions:
                report["output_files_that_would_be_updated"].append(
                    project_relative(result_path)
                )
                policy_dir = (
                    output_root / "runs" / f"stride_{stride}" / policy
                )
                report["output_files_that_would_be_updated"].extend(
                    [
                        project_relative(policy_dir / "all_results.json"),
                        project_relative(policy_dir / "summary.json"),
                    ]
                )
    report["output_files_that_would_be_updated"] = sorted(
        set(report["output_files_that_would_be_updated"])
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-cache", action="store_true")
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help=(
            "Aggregate existing results from "
            "<output-root>/runs/stride_<frame-stride>; does not run models or Gemini."
        ),
    )
    parser.add_argument("--policy", choices=(*POLICIES, "all"))
    parser.add_argument(
        "--frame-stride",
        "--stride",
        dest="frame_stride",
        type=int,
        choices=(3, 6, 12),
        default=6,
        help=(
            "Process every Nth source frame (default: 6). --stride remains "
            "as a backward-compatible alias."
        ),
    )
    parser.add_argument("--video")
    parser.add_argument("--all-videos", action="store_true")
    parser.add_argument("--video-dir", default="data/test_videos")
    parser.add_argument(
        "--qa-file",
        type=Path,
        default=Path("src/understand/qa_test_set/video_test.json"),
        help="Frozen base recorded-video manifest.",
    )
    parser.add_argument(
        "--qa-extension",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional non-overlapping extension manifest. May be repeated; "
            "duplicate videoPath values are rejected."
        ),
    )
    parser.add_argument(
        "--output-root", default="outputs/video_temporal_ablation/full"
    )
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--limit-targets", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--root", default=".")
    parser.add_argument("--api-key-path", default="keys/gemini_api_key.yaml")
    parser.add_argument("--model-version", default="gemini-3.5-flash")
    parser.add_argument(
        "--prompt-file", default="src/understand/prompts/qa_prompt.txt"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    modes = int(args.prepare_cache) + int(bool(args.policy)) + int(bool(args.aggregate))
    if modes != 1:
        raise ValueError(
            "Choose exactly one of --prepare-cache, --policy, or --aggregate."
        )
    if args.video and args.all_videos:
        raise ValueError("Use either --video or --all-videos, not both.")
    if args.resume and args.overwrite:
        raise ValueError("Use either --resume or --overwrite, not both.")
    if args.limit_targets is not None and args.limit_targets < 1:
        raise ValueError("--limit-targets must be positive.")

    qa_file = Path(args.qa_file)
    extension_files = tuple(Path(path) for path in args.qa_extension)
    output_root = Path(args.output_root)
    qa_entries = load_qa_entries(qa_file, extension_files)
    if args.aggregate:
        if args.video or args.all_videos:
            raise ValueError("--aggregate uses every video in --qa-file.")
        expected_videos = [item["videoPath"] for item in qa_entries]
        stride_dir = output_root / "runs" / f"stride_{args.frame_stride}"
        available_policies = tuple(
            policy
            for policy in POLICIES
            if any((stride_dir / policy).glob("*/per_target_results.json"))
        )
        if not available_policies:
            raise FileNotFoundError(
                f"No completed policy results found under: {stride_dir}"
            )
        aggregate_stride_directory(
            stride_dir,
            available_policies,
            expected_videos,
        )
        return
    video_dir = Path(args.video_dir)
    if not video_dir.is_absolute():
        video_dir = PROJECT_ROOT / video_dir
    inventory = benchmark_inventory(qa_entries, video_dir, qa_file)
    if inventory["unannotated_video_files"]:
        print(
            "Unannotated video files (skipped): "
            + ", ".join(inventory["unannotated_video_files"])
        )
    if inventory["missing_video_files"] and not args.dry_run:
        raise FileNotFoundError(
            "Annotated videos are missing: "
            + ", ".join(inventory["missing_video_files"])
        )
    videos = selected_videos(args, qa_entries)

    if args.dry_run:
        dry_run(args, videos, qa_file, qa_entries, output_root)
        return

    if args.prepare_cache:
        for video in videos:
            action, output = cache_preparation_action(
                video,
                qa_entries,
                output_root,
                args.frame_stride,
                resume=args.resume,
                overwrite=args.overwrite,
            )
            if action == "reuse":
                print(f"Reused compatible cache (unchanged): {output}")
            else:
                output = prepare_cache(
                    video,
                    qa_entries,
                    output_root,
                    args.frame_stride,
                    resume=args.resume,
                    overwrite=args.overwrite,
                )
                print(f"Created cache: {output}")
        return

    policies = POLICIES if args.policy == "all" else (args.policy,)
    for video in videos:
        for policy in policies:
            output = run_policy(
                video,
                qa_entries,
                output_root,
                args.frame_stride,
                policy,
                selection_only=args.selection_only,
                limit_targets=args.limit_targets,
                root=args.root,
                api_key_path=args.api_key_path,
                model_version=args.model_version,
                prompt_file=args.prompt_file,
                resume=args.resume,
                overwrite=args.overwrite,
            )
            print(f"Wrote policy result: {output}")

    aggregate_stride_directory(
        output_root / "runs" / f"stride_{args.frame_stride}",
        policies,
        [video.name for video in videos],
    )


if __name__ == "__main__":
    main()
