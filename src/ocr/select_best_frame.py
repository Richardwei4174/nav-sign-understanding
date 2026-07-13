from pathlib import Path
import json
import shutil
from rapidfuzz import fuzz
import re

ROOT = Path(__file__).resolve().parents[2]

TARGET = "5200"
MATCH_THRESHOLD = 70

# offline selection. We test it on a video after we annoted the entire video.


# ocr_results.json from yolo_crop_ocr_video.py
# ↓
# Find target
# ↓
# Choose best frame
# YOLO OCR video uses vid_stride=6 on ~30 FPS video so about 5fps

PROCESSED_FPS = 5
WINDOW_SECONDS = 1
WINDOW_FRAMES = PROCESSED_FPS * WINDOW_SECONDS

ocr_json_path = ROOT / "outputs" / "yolo_ocr_video" / "ocr_results.json"

output_root = ROOT / "outputs" / "frame_selection"
candidates_dir = output_root / "candidates"
selected_dir = output_root / "selected"
selection_json_path = output_root / "selection.json"

candidates_dir.mkdir(parents=True, exist_ok=True)
selected_dir.mkdir(parents=True, exist_ok=True)

for old_file in candidates_dir.glob("*"):
    old_file.unlink()

for old_file in selected_dir.glob("*"):
    old_file.unlink()


def get_detection_text(detection):
    ocr_lines = detection.get("ocr", [])
    return " ".join([line.get("text", "") for line in ocr_lines]).strip()


def get_avg_confidence(detection):
    ocr_lines = detection.get("ocr", [])
    if not ocr_lines:
        return 0.0

    confidences = [line.get("confidence", 0.0) for line in ocr_lines]
    return sum(confidences) / len(confidences)


def has_digit_target(target):
    return any(ch.isdigit() for ch in target)


def extract_tokens(text):
    return re.findall(r"[A-Za-z]+|\d+", text.lower())


def score_match(target, text):
    target_lower = target.lower().strip()
    text_lower = text.lower().strip()

    target_tokens = extract_tokens(target_lower)
    text_tokens = extract_tokens(text_lower)

    # If target contains numbers, require exact numeric token match.
    if has_digit_target(target_lower):
        target_numbers = [tok for tok in target_tokens if tok.isdigit()]
        text_numbers = [tok for tok in text_tokens if tok.isdigit()]

        for num in target_numbers:
            if num not in text_numbers:
                return 0

        # If all target numbers match, then use fuzzy score for remaining words.
        return fuzz.partial_ratio(target_lower, text_lower)

    # If target has no numbers, allow fuzzy text matching.
    return fuzz.partial_ratio(target_lower, text_lower)


with open(ocr_json_path, "r", encoding="utf-8") as f:
    ocr_results = json.load(f)

all_matches = []

for frame_record in ocr_results:
    frame_index = frame_record["frame_index"]

    for detection in frame_record.get("detections", []):
        text = get_detection_text(detection)

        if not text:
            continue

        match_score = score_match(TARGET, text)
        avg_conf = get_avg_confidence(detection)

        if match_score >= MATCH_THRESHOLD:
          all_matches.append({
              "frame_index": frame_index,
              "frame_path": detection["frame_path"],
              "crop_path": detection["crop_path"],
              "bbox_xyxy": detection["bbox_xyxy"],
              "ocr_text": text,
              "match_score": match_score,
              "avg_ocr_confidence": avg_conf,
              "final_score": match_score + (avg_conf * 10),
          })

if not all_matches:
    print(f"No matches found for target: {TARGET}")
    exit()

first_match_frame = all_matches[0]["frame_index"]
window_end_frame = first_match_frame + WINDOW_FRAMES

window_matches = [
    match for match in all_matches
    if first_match_frame <= match["frame_index"] <= window_end_frame
]

best_match = max(window_matches, key=lambda x: x["final_score"])

# Save candidate crops from the 1-second window
saved_candidates = []

for idx, match in enumerate(window_matches):
    src_crop = ROOT / match["crop_path"]

    if src_crop.exists():
        dst_crop = candidates_dir / f"candidate_{idx:02d}_frame_{match['frame_index']:04d}.jpg"
        shutil.copy(src_crop, dst_crop)
        match["saved_candidate_path"] = str(dst_crop.relative_to(ROOT))

    saved_candidates.append(match)

# Save best crop
best_crop_src = ROOT / best_match["crop_path"]
best_crop_dst = selected_dir / "best_crop.jpg"

if best_crop_src.exists():
    shutil.copy(best_crop_src, best_crop_dst)

# Save best full frame
best_frame_src = ROOT / best_match["frame_path"]
best_frame_dst = selected_dir / "best_frame.jpg"

if best_frame_src.exists():
    shutil.copy(best_frame_src, best_frame_dst)

selection_result = {
    "target": TARGET,
    "match_threshold": MATCH_THRESHOLD,
    "window_seconds": WINDOW_SECONDS,
    "first_match_frame": first_match_frame,
    "selected": {
        **best_match,
        "saved_best_crop": str(best_crop_dst.relative_to(ROOT)),
        "saved_best_frame": str(best_frame_dst.relative_to(ROOT)),
    },
    "candidates": saved_candidates,
}

with open(selection_json_path, "w", encoding="utf-8") as f:
    json.dump(selection_result, f, indent=2)

print("Best frame selection complete.")
print(f"Target: {TARGET}")
print(f"First match frame: {first_match_frame}")
print(f"Selected frame: {best_match['frame_index']}")
print(f"OCR text: {best_match['ocr_text']}")
print(f"Match score: {best_match['match_score']:.2f}")
print(f"Average OCR confidence: {best_match['avg_ocr_confidence']:.3f}")
print(f"Saved best crop to: {best_crop_dst}")
print(f"Selection JSON saved to: {selection_json_path}")