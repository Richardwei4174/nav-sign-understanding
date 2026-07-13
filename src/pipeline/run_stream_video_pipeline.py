import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import cv2
from paddleocr import PaddleOCR
from rapidfuzz import fuzz
from ultralytics import YOLOWorld

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(SRC_ROOT))

from src.pipeline.run_multiview_pipeline import run_multiview_from_detections
from src.understand.code.rpi_continuous_testing import GeminiDirectionQA


def load_video_qa(qa_file, video_path):
    with open(qa_file, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    video_name = Path(video_path).name

    for item in qa_data:
        if item["videoPath"] == video_name:
            return item["questions"]

    raise ValueError(f"No QA found for video: {video_name}")


def extract_target_from_question(question_item):
    if "destination" in question_item:
        return question_item["destination"]

    question = question_item["question"].strip()
    question = re.sub(r"^where\s+is\s+", "", question, flags=re.IGNORECASE)
    question = question.rstrip("?").strip()

    return question


def has_digit_target(target):
    return any(ch.isdigit() for ch in target)


def extract_tokens(text):
    return re.findall(r"[A-Za-z]+|\d+", text.lower())


def score_match(target, text):
    target_lower = target.lower().strip()
    text_lower = text.lower().strip()

    target_tokens = extract_tokens(target_lower)
    text_tokens = extract_tokens(text_lower)

    if has_digit_target(target_lower):
        target_numbers = [tok for tok in target_tokens if tok.isdigit()]
        text_numbers = [tok for tok in text_tokens if tok.isdigit()]

        for num in target_numbers:
            if num not in text_numbers:
                return 0

        return fuzz.partial_ratio(target_lower, text_lower)

    return fuzz.partial_ratio(target_lower, text_lower)


def run_ocr(ocr, crop_path):
    ocr_result = ocr.ocr(crop_path, cls=True)

    ocr_lines = []

    if ocr_result and ocr_result[0]:
        for line in ocr_result[0]:
            ocr_lines.append({
                "text": line[1][0],
                "confidence": round(float(line[1][1]), 3),
            })

    return ocr_lines


def get_detection_text(ocr_lines):
    return " ".join([line["text"] for line in ocr_lines]).strip()


def get_avg_confidence(ocr_lines):
    if not ocr_lines:
        return 0.0

    return sum(line["confidence"] for line in ocr_lines) / len(ocr_lines)


def build_detection(candidate):
    return [
        {
            "index": 0,
            "box": candidate["bbox_xyxy"],
            "label": "navigation sign",
            "confidence": 1.0,
        }
    ]


def make_unknown_result(question_item):
    return {
        "question": question_item["question"],
        "expected": question_item["answer"],
        "predicted": "unknown",
        "correct": question_item["answer"] == "unknown",
    }

# save as we go
def save_stream_summary(summary_path, video_path, evaluation_results):
    total_questions_answered = len(evaluation_results)

    total_correct = sum(
        result["correct"] for result in evaluation_results
    )

    accuracy = (
        total_correct / total_questions_answered
        if total_questions_answered > 0 else 0
    )

    summary = {
        "summary": {
            "total_questions": total_questions_answered,
            "total_correct": total_correct,
            "accuracy": accuracy,
        },
        "results": [
            {
                "videoPath": video_path.name,
                "results": evaluation_results,
            }
        ],
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary

def create_target_states(question_items, match_threshold, window_frames):
    target_states = {}

    for i, question_item in enumerate(question_items):
        target = extract_target_from_question(question_item)

        target_states[target] = {
            "target": target,
            "question_item": question_item,
            "question_index": i,
            "match_threshold": match_threshold,
            "window_frames": window_frames,
            "first_match_frame": None,
            "window_end_frame": None,
            "best_candidate": None,
            "candidates": [],
            "status": "searching",
        }

    return target_states


def all_targets_finished(target_states):
    return all(
        state["status"] in ["window_done", "answered"]
        for state in target_states.values()
    )


def update_target_with_detection(
    state,
    frame_idx,
    frame_path,
    crop_path,
    bbox_xyxy,
    ocr_text,
    ocr_lines,
    avg_conf,
    save_debug_images,
    frame,
    crop,
):
    if state["status"] not in ["searching", "window_open"]:
        return

    match_score = score_match(state["target"], ocr_text) if ocr_text else 0

    if match_score < state["match_threshold"]:
        return

    if state["first_match_frame"] is None:
        state["first_match_frame"] = frame_idx
        state["window_end_frame"] = frame_idx + state["window_frames"]
        state["status"] = "window_open"

        print(f"\nFIRST MATCH FOUND for target: {state['target']}")
        print(f"First match frame: {state['first_match_frame']}")
        print(f"Window ends at frame: {state['window_end_frame']}")

    if frame_idx <= state["window_end_frame"]:
        final_score = match_score + (avg_conf * 10)

        candidate = {
            "frame_index": frame_idx,
            "frame_path": frame_path if save_debug_images else None,
            "crop_path": crop_path if save_debug_images else None,
            "frame_image": frame.copy(),
            "crop_image": crop.copy(),
            "bbox_xyxy": bbox_xyxy,
            "ocr_text": ocr_text,
            "ocr": ocr_lines,
            "match_score": match_score,
            "avg_ocr_confidence": avg_conf,
            "final_score": final_score,
        }

        state["candidates"].append(candidate)

        if (
            state["best_candidate"] is None
            or final_score > state["best_candidate"]["final_score"]
        ):
            state["best_candidate"] = candidate
            print(f"  New best candidate for {state['target']}!")


def close_finished_windows(target_states, frame_idx):
    for state in target_states.values():
        if (
            state["status"] == "window_open"
            and state["window_end_frame"] is not None
            and frame_idx > state["window_end_frame"]
        ):
            state["status"] = "window_done"
            print(f"\nFinished window for target: {state['target']}")


def save_selection_for_target(run_dir, target, state):
    safe_target = re.sub(r"[^A-Za-z0-9_-]+", "_", target).strip("_")
    target_dir = run_dir / "selected" / safe_target
    target_dir.mkdir(parents=True, exist_ok=True)

    best_candidate = state["best_candidate"]

    selected_frame_path = target_dir / "best_frame.jpg"
    selected_crop_path = target_dir / "best_crop.jpg"

    if best_candidate["frame_path"] is not None:
        shutil.copy(best_candidate["frame_path"], selected_frame_path)
    else:
        cv2.imwrite(str(selected_frame_path), best_candidate["frame_image"])

    if best_candidate["crop_path"] is not None:
        shutil.copy(best_candidate["crop_path"], selected_crop_path)
    else:
        cv2.imwrite(str(selected_crop_path), best_candidate["crop_image"])

    selection_result = {
        "target": target,
        "match_threshold": state["match_threshold"],
        "window_frames": state["window_frames"],
        "first_match_frame": state["first_match_frame"],
        "selected": {
            "frame_index": best_candidate["frame_index"],
            "frame_path": str(selected_frame_path),
            "crop_path": str(selected_crop_path),
            "bbox_xyxy": best_candidate["bbox_xyxy"],
            "ocr_text": best_candidate["ocr_text"],
            "ocr": best_candidate["ocr"],
            "match_score": best_candidate["match_score"],
            "avg_ocr_confidence": best_candidate["avg_ocr_confidence"],
            "final_score": best_candidate["final_score"],
        },
    }

    selection_path = target_dir / "stream_selection.json"

    with open(selection_path, "w", encoding="utf-8") as f:
        json.dump(selection_result, f, indent=2, ensure_ascii=False)

    return selected_frame_path, selected_crop_path, selection_path

# make sure to make save_debug_images to false when live
def run_stream_video_pipeline(
    video_path,
    output_root,
    qa_file,
    root,
    api_key_path,
    model_version,
    prompt_file,
    match_threshold=70,
    window_frames=5,
    save_debug_images=True,
    ):

    video_path = Path(video_path)
    output_root = Path(output_root)
    video_name = video_path.stem

    run_dir = output_root
    frames_dir = run_dir / "stream_frames"
    crops_dir = run_dir / "stream_crops"

    summary_path = output_root / "stream_summary.json"
    evaluation_results = []
    print(f"run_dir = {run_dir}")
    print(f"output_root = {output_root}")

    print(f"run_dir.exists() = {run_dir.exists()}")
    print(f"run_dir.is_dir() = {run_dir.is_dir()}")
    print(f"run_dir.is_file() = {run_dir.is_file()}")

    output_root.mkdir(parents=True, exist_ok=True)

    if not run_dir.exists():
        run_dir.mkdir()

    (run_dir / "selected").mkdir(exist_ok=True)

    if save_debug_images:
        frames_dir.mkdir(parents=True, exist_ok=True)
        crops_dir.mkdir(parents=True, exist_ok=True)

    save_stream_summary(
        summary_path=summary_path,
        video_path=video_path,
        evaluation_results=evaluation_results,
    )

    question_items = load_video_qa(qa_file, video_path)
    target_states = create_target_states(
        question_items=question_items,
        match_threshold=match_threshold,
        window_frames=window_frames,
    )
    # would take a minute or so to load everythig before seeing this print
    print("\n==============================")
    print("SIMULATED LIVE VIDEO PIPELINE")
    print("==============================")
    print(f"Video: {video_path}")
    print(f"Targets: {list(target_states.keys())}")
    print(f"Questions: {[q['question'] for q in question_items]}")
    print("==============================\n")

    model = YOLOWorld("yolov8s-world.pt")

    classes = [
        "navigation sign",
        "directional sign",
        "wayfinding sign",
        "sign with arrow",
        "directional arrow sign",
        "hallway directional sign",
        "exit sign",
        "",
    ]

    model.set_classes(classes)

    print("Loading PaddleOCR...")
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang="en",
        use_gpu=True,
        show_log=False,
    )

    qa = GeminiDirectionQA(
        root=root,
        api_key_path=api_key_path,
        model_version=model_version,
        prompt_file=prompt_file,
    )

    results = model.predict(
        source=str(video_path),
        conf=0.05,
        iou=0.3,
        agnostic_nms=True,
        vid_stride=6,
        stream=True,
        save=False,
    )

    for frame_idx, result in enumerate(results):
        frame = result.orig_img.copy()
        h, w = frame.shape[:2]

        print(f"\nFrame {frame_idx}: {len(result.boxes)} detections")

        for box_idx, box in enumerate(result.boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]

            frame_path = frames_dir / f"frame_{frame_idx:04d}.jpg"
            crop_path = crops_dir / f"frame_{frame_idx:04d}_box_{box_idx:02d}.jpg"

            if save_debug_images:
                cv2.imwrite(str(frame_path), frame)
                cv2.imwrite(str(crop_path), crop)

            ocr_lines = run_ocr(ocr, crop)
            ocr_text = get_detection_text(ocr_lines)
            avg_conf = get_avg_confidence(ocr_lines)

            print(f"  box {box_idx}: OCR='{ocr_text}' conf={avg_conf:.3f}")

            for target, state in target_states.items():
                update_target_with_detection(
                    state=state,
                    frame_idx=frame_idx,
                    frame_path=frame_path,
                    crop_path=crop_path,
                    bbox_xyxy=[x1, y1, x2, y2],
                    ocr_text=ocr_text,
                    ocr_lines=ocr_lines,
                    avg_conf=avg_conf,
                    save_debug_images=save_debug_images,
                    frame=frame,
                    crop=crop,
                )

        close_finished_windows(target_states, frame_idx)

        if all_targets_finished(target_states):
            print("\nAll targets with matches finished their windows.")
            break

    final_results = []
    found_targets = []
    unknown_targets = []

    for target, state in target_states.items():
        question_item = state["question_item"]

        if state["best_candidate"] is None:
            print(f"\nNo match found for target: {target}. Marking unknown.")

            unknown_result = make_unknown_result(question_item)

            evaluation_results.append(unknown_result)

            save_stream_summary(
                summary_path=summary_path,
                video_path=video_path,
                evaluation_results=evaluation_results,
            )
            final_results.append({
                "target": target,
                "status": "not_found",
                "result": unknown_result,
            })

            unknown_targets.append(target)
            continue

        # print("\n==============================")
        # print(f"BEST STREAM CANDIDATE FOR: {target}")
        # print("==============================")
        best_candidate = state["best_candidate"]
        # print(f"Frame: {best_candidate['frame_index']}")
        # print(f"OCR text: {best_candidate['ocr_text']}")
        # print(f"Match score: {best_candidate['match_score']:.2f}")
        # print(f"OCR confidence: {best_candidate['avg_ocr_confidence']:.3f}")

        selected_frame_path, selected_crop_path, selection_path = save_selection_for_target(
            run_dir=run_dir,
            target=target,
            state=state,
        )

        print(f"Saved selection: {selection_path}")
        print("Calling shared multiview backend...")

        detections = build_detection(best_candidate)

        result = run_multiview_from_detections(
            image_path=selected_frame_path,
            output_root=output_root,
            qa=qa,
            question_items=[question_item],
            detections=detections,
            crop_paths=[selected_crop_path],
            image_output_name=f"{video_name}_{target}",
        )

        if isinstance(result, dict) and "results" in result:
            evaluation_results.extend(result["results"])
        else:
            evaluation_results.append(result)

        save_stream_summary(
            summary_path=summary_path,
            video_path=video_path,
            evaluation_results=evaluation_results,
        )

        found_targets.append(target)

    summary = save_stream_summary(
        summary_path=summary_path,
        video_path=video_path,
        evaluation_results=evaluation_results,
    )

    print("\nSIMULATED LIVE PIPELINE COMPLETE.")
    print(f"Saved summary: {summary_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video",
        default="data/test_videos/IMG_3772.MOV",
        help="Input video path",
    )

    parser.add_argument(
        "--output-root",
        default="outputs/stream_video_pipeline",
        help="Output root",
    )

    parser.add_argument(
        "--qa-file",
        default="src/understand/qa_test_set/rpi_video_test_set.json",
        help="Video QA JSON file",
    )

    parser.add_argument(
        "--root",
        default=".",
        help="Project root used by GeminiDirectionQA",
    )

    parser.add_argument(
        "--api-key-path",
        default="keys/gemini_api_key.yaml",
        help="Path to Gemini API key YAML relative to root",
    )

    parser.add_argument(
        "--model-version",
        default="gemini-3.5-flash",
        help="Gemini model version",
    )

    parser.add_argument(
        "--prompt-file",
        default="src/understand/prompts/qa_prompt.txt",
        help="Prompt file relative to root",
    )

    args = parser.parse_args()

    run_stream_video_pipeline(
        video_path=PROJECT_ROOT / args.video,
        output_root=PROJECT_ROOT / args.output_root,
        qa_file=PROJECT_ROOT / args.qa_file,
        root=args.root,
        api_key_path=args.api_key_path,
        model_version=args.model_version,
        prompt_file=args.prompt_file,
        save_debug_images=False, # false by default
    )